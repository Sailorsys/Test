package com.master.aistudio

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import java.util.concurrent.atomic.AtomicLong
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * غلاف لحدث يُعالَج مرة واحدة بالضبط، حتى لو أُعيد تسجيل مراقب جديد لاحقاً
 * (مثلاً بعد تدوير الشاشة). يمنع هذا إعادة إظهار رسالة خطأ قديمة بمجرد
 * إعادة رسم الشاشة، بعكس LiveData العادية التي تُعيد بث آخر قيمة لأي
 * مراقب جديد يسجّل نفسه.
 */
class Event<out T>(private val content: T) {
    private var hasBeenHandled = false

    fun getContentIfNotHandled(): T? {
        return if (hasBeenHandled) {
            null
        } else {
            hasBeenHandled = true
            content
        }
    }
}

/**
 * أقصى حافة يعمل عليها pipeline معالجة الماسك (union/close/dilate/feather).
 */
private const val MASK_PROCESSING_MAX_EDGE = 1024

class MainViewModel : ViewModel() {

    private val _resultBitmap = MutableLiveData<Bitmap?>()
    val resultBitmap: LiveData<Bitmap?> = _resultBitmap

    private val _maskBitmap = MutableLiveData<Bitmap?>()
    val maskBitmap: LiveData<Bitmap?> = _maskBitmap

    private val _isLoading = MutableLiveData<Boolean>()
    val isLoading: LiveData<Boolean> = _isLoading

    private var currentMaskFile: File? = null
    private var localSamRunner: MobileSamOnnxRunner? = null

    /**
     * قائمة خطوات التحديد المتراكمة، بالترتيب الزمني. كل عنصر يمثل خطوة
     * واحدة: إما ماسك SAM من نقرة واحدة، أو ماسك من رسمة فرشاة حرة واحدة.
     * الاتحاد (union) النهائي يُبنى من هذه القائمة عند كل rebuild، وهذا ما
     * يجعل الرجوع خطوة بخطوة ممكناً بمجرد إزالة آخر عنصر من القائمة.
     */
    private val maskSteps = mutableListOf<Bitmap>()

    /**
     * الماسك المتراكم النهائي (ناتج union كل الخطوات بعد المعالجة الكاملة).
     * يُخزَّن من الآن فصاعداً على دقة المعالجة المصغّرة (MASK_PROCESSING_MAX_EDGE).
     */
    private var accumulatedRawMask: Bitmap? = null
    private val segmentationMutex = Mutex()
    private val maskGeneration = AtomicLong(0L)

    /**
     * نتيجة تحضير بيانات تأثير البوكيه.
     */
    data class BokehPreparationResult(
        val sourceForGpu: Bitmap,
        val depthForGpu: Bitmap,
        val maskForGpu: Bitmap,
        val focusDepth: Float,
        val autoTuning: AutoTuningEstimator.AutoTuningResult
    )

    private val _bokehResult = MutableLiveData<BokehPreparationResult?>()
    val bokehResult: LiveData<BokehPreparationResult?> = _bokehResult

    private val _bokehError = MutableLiveData<Event<Unit>>()
    val bokehError: LiveData<Event<Unit>> = _bokehError

    private var bokehDepthRunner: DepthAnythingRunner? = null
    private var bokehSelfieRunner: SelfieMaskRunner? = null
    private val bokehAutoTuningEstimator = AutoTuningEstimator()
    private val bokehMutex = Mutex()
    private val bokehGeneration = AtomicLong(0L)

    private fun expansionRadiusFor(width: Int, height: Int): Int {
        return (min(width, height) * 0.009f).roundToInt().coerceIn(4, 18)
    }

    /** أقصى حافة MASK_PROCESSING_MAX_EDGE، بنفس نسبة أبعاد المصدر. */
    private fun computeMaskWorkingSize(width: Int, height: Int): Pair<Int, Int> {
        val maxEdge = MASK_PROCESSING_MAX_EDGE
        return when {
            max(width, height) <= maxEdge -> width to height
            width > height -> maxEdge to (height * maxEdge / width)
            else -> (width * maxEdge / height) to maxEdge
        }
    }

    // تنفيذ الأدوات الأربعة الرئيسية عن طريق WorkManager بدل الطلب المباشر.
    fun startProcessing(
        context: Context,
        tool: String,
        imageBitmap: Bitmap,
        fidelity: Float = 0.7f,
        blurStrength: Int = 25
    ) {
        val imageFile = ImageUtils.bitmapToFile(
            context,
            imageBitmap,
            "work_input_${System.currentTimeMillis()}.jpg"
        )

        val dataBuilder = Data.Builder()
        .putString(ImageProcessingWorker.KEY_TOOL, tool)
        .putString(ImageProcessingWorker.KEY_IMAGE_PATH, imageFile.absolutePath)
        .putFloat(ImageProcessingWorker.KEY_FIDELITY, fidelity)
        .putInt(ImageProcessingWorker.KEY_BLUR, blurStrength)

        if (tool == "REMOVE") {
            currentMaskFile?.let {
                dataBuilder.putString(ImageProcessingWorker.KEY_MASK_PATH, it.absolutePath)
            }
        }

        val request = OneTimeWorkRequestBuilder<ImageProcessingWorker>()
        .setInputData(dataBuilder.build())
        .build()

        WorkManager.getInstance(context).enqueueUniqueWork(
            ImageProcessingWorker.UNIQUE_WORK_NAME,
            ExistingWorkPolicy.REPLACE,
            request
        )
    }

    // كل Tap جديد يضيف قناعه إلى القناع المتراكم بدل استبدال النقرة السابقة.
    fun processSegmentPreview(context: Context, imageBitmap: Bitmap, x: Float, y: Float) {
        runLocalSegmentation(
            context = context,
            imageBitmap = imageBitmap,
            localInference = { getLocalSam(context).predictPoint(imageBitmap, x, y) },
            remoteFallback = {
                val imageFile = ImageUtils.bitmapToFile(context, imageBitmap, "input_seg_fallback.jpg")
                val bytes = NetworkManager.segmentPreview(imageFile, x, y)
                bytes?.let(ImageUtils::bytesToBitmap)
            }
        )
    }

    /**
     * يضيف خطوة رسم حر (فرشاة) إلى قائمة خطوات التحديد.
     *
     * ملاحظة الملكية: هذه الدالة تتولى ملكية drawMask بالكامل. إما تُبقى عليه
     * كما هو (إذا كانت أبعاده مطابقة لدقة المعالجة)، أو تصغّره وتحرر الأصل.
     * لا تقم بـ recycle() خارجياً على drawMask بعد تمريره هنا.
     */
    fun addDrawMaskStep(
        context: Context,
        drawMask: Bitmap,
        imageBitmap: Bitmap
    ) {
        val generation = maskGeneration.get()

        viewModelScope.launch(Dispatchers.Default) {
            segmentationMutex.withLock {
                _isLoading.postValue(true)

                var drawMaskWorking: Bitmap? = null

                try {
                    val (workingWidth, workingHeight) =
                    computeMaskWorkingSize(
                        imageBitmap.width,
                        imageBitmap.height
                    )

                    drawMaskWorking =
                    if (
                        drawMask.width == workingWidth &&
                        drawMask.height == workingHeight
                    ) {
                        drawMask
                    } else {
                        Bitmap.createScaledBitmap(
                            drawMask,
                            workingWidth,
                            workingHeight,
                            true
                        ).also {
                            if (
                                it !== drawMask &&
                                !drawMask.isRecycled
                            ) {
                                drawMask.recycle()
                            }
                        }
                    }

                    if (generation != maskGeneration.get()) {
                        if (!drawMaskWorking.isRecycled) {
                            drawMaskWorking.recycle()
                        }
                        drawMaskWorking = null
                        return@withLock
                    }

                    maskSteps.add(drawMaskWorking)
                    drawMaskWorking = null

                    rebuildMaskPipeline(
                        context,
                        imageBitmap.width,
                        imageBitmap.height,
                        generation
                    )

                } catch (error: Exception) {
                    error.printStackTrace()

                    drawMaskWorking?.let {
                        if (!it.isRecycled) {
                            it.recycle()
                        }
                    }

                } finally {
                    _isLoading.postValue(false)
                }
            }
        }
    }

    /**
     * يحذف آخر خطوة تحديد (نقرة SAM أو رسمة فرشاة) ويعيد بناء pipeline
     * الماسك من الخطوات الباقية. لو لم تعد هناك خطوات، يُمسح كل شيء.
     */
    fun undoMaskStep(context: Context, imageBitmap: Bitmap) {
        viewModelScope.launch(Dispatchers.Default) {
            segmentationMutex.withLock {
                if (maskSteps.isEmpty()) {
                    return@withLock
                }

                val removed = maskSteps.removeAt(maskSteps.size - 1)
                if (!removed.isRecycled) {
                    removed.recycle()
                }

                try {
                    rebuildMaskPipeline(
                        context,
                        imageBitmap.width,
                        imageBitmap.height,
                        maskGeneration.get()
                    )
                } catch (error: Exception) {
                    error.printStackTrace()
                }
            }
        }
    }

    private fun runLocalSegmentation(
        context: Context,
        imageBitmap: Bitmap,
        localInference: () -> Bitmap,
        remoteFallback: suspend () -> Bitmap?
    ) {
        val generation = maskGeneration.get()

        viewModelScope.launch(Dispatchers.Default) {
            segmentationMutex.withLock {
                _isLoading.postValue(true)

                var newMaskFullRes: Bitmap? = null
                var newMaskWorking: Bitmap? = null
                

                try {
                    newMaskFullRes = try {
                        localInference()
                    } catch (localError: Exception) {
                        localError.printStackTrace()

                        withContext(Dispatchers.IO) {
                            remoteFallback()
                        } ?: throw localError
                    }

                    // الصورة/جلسة الـ Mask تغيّرت أثناء الـ inference.
                    if (generation != maskGeneration.get()) {
                        newMaskFullRes?.let {
                            if (!it.isRecycled) {
                                it.recycle()
                            }
                        }
                        newMaskFullRes = null
                        return@withLock
                    }

                    val (workingWidth, workingHeight) =
                    computeMaskWorkingSize(
                        imageBitmap.width,
                        imageBitmap.height
                    )

                    newMaskWorking =
                    if (
                        newMaskFullRes!!.width == workingWidth &&
                        newMaskFullRes!!.height == workingHeight
                    ) {
                        newMaskFullRes
                    } else {
                        

                        Bitmap.createScaledBitmap(
                            newMaskFullRes!!,
                            workingWidth,
                            workingHeight,
                            true
                        ).also {
                            if (
                                it !== newMaskFullRes &&
                                !newMaskFullRes!!.isRecycled
                            ) {
                                newMaskFullRes!!.recycle()
                            }

                            newMaskFullRes = null
                        }
                    }

                    // تحقق ثاني قبل نقل الملكية إلى maskSteps.
                    if (generation != maskGeneration.get()) {
                        newMaskWorking?.let {
                            if (!it.isRecycled) {
                                it.recycle()
                            }
                        }
                        newMaskWorking = null
                        return@withLock
                    }

                    maskSteps.add(newMaskWorking!!)
                    newMaskWorking = null
                    

                    rebuildMaskPipeline(
                        context = context,
                        imageWidth = imageBitmap.width,
                        imageHeight = imageBitmap.height,
                        expectedGeneration = generation
                    )

                } catch (error: Exception) {
                    error.printStackTrace()

                    newMaskWorking?.let {
                        if (!it.isRecycled) {
                            it.recycle()
                        }
                    }

                    newMaskFullRes?.let {
                        if (!it.isRecycled) {
                            it.recycle()
                        }
                    }

                    // لا نعيد إظهار Mask قديم إذا أصبحت العملية stale.
                    if (generation == maskGeneration.get()) {
                        _maskBitmap.postValue(
                            accumulatedRawMask?.let { working ->

                                var closed: Bitmap? = null
                                var expanded: Bitmap? = null
                                var feathered: Bitmap? = null
                                var preview: Bitmap? = null

                                try {
                                    closed = ImageUtils.closeMask(
                                        working,
                                        3
                                    )

                                    expanded = ImageUtils.dilateMask(
                                        closed,
                                        expansionRadiusFor(
                                            working.width,
                                            working.height
                                        )
                                    )

                                    feathered = ImageUtils.featherMask(
                                        expanded,
                                        5f
                                    )

                                    preview =
                                    Bitmap.createScaledBitmap(
                                        feathered,
                                        imageBitmap.width,
                                        imageBitmap.height,
                                        true
                                    )

                                    preview
                                } finally {

                                    closed?.let {
                                        if (!it.isRecycled) {
                                            it.recycle()
                                        }
                                    }

                                    expanded?.let {
                                        if (!it.isRecycled) {
                                            it.recycle()
                                        }
                                    }

                                    feathered?.let {
                                        if (
                                            it !== preview &&
                                            !it.isRecycled
                                        ) {
                                            it.recycle()
                                        }
                                    }
                                }
                            }
                        )
                    }

                } finally {
                    newMaskWorking?.let {
                        if (!it.isRecycled) {
                            it.recycle()
                        }
                    }

                    newMaskFullRes?.let {
                        if (!it.isRecycled) {
                            it.recycle()
                        }
                    }

                    _isLoading.postValue(false)
                }
            }
        }
    }

    /**
     * يعيد بناء pipeline الماسك من الصفر اعتماداً على الخطوات الحالية:
     * union كل الخطوات → close → dilate → feather → حفظ الملف + بث المعاينة.
     *
     * يجب استدعاؤها فقط من داخل segmentationMutex.withLock.
     */
    /**
 * يعيد بناء pipeline الماسك من الصفر اعتماداً على الخطوات الحالية:
 * union كل الخطوات → close → dilate → feather → حفظ الملف + بث المعاينة.
 *
 * يجب استدعاؤها فقط من داخل segmentationMutex.withLock.
 *
 * Ownership:
 * - maskSteps: مملوكة للـ ViewModel ولا تُلمس هنا.
 * - accumulatedRawMask: مملوكة للـ ViewModel.
 * - كل الـ intermediate Bitmaps مملوكة لهذه الدالة وتُحرر قبل الخروج.
 * - previewMaskFull تنتقل ملكيتها إلى MainActivity عبر LiveData، لذلك لا نحررها هنا.
 */
    private suspend fun rebuildMaskPipeline(
        context: Context,
        imageWidth: Int,
        imageHeight: Int,
        expectedGeneration: Long
    ) {
        if (expectedGeneration != maskGeneration.get()) {
            return
        }

        val (workingWidth, workingHeight) =
        computeMaskWorkingSize(
            imageWidth,
            imageHeight
        )

        if (maskSteps.isEmpty()) {
            accumulatedRawMask?.let {
                if (!it.isRecycled) {
                    it.recycle()
                }
            }

            accumulatedRawMask = null

            currentMaskFile?.delete()
            currentMaskFile = null

            if (expectedGeneration == maskGeneration.get()) {
                _maskBitmap.postValue(null)
            }

            return
        }

        var combined: Bitmap? = null
        var closedMask: Bitmap? = null
        var expandedMaskWorking: Bitmap? = null
        var previewMaskWorking: Bitmap? = null
        var expandedMaskFull: Bitmap? = null
        var previewMaskFull: Bitmap? = null

        try {

            for (step in maskSteps) {

                val previousCombined = combined

                combined = ImageUtils.unionMasks(
                    previousCombined,
                    step
                )

                if (
                    previousCombined != null &&
                    previousCombined !== combined &&
                    !previousCombined.isRecycled
                ) {
                    previousCombined.recycle()
                }
            }

            val combinedRawMask = combined
            ?: return

            combined = null

            if (expectedGeneration != maskGeneration.get()) {
                if (!combinedRawMask.isRecycled) {
                    combinedRawMask.recycle()
                }
                return
            }

            val previousAccumulated = accumulatedRawMask

            accumulatedRawMask = combinedRawMask

            if (
                previousAccumulated != null &&
                previousAccumulated !== combinedRawMask &&
                !previousAccumulated.isRecycled
            ) {
                previousAccumulated.recycle()
            }

            closedMask = ImageUtils.closeMask(
                combinedRawMask,
                3
            )

            expandedMaskWorking = ImageUtils.dilateMask(
                closedMask,
                expansionRadiusFor(
                    workingWidth,
                    workingHeight
                )
            )

            previewMaskWorking = ImageUtils.featherMask(
                expandedMaskWorking,
                5f
            )

            expandedMaskFull = Bitmap.createScaledBitmap(
                expandedMaskWorking,
                imageWidth,
                imageHeight,
                false
            )

            previewMaskFull = Bitmap.createScaledBitmap(
                previewMaskWorking,
                imageWidth,
                imageHeight,
                true
            )

            if (expectedGeneration != maskGeneration.get()) {
                return
            }

            currentMaskFile?.delete()

            currentMaskFile = ImageUtils.maskBitmapToFile(
                context.applicationContext,
                expandedMaskFull,
                "current_mask.png"
            )

            if (expectedGeneration == maskGeneration.get()) {
                _maskBitmap.postValue(previewMaskFull)
                previewMaskFull = null
            }

        } finally {

            combined?.let {
                if (!it.isRecycled) {
                    it.recycle()
                }
            }

            closedMask?.let {
                if (!it.isRecycled) {
                    it.recycle()
                }
            }

            expandedMaskWorking?.let {
                if (!it.isRecycled) {
                    it.recycle()
                }
            }

            previewMaskWorking?.let {
                if (
                    it !== previewMaskFull &&
                    !it.isRecycled
                ) {
                    it.recycle()
                }
            }

            expandedMaskFull?.let {
                if (!it.isRecycled) {
                    it.recycle()
                }
            }

            previewMaskFull?.let {
                if (!it.isRecycled) {
                    it.recycle()
                }
            }
        }
    }

    /**
     * يمسح كل خطوات التحديد. تُنفَّذ داخل نفس الـ mutex المستخدم في
     * pipeline لتفادي أي race مع عمليات rebuild الجارية.
     */
    /**
 * يمسح كل خطوات التحديد.
 *
 * Ownership:
 * - maskSteps: يحررها ViewModel.
 * - accumulatedRawMask: يحرره ViewModel.
 * - previewMaskBitmap المنشور عبر LiveData: لا يحرره ViewModel،
 *   لأن MainActivity أصبحت مالكته بعد استلامه.
 */
    fun clearMask() {

        // إبطال أي Segmentation/Rebuild قديم فورًا.
        val newGeneration = maskGeneration.incrementAndGet()

        viewModelScope.launch(Dispatchers.Default) {
            segmentationMutex.withLock {

                // لو بدأت جلسة أحدث بعد هذه العملية،
                // لا نلمس حالتها.
                if (newGeneration != maskGeneration.get()) {
                    return@withLock
                }

                maskSteps.forEach { step ->
                    if (!step.isRecycled) {
                        step.recycle()
                    }
                }
                maskSteps.clear()

                accumulatedRawMask?.let {
                    if (!it.isRecycled) {
                        it.recycle()
                    }
                }
                accumulatedRawMask = null

                currentMaskFile?.delete()
                currentMaskFile = null

                _maskBitmap.postValue(null)
            }
        }
    }

    fun prepareBokehEffect(context: Context, sourceBitmap: Bitmap) {
        val generation = bokehGeneration.incrementAndGet()

        viewModelScope.launch(Dispatchers.Default) {
            bokehMutex.withLock {
                _isLoading.postValue(true)

                var depthBitmap: Bitmap? = null
                var maskBitmap: Bitmap? = null
                var sourceForGpu: Bitmap? = null
                var depthForGpu: Bitmap? = null
                var maskForGpu: Bitmap? = null

                try {
                    val appContext = context.applicationContext

                    depthBitmap = getBokehDepthRunner(appContext)
                    .run(sourceBitmap)

                    maskBitmap = getBokehSelfieRunner()
                    .process(appContext, sourceBitmap)

                    val gpuSize = scaleForBokehGpu(sourceBitmap)

                    sourceForGpu = Bitmap.createScaledBitmap(
                        sourceBitmap,
                        gpuSize.first,
                        gpuSize.second,
                        true
                    )

                    depthForGpu = Bitmap.createScaledBitmap(
                        depthBitmap,
                        gpuSize.first,
                        gpuSize.second,
                        true
                    )

                    maskForGpu = Bitmap.createScaledBitmap(
                        maskBitmap,
                        gpuSize.first,
                        gpuSize.second,
                        true
                    )

                    depthBitmap.recycle()
                    depthBitmap = null

                    maskBitmap.recycle()
                    maskBitmap = null

                    val focusDepth = computeBokehFocusDepth(
                        depthForGpu,
                        maskForGpu
                    )

                    val autoTuning = bokehAutoTuningEstimator.estimate(
                        sourceForGpu,
                        depthForGpu,
                        maskForGpu,
                        focusDepth
                    )

                    val result = BokehPreparationResult(
                        sourceForGpu = sourceForGpu,
                        depthForGpu = depthForGpu,
                        maskForGpu = maskForGpu,
                        focusDepth = focusDepth,
                        autoTuning = autoTuning
                    )

                    /*
                 * مهم:
                 * النشر نفسه يتم على Main thread.
                 *
                 * بذلك يصبح فحص generation و clearBokehResult()
                 * متسلسلين على نفس الـ thread، فلا توجد نافذة race
                 * بين "الفحص" و"النشر".
                 */
                    withContext(Dispatchers.Main.immediate) {

                        if (generation != bokehGeneration.get()) {

                            // النتيجة أصبحت قديمة.
                            // لم تنتقل ملكيتها إلى Activity.
                            if (!result.sourceForGpu.isRecycled) {
                                result.sourceForGpu.recycle()
                            }

                            if (!result.depthForGpu.isRecycled) {
                                result.depthForGpu.recycle()
                            }

                            if (!result.maskForGpu.isRecycled) {
                                result.maskForGpu.recycle()
                            }

                        } else {

                            _bokehResult.value = result

                            /*
                         * Ownership transferred to BokehPreparationResult
                         * / MainActivity.
                         */
                            sourceForGpu = null
                            depthForGpu = null
                            maskForGpu = null
                        }
                    }

                } catch (error: Exception) {
                    error.printStackTrace()

                    withContext(Dispatchers.Main.immediate) {
                        if (generation == bokehGeneration.get()) {
                            _bokehError.value = Event(Unit)
                        }
                    }

                } finally {

                    depthBitmap?.let {
                        if (!it.isRecycled) {
                            it.recycle()
                        }
                    }

                    maskBitmap?.let {
                        if (!it.isRecycled) {
                            it.recycle()
                        }
                    }

                    sourceForGpu?.let {
                        if (!it.isRecycled) {
                            it.recycle()
                        }
                    }

                    depthForGpu?.let {
                        if (!it.isRecycled) {
                            it.recycle()
                        }
                    }

                    maskForGpu?.let {
                        if (!it.isRecycled) {
                            it.recycle()
                        }
                    }

                    _isLoading.postValue(false)
                }
            }
        }
    }

    /** يمسح آخر نتيجة بوكيه محضَّرة، دون حذف الـBitmaps نفسها. */
    fun clearBokehResult() {
        /*
     * إبطال أي Bokeh computation ما زال يعمل.
     *
     * أي نتيجة تنتهي بعد هذا السطر ستُعتبر stale
     * ولن يتم نشرها إلى Activity.
     */
        bokehGeneration.incrementAndGet()

        _bokehResult.value = null
    }

    private fun getBokehDepthRunner(context: Context): DepthAnythingRunner {
        return bokehDepthRunner ?: synchronized(this) {
            bokehDepthRunner ?: DepthAnythingRunner(context).also {
                bokehDepthRunner = it
            }
        }
    }

    private fun getBokehSelfieRunner(): SelfieMaskRunner {
        return bokehSelfieRunner ?: synchronized(this) {
            bokehSelfieRunner ?: SelfieMaskRunner().also {
                bokehSelfieRunner = it
            }
        }
    }

    /** أقصى حافة 2000 بكسل، بنفس نسبة أبعاد الصورة الأصلية. */
    private fun scaleForBokehGpu(bitmap: Bitmap): Pair<Int, Int> {
        val maxEdge = 2000
        val width = bitmap.width
        val height = bitmap.height

        return when {
            max(width, height) <= maxEdge -> width to height
            width > height -> maxEdge to (height * maxEdge / width)
            else -> (width * maxEdge / height) to maxEdge
        }
    }

    private fun getLocalSam(context: Context): MobileSamOnnxRunner {
        return localSamRunner ?: synchronized(this) {
            localSamRunner ?: MobileSamOnnxRunner(
                context.applicationContext
            ).also {
                localSamRunner = it
            }
        }
    }

    private fun computeBokehFocusDepth(depth: Bitmap, mask: Bitmap): Float {
        val width = depth.width
        val height = depth.height

        if (width <= 0 || height <= 0) {
            return 0.5f
        }

        val depthPixels = IntArray(width * height)
        val maskPixels = IntArray(width * height)

        depth.getPixels(depthPixels, 0, width, 0, 0, width, height)
        mask.getPixels(maskPixels, 0, width, 0, 0, width, height)

        val values = ArrayList<Float>()
        val step = 4

        for (y in 0 until height step step) {
            val rowOffset = y * width

            for (x in 0 until width step step) {
                val index = rowOffset + x

                if (Color.red(maskPixels[index]) > 170) {
                    values.add(Color.red(depthPixels[index]) / 255f)
                }
            }
        }

        if (values.isEmpty()) {
            return 0.5f
        }

        values.sort()

        return values[values.size / 2]
    }

    override fun onCleared() {
        localSamRunner?.close()
        localSamRunner = null

        maskSteps.forEach { step ->
            if (!step.isRecycled) {
                step.recycle()
            }
        }
        maskSteps.clear()

        accumulatedRawMask?.let {
            if (!it.isRecycled) {
                it.recycle()
            }
        }
        accumulatedRawMask = null

        bokehDepthRunner?.close()
        bokehDepthRunner = null

        bokehSelfieRunner?.close()
        bokehSelfieRunner = null

        currentMaskFile = null
        super.onCleared()
    }
}
