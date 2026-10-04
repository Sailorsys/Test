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
    fun addDrawMaskStep(context: Context, drawMask: Bitmap, imageBitmap: Bitmap) {
        viewModelScope.launch(Dispatchers.Default) {
            segmentationMutex.withLock {
                _isLoading.postValue(true)
                try {
                    val (workingWidth, workingHeight) = computeMaskWorkingSize(
                        imageBitmap.width,
                        imageBitmap.height
                    )

                    val drawMaskWorking = if (
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
                            if (it !== drawMask && !drawMask.isRecycled) {
                                drawMask.recycle()
                            }
                        }
                    }

                    maskSteps.add(drawMaskWorking)
                    rebuildMaskPipeline(context, imageBitmap.width, imageBitmap.height)
                } catch (error: Exception) {
                    error.printStackTrace()
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
                    rebuildMaskPipeline(context, imageBitmap.width, imageBitmap.height)
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
        viewModelScope.launch(Dispatchers.Default) {
            segmentationMutex.withLock {
                _isLoading.postValue(true)
                try {
                    val newMaskFullRes = try {
                        localInference()
                    } catch (localError: Exception) {
                        localError.printStackTrace()
                        withContext(Dispatchers.IO) { remoteFallback() }
                        ?: throw localError
                    }

                    val (workingWidth, workingHeight) = computeMaskWorkingSize(
                        imageBitmap.width,
                        imageBitmap.height
                    )

                    val newMaskWorking = if (
                        newMaskFullRes.width == workingWidth &&
                        newMaskFullRes.height == workingHeight
                    ) {
                        newMaskFullRes
                    } else {
                        Bitmap.createScaledBitmap(
                            newMaskFullRes,
                            workingWidth,
                            workingHeight,
                            true
                        ).also {
                            if (it !== newMaskFullRes && !newMaskFullRes.isRecycled) {
                                newMaskFullRes.recycle()
                            }
                        }
                    }

                    maskSteps.add(newMaskWorking)
                    rebuildMaskPipeline(context, imageBitmap.width, imageBitmap.height)
                } catch (error: Exception) {
                    error.printStackTrace()
                    _maskBitmap.postValue(accumulatedRawMask?.let { working ->
                            val closed = ImageUtils.closeMask(working, 3)
                            val expanded = ImageUtils.dilateMask(
                                closed,
                                expansionRadiusFor(working.width, working.height)
                            )
                            val feathered = ImageUtils.featherMask(expanded, 5f)
                            Bitmap.createScaledBitmap(
                                feathered,
                                imageBitmap.width,
                                imageBitmap.height,
                                true
                            )
                    })
                } finally {
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
    private suspend fun rebuildMaskPipeline(
        context: Context,
        imageWidth: Int,
        imageHeight: Int
    ) {
        val (workingWidth, workingHeight) = computeMaskWorkingSize(imageWidth, imageHeight)

        if (maskSteps.isEmpty()) {
            accumulatedRawMask = null
            currentMaskFile?.delete()
            currentMaskFile = null
            _maskBitmap.postValue(null)
            return
        }

        var combined: Bitmap? = null
        for (step in maskSteps) {
            combined = ImageUtils.unionMasks(combined, step)
        }

        val combinedRawMask = combined ?: run {
            accumulatedRawMask = null
            currentMaskFile?.delete()
            currentMaskFile = null
            _maskBitmap.postValue(null)
            return
        }

        accumulatedRawMask = combinedRawMask

        // 1. Closing صغير (radius=3) يملأ الفجوات الداخلية في القناع.
        val closedMask = ImageUtils.closeMask(combinedRawMask, 3)

        // 2. Dilation رئيسي يوسع القناع عشان يشمل حواف العنصر الخارجية.
        val expandedMaskWorking = ImageUtils.dilateMask(
            closedMask,
            expansionRadiusFor(workingWidth, workingHeight)
        )

        // 3. Feathering على الدقة المصغّرة (رخيص جداً).
        val previewMaskWorking = ImageUtils.featherMask(expandedMaskWorking, 5f)

        // التكبير النهائي فقط يحدث على دقة الصورة الأصلية.
        val expandedMaskFull = Bitmap.createScaledBitmap(
            expandedMaskWorking,
            imageWidth,
            imageHeight,
            false
        )

        val previewMaskFull = Bitmap.createScaledBitmap(
            previewMaskWorking,
            imageWidth,
            imageHeight,
            true
        )

        // 4. ملف القناع المرسل إلى Big LaMa: binary mask بدون feathering.
        currentMaskFile?.delete()
        currentMaskFile = ImageUtils.maskBitmapToFile(
            context.applicationContext,
            expandedMaskFull,
            "current_mask.png"
        )
        _maskBitmap.postValue(previewMaskFull)
    }

    private fun getLocalSam(context: Context): MobileSamOnnxRunner {
        return localSamRunner ?: synchronized(this) {
            localSamRunner ?: MobileSamOnnxRunner(context.applicationContext).also {
                localSamRunner = it
            }
        }
    }

    /**
     * يمسح كل خطوات التحديد. تُنفَّذ داخل نفس الـ mutex المستخدم في
     * pipeline لتفادي أي race مع عمليات rebuild الجارية.
     */
    fun clearMask() {
        viewModelScope.launch(Dispatchers.Default) {
            segmentationMutex.withLock {
                maskSteps.forEach { step ->
                    if (!step.isRecycled) step.recycle()
                }
                maskSteps.clear()
                accumulatedRawMask = null
                currentMaskFile?.delete()
                currentMaskFile = null
                _maskBitmap.postValue(null)
            }
        }
    }

    fun prepareBokehEffect(context: Context, sourceBitmap: Bitmap) {
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

                    depthBitmap = getBokehDepthRunner(appContext).run(sourceBitmap)
                    maskBitmap = getBokehSelfieRunner().process(appContext, sourceBitmap)

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

                    _bokehResult.postValue(
                        BokehPreparationResult(
                            sourceForGpu = sourceForGpu,
                            depthForGpu = depthForGpu,
                            maskForGpu = maskForGpu,
                            focusDepth = focusDepth,
                            autoTuning = autoTuning
                        )
                    )

                    // Ownership transferred to BokehPreparationResult.
                    sourceForGpu = null
                    depthForGpu = null
                    maskForGpu = null

                } catch (error: Exception) {
                    error.printStackTrace()
                    _bokehError.postValue(Event(Unit))

                } finally {
                    depthBitmap?.let {
                        if (!it.isRecycled) it.recycle()
                    }

                    maskBitmap?.let {
                        if (!it.isRecycled) it.recycle()
                    }

                    sourceForGpu?.let {
                        if (!it.isRecycled) it.recycle()
                    }

                    depthForGpu?.let {
                        if (!it.isRecycled) it.recycle()
                    }

                    maskForGpu?.let {
                        if (!it.isRecycled) it.recycle()
                    }

                    _isLoading.postValue(false)
                }
            }
        }
    }

    /** يمسح آخر نتيجة بوكيه محضَّرة، دون حذف الـBitmaps نفسها. */
    fun clearBokehResult() {
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
            if (!step.isRecycled) step.recycle()
        }
        maskSteps.clear()
        accumulatedRawMask = null

        bokehDepthRunner?.close()
        bokehDepthRunner = null

        bokehSelfieRunner?.close()
        bokehSelfieRunner = null

        currentMaskFile = null
        super.onCleared()
    }
}
