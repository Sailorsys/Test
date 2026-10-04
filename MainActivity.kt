package com.master.aistudio

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.view.View
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.work.WorkInfo
import androidx.work.WorkManager
import com.master.aistudio.utils.NativeSmartEnhancer
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.ChipGroup
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.google.android.material.slider.Slider
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtProvider

import java.io.File
import java.util.ArrayDeque
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {

    private enum class Tool {
        FACE,
        FULL,
        REMOVE,
        PORTRAIT,
        AUTO_ENHANCE
    }

    private lateinit var viewModel: MainViewModel
    private lateinit var touchImageView: InteractiveTouchImageView
    private lateinit var beforeAfterSlider: BeforeAfterSliderView
    private lateinit var depthBlurView: DepthBlurGLView
    private lateinit var layoutEmptyState: View
    private lateinit var btnAddImageCenter: MaterialButton
    private lateinit var btnRemoveImage: MaterialButton
    private lateinit var btnSaveImage: MaterialButton
    private lateinit var btnUndo: MaterialButton
    private lateinit var progressBar: LinearProgressIndicator
    private lateinit var tvStatus: TextView
    private lateinit var tvRemoveHint: TextView
    private lateinit var btnExecute: MaterialButton
    private lateinit var btnClearMask: MaterialButton
    private lateinit var toolChipGroup: ChipGroup
    private lateinit var blurSeekBar: Slider
    private lateinit var blurValueText: TextView

    // --- ضوابط وضع الإزالة (نقر / رسم / رجوع خطوة) ---
    private lateinit var maskControlsRow: LinearLayout
    private lateinit var radioGroupMaskMode: RadioGroup
    private lateinit var radioMaskClick: RadioButton
    private lateinit var radioMaskDraw: RadioButton
    private lateinit var btnMaskUndo: MaterialButton

    private var selectedBitmap: Bitmap? = null
    private var beforeBitmap: Bitmap? = null
    private var selectedTool = Tool.FACE
    private var hasMask = false
    private var displayedMaskBitmap: Bitmap? = null

    private var currentBokehResult: MainViewModel.BokehPreparationResult? = null

    private val historyStack = ArrayDeque<Bitmap>()

    private val estimatedDurationFace = 20_000L
    private val estimatedDurationFull = 45_000L
    private val estimatedDurationRemove = 14_000L
    private val estimatedDurationSegment = 1_500L
    private val estimatedDurationPortrait = 6_000L
    private val estimatedDurationAutoEnhance = 10_000L

    private val progressHandler = Handler(Looper.getMainLooper())
    private var progressStartTime = 0L
    private var estimatedDurationMs = 8_000L
    private var baseStatusMessage = ""
    private var progressRunnable: Runnable? = null

    private fun startFakeProgress(estimatedMs: Long, message: String) {
        baseStatusMessage = message
        stopFakeProgress()
        progressBar.isIndeterminate = false
        progressBar.max = 100
        progressStartTime = System.currentTimeMillis()
        estimatedDurationMs = estimatedMs
        progressRunnable =
        object : Runnable {
            override fun run() {
                val elapsed = System.currentTimeMillis() - progressStartTime
                val percent = ((elapsed.toFloat() / estimatedDurationMs) * 95).toInt().coerceIn(0, 95)
                progressBar.progress = percent
                tvStatus.text = "$baseStatusMessage ($percent%)"
                progressHandler.postDelayed(this, 200)
            }
        }
        progressHandler.post(progressRunnable!!)
    }

    private fun stopFakeProgress(completed: Boolean = false) {
        progressRunnable?.let { progressHandler.removeCallbacks(it) }
        progressRunnable = null
        if (completed) {
            progressBar.progress = 100
        }
    }

    private val notificationPermissionLauncher =
    registerForActivityResult(ActivityResultContracts.RequestPermission()) {}

    private fun requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val granted =
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) ==
            PackageManager.PERMISSION_GRANTED
            if (!granted) {
                notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
    }

    private fun releaseDisplayedMask() {
        val oldMask = displayedMaskBitmap
        displayedMaskBitmap = null

        if (oldMask != null && !oldMask.isRecycled) {
            oldMask.recycle()
        }
    }

    private fun clearDisplayedMask() {
        touchImageView.detachMaskBitmap()
        releaseDisplayedMask()
        hasMask = false
    }

    private val pickImageLauncher =
    registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { selectedUri ->
            progressBar.isIndeterminate = true
            progressBar.visibility = View.VISIBLE
            btnAddImageCenter.isEnabled = false
            tvStatus.text = "جاري تحميل الصورة..."

            lifecycleScope.launch {
                val bitmap = withContext(Dispatchers.IO) {
                    loadBitmapFromUri(selectedUri)
                }

                progressBar.isIndeterminate = false
                progressBar.visibility = View.GONE
                btnAddImageCenter.isEnabled = true

                bitmap?.let { b ->
                    clearHistoryStack()

                    // Release everything belonging to the previous image
                    // before installing the new image.
                    clearDisplayedMask()
                    viewModel.clearMask()
                    releaseCurrentBokehResult()

                    selectedBitmap = b
                    beforeBitmap = b

                    touchImageView.setImageBitmap(b)
                    touchImageView.clearMask()

                    hasMask = false
                    radioMaskClick.isChecked = true
                    touchImageView.setTouchMode(
                        InteractiveTouchImageView.TouchMode.CLICK
                    )

                    switchToEditMode()
                    updateUiState()
                    tvStatus.text = "الصورة جاهزة، اختر الأداة واضغط تنفيذ"
                } ?: run {
                    showToast("فشل تحميل الصورة")
                    tvStatus.text = "تحسين صورك بلمسة ذكية"
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        WorkManager.getInstance(this).pruneWork()

        viewModel = ViewModelProvider(this)[MainViewModel::class.java]

        touchImageView = findViewById(R.id.touchImageView)
        beforeAfterSlider = findViewById(R.id.beforeAfterSlider)
        depthBlurView = findViewById(R.id.depthBlurView)
        layoutEmptyState = findViewById(R.id.layoutEmptyState)
        btnAddImageCenter = findViewById(R.id.btnAddImageCenter)
        btnRemoveImage = findViewById(R.id.btnRemoveImage)
        btnSaveImage = findViewById(R.id.btnSaveImage)
        btnUndo = findViewById(R.id.btnUndo)
        progressBar = findViewById(R.id.progressBar)
        tvStatus = findViewById(R.id.tvStatus)
        tvRemoveHint = findViewById(R.id.tvRemoveHint)
        btnExecute = findViewById(R.id.btnExecute)
        btnClearMask = findViewById(R.id.btnClearMask)
        toolChipGroup = findViewById(R.id.toolChipGroup)
        blurSeekBar = findViewById(R.id.blurSeekBar)
        blurValueText = findViewById(R.id.blurValueText)

        maskControlsRow = findViewById(R.id.maskControlsRow)
        radioGroupMaskMode = findViewById(R.id.radioGroupMaskMode)
        radioMaskClick = findViewById(R.id.radioMaskClick)
        radioMaskDraw = findViewById(R.id.radioMaskDraw)
        btnMaskUndo = findViewById(R.id.btnMaskUndo)

        requestNotificationPermissionIfNeeded()

        btnAddImageCenter.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        btnRemoveImage.setOnClickListener {
            resetToEmptyState()
        }

        btnSaveImage.setOnClickListener {
            if (selectedTool == Tool.PORTRAIT) {
                saveBokehResult()
            } else {
                selectedBitmap?.let { bitmap ->
                    val saved = ImageUtils.saveBitmapToGallery(this, bitmap)
                    if (saved) showToast("تم حفظ الصورة في المعرض ✅") else showToast("حدث خطأ أثناء الحفظ")
                }
            }
        }

        btnUndo.setOnClickListener {
            performUndo()
        }

        toolChipGroup.setOnCheckedStateChangeListener { _, checkedIds ->
            val checkedId = checkedIds.firstOrNull() ?: return@setOnCheckedStateChangeListener
            selectedTool =
            when (checkedId) {
                R.id.chipEnhance -> Tool.AUTO_ENHANCE
                R.id.chipEnhanceFull -> Tool.FULL
                R.id.chipRemoveObject -> Tool.REMOVE
                R.id.chipPortraitBlur -> Tool.PORTRAIT
                else -> Tool.FACE
            }

            val isRemove = selectedTool == Tool.REMOVE
            tvRemoveHint.visibility = if (isRemove) View.VISIBLE else View.GONE
            maskControlsRow.visibility = if (isRemove) View.VISIBLE else View.GONE
            blurValueText.visibility = if (selectedTool == Tool.PORTRAIT) View.VISIBLE else View.GONE
            blurSeekBar.visibility = if (selectedTool == Tool.PORTRAIT) View.VISIBLE else View.GONE

            if (isRemove) {
                touchImageView.setTouchMode(
                    if (radioMaskDraw.isChecked) InteractiveTouchImageView.TouchMode.DRAW
                    else InteractiveTouchImageView.TouchMode.CLICK
                )
            } else {
                // أي أداة غير الإزالة → نرجع لوضع النقر دايمًا،
                // عشان الـ Magnifier ما يفضلش شغال بعد ما نسيب أداة الإزالة.
                touchImageView.setTouchMode(InteractiveTouchImageView.TouchMode.CLICK)
            }

            if (selectedTool == Tool.PORTRAIT && currentBokehResult != null) {
                switchToBokehMode()
            } else {
                if (selectedTool != Tool.PORTRAIT) {
                    depthBlurView.visibility = View.GONE
                }
                switchToEditMode()
            }
            updateUiState()
        }

        blurSeekBar.addOnChangeListener { _, value, _ ->
            blurValueText.text = "${value.toInt()}%"
            applyBlurSliderToBokeh(value)
        }

        // --- وضع الإزالة: التبديل بين نقر / رسم ---
        radioGroupMaskMode.setOnCheckedChangeListener { _, checkedId ->
            when (checkedId) {
                R.id.radioMaskClick ->
                touchImageView.setTouchMode(InteractiveTouchImageView.TouchMode.CLICK)
                R.id.radioMaskDraw ->
                touchImageView.setTouchMode(InteractiveTouchImageView.TouchMode.DRAW)
            }
        }

        // --- الرجوع خطوة في الماسك ---
        btnMaskUndo.setOnClickListener {
            val bmp = selectedBitmap ?: return@setOnClickListener
            viewModel.undoMaskStep(this, bmp)
        }

        // --- مستمع اللمس على الصورة ---
        touchImageView.setOnImageTouchListener(
            object : InteractiveTouchImageView.OnImageTouchListener {
                override fun onImageTouched(originalX: Float, originalY: Float) {
                    if (selectedTool != Tool.REMOVE) return
                    selectedBitmap?.let { bitmap ->
                        startFakeProgress(estimatedDurationSegment, "جاري التحديد على الجهاز...")
                        viewModel.processSegmentPreview(this@MainActivity, bitmap, originalX, originalY)
                    }
                }

                override fun onDrawFinished(drawMask: Bitmap) {
                    if (selectedTool != Tool.REMOVE) {
                        drawMask.recycle()
                        return
                    }
                    val bitmap = selectedBitmap
                    if (bitmap == null) {
                        drawMask.recycle()
                        return
                    }
                    // الـViewModel يتولى ملكية drawMask بالكامل — لا نحرره هنا.
                    viewModel.addDrawMaskStep(this@MainActivity, drawMask, bitmap)
                }
        })

        btnExecute.setOnClickListener {
            val bitmap =
            selectedBitmap
            ?: run {
                showToast("يرجى اختيار صورة أولاً")
                return@setOnClickListener
            }

            when (selectedTool) {
                Tool.AUTO_ENHANCE -> {
                    startFakeProgress(estimatedDurationAutoEnhance, "جاري تحسين الإضاءة والتباين تلقائيًا...")
                    Thread {
                        try {
                            val enhancer = NativeSmartEnhancer()
                            val result = enhancer.enhance(bitmap)

                            runOnUiThread {
                                stopFakeProgress(completed = true)
                                pushToHistory(bitmap)
                                selectedBitmap = result
                                touchImageView.setImageBitmap(result)
                                touchImageView.clearMask()
                                hasMask = false
                                beforeBitmap = historyStack.peekLast() ?: bitmap
                                beforeBitmap?.let { before -> beforeAfterSlider.setImages(before, result) }
                                switchToPreviewMode()
                                tvStatus.text = "تم تحسين الصورة بنجاح!"
                                updateUiState()
                            }
                        } catch (e: Exception) {
                            e.printStackTrace()
                            runOnUiThread {
                                stopFakeProgress(completed = false)
                                tvStatus.text = "حدث خطأ أثناء معالجة الصورة"
                            }
                        }
                    }.start()
                }
                Tool.FACE -> {
                    startFakeProgress(estimatedDurationFace, "جاري تحسين الوجه...")
                    viewModel.startProcessing(this, "FACE", bitmap, fidelity = 1.0f)
                }
                Tool.FULL -> {
                    startFakeProgress(estimatedDurationFull, "جاري تحسين الصورة بالكامل...")
                    viewModel.startProcessing(this, "FULL", bitmap, fidelity = 1.0f)
                }
                Tool.REMOVE -> {
                    if (!hasMask) {
                        showToast("حدد عنصرًا واحدًا أو أكثر بالنقر أو بالفرشاة")
                        return@setOnClickListener
                    }
                    startFakeProgress(estimatedDurationRemove, "جاري إزالة العنصر...")
                    viewModel.startProcessing(this, "REMOVE", bitmap)
                }
                Tool.PORTRAIT -> {
                    startFakeProgress(estimatedDurationPortrait, "جاري تجهيز تأثير البورتريه...")
                    viewModel.prepareBokehEffect(this, bitmap)
                }
            }
        }

        btnClearMask.setOnClickListener {
            clearDisplayedMask()
            viewModel.clearMask()
            updateUiState()
        }

        observeViewModel()
        observeBackgroundWork()
        resetToEmptyState()
        cleanupStaleCacheFiles()
    }

    private fun cleanupStaleCacheFiles() {
        val directory = cacheDir ?: return

        Thread {
            try {
                val cutoffTime = System.currentTimeMillis() - TimeUnit.HOURS.toMillis(1)
                directory.listFiles()?.forEach { file ->
                    if (file.isFile && file.lastModified() < cutoffTime) {
                        file.delete()
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }.start()
    }

    private fun applyBlurSliderToBokeh(sliderValue: Float) {
        val half = (sliderValue / 100f).coerceIn(0f, 1f)
        depthBlurView.setBlurStrength(half)

        val curved = Math.pow(half.toDouble(), 0.85).toFloat()
        val relativeBokehSize = 0.018f + curved * 0.055f
        depthBlurView.setMaxBokehSize(relativeBokehSize)
    }

    private fun applyBokehAutoTuning(autoTuning: AutoTuningEstimator.AutoTuningResult) {
        blurSeekBar.value = autoTuning.blurSliderValue

        val colorBoost = (autoTuning.colorBoostSliderValue / 100f).coerceIn(0f, 1f)
        depthBlurView.setBokehColorBoost(colorBoost)

        val lightThreshold =
        (1.0f - (autoTuning.lightThresholdSliderValue / 100f) * 0.55f).coerceIn(0f, 1f)
        depthBlurView.setLightThreshold(lightThreshold)

        val highlightBoost = (autoTuning.highlightBoostSliderValue / 100f).coerceIn(0f, 1f)
        depthBlurView.setHighlightBoost(highlightBoost)
    }

    private fun recycleBokehBitmaps(
        result: MainViewModel.BokehPreparationResult
    ) {
        if (!result.sourceForGpu.isRecycled) {
            result.sourceForGpu.recycle()
        }

        if (!result.depthForGpu.isRecycled) {
            result.depthForGpu.recycle()
        }

        if (!result.maskForGpu.isRecycled) {
            result.maskForGpu.recycle()
        }
    }

    private fun releaseCurrentBokehResult() {
        val previous = currentBokehResult ?: run {
            viewModel.clearBokehResult()
            return
        }

        currentBokehResult = null
        viewModel.clearBokehResult()

        depthBlurView.releaseBitmapReferences {
            recycleBokehBitmaps(previous)
        }
    }

    private fun saveBokehResult() {
        val result = currentBokehResult ?: run {
            showToast("جهّز تأثير البورتريه أولاً")
            return
        }

        btnSaveImage.isEnabled = false

        depthBlurView.exportFullResolution(
            result.sourceForGpu,
            result.depthForGpu,
            result.maskForGpu
        ) { exported ->
            btnSaveImage.isEnabled = true

            if (exported == null) {
                showToast("فشل تصدير الصورة")
                return@exportFullResolution
            }

            val saved = ImageUtils.saveBitmapToGallery(this, exported)
            exported.recycle()

            if (saved) showToast("تم حفظ الصورة في المعرض ✅") else showToast("حدث خطأ أثناء الحفظ")
        }
    }

    private fun observeViewModel() {
        viewModel.isLoading.observe(this) { isLoading ->
            if (isLoading) {
                progressBar.visibility = View.VISIBLE
                btnExecute.isEnabled = false
                btnExecute.alpha = 0.4f
            } else {
                progressBar.visibility = View.GONE
                stopFakeProgress(completed = true)
                updateExecuteButtonState()
            }
        }

        viewModel.maskBitmap.observe(this) { mask ->

            val previousMask = displayedMaskBitmap

            if (mask != null) {
                displayedMaskBitmap = mask
                touchImageView.setMaskBitmap(mask)

                hasMask = true

                if (
                    previousMask != null &&
                    previousMask !== mask &&
                    !previousMask.isRecycled
                ) {
                    previousMask.recycle()
                }

                tvStatus.text =
                "تم تحديث التحديد — يمكنك إضافة نقرة أو رسمة، أو الرجوع خطوة"

            } else {
                displayedMaskBitmap = null
                touchImageView.clearMask()

                hasMask = false

                if (
                    previousMask != null &&
                    !previousMask.isRecycled
                ) {
                    previousMask.recycle()
                }
            }

            updateUiState()
        }

        viewModel.bokehResult.observe(this) { result ->
            if (result == null) {
                return@observe
            }

            stopFakeProgress(completed = true)

            if (result !== currentBokehResult) {
                val previous = currentBokehResult
                currentBokehResult = result

                if (previous != null) {
                    depthBlurView.releaseBitmapReferences {
                        recycleBokehBitmaps(previous)
                    }
                }
            }

            depthBlurView.setFocusDepth(result.focusDepth)
            depthBlurView.setBitmaps(result.sourceForGpu, result.depthForGpu, result.maskForGpu)
            applyBokehAutoTuning(result.autoTuning)

            switchToBokehMode()
            tvStatus.text = "تأثير البورتريه جاهز — حرّك شريط البلور لضبط القوة"
            updateUiState()
        }

        viewModel.bokehError.observe(this) { event ->
            event.getContentIfNotHandled()?.let {
                stopFakeProgress(completed = false)
                showToast("فشل تجهيز تأثير البورتريه، حاول تاني")
                updateUiState()
            }
        }
    }

    private fun observeBackgroundWork() {
        WorkManager.getInstance(this)
        .getWorkInfosForUniqueWorkLiveData(ImageProcessingWorker.UNIQUE_WORK_NAME)
        .observe(this) { workInfos ->
            val workInfo = workInfos?.firstOrNull() ?: return@observe
            when (workInfo.state) {
                WorkInfo.State.RUNNING,
                WorkInfo.State.ENQUEUED -> {
                    progressBar.visibility = View.VISIBLE
                    btnExecute.isEnabled = false
                    btnExecute.alpha = 0.4f
                }
                WorkInfo.State.SUCCEEDED -> {
                    stopFakeProgress(completed = true)
                    val resultPath = workInfo.outputData.getString(ImageProcessingWorker.KEY_RESULT_PATH)
                    resultPath?.let { path ->
                        val resultFile = File(path)
                        val bitmap = ImageUtils.fileToBitmap(resultFile)

                        if (resultFile.exists()) {
                            resultFile.delete()
                        }

                        bitmap?.let { result ->
                            releaseCurrentBokehResult()
                            selectedBitmap?.let { current -> pushToHistory(current) }
                            selectedBitmap = result
                            touchImageView.setImageBitmap(result)
                            touchImageView.clearMask()
                            hasMask = false
                            beforeBitmap = historyStack.peekLast() ?: result
                            beforeBitmap?.let { before -> beforeAfterSlider.setImages(before, result) }
                            switchToPreviewMode()
                            tvStatus.text = "تمت المعالجة بنجاح! قارن قبل/بعد أو تابع التعديل"
                        }
                    }
                    updateUiState()
                }
                WorkInfo.State.FAILED,
                WorkInfo.State.CANCELLED -> {
                    stopFakeProgress(completed = false)
                    progressBar.visibility = View.GONE
                    showToast("حصل خطأ أثناء المعالجة، حاول تاني")
                    updateUiState()
                }
                else -> {}
            }
        }
    }

    // --- حفظ التعديلات وإدارتها (Undo Engine) ---
    private fun pushToHistory(bitmap: Bitmap) {
        if (historyStack.size >= 5) {
            val discarded = historyStack.removeFirst()
            if (!discarded.isRecycled) {
                discarded.recycle()
            }
        }
        historyStack.addLast(bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false))
    }

    private fun clearHistoryStack() {
        while (historyStack.isNotEmpty()) {
            val bitmap = historyStack.removeFirst()
            if (!bitmap.isRecycled) {
                bitmap.recycle()
            }
        }
    }

    private fun performUndo() {
        if (historyStack.isEmpty()) return

        val previousBitmap = historyStack.removeLast()
        selectedBitmap = previousBitmap

        beforeBitmap = historyStack.peekLast() ?: previousBitmap
        beforeAfterSlider.setImages(beforeBitmap!!, selectedBitmap!!)

        releaseCurrentBokehResult()

        clearDisplayedMask()
        viewModel.clearMask()

        touchImageView.setImageBitmap(previousBitmap)
        hasMask = false

        updateUiState()
        tvStatus.text = "تم التراجع عن التعديل الأخير ↩️"
    }

    private fun resetToEmptyState() {
        clearHistoryStack()

        // Release the currently displayed mask first.
        // InteractiveTouchImageView must no longer hold the Bitmap
        // before MainActivity recycles it.
        clearDisplayedMask()

        // Clear ViewModel-owned mask pipeline/history.
        viewModel.clearMask()

        // Release the Bokeh result after GL releases its references.
        releaseCurrentBokehResult()

        selectedBitmap = null
        beforeBitmap = null
        hasMask = false

        touchImageView.setImageBitmap(null)

        radioMaskClick.isChecked = true
        touchImageView.setTouchMode(
            InteractiveTouchImageView.TouchMode.CLICK
        )

        switchToEditMode()
        updateUiState()
        tvStatus.text = "تحسين صورك بلمسة ذكية"
    }

    private fun switchToEditMode() {
        touchImageView.visibility = if (selectedBitmap != null) View.VISIBLE else View.GONE
        beforeAfterSlider.visibility = View.GONE
        depthBlurView.visibility = View.GONE
    }

    private fun switchToPreviewMode() {
        touchImageView.visibility = View.GONE
        beforeAfterSlider.visibility = View.VISIBLE
        depthBlurView.visibility = View.GONE
    }

    private fun switchToBokehMode() {
        touchImageView.visibility = View.GONE
        beforeAfterSlider.visibility = View.GONE
        depthBlurView.visibility = View.VISIBLE
    }

    private fun updateUiState() {
        val hasImage = selectedBitmap != null

        layoutEmptyState.visibility = if (!hasImage) View.VISIBLE else View.GONE
        btnRemoveImage.visibility = if (hasImage) View.VISIBLE else View.GONE

        if (!hasImage) {
            touchImageView.visibility = View.GONE
            beforeAfterSlider.visibility = View.GONE
            depthBlurView.visibility = View.GONE
        } else if (depthBlurView.visibility != View.VISIBLE && beforeAfterSlider.visibility != View.VISIBLE) {
            touchImageView.visibility = View.VISIBLE
        }

        btnSaveImage.isEnabled = hasImage
        btnSaveImage.alpha = if (hasImage) 1.0f else 0.4f

        btnUndo.visibility = if (historyStack.isNotEmpty()) View.VISIBLE else View.GONE

        updateExecuteButtonState()
    }

    private fun isExecuteAllowed(): Boolean {
        if (selectedBitmap == null) return false
        if (selectedTool == Tool.REMOVE && !hasMask) return false
        return true
    }

    private fun updateExecuteButtonState() {
        btnExecute.isEnabled = isExecuteAllowed()
        btnExecute.alpha = if (btnExecute.isEnabled) 1f else 0.4f
        btnExecute.text = if (selectedTool == Tool.REMOVE) "إزالة العنصر المحدد" else "تنفيذ"
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(contentResolver, uri)
                ImageDecoder.decodeBitmap(source) { decoder, _, _ -> decoder.isMutableRequired = true }
            } else {
                @Suppress("DEPRECATION") MediaStore.Images.Media.getBitmap(contentResolver, uri)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun showToast(msg: String) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
    }

    override fun onPause() {
        super.onPause()
        depthBlurView.onPause()
    }

    override fun onResume() {
        super.onResume()
        depthBlurView.onResume()
    }

    override fun onDestroy() {
        clearDisplayedMask()
        releaseCurrentBokehResult()
        super.onDestroy()
    }
}
