package com.master.aistudio

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import java.util.Locale
import kotlin.math.abs
import kotlin.math.min

/**
 * فيو مخصص لعرض صورتين (قبل / بعد) فوق بعض مع خط فاصل قابل للسحب
 * لمقارنة نتيجة المعالجة بصريًا، مع دعم تكبير بإصبعين لفحص التفاصيل عن قرب.
 */
class BeforeAfterSliderView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var beforeBitmap: Bitmap? = null
    private var afterBitmap: Bitmap? = null
    private var dividerX: Float = -1f

    // --- حالة التكبير والتحريك ---
    private var zoomScale = 1f
    private var panX = 0f
    private var panY = 0f
    private var lastTouchX = 0f
    private var lastTouchY = 0f

    // --- متغيرات التحكم في سحب الخط أثناء الزوم ---
    private var isDraggingDivider = false
    private val dividerHitArea = 60f

    // --- تحسين الأداء: تخزين المصفوفات ---
    private val matrixAfter = Matrix()
    private val matrixBefore = Matrix()

    companion object {
        private const val MIN_ZOOM = 1f
        private const val MAX_ZOOM = 4f
    }

    private val dividerPaint = Paint().apply {
        color = Color.WHITE
        strokeWidth = 5f
        isAntiAlias = true
    }

    private val handlePaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val handleBorderPaint = Paint().apply {
        color = Color.parseColor("#7C4DFF")
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    private val labelBgPaint = Paint().apply {
        color = Color.parseColor("#99000000")
        isAntiAlias = true
    }

    private val labelTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        isAntiAlias = true
        textAlign = Paint.Align.LEFT
    }

    private val handleRadius = 28f

    private val scaleGestureDetector = ScaleGestureDetector(
        context,
        object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                zoomScale = (zoomScale * detector.scaleFactor).coerceIn(MIN_ZOOM, MAX_ZOOM)
                clampPan()
                invalidate()
                return true
            }
        }
    )

    private val gestureDetector = GestureDetector(
        context,
        object : GestureDetector.SimpleOnGestureListener() {
            override fun onDoubleTap(e: MotionEvent): Boolean {
                // دبل تاب: رجّع الزوم للوضع الطبيعي فورًا
                zoomScale = 1f
                panX = 0f
                panY = 0f
                invalidate()
                return true
            }
        }
    )

    /** يستدعى بعد وصول نتيجة من السيرفر لعرض المقارنة */
    fun setImages(before: Bitmap, after: Bitmap) {
        beforeBitmap = before
        afterBitmap = after
        dividerX = if (width > 0) width / 2f else -1f
        zoomScale = 1f
        panX = 0f
        panY = 0f
        updateMatrices()
        postInvalidate()
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (dividerX < 0f) dividerX = w / 2f
        dividerX = dividerX.coerceIn(0f, w.toFloat())
        updateMatrices()
    }

    private fun updateMatrices() {
        val after = afterBitmap ?: return
        val before = beforeBitmap ?: return
        if (width == 0 || height == 0) return
        computeFitMatrix(after.width, after.height, matrixAfter)
        computeFitMatrix(before.width, before.height, matrixBefore)
    }

    private fun computeFitMatrix(bw: Int, bh: Int, outMatrix: Matrix) {
        val scale = min(width.toFloat() / bw, height.toFloat() / bh)
        val dx = (width - bw * scale) / 2f
        val dy = (height - bh * scale) / 2f
        outMatrix.setScale(scale, scale)
        outMatrix.postTranslate(dx, dy)
    }

    private fun clampPan() {
        val maxPanX = (width * (zoomScale - 1)) / 2f
        val maxPanY = (height * (zoomScale - 1)) / 2f
        panX = panX.coerceIn(-maxPanX, maxPanX)
        panY = panY.coerceIn(-maxPanY, maxPanY)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val after = afterBitmap ?: return
        val before = beforeBitmap ?: return
        if (width == 0 || height == 0) return

        val divider = dividerX.coerceIn(0f, width.toFloat())

        // 1. رسم الصورة "بعد" بالكامل وتكبيرها/تحريكها مع الزوم
        canvas.save()
        canvas.translate(panX, panY)
        canvas.scale(zoomScale, zoomScale, width / 2f, height / 2f)
        canvas.drawBitmap(after, matrixAfter, null)
        canvas.restore()

        // 2. قص الحاوية الأساسية (View Bounds) عند إحداثي divider بالضبط وبشكل مستقل عن الزوم
        canvas.save()
        canvas.clipRect(0f, 0f, divider, height.toFloat()) // قص مباشر في إحداثيات View الشاشة
        
        // ثم رسم الصورة "قبل" داخل المنطقة المقتطعة مع إعطائها نفس تحويلات التكبير
        canvas.translate(panX, panY)
        canvas.scale(zoomScale, zoomScale, width / 2f, height / 2f)
        canvas.drawBitmap(before, matrixBefore, null)
        canvas.restore()

        // 3. رسم خط السليد والمقبض مرتبطين تماماً بـ View الشاشة (لا يتأثران بالزوم إطلاقاً)
        canvas.drawLine(divider, 0f, divider, height.toFloat(), dividerPaint)
        canvas.drawCircle(divider, height / 2f, handleRadius, handlePaint)
        canvas.drawCircle(divider, height / 2f, handleRadius, handleBorderPaint)

        // 4. رسم العناوين ومؤشر التكبير
        drawLabel(canvas, "قبل", 24f, 24f)
        drawLabel(canvas, "بعد", width - 130f, 24f)

        if (zoomScale > 1.05f) {
            val zoomText = "🔍 x${String.format(Locale.US, "%.1f", zoomScale)}"
            val textWidth = labelTextPaint.measureText(zoomText)
            val padding = 14f
            val rectWidth = textWidth + padding * 2
            val x = (width - rectWidth) / 2f
            drawLabel(canvas, zoomText, x, height - 70f)
        }
    }

    private fun drawLabel(canvas: Canvas, text: String, x: Float, y: Float) {
        val padding = 14f
        val textWidth = labelTextPaint.measureText(text)
        canvas.drawRoundRect(
            x, y, x + textWidth + padding * 2, y + 48f, 12f, 12f, labelBgPaint
        )
        canvas.drawText(text, x + padding, y + 34f, labelTextPaint)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (afterBitmap == null || beforeBitmap == null) return false

        parent?.requestDisallowInterceptTouchEvent(true)
        scaleGestureDetector.onTouchEvent(event)
        gestureDetector.onTouchEvent(event)

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                lastTouchX = event.x
                lastTouchY = event.y

                // فحص لمس الخط مباشرة وفق إحداثيات الشاشة
                isDraggingDivider = abs(event.x - dividerX) < dividerHitArea
            }
            MotionEvent.ACTION_MOVE -> {
                if (scaleGestureDetector.isInProgress) {
                    // أثناء التكبير بإصبعين يتوقف سحب السليدر
                } else if (isDraggingDivider) {
                    // تحريك خط السليدر بشكل مستقل على View الشاشة
                    dividerX = event.x.coerceIn(0f, width.toFloat())
                    invalidate()
                } else if (zoomScale > 1f) {
                    // تحريك الصورة (Pan) عندما يكون الزوم مفعل والإصبع بعيد عن السليدر
                    val dx = event.x - lastTouchX
                    val dy = event.y - lastTouchY
                    panX += dx
                    panY += dy
                    clampPan()
                    lastTouchX = event.x
                    lastTouchY = event.y
                    invalidate()
                } else {
                    // الوضع الطبيعي بدون زوم
                    dividerX = event.x.coerceIn(0f, width.toFloat())
                    invalidate()
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                isDraggingDivider = false
                parent?.requestDisallowInterceptTouchEvent(false)
            }
        }
        return true
    }
}
