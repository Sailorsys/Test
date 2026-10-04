package com.master.aistudio

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.os.Build
import android.util.AttributeSet
import android.view.MotionEvent
import android.widget.Magnifier
import androidx.appcompat.widget.AppCompatImageView
import kotlin.math.max
import kotlin.math.min

class InteractiveTouchImageView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : AppCompatImageView(context, attrs) {

    enum class TouchMode { CLICK, DRAW }

    interface OnImageTouchListener {
        fun onImageTouched(originalX: Float, originalY: Float)
        fun onDrawFinished(drawMask: Bitmap)
    }

    private var touchListener: OnImageTouchListener? = null
    private var maskBitmap: Bitmap? = null
    private var touchMode: TouchMode = TouchMode.CLICK

    private val maskPaint = Paint().apply {
        alpha = 120
    }

    private val brushPreviewPaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        isAntiAlias = true
        alpha = 170
    }

    private val touchPath = Path()
    private var isDrawing = false
    private var downX = 0f
    private var downY = 0f

    // ──────────────────────────────────────────────────────────────
    // عدسة مكبرة (Magnifier) — متاحة من API 28+.
    // تُعرض أثناء الرسم الحر فقط، وتختفي عند رفع الصباع.
    // على API 29+ تُعرض فوق الصباع (setOffset) لمحتوى يظل تحت
    // نقطة اللمس الفعلية. على API 28 تُعرض عند نقطة اللمس مباشرة.
    // ──────────────────────────────────────────────────────────────
    private var magnifier: Magnifier? = null

    private val magnifierSizePx: Int by lazy {
        (resources.displayMetrics.density * 110f).toInt()
    }

    private val magnifierOffsetPx: Float by lazy {
        resources.displayMetrics.density * 90f
    }

    fun setOnImageTouchListener(listener: OnImageTouchListener) {
        this.touchListener = listener
    }

    fun setTouchMode(mode: TouchMode) {
        if (this.touchMode == mode) return
        this.touchMode = mode
        isDrawing = false
        touchPath.reset()
        dismissMagnifier()
        invalidate()
    }

    fun setMaskBitmap(mask: Bitmap?) {
        this.maskBitmap = mask
        invalidate()
    }

    fun detachMaskBitmap(): Bitmap? {
        val old = maskBitmap
        maskBitmap = null
        invalidate()
        return old
    }

    fun clearMask() {
        this.maskBitmap = null
        this.touchPath.reset()
        isDrawing = false
        dismissMagnifier()
        invalidate()
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (drawable == null || touchListener == null) return false

        return when (touchMode) {
            TouchMode.CLICK -> handleClickTouch(event)
            TouchMode.DRAW -> handleDrawTouch(event)
        }
    }

    private fun handleClickTouch(event: MotionEvent): Boolean {
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                isDrawing = true
                downX = event.x
                downY = event.y
                return true
            }
            MotionEvent.ACTION_UP -> {
                if (isDrawing) {
                    isDrawing = false
                    val coords = getBitmapCoordinates(downX, downY)
                    coords?.let {
                        touchListener?.onImageTouched(it[0], it[1])
                    }
                }
                return true
            }
            MotionEvent.ACTION_CANCEL -> {
                isDrawing = false
                return true
            }
        }
        return false
    }

    private fun handleDrawTouch(event: MotionEvent): Boolean {
        val displayedRect = computeDisplayedRect() ?: return false
        val minSide = min(displayedRect.width(), displayedRect.height())
        brushPreviewPaint.strokeWidth = (minSide * 0.04f).coerceAtLeast(6f)

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                isDrawing = true
                touchPath.reset()
                touchPath.moveTo(event.x, event.y)
                showMagnifierAt(event.x, event.y)
                invalidate()
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                if (isDrawing) {
                    touchPath.lineTo(event.x, event.y)
                    showMagnifierAt(event.x, event.y)
                    invalidate()
                }
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                if (isDrawing) {
                    isDrawing = false
                    dismissMagnifier()
                    val drawMask = convertDrawPathToMask(displayedRect)
                    touchPath.reset()
                    invalidate()
                    drawMask?.let { touchListener?.onDrawFinished(it) }
                }
                return true
            }
        }
        return false
    }

    // ───────────────────── Magnifier ─────────────────────

    @Suppress("DEPRECATION")
    private fun ensureMagnifier(): Magnifier? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) return null

        if (magnifier == null) {
            val size = magnifierSizePx
            val builder = Magnifier.Builder(this)
            .setInitialZoom(2.5f)
            .setCornerRadius(size / 2f)
            .setElevation(resources.displayMetrics.density * 8f)

            // setSize موجود منذ API 28 (مُهمل في 29 لكنه يعمل).
            builder.setSize(size, size)

            // API 29+: نُزيح عرض العدسة فوق الصباع، بينما المحتوى
            // المكبَّر يظل عند نقطة اللمس الفعلية.

            magnifier = builder.build()
        }
        return magnifier
    }

    private fun showMagnifierAt(x: Float, y: Float) {
        ensureMagnifier()?.show(x, y)
    }

    private fun dismissMagnifier() {
        magnifier?.dismiss()
    }

    override fun onDetachedFromWindow() {
        dismissMagnifier()
        magnifier = null
        super.onDetachedFromWindow()
    }

    // ─────────────────── نهاية Magnifier ───────────────────

    private fun computeDisplayedRect(): RectF? {
        val currDrawable = drawable ?: return null
        val w = currDrawable.intrinsicWidth.toFloat()
        val h = currDrawable.intrinsicHeight.toFloat()
        if (w <= 0f || h <= 0f) return null
        val rect = RectF(0f, 0f, w, h)
        imageMatrix.mapRect(rect)
        if (rect.width() <= 0f || rect.height() <= 0f) return null
        return rect
    }

    private fun convertDrawPathToMask(displayedRect: RectF): Bitmap? {
        val currDrawable = drawable ?: return null
        val bitmapW = currDrawable.intrinsicWidth
        val bitmapH = currDrawable.intrinsicHeight
        if (bitmapW <= 0 || bitmapH <= 0) return null

        val scaleX = bitmapW / displayedRect.width()
        val scaleY = bitmapH / displayedRect.height()

        val matrix = Matrix()
        matrix.postTranslate(-displayedRect.left, -displayedRect.top)
        matrix.postScale(scaleX, scaleY)

        val scaledPath = Path()
        touchPath.transform(matrix, scaledPath)

        val output = Bitmap.createBitmap(bitmapW, bitmapH, Bitmap.Config.ARGB_8888)

        val canvas = Canvas(output)
        val paint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            isAntiAlias = true
            strokeWidth = brushPreviewPaint.strokeWidth * scaleX
        }
        canvas.drawPath(scaledPath, paint)
        return output
    }

    private fun getBitmapCoordinates(touchX: Float, touchY: Float): FloatArray? {
        val currDrawable = drawable ?: return null

        val drawableWidth = currDrawable.intrinsicWidth
        val drawableHeight = currDrawable.intrinsicHeight

        val rect = RectF(0f, 0f, drawableWidth.toFloat(), drawableHeight.toFloat())
        imageMatrix.mapRect(rect)

        val clampedX = max(rect.left, min(touchX, rect.right))
        val clampedY = max(rect.top, min(touchY, rect.bottom))

        val relativeX = (clampedX - rect.left) / rect.width()
        val relativeY = (clampedY - rect.top) / rect.height()

        val actualX = relativeX * drawableWidth
        val actualY = relativeY * drawableHeight

        return floatArrayOf(actualX, actualY)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (touchMode == TouchMode.DRAW && isDrawing && !touchPath.isEmpty) {
            canvas.drawPath(touchPath, brushPreviewPaint)
        }

        val currDrawable = drawable
        val currentMask = maskBitmap
        if (currentMask != null && currDrawable != null) {
            val rect = RectF(
                0f,
                0f,
                currDrawable.intrinsicWidth.toFloat(),
                currDrawable.intrinsicHeight.toFloat()
            )
            imageMatrix.mapRect(rect)
            canvas.drawBitmap(currentMask, null, rect, maskPaint)
        }
    }
}
