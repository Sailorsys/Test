package com.master.aistudio

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.min

class SelfieMaskRunner {
    companion object {
        private const val TAG = "SelfieMask"
        private const val MODEL_NAME = "modnet_photographic_portrait_matting.onnx"
        private const val INPUT_SIZE = 512
    }

    // private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    @Synchronized
    private fun getSession(context: Context): OrtSession {
        if (ortSession == null) {
            val env = OnnxSessionFactory.environment
            context.assets.open(MODEL_NAME).use { asset ->
                val bytes = asset.readBytes()
                ortSession = OnnxSessionFactory.createSession(
                    modelBytes = bytes,
                    environment = env,
                    logTag = "SelfieMask"
                )
            }
        }
        return ortSession!!
    }

    fun process(context: Context, input: Bitmap): Bitmap {
        val session = getSession(context)
        val prepared = resizeWithLetterbox(input, INPUT_SIZE, INPUT_SIZE)

        val area = INPUT_SIZE * INPUT_SIZE
        val inputFloatArray = FloatArray(1 * 3 * area)
        val pixels = IntArray(area)

        prepared.bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (i in 0 until area) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF

            inputFloatArray[i] = (r - 127.5f) / 127.5f
            inputFloatArray[area + i] = (g - 127.5f) / 127.5f
            inputFloatArray[2 * area + i] = (b - 127.5f) / 127.5f
        }
        val env = OnnxSessionFactory.environment
        val inputShape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputFloatArray), inputShape)

        val inputName = session.inputNames.iterator().next()
        val results = session.run(mapOf(inputName to inputTensor))

        val outputTensor = results[0] as OnnxTensor
        val outputFloatBuffer = outputTensor.floatBuffer
        val floatValues = FloatArray(outputFloatBuffer.remaining())
        outputFloatBuffer.get(floatValues)

        inputTensor.close()
        results.close()

        val maskPixels = IntArray(area)
        var personPixelsCount = 0

        for (i in 0 until area) {
            val alphaVal = floatValues[i].coerceIn(0f, 1f)
            val alpha = (alphaVal * 255f).toInt()

            if (alpha > 128) personPixelsCount++

            // جعل Alpha معتمة دائماً (0xFF) لمنع أندرويد من تطبيق
            // Premultiplied Alpha التخريبي عند تغيير الحجم
            maskPixels[i] = (0xFF shl 24) or (alpha shl 16) or (alpha shl 8) or alpha
        }

        val maskSquare = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        maskSquare.setPixels(maskPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val cropped = Bitmap.createBitmap(
            maskSquare,
            prepared.left,
            prepared.top,
            prepared.contentWidth,
            prepared.contentHeight
        )
        val fullMask = Bitmap.createScaledBitmap(cropped, input.width, input.height, true)

        Log.d(TAG, "Mask coverage=${personPixelsCount * 100 / area}%")

        if (!prepared.bitmap.isRecycled) prepared.bitmap.recycle()
        if (!maskSquare.isRecycled) maskSquare.recycle()
        if (cropped != fullMask && !cropped.isRecycled) cropped.recycle()

        return fullMask
    }

    private fun resizeWithLetterbox(src: Bitmap, dstW: Int, dstH: Int): PreparedBitmap {
        val result = Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888)
        val scale = min(dstW / src.width.toFloat(), dstH / src.height.toFloat())
        val w = (src.width * scale).toInt().coerceIn(1, dstW)
        val h = (src.height * scale).toInt().coerceIn(1, dstH)
        val left = (dstW - w) / 2
        val top = (dstH - h) / 2

        val scaled = Bitmap.createScaledBitmap(src, w, h, true)

        Canvas(result).apply {
            drawColor(Color.BLACK)
            drawBitmap(scaled, left.toFloat(), top.toFloat(), null)
        }

        if (scaled != src && !scaled.isRecycled) {
            scaled.recycle()
        }

        return PreparedBitmap(result, left, top, w, h)
    }

    private data class PreparedBitmap(
        val bitmap: Bitmap,
        val left: Int,
        val top: Int,
        val contentWidth: Int,
        val contentHeight: Int
    )

    fun close() {
        try {
            ortSession?.close()
        } catch (_: Exception) {
        }

        ortSession = null
    }
}
