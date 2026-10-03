package com.master.aistudio

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class MobileSamOnnxRunner(context: Context) : Closeable {

    private companion object {
        const val ENCODER_ASSET = "sam_encoder.onnx"
        const val DECODER_ASSET = "sam_decoder.onnx"

        const val MODEL_SIZE = 1024
        const val EMBEDDING_CHANNELS = 256
        const val EMBEDDING_SIZE = 64
        const val MASK_INPUT_SIZE = 256

        val PIXEL_MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        val PIXEL_STD = floatArrayOf(58.395f, 57.12f, 57.375f)
    }

    private val appContext = context.applicationContext
    private val environment = OrtEnvironment.getEnvironment()

    private val encoderSession: OrtSession = OnnxSessionFactory.createSession(
        modelBytes = loadAsset(ENCODER_ASSET),
        environment = environment,
        logTag = "MobileSAM-Encoder"
    )

    private val decoderSession: OrtSession = OnnxSessionFactory.createSession(
        modelBytes = loadAsset(DECODER_ASSET),
        environment = environment,
        logTag = "MobileSAM-Decoder"
    )

    // كاش الـ embedding
    private var cachedSource: Bitmap? = null
    private var cachedOrigW = 0
    private var cachedOrigH = 0
    private var cachedScale = 0f
    private var cachedEmbedding: FloatArray? = null

    private fun loadAsset(assetName: String): ByteArray {
        return appContext.assets.open(assetName).use { it.readBytes() }
    }

    /** نقطة واحدة — نفس التوقيع القديم */
    fun predictPoint(bitmap: Bitmap, x: Float, y: Float): Bitmap {
        return predict(
            bitmap,
            floatArrayOf(x, y),
            floatArrayOf(1f)
        )
    }

    /**
     * صندوق — نفس التوقيع السابق.
     * مع نماذج Vanish بنحوّل الصندوق لنقطة في المركز.
     */
    fun predictBox(bitmap: Bitmap, xMin: Int, yMin: Int, xMax: Int, yMax: Int): Bitmap {
        val left = min(xMin, xMax).toFloat().coerceIn(0f, (bitmap.width - 1).toFloat())
        val top = min(yMin, yMax).toFloat().coerceIn(0f, (bitmap.height - 1).toFloat())
        val right = max(xMin, xMax).toFloat().coerceIn(0f, (bitmap.width - 1).toFloat())
        val bottom = max(yMin, yMax).toFloat().coerceIn(0f, (bitmap.height - 1).toFloat())
        val cx = (left + right) * 0.5f
        val cy = (top + bottom) * 0.5f
        return predictPoint(bitmap, cx, cy)
    }

    private fun predict(
        bitmap: Bitmap,
        coords: FloatArray,
        labels: FloatArray
    ): Bitmap {
        require(coords.size == labels.size * 2)
        require(labels.isNotEmpty())

        val originalWidth = bitmap.width
        val originalHeight = bitmap.height
        val scale = MODEL_SIZE.toFloat() / max(originalWidth, originalHeight).toFloat()
        val embedding = getEmbedding(bitmap, originalWidth, originalHeight, scale)

        val transformedCoords = FloatArray(coords.size)
        for (i in labels.indices) {
            transformedCoords[i * 2] =
                coords[i * 2].coerceIn(0f, originalWidth.toFloat()) * scale
            transformedCoords[i * 2 + 1] =
                coords[i * 2 + 1].coerceIn(0f, originalHeight.toFloat()) * scale
        }

        val embeddingTensor = createFloatTensor(
            embedding,
            longArrayOf(
                1L,
                EMBEDDING_CHANNELS.toLong(),
                EMBEDDING_SIZE.toLong(),
                EMBEDDING_SIZE.toLong()
            )
        )
        val pointTensor = createFloatTensor(
            transformedCoords,
            longArrayOf(1L, labels.size.toLong(), 2L)
        )
        val labelTensor = createFloatTensor(
            labels,
            longArrayOf(1L, labels.size.toLong())
        )
        val maskInputTensor = createFloatTensor(
            FloatArray(MASK_INPUT_SIZE * MASK_INPUT_SIZE),
            longArrayOf(1L, 1L, MASK_INPUT_SIZE.toLong(), MASK_INPUT_SIZE.toLong())
        )
        val hasMaskInputTensor = createFloatTensor(floatArrayOf(0f), longArrayOf(1L))
        val originalSizeTensor = createFloatTensor(
            floatArrayOf(originalHeight.toFloat(), originalWidth.toFloat()),
            longArrayOf(2L)
        )

        try {
            val inputs = mapOf(
                "image_embeddings" to embeddingTensor,
                "point_coords" to pointTensor,
                "point_labels" to labelTensor,
                "mask_input" to maskInputTensor,
                "has_mask_input" to hasMaskInputTensor,
                "orig_im_size" to originalSizeTensor
            )
            val decoderStartTime = System.currentTimeMillis()
            decoderSession.run(inputs).use { result ->
                android.util.Log.d(
                    "SAM_TIMING",
                    "DECODER took ${System.currentTimeMillis() - decoderStartTime} ms"
                )
                val outTensor = result[0] as OnnxTensor
                val buf = outTensor.floatBuffer.duplicate()
                buf.rewind()

                val shape = outTensor.info.shape
                val maskH: Int
                val maskW: Int
                val logits: FloatArray

                when {
                    shape.size == 4 -> {
                        val numMasks = shape[1].toInt().coerceAtLeast(1)
                        maskH = shape[2].toInt()
                        maskW = shape[3].toInt()
                        val all = FloatArray(numMasks * maskH * maskW)
                        buf.get(all)
                        logits = FloatArray(maskH * maskW)
                        System.arraycopy(all, 0, logits, 0, maskH * maskW)
                    }
                    shape.size == 3 -> {
                        maskH = shape[1].toInt()
                        maskW = shape[2].toInt()
                        logits = FloatArray(maskH * maskW)
                        buf.get(logits)
                    }
                    else -> {
                        maskW = originalWidth
                        maskH = originalHeight
                        logits = FloatArray(maskW * maskH)
                        val n = min(logits.size, buf.remaining())
                        buf.get(logits, 0, n)
                    }
                }

                return logitsToBitmap(logits, maskW, maskH, originalWidth, originalHeight)
            }
        } finally {
            embeddingTensor.close()
            pointTensor.close()
            labelTensor.close()
            maskInputTensor.close()
            hasMaskInputTensor.close()
            originalSizeTensor.close()
        }
    }

    private fun getEmbedding(
        source: Bitmap,
        origW: Int,
        origH: Int,
        scale: Float
    ): FloatArray {
        if (
            cachedSource === source &&
            cachedOrigW == origW &&
            cachedOrigH == origH &&
            cachedEmbedding != null
        ) {
            return cachedEmbedding!!
        }

        val newW = (origW * scale).roundToInt().coerceIn(1, MODEL_SIZE)
        val newH = (origH * scale).roundToInt().coerceIn(1, MODEL_SIZE)

        val scaled = if (source.width == newW && source.height == newH) {
            source
        } else {
            Bitmap.createScaledBitmap(source, newW, newH, true)
        }

        val pixels = IntArray(newW * newH)
        scaled.getPixels(pixels, 0, newW, 0, 0, newW, newH)

        val input = FloatArray(3 * MODEL_SIZE * MODEL_SIZE)
        val plane = MODEL_SIZE * MODEL_SIZE

        for (y in 0 until newH) {
            for (x in 0 until newW) {
                val p = pixels[y * newW + x]
                val r = ((p ushr 16) and 0xFF).toFloat()
                val g = ((p ushr 8) and 0xFF).toFloat()
                val b = (p and 0xFF).toFloat()
                val idx = y * MODEL_SIZE + x
                input[idx] = (r - PIXEL_MEAN[0]) / PIXEL_STD[0]
                input[plane + idx] = (g - PIXEL_MEAN[1]) / PIXEL_STD[1]
                input[2 * plane + idx] = (b - PIXEL_MEAN[2]) / PIXEL_STD[2]
            }
        }

        if (scaled !== source) scaled.recycle()

        val imageTensor = createFloatTensor(
            input,
            longArrayOf(1L, 3L, MODEL_SIZE.toLong(), MODEL_SIZE.toLong())
        )
        return try {
            val encoderStartTime = System.currentTimeMillis()
            encoderSession.run(mapOf("image" to imageTensor)).use { result ->
                android.util.Log.d(
                    "SAM_TIMING",
                    "ENCODER took ${System.currentTimeMillis() - encoderStartTime} ms"
                )
                val embTensor = result[0] as OnnxTensor
                val embBuf = embTensor.floatBuffer.duplicate()
                embBuf.rewind()
                val embedding = FloatArray(EMBEDDING_CHANNELS * EMBEDDING_SIZE * EMBEDDING_SIZE)
                embBuf.get(embedding)

                cachedSource = source
                cachedOrigW = origW
                cachedOrigH = origH
                cachedScale = scale
                cachedEmbedding = embedding
                embedding
            }
        } finally {
            imageTensor.close()
        }
    }

    private fun createFloatTensor(values: FloatArray, shape: LongArray): OnnxTensor {
        val buffer: FloatBuffer = ByteBuffer
            .allocateDirect(values.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        buffer.put(values).rewind()
        return OnnxTensor.createTensor(environment, buffer, shape)
    }

    private fun logitsToBitmap(
        logits: FloatArray,
        maskW: Int,
        maskH: Int,
        origW: Int,
        origH: Int
    ): Bitmap {
        if (maskW == origW && maskH == origH) {
            val pixels = IntArray(maskW * maskH)
            for (i in logits.indices) {
                pixels[i] = if (logits[i] > 0f) Color.WHITE else Color.TRANSPARENT
            }
            return Bitmap.createBitmap(pixels, maskW, maskH, Bitmap.Config.ARGB_8888)
        }

        val small = IntArray(maskW * maskH)
        for (i in logits.indices) {
            small[i] = if (logits[i] > 0f) Color.WHITE else Color.TRANSPARENT
        }
        val smallBmp = Bitmap.createBitmap(small, maskW, maskH, Bitmap.Config.ARGB_8888)
        val full = Bitmap.createScaledBitmap(smallBmp, origW, origH, false)
        if (full !== smallBmp) smallBmp.recycle()
        return full
    }

    override fun close() {
        cachedEmbedding = null
        cachedSource = null
        try { encoderSession.close() } catch (_: Exception) {}
        try { decoderSession.close() } catch (_: Exception) {}
    }
}