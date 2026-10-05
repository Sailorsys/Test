package com.master.aistudio

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BlurMaskFilter
import android.graphics.Canvas
import android.graphics.Paint
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.io.FileOutputStream

object ImageUtils {

    // تحويل ByteArray المستلم من السيرفر إلى Bitmap للعرض
    fun bytesToBitmap(bytes: ByteArray): Bitmap? {
        return try {
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    // حفظ الـ Bitmap في ملف مؤقت لإرساله عبر NetworkManager
    fun bitmapToFile(context: Context, bitmap: Bitmap, fileName: String): File {
        val file = File(context.cacheDir, fileName)
        FileOutputStream(file).use { output ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, output)
        }
        return file
    }

    // حفظ قناع MobileSAM بصيغة PNG فعلية؛ لا تستخدم JPEG مع الأقنعة الثنائية.
    fun maskBitmapToFile(context: Context, bitmap: Bitmap, fileName: String): File {
        val file = File(context.cacheDir, fileName)
        FileOutputStream(file).use { output ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)
        }
        return file
    }

    // دمج قناع جديد مع القناع المتراكم؛ مفيد لتحديد أكثر من عنصر بالنقر المتكرر.
    fun unionMasks(base: Bitmap?, addition: Bitmap): Bitmap {
        if (base == null) {
            return addition.copy(Bitmap.Config.ARGB_8888, false)
        }
        require(base.width == addition.width && base.height == addition.height) {
            "Masks must have identical dimensions"
        }

        val basePixels = IntArray(base.width * base.height)
        val additionPixels = IntArray(addition.width * addition.height)
        val outputPixels = IntArray(basePixels.size)
        base.getPixels(basePixels, 0, base.width, 0, 0, base.width, base.height)
        addition.getPixels(additionPixels, 0, addition.width, 0, 0, addition.width, addition.height)

        for (i in outputPixels.indices) {
            outputPixels[i] = if (
                android.graphics.Color.alpha(basePixels[i]) > 0 ||
                android.graphics.Color.alpha(additionPixels[i]) > 0
            ) {
                android.graphics.Color.WHITE
            } else {
                android.graphics.Color.TRANSPARENT
            }
        }
        return Bitmap.createBitmap(outputPixels, base.width, base.height, Bitmap.Config.ARGB_8888)
    }

    // توسعة binary separable بتمريرين؛ تظهر مباشرة في معاينة القناع قبل الإرسال إلى Big LaMa.
    fun dilateMask(mask: Bitmap, radiusPx: Int): Bitmap {
        val radius = radiusPx.coerceAtLeast(0)
        if (radius == 0) return mask.copy(Bitmap.Config.ARGB_8888, false)

        val width = mask.width
        val height = mask.height
        val source = IntArray(width * height)
        mask.getPixels(source, 0, width, 0, 0, width, height)
        val horizontal = BooleanArray(source.size)
        val output = IntArray(source.size)

        for (y in 0 until height) {
            var active = 0
            for (x in -radius .. radius) {
                if (x in 0 until width && android.graphics.Color.alpha(source[y * width + x]) > 0) {
                    active++
                }
            }
            for (x in 0 until width) {
                horizontal[y * width + x] = active > 0
                val removeX = x - radius
                val addX = x + radius + 1
                if (removeX in 0 until width && android.graphics.Color.alpha(source[y * width + removeX]) > 0) {
                    active--
                }
                if (addX in 0 until width && android.graphics.Color.alpha(source[y * width + addX]) > 0) {
                    active++
                }
            }
        }

        for (x in 0 until width) {
            var active = 0
            for (y in -radius .. radius) {
                if (y in 0 until height && horizontal[y * width + x]) active++
            }
            for (y in 0 until height) {
                output[y * width + x] = if (active > 0) {
                    android.graphics.Color.WHITE
                } else {
                    android.graphics.Color.TRANSPARENT
                }
                val removeY = y - radius
                val addY = y + radius + 1
                if (removeY in 0 until height && horizontal[removeY * width + x]) active--
                if (addY in 0 until height && horizontal[addY * width + x]) active++
            }
        }
        return Bitmap.createBitmap(output, width, height, Bitmap.Config.ARGB_8888)
    }

    /**
 * Erosion separable بتمريرين؛ عكس dilation.
 * يستخدم لـ Closing (dilation + erosion) عشان يملأ الفجوات الصغيرة.
 */
    fun erodeMask(mask: Bitmap, radiusPx: Int): Bitmap {
        val radius = radiusPx.coerceAtLeast(0)
        if (radius == 0) return mask.copy(Bitmap.Config.ARGB_8888, false)

        val width = mask.width
        val height = mask.height
        val source = IntArray(width * height)
        mask.getPixels(source, 0, width, 0, 0, width, height)
        val horizontal = BooleanArray(source.size)
        val output = IntArray(source.size)

        val hWindow = (2 * radius + 1).coerceAtMost(width)
        val vWindow = (2 * radius + 1).coerceAtMost(height)

        for (y in 0 until height) {
            var active = 0
            for (x in -radius .. radius) {
                if (x in 0 until width && android.graphics.Color.alpha(source[y * width + x]) > 0) {
                    active++
                }
            }
            for (x in 0 until width) {
                horizontal[y * width + x] = active >= hWindow
                val removeX = x - radius
                val addX = x + radius + 1
                if (removeX in 0 until width && android.graphics.Color.alpha(source[y * width + removeX]) > 0) {
                    active--
                }
                if (addX in 0 until width && android.graphics.Color.alpha(source[y * width + addX]) > 0) {
                    active++
                }
            }
        }

        for (x in 0 until width) {
            var active = 0
            for (y in -radius .. radius) {
                if (y in 0 until height && horizontal[y * width + x]) active++
            }
            for (y in 0 until height) {
                output[y * width + x] = if (active >= vWindow) {
                    android.graphics.Color.WHITE
                } else {
                    android.graphics.Color.TRANSPARENT
                }
                val removeY = y - radius
                val addY = y + radius + 1
                if (removeY in 0 until height && horizontal[removeY * width + x]) active--
                if (addY in 0 until height && horizontal[addY * width + x]) active++
            }
        }
        return Bitmap.createBitmap(output, width, height, Bitmap.Config.ARGB_8888)
    }

    /**
 * Closing = dilation ثم erosion بنفس الـ radius.
 * يملأ الفجوات الصغيرة جوه القناع ويحافظ على الحجم العام.
 */
    /**
 * Closing = dilation ثم erosion بنفس الـ radius.
 *
 * Ownership:
 * - mask: borrowed، لا نحرره هنا.
 * - dilated: intermediate مملوك لهذه الدالة ويتم تحريره
 *   بعد إنتاج نتيجة erosion.
 */
    fun closeMask(mask: Bitmap, radiusPx: Int): Bitmap {
        val dilated = dilateMask(mask, radiusPx)

        return try {
            erodeMask(dilated, radiusPx)
        } finally {
            if (dilated !== mask && !dilated.isRecycled) {
                dilated.recycle()
            }
        }
    }

    /**
 * Feathering بتنعيم حواف القناع باستخدام BlurMaskFilter.
 * يُستخدم للمعاينة فقط (ليس للإرسال إلى LaMa).
 */
    fun featherMask(mask: Bitmap, blurRadius: Float): Bitmap {
        if (blurRadius <= 0) return mask.copy(Bitmap.Config.ARGB_8888, false)

        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        paint.maskFilter = BlurMaskFilter(blurRadius, BlurMaskFilter.Blur.NORMAL)

        val result = Bitmap.createBitmap(mask.width, mask.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        canvas.drawBitmap(mask, 0f, 0f, paint)
        return result
    }

    // قراءة النتيجة اللي الـ Worker حفظها على القرص بعد المعالجة في الخلفية
    fun fileToBitmap(file: File): Bitmap? {
        return try {
            BitmapFactory.decodeFile(file.absolutePath)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    // حفظ الصورة النهائية في معرض الجهاز (زر "حفظ الصورة")
    fun saveBitmapToGallery(context: Context, bitmap: Bitmap): Boolean {
        val fileName = "MasterAI_${System.currentTimeMillis()}.jpg"
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/MasterAIStudio")
                }
                val uri = context.contentResolver.insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values
                ) ?: return false

                context.contentResolver.openOutputStream(uri)?.use { out ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
                } ?: return false
                true
            } else {
                @Suppress("DEPRECATION")
                val picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
                val appDir = File(picturesDir, "MasterAIStudio")
                if (!appDir.exists()) appDir.mkdirs()
                val file = File(appDir, fileName)
                FileOutputStream(file).use { out ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
                }
                MediaStore.Images.Media.insertImage(
                    context.contentResolver, file.absolutePath, fileName, null
                )
                true
            }
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }
}
