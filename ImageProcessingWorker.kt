package com.master.aistudio

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.ServiceInfo
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.ForegroundInfo
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.master.aistudio.R
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.CancellationException

/**
 * بيشتغل كـ Foreground Service مضمون من النظام، فمش بيتقفل لو المستخدم
 * خرج من التطبيق أو قفل الشاشة. بيعرض إشعار دائم أثناء المعالجة، وإشعار
 * عادي (قابل للتفاعل) لما يخلص.
 */
class ImageProcessingWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    companion object {
        const val UNIQUE_WORK_NAME = "master_ai_studio_processing"
        const val KEY_TOOL = "tool"
        const val KEY_IMAGE_PATH = "image_path"
        const val KEY_MASK_PATH = "mask_path"
        const val KEY_FIDELITY = "fidelity"
        const val KEY_BLUR = "blur"
        const val KEY_RESULT_PATH = "result_path"

        private const val CHANNEL_ID = "image_processing_channel"
        private const val PROGRESS_NOTIFICATION_ID = 1001
        private const val DONE_NOTIFICATION_ID = 1002
    }

    override suspend fun doWork(): Result {
        val tool = inputData.getString(KEY_TOOL) ?: return Result.failure()
        val imagePath = inputData.getString(KEY_IMAGE_PATH) ?: return Result.failure()
        val maskPath = inputData.getString(KEY_MASK_PATH)
        val fidelity = inputData.getFloat(KEY_FIDELITY, 0.7f)
        val blur = inputData.getInt(KEY_BLUR, 25)

        val imageFile = File(imagePath)
        if (!imageFile.exists()) return Result.failure()

        setForeground(createForegroundInfo(statusMessage(tool)))

        val resultBytes = try {
            when (tool) {
                "FACE" -> NetworkManager.enhanceFace(imageFile)

                "FULL" -> NetworkManager.enhanceFull(imageFile)

                "REMOVE" -> {
                    val maskFile = maskPath?.let { File(it) }
                    ?: return Result.failure()

                    NetworkManager.inpaintObject(imageFile, maskFile)
                }

                "PORTRAIT" -> NetworkManager.portraitBlur(imageFile, blur)

                else -> null
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            e.printStackTrace()
            null
        } finally {
            imageFile.delete()
            maskPath?.let { File(it).delete() }
        }

        // تنظيف ملفات الإدخال المؤقتة بعد الاستخدام مباشرة
        imageFile.delete()
        maskPath?.let { File(it).delete() }

        return if (resultBytes != null) {
            val resultFile = File(applicationContext.filesDir, "result_${System.currentTimeMillis()}.jpg")
            FileOutputStream(resultFile).use { it.write(resultBytes) }
            showDoneNotification(success = true)
            Result.success(workDataOf(KEY_RESULT_PATH to resultFile.absolutePath))
        } else {
            showDoneNotification(success = false)
            Result.failure()
        }
    }

    private fun statusMessage(tool: String): String = when (tool) {
        "FACE" -> "جاري تحسين الوجه..."
        "FULL" -> "جاري تحسين الصورة بالكامل..."
        "REMOVE" -> "جاري إزالة العنصر..."
        "PORTRAIT" -> "جاري عمل تأثير البورتريه..."
        else -> "جاري المعالجة..."
    }

    private fun createForegroundInfo(message: String): ForegroundInfo {
        createChannelIfNeeded()
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
        .setContentTitle("Master AI Studio")
        .setContentText(message)
        .setSmallIcon(R.drawable.ic_tool_full)
        .setOngoing(true)
        .setProgress(0, 0, true)
        .build()

        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ForegroundInfo(PROGRESS_NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            ForegroundInfo(PROGRESS_NOTIFICATION_ID, notification)
        }
    }

    private fun showDoneNotification(success: Boolean) {
        createChannelIfNeeded()
        val title = if (success) "تمت المعالجة بنجاح ✅" else "حصل خطأ أثناء المعالجة ❌"
        val text = if (success) "افتح التطبيق لمشاهدة النتيجة" else "افتح التطبيق وجرّب تاني"

        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
        .setContentTitle(title)
        .setContentText(text)
        .setSmallIcon(R.drawable.ic_tool_full)
        .setAutoCancel(true)
        .build()

        val manager = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(DONE_NOTIFICATION_ID, notification)
    }

    private fun createChannelIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val manager = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            val channel = NotificationChannel(
                CHANNEL_ID,
                "معالجة الصور",
                NotificationManager.IMPORTANCE_LOW
            )
            manager.createNotificationChannel(channel)
        }
    }
}
