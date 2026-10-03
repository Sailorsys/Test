package com.master.aistudio.utils

import android.graphics.Bitmap
import android.util.Log

class NativeSmartEnhancer {

    companion object {
        init {
            try {
                // 1. تحميل مكتبة C++ القياسية أولاً
                System.loadLibrary("c++_shared")
                // 2. تحميل مكتبة OpenCV
                System.loadLibrary("opencv_java4")
                // 3. تحميل مكتبتك الخاصة
                System.loadLibrary("smart_enhancer")
                
                Log.d("NativeSmartEnhancer", "تم تحميل جميع المكتبات بنجاح!")
            } catch (e: Throwable) {
                Log.e("NativeSmartEnhancer", "فشل تحميل المكتبات: ${e.message}", e)
                throw RuntimeException("فشل تحميل المكتبات Native: ${e.message}", e)
            }
        }
    }

    private external fun nativeEnhance(src: Bitmap, dst: Bitmap)

    fun enhance(sourceBitmap: Bitmap): Bitmap {
        val config = sourceBitmap.config ?: Bitmap.Config.ARGB_8888
        val outputBitmap = Bitmap.createBitmap(
            sourceBitmap.width,
            sourceBitmap.height,
            config
        )
        
        nativeEnhance(sourceBitmap, outputBitmap)
        
        return outputBitmap
    }
}
