package com.master.aistudio

import android.util.Log
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

/**
 * مصنع موحّد لإنشاء جلسات ONNX Runtime.
 *
 * الإعدادات المُختارة مبنية على قياسات فعلية على أجهزة حقيقية:
 * - XNNPACK: أسرع بحوالي 10% من CPU على MobileSAM، مستقر على كل
 *   أجهزة ARM، ومفيش أي اعتماد على تعريفات الشركة المصنعة.
 * - NNAPI: مستبعد تماماً. أظهر فشل قاطع على Exynos 990
 *   (FAILED_TRANSACTION من تعريفات Samsung)، وسلوكه على أجهزة
 *   Snapdragon غير مضمون.
 *
 * كل الإعدادات المشتركة بتتظبط هنا من مكان واحد، فأي تعديل مستقبلي
 * (عدد الخيوط، مزود التنفيذ، مستوى التحسين...) ينعكس على كل الموديلات
 * تلقائياً.
 */
object OnnxSessionFactory {

    private const val TAG = "OnnxSessionFactory"

    // 4 خيوط = أفضل توازن على معالجات big.LITTLE.
    // 1 خيط inter-op = التنفيذ المتسلسل أسرع من المتوازي على الموبايل
    // (الـ inter-op parallelism مفيد أساساً عند خدمة عدة طلبات متزامنة).
    private const val INTRA_OP_THREADS = 4
    private const val INTER_OP_THREADS = 1

    /**
     * ينشئ جلسة ONNX Runtime جاهزة بإعدادات موحّدة.
     *
     * @param modelBytes بايتات النموذج (مقروءة من assets أو ملف)
     * @param environment بيئة ONNX (singleton على مستوى التطبيق)
     * @param logTag وسم للـ logging يميّز الموديل الحالي
     * @return جلسة جاهزة للاستدلال. مسؤولية المستدعي إغلاقها.
     */
    fun createSession(
        modelBytes: ByteArray,
        environment: OrtEnvironment,
        logTag: String
    ): OrtSession {
        val startTime = System.currentTimeMillis()

        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(INTRA_OP_THREADS)
            setInterOpNumThreads(INTER_OP_THREADS)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

            try {
                addXnnpack(emptyMap<String, String>())
            } catch (e: Exception) {
                // لو XNNPACK فشل لأي سبب (نادر جداً)، الجلسة بتكمل على
                // CPU العادي بدل ما نرمي exception ونوقف التطبيق.
                Log.w(TAG, "[$logTag] XNNPACK failed, falling back to CPU", e)
            }
        }

        return try {
            environment.createSession(modelBytes, options).also {
                Log.d(
                    TAG,
                    "[$logTag] Session created in ${System.currentTimeMillis() - startTime} ms"
                )
            }
        } finally {
            // SessionOptions يحتل ذاكرة native — لازم يُغلق بعد createSession
            options.close()
        }
    }
}