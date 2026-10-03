package com.master.aistudio

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.max
import kotlin.math.min

/**
 * يحلل صور المصدر والعمق والقناع على المعالج فقط (بدون أي استدعاء GL) ليقترح
 * قيماً أولية ذكية للسلايدرات الأربعة الموجودة حالياً في MainActivity، بدل
 * تركها دائماً على نفس القيم الثابتة بغض النظر عن نوع المشهد (ليلي خافت،
 * إضاءة متوسطة، أو نهاري ساطع).
 *
 * تصميم هذا الكلاس مقصود أن يكون على مستوى "قيمة السلايدر من 0 إلى 100"
 * وليس على مستوى GL uniforms مباشرة، لأن معادلات تحويل السلايدر إلى قيم
 * الرندرر (curved bokeh size، مدى lightThreshold، إلخ) موجودة بالفعل داخل
 * MainActivity.setupSliders / applyCurrentSliderValuesToGL. تكرار هذه
 * المعادلات هنا كان سيُنتج مصدرين مختلفين للحقيقة قابلين للانحراف عن بعضهما
 * مستقبلاً؛ لذلك هذا الكلاس يقترح فقط "أين يجب أن يقف كل سلايدر"، وتبقى
 * MainActivity هي المسؤولة الوحيدة عن كيفية ترجمة ذلك إلى GL.
 *
 * القراءة على العمق والقناع هنا تفترض الترميز الحالي فقط: عمق grayscale
 * عادي مباشر (القناة الحمراء = قيمة العمق نفسها 0..255)، وقناع بنفس المبدأ
 * (القناة الحمراء = شفافية الشخص 0..255)، بدون أي ترميز 16-bit.
 */
class AutoTuningEstimator(
    private val targetSampleCount: Int = DEFAULT_TARGET_SAMPLE_COUNT
) {

    /**
     * القيم المقترحة، كل واحدة جاهزة للإسناد مباشرة إلى value الخاصة بالسلايدر
     * المقابل (Slider من Material Components)، بنفس مدى 0 إلى 100 المستخدم
     * حالياً في MainActivity.
     */
    data class AutoTuningResult(
        val blurSliderValue: Float,
        val colorBoostSliderValue: Float,
        val lightThresholdSliderValue: Float,
        val highlightBoostSliderValue: Float
    )

    private data class SceneStatistics(
        val meanBackgroundLuminance: Float,
        val backgroundHighlightPercentileLuminance: Float,
        val meanBackgroundSaturation: Float,
        val medianBackgroundDepth: Float,
        val highlightPixelRatio: Float,
        val hasBackgroundSamples: Boolean
    )

    /**
     * يحلل المشهد ويقترح قيم السلايدرات الأربعة.
     *
     * @param source صورة المصدر (يفضَّل استخدام نفس sourceForGpu الموجودة
     * أصلاً في MainActivity، لأنها بنفس دقة وحجم depth وmask تماماً).
     * @param depth صورة العمق المطابقة لنفس أبعاد [source] تماماً.
     * @param mask صورة قناع الشخص المطابقة لنفس أبعاد [source] تماماً.
     * @param focusDepth عمق نقطة التركيز المحسوب مسبقاً عبر
     * computeFocusDepthFast الموجودة أصلاً في MainActivity، لتفادي حساب نفس
     * القيمة مرتين.
     */
    fun estimate(
        source: Bitmap,
        depth: Bitmap,
        mask: Bitmap,
        focusDepth: Float
    ): AutoTuningResult {

        val width = source.width
        val height = source.height

        if (width <= 0 || height <= 0) {
            return fallbackResult()
        }

        val statistics = collectSceneStatistics(source, depth, mask, width, height)

        if (!statistics.hasBackgroundSamples) {
            return fallbackResult()
        }

        /*
         * الأربع Sliders في activity_main.xml مضبوطة بـ stepSize = 1.0، أي أن
         * قيمتها يجب أن تكون دائماً رقماً صحيحاً بالضبط (BaseSlider.validateValues
         * يرمي IllegalStateException عند أي قيمة عشرية مثل 57.575073). المعادلات
         * أعلاه تُنتج أرقاماً عشرية دقيقة، لذلك يجب تقريبها هنا في نقطة واحدة
         * قبل إعادتها، بدل تعديل كل معادلة على حدة.
         */
        return AutoTuningResult(
            blurSliderValue = roundToSliderStep(
                estimateBlurSliderValue(statistics, focusDepth)
            ),
            colorBoostSliderValue = roundToSliderStep(
                estimateColorBoostSliderValue(statistics)
            ),
            lightThresholdSliderValue = roundToSliderStep(
                estimateLightThresholdSliderValue(statistics)
            ),
            highlightBoostSliderValue = roundToSliderStep(
                estimateHighlightBoostSliderValue(statistics)
            )
        )
    }

    private fun roundToSliderStep(value: Float): Float {
        return kotlin.math.round(value).coerceIn(0f, 100f)
    }

    private fun collectSceneStatistics(
        source: Bitmap,
        depth: Bitmap,
        mask: Bitmap,
        width: Int,
        height: Int
    ): SceneStatistics {

        val totalPixels = width * height
        val step = max(1, totalPixels / targetSampleCount)

        val sourcePixels = IntArray(width * height)
        val depthPixels = IntArray(width * height)
        val maskPixels = IntArray(width * height)

        source.getPixels(sourcePixels, 0, width, 0, 0, width, height)
        depth.getPixels(depthPixels, 0, width, 0, 0, width, height)
        mask.getPixels(maskPixels, 0, width, 0, 0, width, height)

        val luminanceSamples = ArrayList<Float>(targetSampleCount)
        val depthSamples = ArrayList<Float>(targetSampleCount)

        var saturationSum = 0f
        var backgroundSampleCount = 0
        var highlightSampleCount = 0

        var index = 0

        while (index < totalPixels) {
            val maskAlpha = Color.red(maskPixels[index]) / 255f

            if (maskAlpha < BACKGROUND_MASK_THRESHOLD) {
                val pixel = sourcePixels[index]

                val red = Color.red(pixel) / 255f
                val green = Color.green(pixel) / 255f
                val blue = Color.blue(pixel) / 255f

                val luminance = luminanceOf(red, green, blue)
                val saturation = saturationOf(red, green, blue)

                val depthValue = Color.red(depthPixels[index]) / 255f

                luminanceSamples.add(luminance)
                depthSamples.add(depthValue)

                saturationSum += saturation
                backgroundSampleCount++

                if (luminance >= HIGHLIGHT_PIXEL_LUMINANCE) {
                    highlightSampleCount++
                }
            }

            index += step
        }

        if (backgroundSampleCount == 0) {
            return SceneStatistics(
                meanBackgroundLuminance = 0.5f,
                backgroundHighlightPercentileLuminance = 0.8f,
                meanBackgroundSaturation = 0.3f,
                medianBackgroundDepth = 0.5f,
                highlightPixelRatio = 0f,
                hasBackgroundSamples = false
            )
        }

        luminanceSamples.sort()
        depthSamples.sort()

        val meanLuminance = luminanceSamples.sum() / luminanceSamples.size

        val percentileIndex = (
            (luminanceSamples.size - 1) * HIGHLIGHT_PERCENTILE
            ).toInt().coerceIn(0, luminanceSamples.size - 1)

        val highlightPercentileLuminance = luminanceSamples[percentileIndex]

        val medianDepth = depthSamples[depthSamples.size / 2]

        val meanSaturation = saturationSum / backgroundSampleCount

        val highlightRatio = highlightSampleCount.toFloat() / backgroundSampleCount

        return SceneStatistics(
            meanBackgroundLuminance = meanLuminance,
            backgroundHighlightPercentileLuminance = highlightPercentileLuminance,
            meanBackgroundSaturation = meanSaturation,
            medianBackgroundDepth = medianDepth,
            highlightPixelRatio = highlightRatio,
            hasBackgroundSamples = true
        )
    }

    /**
     * كلما كان المشهد أغمق (ليلي) تحتاج قوة تمويه أعلى لإبراز أثر البوكيه،
     * وكلما كان الفصل العمقي بين الشخص والخلفية صغيراً تحتاج تمويهاً أعلى
     * أيضاً لتعويض غياب الفصل الطبيعي.
     */
    private fun estimateBlurSliderValue(
        statistics: SceneStatistics,
        focusDepth: Float
    ): Float {

        val nightFactor = (1f - statistics.meanBackgroundLuminance).coerceIn(0f, 1f)

        val depthSeparation = kotlin.math.abs(
            statistics.medianBackgroundDepth - focusDepth
        ).coerceIn(0f, 1f)

        val flatDepthFactor = 1f - depthSeparation

        val value = BLUR_BASE +
            flatDepthFactor * BLUR_DEPTH_WEIGHT +
            nightFactor * BLUR_NIGHT_WEIGHT

        return value.coerceIn(BLUR_MIN, BLUR_MAX)
    }

    /**
     * خلفية منخفضة التشبع (شائعة في مشاهد الليل تحت إضاءة أبيض/أصفر باهتة)
     * تحتاج تعزيز ألوان أعلى ليظهر البوكيه نابضاً بالحياة، بينما خلفية
     * مشبعة أصلاً (غروب، أضواء نيون ملوّنة) تحتاج تعزيزاً أقل لتفادي تشبّع
     * زائد يبدو غير طبيعي.
     */
    private fun estimateColorBoostSliderValue(statistics: SceneStatistics): Float {
        val desaturation = (COLOR_BOOST_REFERENCE_SATURATION -
            statistics.meanBackgroundSaturation).coerceIn(0f, 1f)

        val value = desaturation * COLOR_BOOST_WEIGHT

        return value.coerceIn(COLOR_BOOST_MIN, COLOR_BOOST_MAX)
    }

    /**
     * الهدف أن يقف u_lightThreshold الفعلي قريباً من بيرسنتايل السطوع
     * المرتفع في الخلفية، بحيث تُلتقط فعلاً أسطع نقاط المشهد كـ"إضاءة" دون
     * أن تُغرق كامل الخلفية بوزن الإضاءة. بما أن معادلة MainActivity تحوّل
     * قيمة السلايدر إلى threshold ضمن المدى [0.45, 1.0] فقط
     * (threshold = 1.0 - (value/100) * 0.55)، يُقيَّد الهدف هنا لنفس المدى
     * القابل للتحقيق فعلياً عبر السلايدر.
     */
    private fun estimateLightThresholdSliderValue(statistics: SceneStatistics): Float {
        val targetThreshold = statistics.backgroundHighlightPercentileLuminance
            .coerceIn(LIGHT_THRESHOLD_TARGET_MIN, LIGHT_THRESHOLD_TARGET_MAX)

        val value = (1f - targetThreshold) / LIGHT_THRESHOLD_SLIDER_SCALE * 100f

        return value.coerceIn(0f, 100f)
    }

    /**
     * مشاهد ليلية بها نقاط ضوء قليلة لكنها شديدة السطوع (أعمدة إنارة، لافتات)
     * تستفيد من highlightBoost عالٍ لتتحول هذه النقاط إلى بقع متوهجة واضحة.
     * مشاهد نهارية بها مساحات سطوع واسعة (سماء، جدران مضاءة بالشمس) تحتاج
     * قيمة أقل حتى لا تُحرق هذه المساحات بالكامل.
     */
    private fun estimateHighlightBoostSliderValue(statistics: SceneStatistics): Float {
        val nightFactor = (1f - statistics.meanBackgroundLuminance).coerceIn(0f, 1f)

        val sparsity = (1f - min(
            1f,
            statistics.highlightPixelRatio * HIGHLIGHT_RATIO_SPARSITY_SCALE
        )).coerceIn(0f, 1f)

        val value = HIGHLIGHT_BOOST_BASE +
            nightFactor * sparsity * HIGHLIGHT_BOOST_NIGHT_POINT_LIGHT_WEIGHT

        return value.coerceIn(HIGHLIGHT_BOOST_MIN, HIGHLIGHT_BOOST_MAX)
    }

    private fun luminanceOf(red: Float, green: Float, blue: Float): Float {
        return LUMINANCE_RED_WEIGHT * red +
            LUMINANCE_GREEN_WEIGHT * green +
            LUMINANCE_BLUE_WEIGHT * blue
    }

    private fun saturationOf(red: Float, green: Float, blue: Float): Float {
        val maxChannel = max(red, max(green, blue))
        val minChannel = min(red, min(green, blue))

        if (maxChannel <= 0f) {
            return 0f
        }

        return (maxChannel - minChannel) / maxChannel
    }

    private fun fallbackResult(): AutoTuningResult {
        return AutoTuningResult(
            blurSliderValue = FALLBACK_BLUR_SLIDER_VALUE,
            colorBoostSliderValue = FALLBACK_COLOR_BOOST_SLIDER_VALUE,
            lightThresholdSliderValue = FALLBACK_LIGHT_THRESHOLD_SLIDER_VALUE,
            highlightBoostSliderValue = FALLBACK_HIGHLIGHT_BOOST_SLIDER_VALUE
        )
    }

    companion object {

        private const val DEFAULT_TARGET_SAMPLE_COUNT = 20000

        private const val BACKGROUND_MASK_THRESHOLD = 0.5f
        private const val HIGHLIGHT_PIXEL_LUMINANCE = 0.75f
        private const val HIGHLIGHT_PERCENTILE = 0.90f

        private const val LUMINANCE_RED_WEIGHT = 0.2126f
        private const val LUMINANCE_GREEN_WEIGHT = 0.7152f
        private const val LUMINANCE_BLUE_WEIGHT = 0.0722f

        private const val BLUR_BASE = 20f
        private const val BLUR_DEPTH_WEIGHT = 30f
        private const val BLUR_NIGHT_WEIGHT = 25f
        private const val BLUR_MIN = 15f
        private const val BLUR_MAX = 85f

        private const val COLOR_BOOST_REFERENCE_SATURATION = 0.55f
        private const val COLOR_BOOST_WEIGHT = 160f
        private const val COLOR_BOOST_MIN = 10f
        private const val COLOR_BOOST_MAX = 35f

        private const val LIGHT_THRESHOLD_TARGET_MIN = 0.45f
        private const val LIGHT_THRESHOLD_TARGET_MAX = 0.95f
        private const val LIGHT_THRESHOLD_SLIDER_SCALE = 0.55f

        private const val HIGHLIGHT_RATIO_SPARSITY_SCALE = 12f
        private const val HIGHLIGHT_BOOST_BASE = 15f
        private const val HIGHLIGHT_BOOST_NIGHT_POINT_LIGHT_WEIGHT = 55f
        private const val HIGHLIGHT_BOOST_MIN = 10f
        private const val HIGHLIGHT_BOOST_MAX = 90f

        private const val FALLBACK_BLUR_SLIDER_VALUE = 40f
        private const val FALLBACK_COLOR_BOOST_SLIDER_VALUE = 50f
        private const val FALLBACK_LIGHT_THRESHOLD_SLIDER_VALUE = 65f
        private const val FALLBACK_HIGHLIGHT_BOOST_SLIDER_VALUE = 40f
    }
}
