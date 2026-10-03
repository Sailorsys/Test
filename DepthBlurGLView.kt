package com.master.aistudio

import android.content.Context
import android.graphics.Bitmap
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.util.AttributeSet
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class DepthBlurGLView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : GLSurfaceView(context, attrs) {

    private val renderer = DepthBlurRenderer()

    init {
        setEGLContextClientVersion(3)
        setRenderer(renderer)
        renderMode = RENDERMODE_WHEN_DIRTY

        /*
         * بدون هذا، يهدم أندرويد سياق EGL بالكامل تلقائياً كل مرة يروح
         * فيها التطبيق للخلفية ثم يرجع (Home، تبديل تطبيق، إلخ)، والكود
         * الحالي لا يعيد رفع الصور (mainTextureId/depthTextureId/maskTextureId)
         * بعد إعادة بناء السياق — فتظهر الشاشة سوداء أو تالفة عند العودة
         * للتطبيق أثناء عرض تأثير البوكيه. preserveEGLContextOnPause يبقي
         * السياق حياً طوال الوقت، فيتفادى المشكلة من جذرها بسطر واحد.
         */
        preserveEGLContextOnPause = true
    }

    fun setBitmaps(main: Bitmap, depth: Bitmap, mask: Bitmap) {
        queueEvent {
            renderer.updateTextures(main, depth, mask)
            requestRender()
        }
    }

    fun setBlurStrength(strength: Float) {
        queueEvent {
            renderer.blurStrength = strength.coerceIn(0f, 1f)
            requestRender()
        }
    }

    fun setFocusDepth(focus: Float) {
        queueEvent {
            renderer.focusDepth = focus.coerceIn(0f, 1f)
            requestRender()
        }
    }

    fun setLightThreshold(threshold: Float) {
        queueEvent {
            renderer.lightThreshold = threshold.coerceIn(0f, 1f)
            requestRender()
        }
    }

    fun setMaxBokehSize(relativeSize: Float) {
        queueEvent {
            renderer.maxBokehSize = relativeSize.coerceIn(0f, 1f)
            requestRender()
        }
    }

    fun setFeetProtection(x: Float, y: Float, radius: Float, aspect: Float) {
        queueEvent {
            renderer.feetX = x
            renderer.feetY = y
            renderer.feetRadius = radius
            renderer.imageAspect = aspect
            requestRender()
        }
    }

    fun setBokehColorBoost(boost: Float) {
        queueEvent {
            renderer.bokehColorBoost = boost.coerceIn(0f, 1f)
            requestRender()
        }
    }

    fun setHighlightBoost(boost: Float) {
        queueEvent {
            renderer.highlightBoost = boost.coerceIn(0f, 1f)
            requestRender()
        }
    }

    fun exportFullResolution(
        source: Bitmap,
        depth: Bitmap,
        mask: Bitmap,
        onComplete: (Bitmap?) -> Unit
    ) {
        queueEvent {
            val result = try {
                renderer.renderFullResolution(source, depth, mask)
            } catch (e: Exception) {
                android.util.Log.e("DepthBlurGLView", "Export failed", e)
                null
            }
            post { onComplete(result) }
        }
    }

    private inner class DepthBlurRenderer : Renderer {

        var blurStrength: Float = 0.10f
        var focusDepth: Float = 0.5f
        var lightThreshold: Float = 0.30f
        var maxBokehSize: Float = 40f
        var bokehColorBoost: Float = 0.5f
        var highlightBoost: Float = 0.5f
        var feetX: Float = 0f
        var feetY: Float = 0f
        var feetRadius: Float = 0f
        var imageAspect: Float = 1f

        private var viewportWidth = 0
        private var viewportHeight = 0
        private var imageWidth = 1
        private var imageHeight = 1

        private var scaleX = 1.0f
        private var scaleY = 1.0f

        private var mainTextureId = 0
        private var depthTextureId = 0
        private var maskTextureId = 0

        private var fboId = 0
        private var fboTextureId = 0

        private var pendingMain: Bitmap? = null
        private var pendingDepth: Bitmap? = null
        private var pendingMask: Bitmap? = null
        private var hasNewBitmaps = false

        private var pass1Program = 0
        private var pass2QuadProgram = 0

        private lateinit var quadBuffer: FloatBuffer

        fun renderFullResolution(main: Bitmap, depth: Bitmap, mask: Bitmap): Bitmap? {
            if (pass1Program == 0) return null

            val w = main.width
            val h = main.height

            val exportMainTex = loadTexture(main, 0)
            val exportDepthTex = loadTexture(depth, 0)
            val exportMaskTex = loadTexture(mask, 0)

            val fboArr = IntArray(1)
            val texArr = IntArray(1)
            GLES30.glGenFramebuffers(1, fboArr, 0)
            GLES30.glGenTextures(1, texArr, 0)
            val exportFboId = fboArr[0]
            val exportFboTex = texArr[0]

            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, exportFboTex)
            GLES30.glTexImage2D(
                GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA, w, h, 0,
                GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
            )
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, exportFboId)
            GLES30.glFramebufferTexture2D(
                GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
                GLES30.GL_TEXTURE_2D, exportFboTex, 0
            )

            val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
            if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) {
                android.util.Log.e("DepthBlurGLView", "Export FBO incomplete: $status")
                cleanupExportResources(exportMainTex, exportDepthTex, exportMaskTex, exportFboId, exportFboTex)
                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
                return null
            }

            GLES30.glViewport(0, 0, w, h)
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
            GLES30.glUseProgram(pass1Program)

            bindTexture(exportMainTex, GLES30.GL_TEXTURE0, "u_mainTex", pass1Program)
            bindTexture(exportDepthTex, GLES30.GL_TEXTURE1, "u_depthTex", pass1Program)
            bindTexture(exportMaskTex, GLES30.GL_TEXTURE2, "u_maskTex", pass1Program)

            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_blurStrength"), blurStrength)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_focusDepth"), focusDepth)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_maxBokehSize"), maxBokehSize)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_lightThreshold"), lightThreshold)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_highlightBoost"), highlightBoost)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_colorBoost"), bokehColorBoost)
            GLES30.glUniform2f(GLES30.glGetUniformLocation(pass1Program, "u_scale"), 1.0f, 1.0f)
            GLES30.glUniform2f(GLES30.glGetUniformLocation(pass1Program, "u_feetPoint"), feetX, feetY)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_feetRadius"), feetRadius)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_imageAspect"), imageAspect)
            drawQuad(pass1Program)

            val buffer = ByteBuffer.allocateDirect(w * h * 4).order(ByteOrder.nativeOrder())
            GLES30.glReadPixels(0, 0, w, h, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, buffer)
            buffer.rewind()

            val rawBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            rawBitmap.copyPixelsFromBuffer(buffer)

            val flipMatrix = android.graphics.Matrix().apply { postScale(1f, -1f, w / 2f, h / 2f) }
            val finalBitmap = Bitmap.createBitmap(rawBitmap, 0, 0, w, h, flipMatrix, true)
            if (finalBitmap != rawBitmap) rawBitmap.recycle()

            cleanupExportResources(exportMainTex, exportDepthTex, exportMaskTex, exportFboId, exportFboTex)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
            GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
            requestRender()

            return finalBitmap
        }

        private fun cleanupExportResources(
            mainTex: Int, depthTex: Int, maskTex: Int, fboId: Int, fboTex: Int
        ) {
            GLES30.glDeleteTextures(3, intArrayOf(mainTex, depthTex, maskTex), 0)
            GLES30.glDeleteFramebuffers(1, intArrayOf(fboId), 0)
            GLES30.glDeleteTextures(1, intArrayOf(fboTex), 0)
        }

        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            GLES30.glClearColor(0f, 0f, 0f, 1f)

            initBuffers()
            initPass1Shader()
            initPass2QuadShader()
        }

        override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
            viewportWidth = width
            viewportHeight = height
            GLES30.glViewport(0, 0, width, height)
            setupFBO(width, height)
            updateAspectScale()
        }

        override fun onDrawFrame(gl: GL10?) {
            if (hasNewBitmaps) {
                uploadBitmapsGL()
                hasNewBitmaps = false
            }

            if (mainTextureId == 0) return

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)
            GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

            GLES30.glUseProgram(pass1Program)
            bindTexture(mainTextureId, GLES30.GL_TEXTURE0, "u_mainTex", pass1Program)
            bindTexture(depthTextureId, GLES30.GL_TEXTURE1, "u_depthTex", pass1Program)
            bindTexture(maskTextureId, GLES30.GL_TEXTURE2, "u_maskTex", pass1Program)

            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_blurStrength"), blurStrength)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_focusDepth"), focusDepth)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_maxBokehSize"), maxBokehSize)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_lightThreshold"), lightThreshold)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_highlightBoost"), highlightBoost)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_colorBoost"), bokehColorBoost)
            GLES30.glUniform2f(GLES30.glGetUniformLocation(pass1Program, "u_scale"), scaleX, scaleY)
            GLES30.glUniform2f(GLES30.glGetUniformLocation(pass1Program, "u_feetPoint"), feetX, feetY)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_feetRadius"), feetRadius)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_imageAspect"), imageAspect)
            drawQuad(pass1Program)

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
            GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

            GLES30.glUseProgram(pass2QuadProgram)
            bindTexture(fboTextureId, GLES30.GL_TEXTURE0, "u_fboTex", pass2QuadProgram)
            drawQuad(pass2QuadProgram)
        }

        fun updateTextures(main: Bitmap, depth: Bitmap, mask: Bitmap) {
            pendingMain = main
            pendingDepth = depth
            pendingMask = mask
            hasNewBitmaps = true
        }

        private fun updateAspectScale() {
            if (viewportWidth <= 0 || viewportHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) return

            val imageAspect = imageWidth.toFloat() / imageHeight.toFloat()
            val viewAspect = viewportWidth.toFloat() / viewportHeight.toFloat()

            if (imageAspect > viewAspect) {
                scaleX = 1.0f
                scaleY = viewAspect / imageAspect
            } else {
                scaleX = imageAspect / viewAspect
                scaleY = 1.0f
            }
        }

        private fun initBuffers() {
            val quadCoords = floatArrayOf(
                -1.0f, 1.0f,
                -1.0f, -1.0f,
                1.0f, 1.0f,
                1.0f, -1.0f
            )
            quadBuffer = ByteBuffer.allocateDirect(quadCoords.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(quadCoords)
            quadBuffer.position(0)
        }

        private fun setupFBO(width: Int, height: Int) {
            if (fboId != 0) {
                GLES30.glDeleteFramebuffers(1, intArrayOf(fboId), 0)
                GLES30.glDeleteTextures(1, intArrayOf(fboTextureId), 0)
            }

            val fboArr = IntArray(1)
            val texArr = IntArray(1)

            GLES30.glGenFramebuffers(1, fboArr, 0)
            GLES30.glGenTextures(1, texArr, 0)

            fboId = fboArr[0]
            fboTextureId = texArr[0]

            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fboTextureId)
            GLES30.glTexImage2D(
                GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA, width, height, 0,
                GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
            )

            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)
            GLES30.glFramebufferTexture2D(
                GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
                GLES30.GL_TEXTURE_2D, fboTextureId, 0
            )

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        }

        private fun uploadBitmapsGL() {
            if (pendingMain != null && !pendingMain!!.isRecycled) {
                imageWidth = pendingMain!!.width
                imageHeight = pendingMain!!.height
                updateAspectScale()
            }
            mainTextureId = loadTexture(pendingMain, mainTextureId)
            depthTextureId = loadTexture(pendingDepth, depthTextureId)
            maskTextureId = loadTexture(pendingMask, maskTextureId)
        }

        private fun loadTexture(bitmap: Bitmap?, oldId: Int): Int {
            if (bitmap == null || bitmap.isRecycled) return oldId
            var id = oldId
            if (id == 0) {
                val tex = IntArray(1)
                GLES30.glGenTextures(1, tex, 0)
                id = tex[0]
            }

            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, id)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

            GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0)
            return id
        }

        private fun bindTexture(textureId: Int, textureUnit: Int, uniformName: String, program: Int) {
            GLES30.glActiveTexture(textureUnit)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
            GLES30.glUniform1i(GLES30.glGetUniformLocation(program, uniformName), textureUnit - GLES30.GL_TEXTURE0)
        }

        private fun drawQuad(program: Int) {
            val aPosLoc = GLES30.glGetAttribLocation(program, "a_position")
            GLES30.glEnableVertexAttribArray(aPosLoc)
            GLES30.glVertexAttribPointer(aPosLoc, 2, GLES30.GL_FLOAT, false, 0, quadBuffer)
            GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
            GLES30.glDisableVertexAttribArray(aPosLoc)
        }

        private fun initPass1Shader() {
            val vs = """#version 300 es
                in vec2 a_position;
                uniform vec2 u_scale;
                out vec2 v_texCoord;
                void main() {
                    v_texCoord = vec2((a_position.x + 1.0) * 0.5, (1.0 - a_position.y) * 0.5);
                    gl_Position = vec4(a_position * u_scale, 0.0, 1.0);
                }
            """.trimIndent()

            val fs = """#version 300 es
precision highp float;

in vec2 v_texCoord;
uniform sampler2D u_mainTex;
uniform sampler2D u_depthTex;
uniform sampler2D u_maskTex;
uniform float u_blurStrength;
uniform float u_focusDepth;
uniform float u_maxBokehSize;
uniform float u_lightThreshold;
uniform float u_highlightBoost;
uniform float u_colorBoost;
uniform vec2 u_feetPoint;
uniform float u_feetRadius;
uniform float u_imageAspect;
out vec4 fragColor;

const float GOLDEN_ANGLE = 2.39996323;
const int SAMPLES = 280;

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
    vec4 baseColor = texture(u_mainTex, v_texCoord);
    float depth    = texture(u_depthTex, v_texCoord).r;
    float mask     = texture(u_maskTex,  v_texCoord).r;

    float diff = depth - u_focusDepth;
    float ad = abs(diff);
    const float SIGMA_BG = 0.150;
    const float SIGMA_FG = 0.100;
    float cocBg = 1.0 - exp(-ad * ad / (2.0 * SIGMA_BG * SIGMA_BG));
    float cocFg = (1.0 - exp(-ad * ad / (2.0 * SIGMA_FG * SIGMA_FG))) * 0.55;
    float coc = mix(cocBg, cocFg, step(diff, 0.0));

    float depthProtect = smoothstep(0.25, 0.75, mask);

    vec2 feetDelta = v_texCoord - u_feetPoint;
    feetDelta.x *= u_imageAspect;
    float feetDist = length(feetDelta);
    float safeRadius = max(u_feetRadius, 0.001);
    float spatialProtect = 1.0 - smoothstep(0.0, safeRadius, feetDist);

    float protect = max(depthProtect, spatialProtect);
    coc *= (1.0 - protect);

    vec2 texSize = vec2(textureSize(u_mainTex, 0));
    float maxRadiusPx = u_maxBokehSize * texSize.x;
    float radius = coc * u_blurStrength * maxRadiusPx;

float bottomProtect = smoothstep(0.55, 0.85, v_texCoord.y);
radius *= (1.0 - bottomProtect);

    if (radius < 1.0) {
        fragColor = baseColor;
        return;
    }

    vec2 texel = 1.0 / vec2(textureSize(u_mainTex, 0));
    float lumC = lum(baseColor.rgb);
    float lumL = lum(texture(u_mainTex, v_texCoord - vec2(texel.x * 8.0, 0.0)).rgb);
    float lumR = lum(texture(u_mainTex, v_texCoord + vec2(texel.x * 8.0, 0.0)).rgb);
    float lumU = lum(texture(u_mainTex, v_texCoord - vec2(0.0, texel.y * 8.0)).rgb);
    float lumD = lum(texture(u_mainTex, v_texCoord + vec2(0.0, texel.y * 8.0)).rgb);

    float dX = abs(lumC - (lumL + lumR) * 0.5);
float dY = abs(lumC - (lumU + lumD) * 0.5);
float stretchSignal = dX - dY * 1.25;
float vStretch = smoothstep(0.10, 0.28, stretchSignal);

float kx = 1.0;
float ky = 1.0;

    vec3 accum = vec3(0.0);
    float wsum = 0.0;

    for (int i = 0; i < SAMPLES; i++) {
        float fi = float(i);
        float r = sqrt((fi + 0.5) / float(SAMPLES));
        float theta = fi * GOLDEN_ANGLE;

        vec2 offset = vec2(cos(theta) * r * kx,
                           sin(theta) * r * ky) * radius * texel;
        vec2 uv = v_texCoord + offset;

        vec3 col = texture(u_mainTex, uv).rgb;
        float l  = lum(col);

        float highlight = smoothstep(u_lightThreshold, 1.0, l);
        float wHighlight = 1.0 + u_highlightBoost * 45.0 * highlight;

        float neighborMask = texture(u_maskTex, uv).r;
        float neighborProtect = smoothstep(0.25, 0.75, neighborMask);
        float antiLeak = 1.0 - neighborProtect * (1.0 - protect);

        float w = wHighlight * antiLeak;

        accum += col * w;
        wsum  += w;
    }

    vec3 blurred = accum / max(wsum, 1e-4);

if (u_colorBoost > 0.001) {
    float maxColor = max(blurred.r, max(blurred.g, blurred.b));
    float highlightMask = smoothstep(u_lightThreshold, 1.0, maxColor);
    vec3 hsv = rgb2hsv(blurred);
    float boost = u_colorBoost * 2.0 * highlightMask;
    if (boost > 0.0) {
        hsv.y = pow(hsv.y, 1.0 / (1.0 + boost));
    }
    blurred = hsv2rgb(hsv);
}

    fragColor = vec4(blurred, 1.0);
}
            """.trimIndent()

            pass1Program = createProgram(vs, fs)
        }

        private fun initPass2QuadShader() {
            val vs = """#version 300 es
                in vec2 a_position;
                out vec2 v_texCoord;
                void main() {
                    v_texCoord = vec2((a_position.x + 1.0) * 0.5, (a_position.y + 1.0) * 0.5);
                    gl_Position = vec4(a_position, 0.0, 1.0);
                }
            """.trimIndent()

            val fs = """#version 300 es
                precision mediump float;
                in vec2 v_texCoord;
                uniform sampler2D u_fboTex;
                out vec4 fragColor;
                void main() {
                    fragColor = texture(u_fboTex, v_texCoord);
                }
            """.trimIndent()

            pass2QuadProgram = createProgram(vs, fs)
        }

        private fun createProgram(vertexSrc: String, fragmentSrc: String): Int {
            val vs = loadShader(GLES30.GL_VERTEX_SHADER, vertexSrc)
            val fs = loadShader(GLES30.GL_FRAGMENT_SHADER, fragmentSrc)
            val prog = GLES30.glCreateProgram()
            GLES30.glAttachShader(prog, vs)
            GLES30.glAttachShader(prog, fs)
            GLES30.glLinkProgram(prog)
            return prog
        }

        private fun loadShader(type: Int, shaderCode: String): Int {
            val cleanShaderCode = shaderCode.trimStart()
            val shader = GLES30.glCreateShader(type)
            GLES30.glShaderSource(shader, cleanShaderCode)
            GLES30.glCompileShader(shader)

            val compileStatus = IntArray(1)
            GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, compileStatus, 0)

            val shaderType = if (type == GLES30.GL_VERTEX_SHADER) "VERTEX" else "FRAGMENT"

            if (compileStatus[0] == 0) {
                val log = GLES30.glGetShaderInfoLog(shader)
                android.util.Log.e("BOKEH_SHADER", "$shaderType SHADER COMPILE FAILED:\n$log")
                android.util.Log.e("BOKEH_SHADER",
                "First chars: ${cleanShaderCode.take(50).replace("\n", "\\n")}")
            } else {
                android.util.Log.d("BOKEH_SHADER", "$shaderType shader compiled successfully")
            }

            return shader
        }
    }
}
