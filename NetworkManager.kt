package com.master.aistudio

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

object NetworkManager {

// رابط الـ Space الجديد بدل رابط Lightning
private const val BASE_URL = "https://xudress8-image-to-prompts.hf.space/"

// توكين القراءة فقط اللي عملته من huggingface.co/settings/tokens
// (استبدل القيمة دي بالتوكين بتاعك فعليًا)
private const val HF_TOKEN = "my token "

// Interceptor بيضيف الـ Authorization header تلقائيًا لكل طلب بيتبعت،
// من غير ما نضطر نكرر نفس الكود في كل دالة على حدة
private val authInterceptor = Interceptor { chain ->
val original = chain.request()
val requestWithAuth = original.newBuilder()
.addHeader("Authorization", "Bearer $HF_TOKEN")
.build()
chain.proceed(requestWithAuth)
}

private val client = OkHttpClient.Builder()
.addInterceptor(authInterceptor)
.connectTimeout(60, TimeUnit.SECONDS)
.readTimeout(180, TimeUnit.SECONDS)
.writeTimeout(180, TimeUnit.SECONDS)
.build()

// 1. تحسين الوجه (CodeFormer)
suspend fun enhanceFace(imageFile: File): ByteArray? = withContext(Dispatchers.IO) {
val requestBody = MultipartBody.Builder()
.setType(MultipartBody.FORM)
.addFormDataPart(
"file",
imageFile.name,
RequestBody.create("image/jpeg".toMediaTypeOrNull(), imageFile)
)
.build()

val request = Request.Builder()
.url("${BASE_URL}api/v1/enhance-face")
.post(requestBody)
.build()

executeRequest(request)
}

// 2. تحسين الصورة الكاملة (Real-ESRGAN)
suspend fun enhanceFull(imageFile: File): ByteArray? = withContext(Dispatchers.IO) {
val requestBody = MultipartBody.Builder()
.setType(MultipartBody.FORM)
.addFormDataPart(
"file",
imageFile.name,
RequestBody.create("image/jpeg".toMediaTypeOrNull(), imageFile)
)
.build()

val request = Request.Builder()
.url("${BASE_URL}api/v1/enhance-full")
.post(requestBody)
.build()

executeRequest(request)
}

// 3️⃣ أ. معاينة التحديد بالرسم الحر / Bounding Box (جديدة)
suspend fun segmentPreviewBox(
imageFile: File,
xMin: Int,
yMin: Int,
xMax: Int,
yMax: Int
): ByteArray? = withContext(Dispatchers.IO) {
val requestBody = MultipartBody.Builder()
.setType(MultipartBody.FORM)
.addFormDataPart(
"file",
imageFile.name,
RequestBody.create("image/jpeg".toMediaTypeOrNull(), imageFile)
)
.addFormDataPart("x_min", xMin.toString())
.addFormDataPart("y_min", yMin.toString())
.addFormDataPart("x_max", xMax.toString())
.addFormDataPart("y_max", yMax.toString())
.build()

val request = Request.Builder()
.url("${BASE_URL}api/v1/segment-preview")
.post(requestBody)
.build()

executeRequest(request)
}

// 3️⃣ ب. معاينة التحديد بلمس نقطة واحدة (القديمة)
suspend fun segmentPreview(imageFile: File, x: Float, y: Float): ByteArray? = withContext(Dispatchers.IO) {
val requestBody = MultipartBody.Builder()
.setType(MultipartBody.FORM)
.addFormDataPart(
"file",
imageFile.name,
RequestBody.create("image/jpeg".toMediaTypeOrNull(), imageFile)
)
.addFormDataPart("x", x.toInt().toString())
.addFormDataPart("y", y.toInt().toString())
.build()

val request = Request.Builder()
.url("${BASE_URL}api/v1/segment-preview")
.post(requestBody)
.build()

executeRequest(request)
}

// 4. إزالة العنصر (LaMa Inpainting)
suspend fun inpaintObject(imageFile: File, maskFile: File): ByteArray? = withContext(Dispatchers.IO) {
val requestBody = MultipartBody.Builder()
.setType(MultipartBody.FORM)
.addFormDataPart(
"image_file",
imageFile.name,
RequestBody.create("image/jpeg".toMediaTypeOrNull(), imageFile)
)
.addFormDataPart(
"mask_file",
maskFile.name,
RequestBody.create("image/png".toMediaTypeOrNull(), maskFile)
)
.build()

val request = Request.Builder()
.url("${BASE_URL}api/v1/inpaint")
.post(requestBody)
.build()

executeRequest(request)
}

// 5. البورتريه (فصل الشخص وبلور الخلفية - RVM)
suspend fun portraitBlur(imageFile: File, blurStrength: Int = 25): ByteArray? = withContext(Dispatchers.IO) {
val requestBody = MultipartBody.Builder()
.setType(MultipartBody.FORM)
.addFormDataPart(
"file",
imageFile.name,
RequestBody.create("image/jpeg".toMediaTypeOrNull(), imageFile)
)
.addFormDataPart("blur_strength", blurStrength.toString())
.build()

val request = Request.Builder()
.url("${BASE_URL}api/v1/portrait-blur")
.post(requestBody)
.build()

executeRequest(request)
}

private fun executeRequest(request: Request): ByteArray? {
return try {
val response = client.newCall(request).execute()
if (response.isSuccessful) {
response.body?.bytes()
} else {
null
}
} catch (e: IOException) {
e.printStackTrace()
null
}
}
}