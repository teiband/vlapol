package com.vlapol.streamer

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import java.io.ByteArrayOutputStream

/** Converts an ARCore CPU image (YUV_420_888) to JPEG, handling arbitrary row/pixel strides. */
object YuvToJpeg {

    fun convert(image: Image, quality: Int): ByteArray {
        val w = image.width
        val h = image.height
        val nv21 = toNv21(image, w, h)
        val out = ByteArrayOutputStream(w * h / 4)
        YuvImage(nv21, ImageFormat.NV21, w, h, null).compressToJpeg(Rect(0, 0, w, h), quality, out)
        return out.toByteArray()
    }

    private fun toNv21(image: Image, w: Int, h: Int): ByteArray {
        val out = ByteArray(w * h * 3 / 2)
        val planes = image.planes

        // Y plane
        val y = planes[0]
        val yBuf = y.buffer
        val yRow = y.rowStride
        val yPix = y.pixelStride
        var pos = 0
        if (yPix == 1) {
            for (row in 0 until h) {
                yBuf.position(row * yRow)
                yBuf.get(out, pos, w)
                pos += w
            }
        } else {
            for (row in 0 until h) {
                val base = row * yRow
                for (col in 0 until w) out[pos++] = yBuf.get(base + col * yPix)
            }
        }

        // Interleaved V/U (NV21 order)
        val u = planes[1]
        val v = planes[2]
        val uBuf = u.buffer
        val vBuf = v.buffer
        val uRow = u.rowStride
        val vRow = v.rowStride
        val uPix = u.pixelStride
        val vPix = v.pixelStride
        for (row in 0 until h / 2) {
            for (col in 0 until w / 2) {
                out[pos++] = vBuf.get(row * vRow + col * vPix)
                out[pos++] = uBuf.get(row * uRow + col * uPix)
            }
        }
        return out
    }
}
