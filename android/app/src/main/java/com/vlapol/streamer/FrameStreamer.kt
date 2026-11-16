package com.vlapol.streamer

import android.os.Handler
import android.os.Looper
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import java.nio.ByteBuffer
import java.util.concurrent.TimeUnit

/** WebSocket client with auto-reconnect and frame dropping when the link is congested. */
class FrameStreamer {

    companion object {
        private const val TAG = "FrameStreamer"
        /** Drop frames instead of queueing when more than this many bytes are pending. */
        private const val MAX_QUEUE_BYTES = 1_500_000L
    }

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .pingInterval(5, TimeUnit.SECONDS)
        .build()
    private val handler = Handler(Looper.getMainLooper())

    @Volatile private var socket: WebSocket? = null
    @Volatile private var url: String? = null
    @Volatile var state: String = "disconnected"
        private set
    @Volatile var dropped: Long = 0
        private set

    fun connect(newUrl: String) {
        url = newUrl
        socket?.close(1000, "reconnect")
        socket = null
        open()
    }

    private fun open() {
        val u = url ?: return
        state = "connecting"
        val request = try { Request.Builder().url(u).build() } catch (e: IllegalArgumentException) {
            state = "bad url"; return
        }
        client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                socket = webSocket
                state = "connected"
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                if (socket === webSocket) { socket = null; state = "closed" }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.w(TAG, "websocket failure: ${t.message}")
                if (socket === webSocket || socket == null) {
                    socket = null
                    state = "retrying"
                    handler.postDelayed({ if (socket == null && url == u) open() }, 1000)
                }
            }
        })
    }

    /** Returns false if the frame was not sent (not connected or congested). */
    fun send(headerJson: String, jpeg: ByteArray): Boolean {
        val ws = socket ?: return false
        if (ws.queueSize() > MAX_QUEUE_BYTES) {
            dropped++
            return false
        }
        val header = headerJson.toByteArray(Charsets.UTF_8)
        val buf = ByteBuffer.allocate(4 + header.size + jpeg.size)
        buf.putInt(header.size)
        buf.put(header)
        buf.put(jpeg)
        buf.flip()
        return ws.send(buf.toByteString())
    }

    fun close() {
        url = null
        socket?.close(1000, "bye")
        socket = null
        client.dispatcher.executorService.shutdown()
    }
}
