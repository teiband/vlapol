package com.vlapol.streamer

import android.Manifest
import android.content.pm.PackageManager
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.UnavailableException
import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Minimal ARCore streamer: for every camera frame it sends
 *   [4-byte big-endian header length][UTF-8 JSON header][JPEG bytes]
 * over a WebSocket to the PC. The PC side (vlapol.phone.server) records or runs the policy.
 *
 * The header contains the ARCore camera pose (sensor-aligned, OpenGL convention:
 * +X right, +Y up, camera looks along -Z), the CPU-image intrinsics and the timestamp.
 * Both the pose and the JPEG are in the sensor's native (landscape) orientation.
 */
class MainActivity : AppCompatActivity(), GLSurfaceView.Renderer {

    companion object {
        private const val TAG = "VlapolStreamer"
        private const val TARGET_WIDTH = 640
        private const val JPEG_QUALITY = 90

        private const val VERTEX_SHADER = """
            attribute vec4 a_Position;
            attribute vec2 a_TexCoord;
            varying vec2 v_TexCoord;
            void main() {
                gl_Position = a_Position;
                v_TexCoord = a_TexCoord;
            }
        """
        private const val FRAGMENT_SHADER = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            varying vec2 v_TexCoord;
            uniform samplerExternalOES u_Texture;
            void main() {
                gl_FragColor = texture2D(u_Texture, v_TexCoord);
            }
        """
        private val QUAD_NDC = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)

        private fun floatBuffer(n: Int): FloatBuffer =
            ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    }

    // Camera preview (full-screen quad with the ARCore camera texture)
    private var previewProgram = 0
    private var positionAttrib = 0
    private var texCoordAttrib = 0
    private val quadCoords: FloatBuffer = floatBuffer(8).apply { put(QUAD_NDC); position(0) }
    private val quadTexCoords: FloatBuffer = floatBuffer(8)
    private var texCoordsReady = false
    private var surfaceWidth = 0
    private var surfaceHeight = 0
    @Volatile private var geometryDirty = true

    private var session: Session? = null
    private var installRequested = false
    private lateinit var glView: GLSurfaceView
    private lateinit var statusView: TextView
    private lateinit var urlField: EditText
    private val streamer = FrameStreamer()

    private var seq = 0L
    private var fpsCounter = 0
    private var fpsWindowStart = System.nanoTime()
    private var lastStatus = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        glView = GLSurfaceView(this).apply {
            preserveEGLContextOnPause = true
            setEGLContextClientVersion(2)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            setRenderer(this@MainActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }

        urlField = EditText(this).apply {
            setText(getPreferences(MODE_PRIVATE).getString("url", "ws://127.0.0.1:8765"))
            setSingleLine()
            setTextColor(0xFFFFFFFF.toInt())
            minWidth = 600
        }
        val connectButton = Button(this).apply {
            text = "Connect"
            setOnClickListener {
                val url = urlField.text.toString().trim()
                getPreferences(MODE_PRIVATE).edit().putString("url", url).apply()
                streamer.connect(url)
            }
        }
        statusView = TextView(this).apply {
            setTextColor(0xFF00FF00.toInt())
            textSize = 16f
        }
        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 24, 24, 24)
            addView(LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                addView(urlField)
                addView(connectButton)
            })
            addView(statusView)
        }
        val root = FrameLayout(this).apply {
            addView(glView)
            addView(controls, FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.START))
        }
        setContentView(root)
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 0)
            return
        }
        if (session == null) {
            try {
                when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                    ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                        installRequested = true
                        return
                    }
                    ArCoreApk.InstallStatus.INSTALLED -> {}
                    else -> {}
                }
                session = createSession()
            } catch (e: UnavailableException) {
                setStatus("ARCore unavailable: ${e.javaClass.simpleName}")
                return
            }
        }
        try {
            session?.resume()
        } catch (e: Exception) {
            setStatus("Camera not available: ${e.message}")
            session = null
            return
        }
        glView.onResume()
    }

    override fun onPause() {
        super.onPause()
        glView.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        streamer.close()
        session?.close()
        session = null
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (results.isEmpty() || results[0] != PackageManager.PERMISSION_GRANTED) {
            setStatus("Camera permission is required")
        }
    }

    private fun createSession(): Session {
        val s = Session(this)
        // Pick the CPU-image resolution closest to TARGET_WIDTH (default ARCore CPU image is 640x480).
        val filter = CameraConfigFilter(s)
            .setFacingDirection(CameraConfig.FacingDirection.BACK)
        val configs = s.getSupportedCameraConfigs(filter)
        val best = configs.minByOrNull { kotlin.math.abs(it.imageSize.width - TARGET_WIDTH) }
        if (best != null) s.cameraConfig = best

        val config = Config(s).apply {
            // Auto focus: the scene is close to the gripper; fixed focus is set near infinity.
            focusMode = Config.FocusMode.AUTO
            updateMode = Config.UpdateMode.BLOCKING
            planeFindingMode = Config.PlaneFindingMode.DISABLED
            lightEstimationMode = Config.LightEstimationMode.DISABLED
            depthMode = Config.DepthMode.DISABLED
        }
        s.configure(config)
        return s
    }

    // ---------------------------------------------------------------- GL renderer

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        val tex = IntArray(1)
        GLES20.glGenTextures(1, tex, 0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, tex[0])
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        cameraTextureId = tex[0]

        previewProgram = GLES20.glCreateProgram().also { prog ->
            GLES20.glAttachShader(prog, compileShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER))
            GLES20.glAttachShader(prog, compileShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER))
            GLES20.glLinkProgram(prog)
        }
        positionAttrib = GLES20.glGetAttribLocation(previewProgram, "a_Position")
        texCoordAttrib = GLES20.glGetAttribLocation(previewProgram, "a_TexCoord")
        texCoordsReady = false
        geometryDirty = true
    }

    private fun compileShader(type: Int, source: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, source)
        GLES20.glCompileShader(shader)
        val ok = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, ok, 0)
        if (ok[0] == 0) Log.e(TAG, "shader compile failed: " + GLES20.glGetShaderInfoLog(shader))
        return shader
    }

    /** Draws what the camera sees. Only the screen preview; the streamed JPEG is unaffected. */
    private fun drawPreview(frame: com.google.ar.core.Frame) {
        if (frame.timestamp == 0L) return  // no camera image yet
        if (!texCoordsReady || frame.hasDisplayGeometryChanged()) {
            quadCoords.position(0)
            quadTexCoords.position(0)
            frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, quadCoords,
                Coordinates2d.TEXTURE_NORMALIZED, quadTexCoords)
            texCoordsReady = true
        }
        quadCoords.position(0)
        quadTexCoords.position(0)
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glUseProgram(previewProgram)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTextureId)
        GLES20.glVertexAttribPointer(positionAttrib, 2, GLES20.GL_FLOAT, false, 0, quadCoords)
        GLES20.glVertexAttribPointer(texCoordAttrib, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)
        GLES20.glEnableVertexAttribArray(positionAttrib)
        GLES20.glEnableVertexAttribArray(texCoordAttrib)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(positionAttrib)
        GLES20.glDisableVertexAttribArray(texCoordAttrib)
    }

    private var cameraTextureId = -1

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        surfaceWidth = width
        surfaceHeight = height
        geometryDirty = true
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val s = session ?: return
        if (cameraTextureId < 0) return
        s.setCameraTextureName(cameraTextureId)
        if (geometryDirty && surfaceWidth > 0) {
            // Done here because the session may not exist yet when the surface is created.
            @Suppress("DEPRECATION")
            s.setDisplayGeometry(windowManager.defaultDisplay.rotation, surfaceWidth, surfaceHeight)
            geometryDirty = false
        }

        val frame = try { s.update() } catch (e: Exception) {
            Log.w(TAG, "update failed", e); return
        }
        drawPreview(frame)
        val camera = frame.camera
        val tracking = camera.trackingState

        val image = try { frame.acquireCameraImage() } catch (e: NotYetAvailableException) {
            return
        }
        val jpeg: ByteArray
        val width: Int
        val height: Int
        try {
            width = image.width
            height = image.height
            jpeg = YuvToJpeg.convert(image, JPEG_QUALITY)
        } finally {
            image.close()
        }

        val pose = camera.pose
        val t = FloatArray(3).also { pose.getTranslation(it, 0) }
        val q = FloatArray(4).also { pose.getRotationQuaternion(it, 0) } // x, y, z, w
        val intr = camera.imageIntrinsics
        val f = intr.focalLength
        val c = intr.principalPoint
        val dims = intr.imageDimensions

        val header = JSONObject().apply {
            put("seq", seq++)
            put("t_ns", frame.timestamp)
            put("tracking", tracking.name)
            if (tracking != TrackingState.TRACKING) {
                put("tracking_failure", camera.trackingFailureReason.name)
            }
            put("pose", JSONArray().apply {
                t.forEach { put(it.toDouble()) }
                q.forEach { put(it.toDouble()) }
            })
            put("K", JSONArray().apply {
                put(f[0].toDouble()); put(f[1].toDouble()); put(c[0].toDouble()); put(c[1].toDouble())
            })
            put("intr_size", JSONArray().apply { put(dims[0]); put(dims[1]) })
            put("size", JSONArray().apply { put(width); put(height) })
        }
        val sent = streamer.send(header.toString(), jpeg)

        fpsCounter++
        val now = System.nanoTime()
        if (now - fpsWindowStart > 1_000_000_000L) {
            val fps = fpsCounter * 1e9 / (now - fpsWindowStart)
            fpsCounter = 0
            fpsWindowStart = now
            var status = String.format(
                "%s | %.1f fps | %dx%d | %s | dropped %d%s",
                streamer.state, fps, width, height, tracking.name, streamer.dropped,
                if (!sent) " (not sending)" else "")
            if (tracking != TrackingState.TRACKING) {
                status += " | not tracking (${camera.trackingFailureReason.name}): move the phone slowly sideways"
            }
            setStatus(status)
        }
    }

    private fun setStatus(msg: String) {
        if (msg == lastStatus) return
        lastStatus = msg
        runOnUiThread { statusView.text = msg }
    }
}
