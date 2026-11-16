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
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.UnavailableException
import org.json.JSONArray
import org.json.JSONObject
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
    }

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
    }

    private var cameraTextureId = -1

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        session?.setDisplayGeometry(windowManager.defaultDisplay.rotation, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val s = session ?: return
        if (cameraTextureId < 0) return
        s.setCameraTextureName(cameraTextureId)

        val frame = try { s.update() } catch (e: Exception) {
            Log.w(TAG, "update failed", e); return
        }
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
            setStatus(String.format(
                "%s | %.1f fps | %dx%d | %s | dropped %d%s",
                streamer.state, fps, width, height, tracking.name, streamer.dropped,
                if (!sent) " (not sending)" else ""))
        }
    }

    private fun setStatus(msg: String) {
        if (msg == lastStatus) return
        lastStatus = msg
        runOnUiThread { statusView.text = msg }
    }
}
