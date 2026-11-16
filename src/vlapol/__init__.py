"""vlapol: VLA testbed with a MuJoCo pick-and-place task and phone-based data collection."""

import ctypes.util
import os
import sys

# Headless Linux (Colab, CI): render offscreen unless the user chose a backend. Must run before
# `import mujoco`. Only pick a backend whose library exists, since mujoco fails at import otherwise.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and "MUJOCO_GL" not in os.environ:
    for _backend, _lib in (("egl", "EGL"), ("osmesa", "OSMesa")):
        if ctypes.util.find_library(_lib):
            os.environ["MUJOCO_GL"] = _backend
            break

__version__ = "0.1.0"
