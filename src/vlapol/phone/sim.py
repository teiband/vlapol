"""Simulated phone for testing the pipeline without hardware.

    python -m vlapol.phone.sim --url ws://127.0.0.1:8765 --seconds 20

Generates frames with a known camera trajectory and two finger tags whose distance oscillates,
encoded exactly like the Android app sends them.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from ..geometry import GL_TO_CV, T_to_pose7, make_T
from ..rig import GripperConfig
from .protocol import encode

W, H = 640, 480
K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def camera_pose(t: float) -> np.ndarray:
    """T_world_cam (OpenCV convention) along a smooth 3D path."""
    pos = [0.10 * np.sin(0.5 * t), 0.05 * np.sin(0.3 * t), 0.08 * t / 10]
    rot = [0.1 * np.sin(0.4 * t), 0.2 * np.sin(0.25 * t), 0.15 * np.sin(0.2 * t)]
    return make_T(pos, rot)


def gripper_dist(t: float, cfg: GripperConfig) -> float:
    a = 0.5 + 0.5 * np.cos(0.8 * t)
    return cfg.closed_dist_m + a * (cfg.open_dist_m - cfg.closed_dist_m)


def _texture(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (H // 8, W // 8, 3), dtype=np.uint8)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


def render(t: float, cfg: GripperConfig, bg: np.ndarray) -> np.ndarray:
    img = bg.copy()
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.aruco_dict))
    half = gripper_dist(t, cfg) / 2
    s = cfg.tag_size_m / 2
    for tag_id, x in ((cfg.left_id, -half), (cfg.right_id, half)):
        c = np.array([x, 0.04, 0.12])
        q = s * 1.4  # white quiet zone
        for size, marker in ((q, None), (s, cv2.aruco.generateImageMarker(d, tag_id, 120))):
            corners3 = c + np.array([[-size, -size, 0], [size, -size, 0], [size, size, 0], [-size, size, 0]])
            uv = (K @ corners3.T).T
            uv = (uv[:, :2] / uv[:, 2:]).astype(np.float32)
            src = marker if marker is not None else np.full((10, 10), 255, np.uint8)
            h, w = src.shape
            M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), uv)
            warped = cv2.warpPerspective(cv2.cvtColor(src, cv2.COLOR_GRAY2BGR), M, (W, H), flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST)
            img[mask > 0] = warped[mask > 0]
    return img


def make_frame(seq: int, t: float, cfg: GripperConfig, bg: np.ndarray, tracking: str = "TRACKING"):
    T_gl = camera_pose(t) @ GL_TO_CV  # back to ARCore's OpenGL camera convention
    header = {
        "seq": seq,
        "t_ns": int(t * 1e9) + 1_000_000_000,
        "tracking": tracking,
        "pose": [float(v) for v in T_to_pose7(T_gl)],
        "K": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
        "intr_size": [W, H],
        "size": [W, H],
    }
    ok, jpeg = cv2.imencode(".jpg", render(t, cfg, bg), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return header, jpeg.tobytes()


def run(url: str, seconds: float, fps: float = 30.0, cfg: GripperConfig | None = None) -> int:
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    cfg = cfg or GripperConfig()
    bg = _texture()
    n = int(seconds * fps)
    with connect(url, max_size=None, compression=None) as ws:
        t0 = time.monotonic()
        for i in range(n):
            header, jpeg = make_frame(i, i / fps, cfg, bg)
            try:
                ws.send(encode(header, jpeg))
            except ConnectionClosed:  # receiver finished
                return i
            delay = t0 + (i + 1) / fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    return n


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://127.0.0.1:8765")
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--fps", type=float, default=30)
    args = p.parse_args(argv)
    print(f"sent {run(args.url, args.seconds, args.fps)} frames")


if __name__ == "__main__":
    main()
