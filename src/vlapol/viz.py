"""Render a chunk of relative TCP actions as a 3D trajectory projected into the camera image."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import vec6_to_T


def chunk_to_tcp_poses(actions: np.ndarray) -> np.ndarray:
    """Integrate incremental actions (N,>=6) into poses relative to the current TCP: (N+1,4,4)."""
    poses = [np.eye(4)]
    for a in actions:
        poses.append(poses[-1] @ vec6_to_T(a[:6]))
    return np.stack(poses)


def project(K: np.ndarray, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = pts_cam[:, 2]
    valid = z > 1e-3
    uv = (K @ pts_cam.T).T
    uv = uv[:, :2] / np.where(valid, z, 1.0)[:, None]
    return uv, valid


def _grip_color(g: float) -> tuple[int, int, int]:
    # open (1) -> green, closed (0) -> red   (BGR)
    g = float(np.clip(g, 0, 1))
    return (0, int(255 * g), int(255 * (1 - g)))


def draw_action_chunk(
    image: np.ndarray,
    K: np.ndarray,
    T_cam_tcp: np.ndarray,
    actions: np.ndarray,
    color: tuple[int, int, int] | None = None,
    thickness: int = 2,
    axes_len: float = 0.02,
    label: str | None = None,
) -> np.ndarray:
    """Draw the predicted/ground-truth trajectory of the TCP on `image` (in place).

    actions: (N, 7) incremental relative actions as produced by vlapol.process.compute_actions.
    """
    poses = chunk_to_tcp_poses(actions)
    T_cam = T_cam_tcp[None] @ poses               # TCP poses in camera frame
    pts = T_cam[:, :3, 3]
    uv, valid = project(K, pts)
    n = len(pts)
    grip = np.concatenate([[actions[0, 6] if len(actions) else 1.0], actions[:, 6]]) if actions.shape[1] > 6 else np.ones(n)

    overlay = image.copy()
    for k in range(n - 1):
        if not (valid[k] and valid[k + 1]):
            continue
        c = color or _grip_color(grip[k + 1])
        p0, p1 = tuple(np.round(uv[k]).astype(int)), tuple(np.round(uv[k + 1]).astype(int))
        cv2.line(overlay, p0, p1, c, thickness, cv2.LINE_AA)
    # fade: blend the polyline with decreasing opacity is costly; a single blend is enough
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, dst=image)

    # arrow from current TCP toward a point a few steps ahead (direction of motion)
    look = min(n - 1, max(1, n // 3))
    if valid[0] and valid[look] and np.linalg.norm(uv[look] - uv[0]) > 3:
        cv2.arrowedLine(image, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[look]).astype(int)),
                        color or (255, 255, 255), thickness + 1, cv2.LINE_AA, tipLength=0.25)
    # current TCP marker
    if valid[0]:
        cv2.circle(image, tuple(np.round(uv[0]).astype(int)), 5, (255, 255, 255), -1, cv2.LINE_AA)

    # orientation axes at the end of the chunk
    end = T_cam[-1]
    axes = end[:3, 3][None] + axes_len * end[:3, :3].T
    pa, va = project(K, np.vstack([end[:3, 3][None], axes]))
    if va.all():
        o = tuple(np.round(pa[0]).astype(int))
        for k, c in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.line(image, o, tuple(np.round(pa[k + 1]).astype(int)), c, 2, cv2.LINE_AA)
    if label:
        cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color or (255, 255, 255), 2)
    return image
