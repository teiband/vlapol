"""Gripper opening from two ArUco tags glued on the mockup's fingers (as in UMI)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .rig import GripperConfig


@dataclass
class GripperObs:
    dist_m: float | None                       # distance between tag centers, None if not both seen
    centers: dict[int, np.ndarray]             # tag id -> 3D center in camera frame
    corners: dict[int, np.ndarray]             # tag id -> (4,2) image corners


class GripperTagDetector:
    def __init__(self, cfg: GripperConfig):
        self.cfg = cfg
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.aruco_dict))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        s = cfg.tag_size_m / 2
        # Corner order of ArUco: top-left, top-right, bottom-right, bottom-left.
        self.obj_pts = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)

    def detect(self, image_bgr: np.ndarray, K: np.ndarray) -> GripperObs:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        corners, ids, _ = self.detector.detectMarkers(gray)
        centers, img_corners = {}, {}
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                i = int(i)
                if i not in (self.cfg.left_id, self.cfg.right_id):
                    continue
                c = c.reshape(4, 2).astype(np.float64)
                ok, _, tvec = cv2.solvePnP(self.obj_pts, c, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    centers[i] = tvec.reshape(3)
                    img_corners[i] = c
        dist = None
        if self.cfg.left_id in centers and self.cfg.right_id in centers:
            dist = float(np.linalg.norm(centers[self.cfg.left_id] - centers[self.cfg.right_id]))
        return GripperObs(dist, centers, img_corners)

    def normalize(self, dist_m: float | None) -> float | None:
        """Map tag distance to [0 (closed), 1 (open)]."""
        if dist_m is None:
            return None
        lo, hi = self.cfg.closed_dist_m, self.cfg.open_dist_m
        return float(np.clip((dist_m - lo) / max(hi - lo, 1e-6), 0.0, 1.0))


def fill_gaps(values: np.ndarray, max_gap: int | None = None) -> np.ndarray:
    """Linearly interpolate NaNs; edges are held constant. Returns all-NaN input unchanged."""
    v = np.asarray(values, dtype=np.float64).copy()
    ok = ~np.isnan(v)
    if not ok.any():
        return v
    idx = np.arange(len(v))
    filled = np.interp(idx, idx[ok], v[ok])
    if max_gap is not None:
        # leave long gaps as NaN
        run = 0
        for i in range(len(v)):
            run = run + 1 if not ok[i] else 0
            if run > max_gap:
                filled[i - run + 1 : i + 1] = np.nan
    return filled


def draw(image: np.ndarray, obs: GripperObs, normalized: float | None) -> None:
    for c in obs.corners.values():
        cv2.polylines(image, [c.astype(np.int32)], True, (255, 200, 0), 2)
    txt = "gripper: --" if normalized is None else f"gripper: {normalized:.2f} ({obs.dist_m * 100:.1f} cm)"
    cv2.putText(image, txt, (10, image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
