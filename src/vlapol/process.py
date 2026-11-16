"""Turn a raw episode into a fixed-rate sequence of observations and relative TCP actions.

Action convention (what the VLA predicts), for step i at the export rate:
    action_i = [dx, dy, dz, drx, dry, drz, gripper_{i+1}]
where (dx..drz) is the pose of TCP_{i+1} expressed in the frame of TCP_i (translation in meters,
rotation vector in rad), and gripper is the normalized opening (0 closed, 1 open) at step i+1.
Because it is relative to the current TCP frame, no world frame or robot base is needed: the
same action can be rendered on the camera image at eval time or sent to a real arm's EE controller.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .episode_io import Episode
from .geometry import T_to_vec6, arcore_pose_to_T_world_cam, relative
from .gripper import GripperTagDetector, fill_gaps
from .rig import Rig

ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]


@dataclass
class ProcessedEpisode:
    task: str
    frame_idx: np.ndarray       # (N,) indices into the raw episode
    t: np.ndarray               # (N,) seconds since episode start
    T_world_tcp: np.ndarray     # (N,4,4)
    gripper: np.ndarray         # (N,) normalized opening
    gripper_dist_m: np.ndarray  # (N,) raw finger-tag distance (gaps interpolated)
    actions: np.ndarray         # (N,7)
    K: np.ndarray               # 3x3 intrinsics of the raw images


def gripper_series(ep: Episode, rig: Rig, cache: bool = True) -> np.ndarray:
    """Raw tag distance per frame (NaN when not both tags detected); cached next to the episode."""
    cache_path = ep.dir / "gripper_dist.npy"
    if cache and cache_path.exists():
        d = np.load(cache_path)
        if len(d) == len(ep):
            return d
    det = GripperTagDetector(rig.gripper)
    d = np.full(len(ep), np.nan)
    for i, f in enumerate(ep.frames):
        obs = det.detect(ep.image(i), f.K)
        if obs.dist_m is not None:
            d[i] = obs.dist_m
    if cache:
        np.save(cache_path, d)
    return d


def resample_indices(t_ns: np.ndarray, fps: float) -> np.ndarray:
    """Indices of the raw frames closest to a uniform grid at `fps` (duplicates removed)."""
    t = (t_ns - t_ns[0]) / 1e9
    grid = np.arange(0.0, t[-1] + 1e-9, 1.0 / fps)
    idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    prev = np.clip(idx - 1, 0, len(t) - 1)
    idx = np.where(np.abs(t[prev] - grid) < np.abs(t[idx] - grid), prev, idx)
    return np.unique(idx)


def compute_actions(T_world_tcp: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    n = len(T_world_tcp)
    actions = np.zeros((n, 7), dtype=np.float64)
    for i in range(n):
        j = min(i + 1, n - 1)
        actions[i, :6] = T_to_vec6(relative(T_world_tcp[i], T_world_tcp[j]))
        actions[i, 6] = gripper[j]
    return actions


def trim_to_tracking(ok: np.ndarray) -> tuple[int, int] | None:
    """Longest contiguous run of tracked frames (start, end_exclusive)."""
    best, start = None, None
    for i, v in enumerate(list(ok) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def process_episode(path: str | Path, rig: Rig, fps: float | None = None) -> ProcessedEpisode:
    ep = Episode(path)
    if len(ep) < 2:
        raise ValueError(f"{path}: too few frames")
    fps = fps or rig.fps
    tracked = np.array([f.tracking_ok for f in ep.frames])
    run = trim_to_tracking(tracked)
    if run is None or run[1] - run[0] < 2:
        raise ValueError(f"{path}: no tracked segment")

    dist = gripper_series(ep, rig)
    det = GripperTagDetector(rig.gripper)
    grip = np.array([np.nan if np.isnan(x) else det.normalize(x) for x in dist])
    grip = fill_gaps(grip)
    dist_f = fill_gaps(dist)
    if np.isnan(grip).all():
        grip = np.ones_like(grip)  # no tags seen at all: assume open
        dist_f = np.full_like(dist_f, rig.gripper.open_dist_m)

    s, e = run
    t_ns = ep.timestamps()[s:e]
    sel = resample_indices(t_ns, fps) + s

    T_world_tcp = np.stack([arcore_pose_to_T_world_cam(ep.frames[i].pose7) @ rig.T_cam_tcp for i in sel])
    g = grip[sel]
    return ProcessedEpisode(
        task=ep.task,
        frame_idx=sel,
        t=(ep.timestamps()[sel] - ep.timestamps()[sel[0]]) / 1e9,
        T_world_tcp=T_world_tcp,
        gripper=g,
        gripper_dist_m=dist_f[sel],
        actions=compute_actions(T_world_tcp, g),
        K=ep.frames[sel[0]].K,
    )
