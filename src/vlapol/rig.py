"""Rig configuration (camera->TCP extrinsic, gripper tags, export settings)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .geometry import make_T

DEFAULT_RIG = Path(__file__).resolve().parents[2] / "configs" / "rig.yaml"


@dataclass
class GripperConfig:
    aruco_dict: str = "DICT_4X4_50"
    left_id: int = 0
    right_id: int = 1
    tag_size_m: float = 0.015
    closed_dist_m: float = 0.02
    open_dist_m: float = 0.08


@dataclass
class Rig:
    T_cam_tcp: np.ndarray
    gripper: GripperConfig
    fps: int = 10
    image_size: tuple[int, int] = (480, 640)  # (H, W)
    raw: dict | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Rig":
        path = Path(path) if path else DEFAULT_RIG
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Rig":
        t = raw.get("T_cam_tcp", {})
        exp = raw.get("export", {})
        return cls(
            T_cam_tcp=make_T(t.get("translation", [0, 0, 0]), t.get("rotvec", [0, 0, 0])),
            gripper=GripperConfig(**raw.get("gripper", {})),
            fps=int(exp.get("fps", 10)),
            image_size=tuple(exp.get("image_size", [480, 640])),
            raw=raw,
        )
