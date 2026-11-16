"""Wire format of the Android streamer: [u32 BE header_len][JSON header][JPEG bytes]."""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PhoneFrame:
    seq: int
    t_ns: int
    tracking: str                 # "TRACKING" | "PAUSED" | "STOPPED"
    pose7: np.ndarray             # ARCore pose [tx,ty,tz,qx,qy,qz,qw], OpenGL camera convention
    K: np.ndarray                 # 3x3 intrinsics for the transmitted image size
    size: tuple[int, int]         # (width, height)
    jpeg: bytes
    tracking_failure: str | None = None
    recv_time: float = 0.0        # PC wall clock (time.time()) at reception
    _image: np.ndarray | None = field(default=None, repr=False)

    @property
    def tracking_ok(self) -> bool:
        return self.tracking == "TRACKING"

    def image(self) -> np.ndarray:
        """Decoded BGR image (cached)."""
        if self._image is None:
            self._image = cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR)
        return self._image

    def header(self) -> dict:
        h = {
            "seq": self.seq,
            "t_ns": self.t_ns,
            "tracking": self.tracking,
            "pose": [float(x) for x in self.pose7],
            "K": [float(self.K[0, 0]), float(self.K[1, 1]), float(self.K[0, 2]), float(self.K[1, 2])],
            "size": list(self.size),
        }
        if self.tracking_failure:
            h["tracking_failure"] = self.tracking_failure
        return h


def _intrinsics(header: dict) -> np.ndarray:
    fx, fy, cx, cy = header["K"]
    w, h = header["size"]
    iw, ih = header.get("intr_size", header["size"])
    # Intrinsics are reported for the CPU image; rescale defensively if sizes differ.
    sx, sy = w / iw, h / ih
    return np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=np.float64)


def decode(msg: bytes, recv_time: float = 0.0) -> PhoneFrame:
    (n,) = struct.unpack(">I", msg[:4])
    header = json.loads(msg[4 : 4 + n].decode("utf-8"))
    jpeg = bytes(msg[4 + n :])
    return frame_from_header(header, jpeg, recv_time)


def frame_from_header(header: dict, jpeg: bytes, recv_time: float = 0.0) -> PhoneFrame:
    return PhoneFrame(
        seq=int(header["seq"]),
        t_ns=int(header["t_ns"]),
        tracking=header["tracking"],
        pose7=np.asarray(header["pose"], dtype=np.float64),
        K=_intrinsics(header),
        size=tuple(header["size"]),
        jpeg=jpeg,
        tracking_failure=header.get("tracking_failure"),
        recv_time=recv_time,
    )


def encode(header: dict, jpeg: bytes) -> bytes:
    h = json.dumps(header).encode("utf-8")
    return struct.pack(">I", len(h)) + h + jpeg
