"""On-disk raw episode format.

<session>/episode_000007/
    episode.json     {"task": ..., "created": ..., "num_frames": ...}
    frames.jsonl     one PhoneFrame header per line (+ "file", "recv_time")
    images/000000.jpg ...  (JPEGs exactly as streamed, no re-encoding)
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from .phone.protocol import PhoneFrame, frame_from_header


class EpisodeWriter:
    def __init__(self, session_dir: str | Path, task: str):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.dir = self.session_dir / f"episode_{next_episode_index(self.session_dir):06d}"
        (self.dir / "images").mkdir(parents=True)
        self.task = task
        self.n = 0
        self.n_lost = 0
        self._meta = open(self.dir / "frames.jsonl", "w")

    def add(self, frame: PhoneFrame) -> None:
        name = f"{self.n:06d}.jpg"
        (self.dir / "images" / name).write_bytes(frame.jpeg)
        rec = frame.header() | {"file": name, "recv_time": frame.recv_time}
        self._meta.write(json.dumps(rec) + "\n")
        self.n += 1
        self.n_lost += 0 if frame.tracking_ok else 1

    def close(self) -> Path:
        self._meta.close()
        info = {"task": self.task, "created": time.time(), "num_frames": self.n, "frames_not_tracking": self.n_lost}
        (self.dir / "episode.json").write_text(json.dumps(info, indent=2))
        return self.dir

    def discard(self) -> None:
        self._meta.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def next_episode_index(session_dir: Path) -> int:
    existing = [int(p.name.split("_")[1]) for p in session_dir.glob("episode_*") if p.name.split("_")[1].isdigit()]
    return max(existing, default=-1) + 1


def list_episodes(root: str | Path) -> list[Path]:
    """All finished episodes below root (a session dir or a directory of sessions)."""
    return sorted(p.parent for p in Path(root).rglob("episode.json"))


class Episode:
    def __init__(self, path: str | Path):
        self.dir = Path(path)
        self.info = json.loads((self.dir / "episode.json").read_text())
        self.records = [json.loads(line) for line in (self.dir / "frames.jsonl").read_text().splitlines() if line]
        self.frames = [frame_from_header(r, b"", r.get("recv_time", 0.0)) for r in self.records]

    @property
    def task(self) -> str:
        return self.info["task"]

    def __len__(self) -> int:
        return len(self.records)

    def image(self, i: int) -> np.ndarray:
        return cv2.imread(str(self.dir / "images" / self.records[i]["file"]), cv2.IMREAD_COLOR)

    def timestamps(self) -> np.ndarray:
        return np.array([f.t_ns for f in self.frames], dtype=np.int64)
