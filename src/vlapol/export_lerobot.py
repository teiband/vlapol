"""Export raw phone episodes to a LeRobotDataset (e.g. for SmolVLA fine-tuning).

    vlapol-export-lerobot data/raw --repo-id local/umi_phone --root data/lerobot/umi_phone

Features:
    observation.images.wrist  video (H, W, 3)   wrist camera image (phone on the mockup)
    observation.state         float32 (1,)      gripper opening, 0 closed .. 1 open
    action                    float32 (7,)      [dx, dy, dz, drx, dry, drz, gripper]
                                                relative TCP motion to the next step (see vlapol.process)
"""
from __future__ import annotations

import argparse
import inspect
import logging

import cv2
import numpy as np

from .episode_io import Episode, list_episodes
from .process import ACTION_NAMES, process_episode
from .rig import Rig

log = logging.getLogger("vlapol.export")

IMAGE_KEY = "observation.images.wrist"


def _import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # lerobot >= 0.3
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # older layout
    return LeRobotDataset


def features(image_size: tuple[int, int]) -> dict:
    h, w = image_size
    return {
        IMAGE_KEY: {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
    }


def iter_frames(ep_dir, rig: Rig):
    """Yield (image_rgb, state, action, task) at the export rate."""
    pe = process_episode(ep_dir, rig)
    ep = Episode(ep_dir)
    h, w = rig.image_size
    for i, raw_idx in enumerate(pe.frame_idx):
        img = ep.image(int(raw_idx))
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        yield (cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
               np.array([pe.gripper[i]], np.float32),
               pe.actions[i].astype(np.float32),
               pe.task)


def export(raw_root, repo_id: str, root: str | None, rig: Rig) -> int:
    LeRobotDataset = _import_lerobot_dataset()
    ds = LeRobotDataset.create(repo_id=repo_id, fps=rig.fps, features=features(rig.image_size),
                               root=root, robot_type="umi_phone_mockup", use_videos=True)
    task_as_arg = "task" in inspect.signature(ds.add_frame).parameters
    n_ep = 0
    for ep_dir in list_episodes(raw_root):
        try:
            frames = list(iter_frames(ep_dir, rig))
        except ValueError as e:
            log.warning("skipping %s: %s", ep_dir, e)
            continue
        for img, state, action, task in frames:
            frame = {IMAGE_KEY: img, "observation.state": state, "action": action}
            if task_as_arg:
                ds.add_frame(frame, task=task)
            else:
                ds.add_frame(frame | {"task": task})
        ds.save_episode()
        n_ep += 1
        log.info("exported %s (%d frames)", ep_dir, len(frames))
    if hasattr(ds, "finalize"):
        ds.finalize()
    return n_ep


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw_root", help="session dir or directory containing sessions")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None, help="output directory (default: LeRobot cache)")
    p.add_argument("--rig", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    n = export(args.raw_root, args.repo_id, args.root, Rig.load(args.rig))
    print(f"exported {n} episodes")


if __name__ == "__main__":
    main()
