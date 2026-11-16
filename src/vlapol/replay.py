"""Replay a recorded episode with the ground-truth future action chunk drawn on each frame.

Use this to sanity-check labels and the camera->TCP extrinsic before training: the drawn
trajectory should start at the fingertip midpoint and follow where the gripper actually goes.

    vlapol-replay data/raw/session1/episode_000000 --chunk 10 --out replay.mp4
"""
from __future__ import annotations

import argparse

import cv2

from .episode_io import Episode
from .process import process_episode
from .rig import Rig
from .viz import draw_action_chunk


def render(ep_dir, rig: Rig, chunk: int, fps: float | None = None):
    pe = process_episode(ep_dir, rig, fps)
    ep = Episode(ep_dir)
    for i, raw_idx in enumerate(pe.frame_idx):
        img = ep.image(int(raw_idx))
        draw_action_chunk(img, ep.frames[int(raw_idx)].K, rig.T_cam_tcp, pe.actions[i : i + chunk],
                          label=f"{pe.task}  t={pe.t[i]:.1f}s  grip={pe.gripper[i]:.2f}")
        yield img, pe


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode")
    p.add_argument("--rig", default=None)
    p.add_argument("--chunk", type=int, default=10, help="number of future steps to draw")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--out", default=None, help="write an mp4 instead of showing a window")
    args = p.parse_args(argv)
    rig = Rig.load(args.rig)
    fps = args.fps or rig.fps
    writer = None
    for img, _ in render(args.episode, rig, args.chunk, fps):
        if args.out:
            if writer is None:
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
            writer.write(img)
        else:
            cv2.imshow("replay", img)
            if (cv2.waitKey(int(1000 / fps)) & 0xFF) in (ord("q"), 27):
                break
    if writer is not None:
        writer.release()
        print(f"wrote {args.out}")
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
