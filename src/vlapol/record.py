"""Record demonstrations streamed from the phone on the UMI mockup.

    vlapol-record --out data/raw/cup_session1 --task "put the cup on the plate"

Keys in the preview window:  SPACE start/stop episode | d delete last episode | q quit
Connect the phone via USB with `adb reverse tcp:8765 tcp:8765` and use ws://127.0.0.1:8765
in the app, or use the PC's LAN IP over Wi-Fi.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import cv2

from . import gripper as grip_mod
from .episode_io import EpisodeWriter
from .phone.server import PhoneServer
from .rig import Rig

log = logging.getLogger("vlapol.record")


def _overlay(img, frame, recording: bool, writer: EpisodeWriter | None, det, n_saved: int):
    obs = det.detect(img, frame.K)
    grip_mod.draw(img, obs, det.normalize(obs.dist_m))
    if not frame.tracking_ok:
        cv2.putText(img, f"TRACKING LOST ({frame.tracking_failure})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if recording:
        cv2.circle(img, (img.shape[1] - 25, 25), 10, (0, 0, 255), -1)
        cv2.putText(img, f"REC {writer.n}", (img.shape[1] - 120, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, f"saved episodes: {n_saved}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def run(args) -> None:
    rig = Rig.load(args.rig)
    det = grip_mod.GripperTagDetector(rig.gripper)
    out = Path(args.out)
    saved: list[Path] = []
    writer: EpisodeWriter | None = None

    def start():
        nonlocal writer
        writer = EpisodeWriter(out, args.task)
        log.info("recording %s", writer.dir)

    def stop():
        nonlocal writer
        if writer is None:
            return
        if writer.n < 5:
            writer.discard()
            log.warning("episode too short, discarded")
        else:
            d = writer.close()
            saved.append(d)
            log.info("saved %s (%d frames, %d not tracking)", d, writer.n, writer.n_lost)
        writer = None

    with PhoneServer(port=args.port) as srv:
        log.info("waiting for the phone on port %d ...", args.port)
        last = None
        ep_start = None
        while True:
            frames = srv.drain()
            if writer is not None:
                for f in frames:
                    writer.add(f)
            frame = frames[-1] if frames else last

            if args.headless:
                if frame is None:
                    time.sleep(0.01)
                    continue
                if writer is None:
                    if len(saved) >= args.episodes:
                        break
                    start()
                    ep_start = time.monotonic()
                elif time.monotonic() - ep_start >= args.episode_seconds:
                    stop()
                time.sleep(0.01)
                last = frame
                continue

            if frame is not None and frame is not last:
                img = _overlay(frame.image().copy(), frame, writer is not None, writer, det, len(saved))
                cv2.imshow("vlapol-record", img)
            last = frame
            key = cv2.waitKey(10) & 0xFF
            if key == ord(" "):
                stop() if writer is not None else start()
            elif key == ord("d") and writer is None and saved:
                d = saved.pop()
                shutil.rmtree(d, ignore_errors=True)
                log.info("deleted %s", d)
            elif key in (ord("q"), 27):
                stop()
                break
        stop()
    if not args.headless:
        cv2.destroyAllWindows()
    log.info("done: %d episodes in %s", len(saved), out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="session directory for the episodes")
    p.add_argument("--task", required=True, help="language instruction for these episodes")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--rig", default=None, help="rig yaml (default: configs/rig.yaml)")
    p.add_argument("--headless", action="store_true", help="no window; record --episodes of --episode-seconds")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--episode-seconds", type=float, default=10.0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
