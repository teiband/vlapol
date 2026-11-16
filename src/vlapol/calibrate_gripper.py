"""Measure the finger-tag distance for fully closed / fully open and write it to the rig yaml.

    vlapol-calibrate-gripper [--rig configs/rig.yaml]

Hold the mockup closed and press 'c', fully open and press 'o', then 's' to save, 'q' to quit.
Each key averages the last ~1 s of detections.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import cv2
import numpy as np
import yaml

from .gripper import GripperTagDetector, draw
from .phone.server import PhoneServer
from .rig import DEFAULT_RIG, Rig


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rig", default=str(DEFAULT_RIG))
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)

    rig = Rig.load(args.rig)
    det = GripperTagDetector(rig.gripper)
    hist: collections.deque = collections.deque(maxlen=30)
    closed = opened = None
    with PhoneServer(port=args.port) as srv:
        last = None
        while True:
            f = srv.latest(timeout=0.05)
            if f is not None and f is not last:
                last = f
                img = f.image().copy()
                obs = det.detect(img, f.K)
                if obs.dist_m is not None:
                    hist.append(obs.dist_m)
                draw(img, obs, det.normalize(obs.dist_m))
                txt = f"closed={closed and round(closed * 100, 2)} cm  open={opened and round(opened * 100, 2)} cm  [c/o/s/q]"
                cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("calibrate gripper", img)
            k = cv2.waitKey(10) & 0xFF
            if k == ord("c") and hist:
                closed = float(np.median(hist))
            elif k == ord("o") and hist:
                opened = float(np.median(hist))
            elif k == ord("s") and closed is not None and opened is not None:
                raw = yaml.safe_load(Path(args.rig).read_text())
                raw.setdefault("gripper", {}).update(closed_dist_m=round(closed, 5), open_dist_m=round(opened, 5))
                Path(args.rig).write_text(yaml.safe_dump(raw, sort_keys=False))
                print(f"saved closed={closed:.4f} m, open={opened:.4f} m to {args.rig}")
            elif k in (ord("q"), 27):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
