"""Print live statistics of the phone stream: rate, resolution, intrinsics, tracking, motion.

    vlapol-phone-check [--port 8765] [--seconds 30]

Move the phone around: 'moved' should follow your motion in centimeters (ARCore is metric).

To check the metric scale, lay the phone down for a few seconds, slide it a known distance along a
ruler and lay it down again. Each time the phone comes to rest at a new place, the tool prints how
far that is from the previous rest position and how much the phone turned in between. Don't turn
it: the lens is off-center, so turning moves the camera too.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ..geometry import arcore_pose_to_T_world_cam
from .server import PhoneServer


STILL_SPREAD_M = 0.004   # the phone counts as still if it stays within this radius ...
STILL_SECONDS = 2        # ... for this many consecutive seconds
NEW_REST_M = 0.01        # a rest position this far from the last one is a new one


class RestTracker:
    """Finds the positions where the phone lies still and measures the steps between them."""

    def __init__(self) -> None:
        self.rest = None          # last confirmed rest position
        self._candidate = None
        self._count = 0

    def update(self, positions: np.ndarray) -> float | None:
        """Feed one second of positions (n, 3). Returns the step in meters when a new rest is found."""
        if len(positions) == 0:
            self._candidate, self._count = None, 0
            return None
        mean = positions.mean(axis=0)
        if np.linalg.norm(positions - mean, axis=1).max() > STILL_SPREAD_M:
            self._candidate, self._count = None, 0
            return None
        if self._candidate is not None and np.linalg.norm(mean - self._candidate) < STILL_SPREAD_M:
            self._count += 1
        else:
            self._candidate, self._count = mean, 1
        if self._count < STILL_SECONDS:
            return None
        if self.rest is None:
            self.rest = self._candidate
            return None
        step = float(np.linalg.norm(self._candidate - self.rest))
        if step < NEW_REST_M:
            return None
        self.rest = self._candidate
        return step


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl+C)")
    args = p.parse_args(argv)

    with PhoneServer(port=args.port) as srv:
        print(f"listening on port {args.port}; waiting for the phone ...")
        first = srv.latest()
        print(f"connected: {first.size[0]}x{first.size[1]}  fx={first.K[0,0]:.1f} fy={first.K[1,1]:.1f} "
              f"cx={first.K[0,2]:.1f} cy={first.K[1,2]:.1f}")
        srv.drain()
        start = time.monotonic()
        origin = None
        rests = RestTracker()
        rest_R = None  # camera orientation at the last rest position
        try:
            while not args.seconds or time.monotonic() - start < args.seconds:
                time.sleep(1.0)
                frames = srv.drain()
                if not frames:
                    print("no frames in the last second")
                    continue
                f = frames[-1]
                dt = (frames[-1].t_ns - frames[0].t_ns) / 1e9
                rate = (len(frames) - 1) / dt if dt > 0 else 0.0
                gaps = np.diff([x.seq for x in frames])
                lost_seq = int(np.sum(gaps[gaps > 1] - 1)) if len(gaps) else 0
                latency = (time.time() - f.recv_time) * 1000
                kb = np.mean([len(x.jpeg) for x in frames]) / 1024
                line = f"{rate:5.1f} fps  {kb:5.1f} kB/frame  skipped {lost_seq:3d}  {f.tracking:8s}"
                tracked = [arcore_pose_to_T_world_cam(x.pose7) for x in frames if x.tracking_ok]
                last_rest = rests.rest
                step = rests.update(np.array([T[:3, 3] for T in tracked]).reshape(-1, 3))
                turned = None
                if rests.rest is not last_rest:  # came to rest at a new place
                    R = tracked[-1][:3, :3]
                    if rest_R is not None:
                        cos = (np.trace(rest_R.T @ R) - 1) / 2
                        turned = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
                    rest_R = R
                if f.tracking_ok:
                    T = arcore_pose_to_T_world_cam(f.pose7)
                    origin = T if origin is None else origin
                    moved = np.linalg.norm(T[:3, 3] - origin[:3, 3]) * 100
                    line += f"  moved {moved:6.1f} cm from first tracked pose"
                elif f.tracking_failure:
                    line += f"  ({f.tracking_failure})"
                print(line + f"  [last frame {latency:.0f} ms ago]", flush=True)
                if step is not None:
                    # The lens is off-center, so turning the phone also moves the camera.
                    print(f"    --> came to rest {step * 100:.1f} cm from the previous rest position"
                          f" (turned {turned:.1f} deg)", flush=True)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
