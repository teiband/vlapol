"""Print live statistics of the phone stream: rate, resolution, intrinsics, tracking, motion.

    vlapol-phone-check [--port 8765] [--seconds 30]

Move the phone around: 'moved' should follow your motion in centimeters (ARCore is metric).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ..geometry import arcore_pose_to_T_world_cam
from .server import PhoneServer


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
                if f.tracking_ok:
                    T = arcore_pose_to_T_world_cam(f.pose7)
                    origin = T if origin is None else origin
                    moved = np.linalg.norm(T[:3, 3] - origin[:3, 3]) * 100
                    line += f"  moved {moved:6.1f} cm from first tracked pose"
                elif f.tracking_failure:
                    line += f"  ({f.tracking_failure})"
                print(line + f"  [last frame {latency:.0f} ms ago]", flush=True)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
