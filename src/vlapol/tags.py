"""Generate a printable sheet with the two finger tags at their exact physical size.

    vlapol-make-tags --out tags.png        # print at 100% scale (no 'fit to page')
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from .rig import Rig


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rig", default=None)
    p.add_argument("--out", default="gripper_tags.png")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--copies", type=int, default=3, help="spare copies of each tag")
    args = p.parse_args(argv)

    g = Rig.load(args.rig).gripper
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, g.aruco_dict))
    px = int(round(g.tag_size_m / 0.0254 * args.dpi))
    margin = px // 2
    tiles = []
    for tag_id in (g.left_id, g.right_id):
        img = cv2.aruco.generateImageMarker(d, tag_id, px)
        tile = np.full((px + 2 * margin, px + 2 * margin), 255, np.uint8)
        tile[margin:margin + px, margin:margin + px] = img
        cv2.putText(tile, f"id {tag_id}", (4, tile.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, px / 400, 0, 1)
        tiles.append(np.hstack([tile] * args.copies))
    sheet = np.vstack(tiles)
    # 50 mm scale bar to verify the printed scale with a ruler
    bar = int(round(0.05 / 0.0254 * args.dpi))
    strip = np.full((px, max(sheet.shape[1], bar + 20)), 255, np.uint8)
    cv2.rectangle(strip, (10, px // 3), (10 + bar, px // 3 + max(2, px // 20)), 0, -1)
    cv2.putText(strip, "50 mm", (10, px - 6), cv2.FONT_HERSHEY_SIMPLEX, px / 400, 0, 1)
    if strip.shape[1] > sheet.shape[1]:
        sheet = np.pad(sheet, ((0, 0), (0, strip.shape[1] - sheet.shape[1])), constant_values=255)
    sheet = np.vstack([sheet, strip[:, : sheet.shape[1]]])
    cv2.imwrite(args.out, sheet)
    print(f"wrote {args.out}: tags {g.tag_size_m * 1000:.1f} mm at {args.dpi} dpi — print at 100% scale")


if __name__ == "__main__":
    main()
