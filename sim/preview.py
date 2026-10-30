#!/usr/bin/env python3
"""Проверка рендера: вырезки вокруг целей по разметке (увеличенные), чтобы глазами сверить, что дрон
на месте метки и какого он размера. python sim/preview.py out/fv [кадр]"""
import json, os, sys
import cv2
import numpy as np


def main():
    out = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    meta = json.load(open(os.path.join(out, "meta.json"), encoding="utf-8"))
    for cam in ("wide", "fine"):
        gt_path = os.path.join(out, f"{meta['name']}_{cam}.mp4.gt.json")
        png = os.path.join(out, cam, f"f_{k:06d}.png")
        if not (os.path.exists(gt_path) and os.path.exists(png)):
            continue
        img = cv2.imread(png)
        rec = json.load(open(gt_path))["frames"][k]
        crops = []
        for o in rec:
            x, y, r = int(round(o["x"])), int(round(o["y"])), 24
            c = np.zeros((2 * r + 1, 2 * r + 1, 3), np.uint8)
            y0, x0 = max(y - r, 0), max(x - r, 0)
            sub = img[y0:y + r + 1, x0:x + r + 1]
            c[y0 - (y - r):y0 - (y - r) + sub.shape[0], x0 - (x - r):x0 - (x - r) + sub.shape[1]] = sub
            c = cv2.resize(c, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(c, (c.shape[1] // 2, c.shape[0] // 2), (0, 0, 255), cv2.MARKER_CROSS, 30, 1)
            cv2.putText(c, f"{o['kind']} {o['id']} {o['dist']:.0f}m {o['px']:.1f}px", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            crops.append(c)
        small = cv2.resize(img, (1152, int(img.shape[0] * 1152 / img.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out, f"preview_{cam}_{k}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if crops:
            cv2.imwrite(os.path.join(out, f"targets_{cam}_{k}.png"), np.hstack(crops))
        print(cam, k, f"{len(rec)} целей в кадре:", [(o["kind"], o["id"], o["px"]) for o in rec])


if __name__ == "__main__":
    main()
