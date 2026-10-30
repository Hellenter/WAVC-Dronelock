#!/usr/bin/env python3
"""Склейка 12 снимков по кругу (sim.gd --panorama) в полосу 360° с подписями секторов.
    python sim/panorama.py out/pano      -> out/pano/pano.jpg"""
import json, os, sys
import cv2
import numpy as np


def main():
    out = sys.argv[1]
    info = json.load(open(os.path.join(out, "pano.json"), encoding="utf-8"))
    shots = [cv2.imread(os.path.join(out, f"pano_{i:02d}.png")) for i in range(len(info["shots"]))]
    strip = np.hstack(shots)
    w = strip.shape[1]
    bar = np.full((44, w, 3), 30, np.uint8)
    for sc in info.get("sectors", []):
        a0, a1 = float(sc.get("az_from", 0)), float(sc.get("az_to", 360))
        x0, x1 = int(a0 / 360 * w), int(((a1 - a0) % 360 or 360) / 360 * w) + int(a0 / 360 * w)
        for x in (x0, x1 % w):
            cv2.line(strip, (x, 0), (x, strip.shape[0] - 1), (0, 255, 255), 2)
        name = str(sc.get("name", sc.get("type", "")))
        cv2.putText(bar, f"{name}  {a0:.0f}-{a1:.0f} deg", ((x0 + x1) // 2 - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    for az in range(0, 360, 30):
        x = int(az / 360 * w)
        cv2.putText(bar, f"{az}", (x + 4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    img = np.vstack([bar, strip])
    cv2.imwrite(os.path.join(out, "pano.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("pano.jpg", img.shape[1], "x", img.shape[0])


if __name__ == "__main__":
    main()
