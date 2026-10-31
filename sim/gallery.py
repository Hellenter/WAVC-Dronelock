#!/usr/bin/env python3
"""Склейка галереи погоды (sim.gd --gallery) в одну картинку: широкая камера и врезка точной на ближайший
дрон, подпись — погода, время, солнце, видимость.  python sim/gallery.py out/g -> out/g/gallery.jpg"""
import json, os, sys
import cv2
import numpy as np

TILE_W = 576


def caption(s):
    t = s.get("label", "")
    if "time_now" in s:
        h = float(s["time_now"])
        t += f"  {int(h):02d}:{int(round(h * 60)) % 60:02d}"
    t2 = f"sun {s.get('sun_el', 0):.1f} deg  clouds {s.get('clouds', 0):.2f}  vis {s.get('visibility', 0):.0f} m"
    if s.get("rain", 0):
        t2 += f"  rain {s['rain']:.2f}"
    if s.get("snow", 0):
        t2 += f"  snow {s['snow']:.2f}"
    return t, t2


def main():
    out = sys.argv[1]
    info = json.load(open(os.path.join(out, "gallery.json"), encoding="utf-8"))
    tiles = []
    for i, s in enumerate(info["shots"]):
        img = cv2.imread(os.path.join(out, f"gallery_{i:02d}.png"))
        img = cv2.resize(img, (TILE_W, int(img.shape[0] * TILE_W / img.shape[1])), interpolation=cv2.INTER_AREA)
        fine = cv2.imread(os.path.join(out, f"gallery_{i:02d}_fine.png"))
        if fine is not None and s.get("fine_dist"):
            fw = TILE_W // 3
            fine = cv2.resize(fine, (fw, int(fine.shape[0] * fw / fine.shape[1])), interpolation=cv2.INTER_AREA)
            y0 = img.shape[0] - fine.shape[0] - 4
            img[y0:y0 + fine.shape[0], -fine.shape[1] - 4:-4] = fine
            cv2.rectangle(img, (img.shape[1] - fine.shape[1] - 5, y0 - 1), (img.shape[1] - 4, y0 + fine.shape[0]), (0, 255, 255), 1)
            cv2.putText(img, f"fine: drone {s['fine_dist']:.0f} m", (img.shape[1] - fine.shape[1] - 2, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        bar = np.full((40, TILE_W, 3), 28, np.uint8)
        t1, t2 = caption(s)
        cv2.putText(bar, t1, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, t2, (6, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, img]))
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    cv2.imwrite(os.path.join(out, "gallery.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("gallery.jpg:", len(info["shots"]), "кадров")


if __name__ == "__main__":
    main()
