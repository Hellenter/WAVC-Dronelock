"""Сверка нативного детектора с эталонной Python-версией (detector/detector.py)
и замер скорости обоих на одних и тех же кадрах.

    python native/python/compare_with_python.py detector/data/sky16mp.mp4 --frames 20
"""
import argparse, os, sys, time
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, "..", "..", "detector"))
import cv2  # noqa: E402
from detector import PointDetector  # noqa: E402
from run_demo import load_frames  # noqa: E402
from dronolok_native import NativePointDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    frames = load_frames(a.video, max_frames=a.frames)
    print(f"{len(frames)} кадров {frames[0].shape[1]}x{frames[0].shape[0]}")
    cv2.setNumThreads(1)
    py = PointDetector(workers=1)
    nat = NativePointDetector(threads=1)
    total, same, diff = 0, 0, 0
    for f in frames:
        a_ = sorted(py.detect(f))
        b_ = sorted(nat.detect(f))
        total += len(a_)
        sa = {(x, y, c, p) for x, y, c, _, p in a_}
        sb = {(x, y, c, p) for x, y, c, _, p in b_}
        same += len(sa & sb)
        diff += len(sa ^ sb)
    print(f"кандидатов Python: {total}, совпало: {same}, различий: {diff}")

    def bench(det, label):
        for f in frames[:3]:
            det.detect(f)
        t = time.perf_counter()
        for f in frames:
            det.detect(f)
        ms = (time.perf_counter() - t) / len(frames) * 1000
        print(f"{label:40s} {ms:7.2f} мс/кадр  {1000 / ms:7.1f} FPS")

    bench(py, "Python+OpenCV, 1 поток")
    cv2.setNumThreads(1)
    bench(PointDetector(workers=a.threads), f"Python+OpenCV, {a.threads} потока")
    for isa in ("scalar", "sse2", "avx2", "neon"):
        try:
            d = NativePointDetector(threads=1, isa=isa)
        except Exception:
            continue
        if not d.backend.endswith(isa):
            continue
        bench(d, f"нативное {d.backend}, 1 поток")
        bench(NativePointDetector(threads=a.threads, isa=isa), f"нативное {d.backend}, {a.threads} потока")
    return 0 if diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
