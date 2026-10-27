"""Замер скорости вариантов на кадрах 4608x3456 (только обработка, без декодирования).

    python bench.py data/sky16mp.mp4
"""
import sys, time, json
import numpy as np, cv2
from detector import PointDetector, Tracker
from run_demo import load_frames


def naive_8neighbors(g, thr=10):
    """Текущий подход для сравнения: точка темнее/светлее всех 8 соседей на thr (NumPy, векторно)."""
    c = g[1:-1, 1:-1].astype(np.int16)
    dark = np.ones(c.shape, bool)
    bright = np.ones(c.shape, bool)
    h, w = g.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            n = g[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx].astype(np.int16)
            dark &= n - c >= thr
            bright &= c - n >= thr
    return np.count_nonzero(dark | bright)


def timeit(fn, frames, warm=2):
    for f in frames[:warm]:
        fn(f)
    t = time.perf_counter()
    for f in frames:
        fn(f)
    return (time.perf_counter() - t) / len(frames) * 1000


def main():
    frames = load_frames(sys.argv[1], max_frames=20)
    h, w = frames[0].shape
    rows = []
    ncpu = cv2.getNumThreads()
    rows.append(("наивный 8 соседей, NumPy", 1, timeit(naive_8neighbors, frames[:5])))
    cv2.setNumThreads(1)
    for pool in (1, 2):
        for workers in sorted({1, 2, ncpu}):
            det = PointDetector(pool=pool, workers=workers)
            trk = Tracker((h, w))
            rows.append((f"морфология, pool={pool}", workers,
                         timeit(lambda f: trk.update(det.detect(f)), frames)))
    cv2.setNumThreads(ncpu)
    print(f"Кадр {w}x{h}, CPU потоков доступно: {ncpu}")
    out = []
    for name, th, ms in rows:
        print(f"{name:32s} потоков={th}  {ms:7.1f} мс/кадр  {1000 / ms:6.1f} FPS")
        out.append(dict(variant=name, threads=th, ms=round(ms, 1), fps=round(1000 / ms, 1)))
    json.dump(out, open(sys.argv[1] + ".bench.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
