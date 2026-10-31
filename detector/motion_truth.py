"""Автоматическая разметка летящих целей для роликов с неподвижной камерой (записи War Thunder и т.п.).

Правда — то, что движется на небе: точка есть на кадре t, но её нет на тех же местах на кадрах
t-D и t+D (разность кадров), выше кромки леса/крыш. Отклики связываются в треки, остаются только
треки, которые прожили не меньше --min-len кадров и пролетели не меньше --min-travel пикс.
Облака за 2*D кадров почти не сдвигаются, листва — ниже кромки. Пишет <видео>.gt.json в формате
synth_horizon.py (все цели — "drone", группа "moving"), который понимают dronolok_bench и приложение.

    python motion_truth.py wt4.mp4 [--scale 0.5] [--d 6]

Это не ручная разметка: зависшую цель и цель на фоне крон она не видит, а самолёт и дрон не различает.
"""
import argparse, json, subprocess
import numpy as np, cv2


def frames(path, w, h):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h}:flags=area", "-pix_fmt", "gray",
                          "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    n = w * h
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--d", type=int, default=6, help="разнос кадров для разности")
    ap.add_argument("--thr", type=float, default=10.0)
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--min-travel", type=float, default=40.0, help="пикс. исходного кадра")
    ap.add_argument("--sky-from", type=int, default=300, help="кромку искать по медиане кадров отсюда (+30)")
    a = ap.parse_args()
    W0, H0, fps = (int(v) if i < 2 else v for i, v in enumerate(probe(a.video)))
    w, h = int(W0 * a.scale) // 2 * 2, int(H0 * a.scale) // 2 * 2
    sx, sy = W0 / w, H0 / h
    D = a.d
    fr = []
    for i, f in enumerate(frames(a.video, w, h)):   # кромка — по медиане кадров, когда камера уже стоит
        if i >= a.sky_from:
            fr.append(f)
        if len(fr) >= 30:
            break
    if len(fr) < 5:
        raise SystemExit("ролик короче --sky-from")
    sky = sky_mask(np.median(np.stack(fr), 0).astype(np.float32))
    dets = []          # по кадрам: список (x, y, area)
    ring = []
    for i, f in enumerate(frames(a.video, w, h)):
        ring.append(f.astype(np.int16))
        if len(ring) > 2 * D + 1:
            ring.pop(0)
        dets.append([])
        if len(ring) < 2 * D + 1:
            continue
        t = i - D
        c, p, n = ring[D], ring[0], ring[2 * D]
        dd = np.minimum(np.abs(c - p), np.abs(c - n)).astype(np.uint8)
        m = ((dd > a.thr) & sky).astype(np.uint8)
        if m.mean() > 0.002:   # меняется всё небо — камера поворачивается или переключается: не размечаем
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        k, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
        for j in range(1, k):
            x0, y0, bw, bh, ar = st[j]
            if ar < 2 or bw > 60 or bh > 60:
                continue
            dets[t].append((cen[j][0], cen[j][1], ar))
    N = len(dets)
    # треки: ближайший сосед, пропуск до 8 кадров
    tracks, live = [], []
    for t in range(N):
        used = set()
        for tr in live:
            lx, ly, lt = tr["pts"][-1]
            vx, vy = tr["v"]
            gap = t - lt
            px, py = lx + vx * gap, ly + vy * gap
            best, bd = None, 6.0 + 3.0 * gap
            for j, (x, y, ar) in enumerate(dets[t]):
                if j in used:
                    continue
                d = np.hypot(x - px, y - py)
                if d < bd:
                    best, bd = j, d
            if best is not None:
                x, y, _ = dets[t][best]
                used.add(best)
                tr["v"] = (0.7 * vx + 0.3 * (x - lx) / gap, 0.7 * vy + 0.3 * (y - ly) / gap)
                tr["pts"].append((x, y, t))
        keep = []
        for tr in live:
            if t - tr["pts"][-1][2] > 8:
                tracks.append(tr)
            else:
                keep.append(tr)
        live = keep
        for j, (x, y, ar) in enumerate(dets[t]):
            if j not in used:
                live.append(dict(pts=[(x, y, t)], v=(0.0, 0.0)))
    tracks += live
    gt = [[] for _ in range(N)]
    nt = 0
    for tr in tracks:
        pts = tr["pts"]
        if len(pts) < a.min_len:
            continue
        travel = max(np.hypot((x - pts[0][0]) * sx, (y - pts[0][1]) * sy) for x, y, _ in pts)
        if travel < a.min_travel:
            continue
        nt += 1
        # между попаданиями — по прямой (цель на кадре есть, просто разность её не взяла)
        for (x0, y0, t0), (x1, y1, t1) in zip(pts, pts[1:]):
            for t in range(t0, t1):
                u = (t - t0) / (t1 - t0)
                gt[t].append(dict(kind="drone", bright=False, x=round((x0 + u * (x1 - x0)) * sx, 1),
                                  y=round((y0 + u * (y1 - y0)) * sy, 1), id=nt, group="moving"))
        x, y, t = pts[-1]
        gt[t].append(dict(kind="drone", bright=False, x=round(x * sx, 1), y=round(y * sy, 1), id=nt, group="moving"))
    json.dump(dict(w=W0, h=H0, fps=fps, frames=gt), open(a.video + ".gt.json", "w"))
    print(f"{a.video}: {N} кадров, целей {nt}, кадров с целью {sum(1 for g in gt if g)}")


def sky_mask(med):
    """Небо: выше кромки — первого сверху места земли (текстура или заметно темнее неба над ним),
    под которым земли большинство."""
    g = cv2.GaussianBlur(med, (0, 0), 1.0)
    tex = cv2.blur(np.abs(cv2.Laplacian(g, cv2.CV_32F)), (9, 9)) > 2.5
    top = np.maximum.accumulate(cv2.blur(g, (15, 15)), 0)
    gnd = tex | (g < 0.7 * top)
    h, w = gnd.shape
    below = np.cumsum(gnd[::-1], 0)[::-1]          # земли ниже, включая строку
    rows = np.arange(h)[:, None]
    edge = gnd & (2 * below > (h - rows))
    line = np.where(edge.any(0), edge.argmax(0), h).astype(np.float32)
    pad = np.pad(line, 15, mode="edge")      # медиана по 31 столбцу: облако не вырезает "колодец" в небе
    line = np.median(np.lib.stride_tricks.sliding_window_view(pad, 31), 1).astype(np.float32)
    line = cv2.erode(line[None, :], np.ones((1, 9))).ravel()
    return rows < (line[None, :] - 4)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "csv=p=0", path], capture_output=True,
                         text=True).stdout.strip().split(",")
    n, d = out[2].split("/")
    return int(out[0]), int(out[1]), float(n) / float(d)


if __name__ == "__main__":
    main()
