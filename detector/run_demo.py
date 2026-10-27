"""Зацикленный прогон видео через детектор: разметка целей, запись демо, замер FPS.

    python run_demo.py data/sky16mp.mp4 --loops 3 --out demo.mp4
    python run_demo.py real.mp4 --upscale 4608x3456 --show      # окно на экране

Кадры заранее декодируются в память (камера отдаёт сырые кадры, декодирование
h264 не должно входить в замер). Время детекции и трекинга меряется отдельно
от отрисовки и записи демо.
"""
import argparse, json, os, time
import numpy as np, cv2
from detector import PointDetector, Tracker


def load_frames(path, upscale=None, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if upscale:
            g = cv2.resize(g, upscale, interpolation=cv2.INTER_LINEAR)
        frames.append(g)
    return frames


def render(gray, tracks, dets, clutter=(), view_w=1920, panel=360, zoom=6, crop=40):
    h, w = gray.shape
    s = view_w / w
    view = cv2.cvtColor(cv2.resize(gray, (view_w, int(h * s)), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_GRAY2BGR)
    for x, y, *_ in dets:   # сырые кандидаты — мелкие серые точки
        cv2.circle(view, (int(x * s), int(y * s)), 2, (150, 150, 150), -1)
    for x, y, *_ in clutter:   # подавленный статичный клаттер — серые квадраты
        cv2.rectangle(view, (int(x * s) - 6, int(y * s) - 6), (int(x * s) + 6, int(y * s) + 6), (120, 120, 120), 1)
    canvas = np.full((view.shape[0], view_w + panel, 3), 30, np.uint8)
    good = [t for t in tracks if not t["clutter"]]
    good.sort(key=lambda t: -t["c"])
    for t in tracks:
        col = (128, 128, 128) if t["clutter"] else ((0, 255, 255) if t["pol"] > 0 else (0, 0, 255))
        cx, cy = int(t["x"] * s), int(t["y"] * s)
        cv2.rectangle(view, (cx - 12, cy - 12), (cx + 12, cy + 12), col, 2)
        if not t["clutter"]:
            pts = np.array([(int(px * s), int(py * s)) for px, py in t["path"]], np.int32)
            cv2.polylines(view, [pts], False, col, 1)
            cv2.putText(view, f'#{t["id"]}', (cx + 14, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, .5, col, 1)
    canvas[:, :view_w] = view
    # увеличенные вырезки найденных целей (на общем плане дрон — 1-3 пикселя)
    tsz = crop * 2 * zoom // 2
    ypos = 10
    for t in good[: (view.shape[0] - 10) // (tsz + 30)]:
        x, y = int(t["x"]), int(t["y"])
        x0, y0 = np.clip(x - crop // 2, 0, w - crop), np.clip(y - crop // 2, 0, h - crop)
        c = gray[y0:y0 + crop, x0:x0 + crop]
        c = cv2.resize(c, (tsz, tsz), interpolation=cv2.INTER_NEAREST)
        x1 = view_w + (panel - tsz) // 2
        canvas[ypos + 20:ypos + 20 + tsz, x1:x1 + tsz] = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        sp = np.hypot(t["vx"], t["vy"])
        txt = f'#{t["id"]} {"bright" if t["pol"] > 0 else "dark"} c={t["c"]:.0f} v={sp:.1f}px/f'
        cv2.putText(canvas, txt, (view_w + 8, ypos + 14), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        ypos += tsz + 30
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--loops", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--upscale", default="", help="WxH, напр. 4608x3456")
    ap.add_argument("--pool", type=int, default=2)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--thr", type=float, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="потоков-полос на кадр")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--umat", action="store_true", help="OpenCL/GPU через cv2.UMat")
    ap.add_argument("--learn-mask", type=int, default=0, help="N первых кадров для авто-маски неба")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stats", default="")
    a = ap.parse_args()
    # параллелим полосами кадра; внутренний параллелизм OpenCV при этом только мешает
    cv2.setNumThreads(a.threads or (1 if a.workers > 1 else 0))
    up = tuple(map(int, a.upscale.split("x"))) if a.upscale else None
    t0 = time.time()
    frames = load_frames(a.video, up, a.max_frames or None)
    print(f"загружено {len(frames)} кадров {frames[0].shape[1]}x{frames[0].shape[0]} за {time.time() - t0:.1f} c")
    gt_path = a.video + ".gt.json"
    gt = json.load(open(gt_path))["frames"] if os.path.exists(gt_path) and not up else None

    det = PointDetector(k=a.k, pool=a.pool, thr=a.thr, workers=a.workers, use_umat=a.umat)
    if a.learn_mask:
        m = det.learn_mask(frames[:a.learn_mask])
        print(f"авто-маска: исключено {100 * (1 - m.mean()):.1f}% тайлов (лес/земля)")
    trk = Tracker(frames[0].shape)
    writer = None
    times, parts = [], []
    hit = {"drone": [0, 0], "bird": [0, 0]}
    fp_frames, fp_total = 0, 0
    n = len(frames)
    for i in range(n * a.loops):
        fi = i % n
        g = frames[fi]
        if fi == 0:
            trk.reset_tracks()   # видео зациклено — трек не должен "прыгать" через склейку
        t1 = time.perf_counter()
        dets = det.detect(g)
        tracks = trk.update(dets)
        dt = time.perf_counter() - t1
        if i >= 3:
            times.append(dt)
            parts.append(det.timing)
        if gt is not None and i >= n:   # оцениваем после первого прохода (прогрев трекера)
            live = [t for t in tracks if not t["clutter"]]
            matched = set()
            for o in gt[fi]:
                k = o["kind"]
                hit[k][1] += 1
                for j, t in enumerate(live):
                    if abs(t["x"] - o["x"]) < 10 and abs(t["y"] - o["y"]) < 10:
                        hit[k][0] += 1
                        matched.add(j)
                        break
            fp = len(live) - len(matched)
            fp_total += fp
            fp_frames += 1
        if a.out or a.show:
            vis = render(g, tracks, dets, trk.clutter_dets)
            fps_now = 1.0 / np.mean(times[-20:]) if times else 0
            cv2.putText(vis, f"{g.shape[1]}x{g.shape[0]}  detect+track {dt * 1000:.0f} ms  ~{fps_now:.1f} FPS  "
                             f"cands={len(dets)}  tracks={sum(not t['clutter'] for t in tracks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
            if a.out:
                if writer is None:
                    writer = cv2.VideoWriter(a.out + ".tmp.avi", cv2.VideoWriter_fourcc(*"MJPG"), 25,
                                             (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            if a.show:
                cv2.imshow("dronolok", vis)
                if cv2.waitKey(1) == 27:
                    break
    if writer is not None:
        writer.release()
        os.system(f'ffmpeg -loglevel error -y -i "{a.out}.tmp.avi" -c:v libx264 -crf 23 -pix_fmt yuv420p "{a.out}" '
                  f'&& rm "{a.out}.tmp.avi"')
    ms = np.array(times) * 1000
    res = dict(mask=bool(a.learn_mask), resolution=f"{frames[0].shape[1]}x{frames[0].shape[0]}", pool=a.pool, k=a.k,
               threads=cv2.getNumThreads(), frames=len(ms),
               mean_ms=round(float(ms.mean()), 1), p95_ms=round(float(np.percentile(ms, 95)), 1),
               fps=round(1000 / float(ms.mean()), 1),
               detect_ms=round(float(np.mean([p["detect_ms"] for p in parts])), 1), workers=a.workers)
    if gt is not None:
        res["recall_drone"] = round(hit["drone"][0] / max(1, hit["drone"][1]), 3)
        res["recall_bird"] = round(hit["bird"][0] / max(1, hit["bird"][1]), 3)
        res["false_tracks_per_frame"] = round(fp_total / max(1, fp_frames), 2)
    print(json.dumps(res, ensure_ascii=False))
    if a.stats:
        with open(a.stats, "a") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
