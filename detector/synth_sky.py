"""Генератор синтетического 16 Мп видео неба с удалёнными дронами.

Сцена: градиент неба, движущиеся облака, качающаяся кромка леса с просветами,
провода, 4 дрона (1.5-4 px, один подсвечен лазером = яркий), 3 ласточки.
Пишет gray-видео (h264, почти без потерь) и ground truth в JSON.

    python synth_sky.py --out sky16mp.mp4 --frames 100

--cls добавляет сцену для проверки классификатора дрон/птица: птицы летят ровно
(по траектории их не отличить), но машут крыльями 2-15 Гц, часть — пачками с
планированием; стая из 5 птиц; чайка кружит без взмахов (заведомо трудный случай);
дроны зависают, разворачиваются и снова разгоняются.

    python synth_sky.py --out cls.mp4 --w 1920 --h 1080 --frames 400 --cls
"""
import argparse, json, subprocess
import numpy as np, cv2


def fractal_noise(h, w, rng, octaves=6, base=8, persistence=0.55):
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = base * 2 ** o
        sh, sw = max(2, h * cells // w), cells
        n = rng.standard_normal((sh + 1, sw + 1)).astype(np.float32)
        out += amp * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= persistence
    return out / total


def blob(size, rng):
    """Маленький силуэт дрона/птицы (субпиксельный, с размытием оптики)."""
    s = 9
    k = np.zeros((s * 8, s * 8), np.float32)
    c = s * 4
    r = size * 4 / 2
    cv2.ellipse(k, (c, c), (int(r), int(r * 0.6)), rng.uniform(0, 180), 0, 360, 1.0, -1)
    k = cv2.resize(k, (s, s), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(k, (0, 0), 0.6)


def bird_kernel(size, span, angle):
    """Птица сверху/снизу: тело и крылья; span 0..1 — насколько раскрыты крылья."""
    s = 9
    k = np.zeros((s * 8, s * 8), np.float32)
    c = s * 4
    r = size * 4 / 2
    cv2.ellipse(k, (c, c), (max(1, int(r * 0.45)), max(1, int(r * 0.3))), angle, 0, 360, 1.0, -1)
    if span > 0.05:
        cv2.ellipse(k, (c, c), (max(1, int(r * 1.3 * span)), max(1, int(r * 0.22))), angle + 90, 0, 360, 1.0, -1)
    k = cv2.resize(k, (s, s), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(k, (0, 0), 0.6)


def cls_objects(W, H, fps, rng):
    """Цели для проверки классификатора (свой генератор: основная сцена не меняется)."""
    objs = []
    def heading(speed):
        a = rng.uniform(0, 2 * np.pi)
        return np.array([np.cos(a), np.sin(a)]) * speed
    for i in range(6):   # ровно летящие птицы с взмахами; нечётные — пачками с планированием
        objs.append(dict(kind="bird", model="flap", bright=False, size=rng.uniform(2.0, 4.0),
                         p=np.array([rng.uniform(.15, .85) * W, rng.uniform(.1, .6) * H]),
                         v=heading(rng.uniform(2, 7)), turn=rng.normal(0, 0.01),
                         freq=rng.uniform(2, 15), phase=rng.uniform(0, 1),
                         bursts=(i % 2 == 1), on=rng.uniform(0.4, 1.0), off=rng.uniform(0.5, 1.5),
                         contrast=rng.uniform(30, 60), angle=rng.uniform(0, 180)))
    lead = np.array([rng.uniform(.3, .7) * W, rng.uniform(.15, .4) * H])   # стая
    vf, ff = heading(rng.uniform(3, 6)), rng.uniform(3, 8)
    for i in range(5):
        objs.append(dict(kind="bird", model="flap", bright=False, size=rng.uniform(2.0, 3.5),
                         p=lead + rng.uniform(-60, 60, 2), v=vf.copy(), turn=0.004,
                         freq=ff * rng.uniform(0.9, 1.1), phase=rng.uniform(0, 1), bursts=False,
                         contrast=rng.uniform(30, 50), angle=rng.uniform(0, 180)))
    objs.append(dict(kind="bird", model="soar", bright=False, size=3.5,   # кружит без взмахов
                     p=np.array([0.5 * W, 0.3 * H]), v=heading(4.0), turn=2 * np.pi / (fps * 8),
                     freq=0, phase=0, bursts=False, contrast=40, angle=0))
    for i in range(4):   # дроны: разгон, полёт, торможение, зависание, разворот
        objs.append(dict(kind="drone", model="hover", bright=False, size=rng.uniform(1.5, 3.5),
                         p=np.array([rng.uniform(.15, .85) * W, rng.uniform(.1, .6) * H]),
                         v=np.zeros(2), cruise=heading(rng.uniform(2, 6)), t0=rng.uniform(0, 4),
                         contrast=rng.uniform(25, 50), angle=rng.uniform(0, 180)))
    return objs


def cls_step(o, f, fps, W, H, rng):
    """Сдвигает цель на кадр; возвращает ядро силуэта и множитель контраста."""
    t = f / fps
    if o["model"] == "hover":
        # цикл 6 с: 2.5 с полёт, 1 с торможение, 2 с висит, 0.5 с разгон в новом направлении
        ph = (t + o["t0"]) % 6.0
        if ph < 0.05 and f > 0 and not o.get("turned"):
            c, s_ = np.cos(rng.uniform(0.5, 2.5)), np.sin(rng.uniform(0.5, 2.5))
            o["cruise"] = np.array([c * o["cruise"][0] - s_ * o["cruise"][1], s_ * o["cruise"][0] + c * o["cruise"][1]])
            o["turned"] = True
        if ph >= 0.05:
            o["turned"] = False
        k = 1.0 if ph < 2.5 else max(0.0, 1 - (ph - 2.5)) if ph < 3.5 else 0.0 if ph < 5.5 else (ph - 5.5) / 0.5
        o["v"] = o["cruise"] * k
        o["p"] = o["p"] + o["v"] + rng.normal(0, 0.15, 2)          # дрожь в висении
        return blob_cache(o, 1.0), 1.0 + 0.05 * np.sin(t * 0.7) + rng.normal(0, 0.03)
    a = o["turn"]
    o["v"] = np.array([np.cos(a) * o["v"][0] - np.sin(a) * o["v"][1], np.sin(a) * o["v"][0] + np.cos(a) * o["v"][1]])
    o["p"] = o["p"] + o["v"]
    if o["freq"] <= 0:
        return blob_cache(o, 1.0), 1.0
    flapping = True
    if o["bursts"]:
        flapping = (t + o["phase"] * 3) % (o["on"] + o["off"]) < o["on"]
    span = 0.5 + 0.5 * np.cos(2 * np.pi * (o["freq"] * t + o["phase"])) if flapping else 1.0
    if flapping:   # тело качается вверх-вниз в такт взмахам
        o["p"] = o["p"] + np.array([0.0, 0.25 * o["size"] * np.sin(2 * np.pi * (o["freq"] * t + o["phase"]))]) * 0.2
    return bird_kernel(o["size"], span, o["angle"]), 1.0


def blob_cache(o, _):
    if "k" not in o:
        o["k"] = bird_kernel(o["size"], 1.0, o["angle"]) if o["kind"] == "bird" else blob(o["size"], np.random.default_rng(1))
    return o["k"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sky16mp.mp4")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--w", type=int, default=4608)
    ap.add_argument("--h", type=int, default=3456)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--insulators", type=int, default=0,
                    help="статичные ложные цели на проводах (катушки ЛЭП), мерцают")
    ap.add_argument("--cls", action="store_true",
                    help="добавить сцену для классификатора: машущие и парящие птицы, стая, зависающие дроны")
    ap.add_argument("--drone-contrast", default="25,50",
                    help="контраст дронов, мин,макс (тусклые дальние цели: 6,14)")
    ap.add_argument("--noise", type=float, default=2.0, help="СКО шума сенсора, уровни яркости")
    a = ap.parse_args()
    W, H, N = a.w, a.h, a.frames
    c_lo, c_hi = (float(v) for v in a.drone_contrast.split(","))
    rng = np.random.default_rng(a.seed)

    # --- статичные слои ---
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    sky = (205 - 45 * yy) * np.ones((1, W), np.float32)            # градиент неба
    pad = 400
    cl = fractal_noise(H + pad, W + pad, rng, octaves=7, base=6)
    cl = np.clip((cl - 0.05) * 3.0, 0, 1)                           # облачность
    shade = fractal_noise(H + pad, W + pad, rng, octaves=6, base=12)
    clouds = cl * (35 + 25 * shade)                                  # светлые облака с тенями
    clouds -= np.clip(cl - 0.6, 0, 1) * 60                           # тёмные "днища"

    tree_top = (H * 0.86 + H * 0.05 * fractal_noise(1, W, rng, octaves=7, base=10)[0]).astype(np.float32)
    leaves = fractal_noise(H, W, rng, octaves=3, base=400)          # мелкая листва
    wires = []
    for y0, sag in [(0.70, 0.04), (0.74, 0.05)]:
        xs = np.arange(W)
        wires.append(H * y0 + H * sag * ((xs / W - 0.5) * 2) ** 2)

    # катушки на проводах: свой генератор, чтобы остальное видео не менялось
    irng = np.random.default_rng(a.seed + 100)
    insul = []
    for i in range(a.insulators):
        x = int(irng.uniform(0.05, 0.95) * W)
        wy = wires[i % len(wires)]
        insul.append(dict(x=x, y=int(wy[x]) + 1, k=blob(irng.uniform(2.0, 3.5), irng),
                          c=irng.uniform(25, 45)))

    # --- цели ---
    objs = []
    for i in range(4):   # дроны: плавное движение
        objs.append(dict(kind="drone", bright=(i == 3), size=rng.uniform(1.5, 4.0),
                         p=np.array([rng.uniform(.15, .85) * W, rng.uniform(.1, .65) * H]),
                         v=rng.normal(0, 3, 2), contrast=rng.uniform(c_lo, c_hi)))
    for i in range(3):   # ласточки: быстрые, рваные
        objs.append(dict(kind="bird", bright=False, size=rng.uniform(2.0, 4.5),
                         p=np.array([rng.uniform(.1, .9) * W, rng.uniform(.1, .6) * H]),
                         v=rng.normal(0, 12, 2), contrast=rng.uniform(40, 70)))
    for o in objs:
        o["k"] = blob(o["size"], rng)
    cobjs = cls_objects(W, H, a.fps, np.random.default_rng(a.seed + 200)) if a.cls else []
    crng = np.random.default_rng(a.seed + 300)

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "fast",
         "-crf", "12", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    gt = []
    X = np.arange(W, dtype=np.float32)[None, :]
    Y = np.arange(H, dtype=np.float32)[:, None]
    for f in range(N):
        dx, dy = 0.6 * f, 0.15 * f                                   # дрейф облаков
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = sky + cv2.warpAffine(clouds, M, (W, H), flags=cv2.INTER_LINEAR)
        # провода
        for wy in wires:
            m = np.abs(Y - wy[None, :]) < 0.7
            img[m] -= 45
        # лес: качание по столбцам, просветы неба между листьями
        sway = 3 * np.sin(f * 0.25 + X[0] * 0.004)
        top = tree_top + sway
        tree = (Y > top[None, :]).astype(np.float32)
        gaps = (leaves > 0.9).astype(np.float32) * (Y > top[None, :] + 15)
        img = img * (1 - tree) + tree * (55 + 20 * leaves)
        img = img * (1 - gaps) + gaps * 170
        for o in insul:   # отблеск: видна в ~3/4 кадров, контраст гуляет
            if irng.random() < 0.75:
                k = o["k"]
                s = k.shape[0] // 2
                img[o["y"] - s:o["y"] + s + 1, o["x"] - s:o["x"] + s + 1] -= o["c"] * irng.uniform(0.7, 1.0) * k / k.max()
        rec = []
        for o in objs:
            if o["kind"] == "bird":
                o["v"] = o["v"] * 0.9 + rng.normal(0, 4, 2)
            else:
                o["v"] = o["v"] * 0.98 + rng.normal(0, 0.3, 2)
            o["p"] = o["p"] + o["v"]
            o["p"][0] %= W
            o["p"][1] = np.clip(o["p"][1], 50, 0.8 * H)
            k = o["k"]
            if o["kind"] == "bird":                                   # взмахи крыльев
                k = k * (0.6 + 0.4 * abs(np.sin(f * 1.3)))
            x, y = int(o["p"][0]), int(o["p"][1])
            s = k.shape[0] // 2
            if s <= x < W - s - 1:
                sl = img[y - s:y + s + 1, x - s:x + s + 1]
                sl += (o["contrast"] if o["bright"] else -o["contrast"]) * k / k.max()
                rec.append(dict(kind=o["kind"], bright=o["bright"], x=x, y=y))
        for o in cobjs:
            k, cm = cls_step(o, f, a.fps, W, H, crng)
            o["p"][0] %= W
            o["p"][1] = np.clip(o["p"][1], 50, 0.8 * H)
            x, y = int(round(o["p"][0])), int(round(o["p"][1]))
            s = k.shape[0] // 2
            if s <= x < W - s - 1:
                img[y - s:y + s + 1, x - s:x + s + 1] -= o["contrast"] * cm * k / max(1e-6, blob_cache(o, 1).max())
                rec.append(dict(kind=o["kind"], bright=False, x=x, y=y, model=o["model"],
                                obj=cobjs.index(o), freq=round(float(o.get("freq", 0)), 1),
                                bursts=bool(o.get("bursts", False))))
        gt.append(rec)
        img += rng.normal(0, a.noise, img.shape).astype(np.float32)     # шум сенсора
        proc.stdin.write(np.clip(img, 0, 255).astype(np.uint8).tobytes())
        if f % 10 == 0:
            print("frame", f, flush=True)
    proc.stdin.close(); proc.wait()
    json.dump(dict(w=W, h=H, fps=a.fps, frames=gt), open(a.out + ".gt.json", "w"))


if __name__ == "__main__":
    main()
