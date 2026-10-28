"""Синтетика для неподвижной широкой камеры: горизонт с лесом, дальние тусклые дроны.

В отличие от synth_sky.py всё задано во времени, а не в кадрах: скорости в пикс./с,
взмахи в Гц, длительность в секундах. Один и тот же --seed при разной --fps даёт ту же
сцену, снятую с другой частотой кадров (проверка, что трекер считает в секундах).

Сцена: небо светлее к горизонту (дымка), облака дрейфуют, кромка леса качается и
дрожит (марево у горизонта), просветы в кронах. Цели (поле "group" в разметке):
  far   — дальние тусклые дроны в полосе над лесом (1.5-2.5 пикс., контраст --far-contrast)
  high  — более близкие дроны выше в небе, появляются в разное время (задержка обнаружения)
  pair  — пара дронов строем в 20 пикс.: второй слабее и в том же тайле
  bird  — ласточки (рваный полёт, взмахи 8-12 Гц) и чайки (ровно, 3 Гц)

Пишет y4m (gray, без сжатия) и разметку <out>.gt.json:

    python synth_horizon.py --out hz.y4m --w 2304 --h 1728 --fps 25 --seconds 10
"""
import argparse, json
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


def blob(size, rng, aspect=0.6):
    s = 11
    k = np.zeros((s * 8, s * 8), np.float32)
    c = s * 4
    r = size * 4 / 2
    cv2.ellipse(k, (c, c), (max(1, int(r)), max(1, int(r * aspect))), rng.uniform(0, 180), 0, 360, 1.0, -1)
    k = cv2.resize(k, (s, s), interpolation=cv2.INTER_AREA)
    k = cv2.GaussianBlur(k, (0, 0), 0.6)
    return k / k.max()


def stamp(img, k, x, y, amp):
    """Субпиксельная вставка: ядро сдвигается на дробную часть координаты."""
    s = k.shape[0] // 2
    xi, yi = int(np.floor(x)), int(np.floor(y))
    H, W = img.shape
    if not (s + 1 <= xi < W - s - 2 and s + 1 <= yi < H - s - 2):
        return False
    M = np.float32([[1, 0, x - xi], [0, 1, y - yi]])
    kk = cv2.warpAffine(k, M, k.shape[::-1], flags=cv2.INTER_LINEAR)
    img[yi - s:yi + s + 1, xi - s:xi + s + 1] += amp * kk
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hz.y4m")
    ap.add_argument("--w", type=int, default=2304)
    ap.add_argument("--h", type=int, default=1728)
    ap.add_argument("--fps", type=float, default=25)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--noise", type=float, default=2.0, help="СКО шума матрицы, уровни яркости")
    ap.add_argument("--far-contrast", default="7,12", help="контраст дальних дронов, мин,макс")
    ap.add_argument("--horizon", type=float, default=0.62, help="кромка леса, доля высоты кадра")
    ap.add_argument("--clouds", type=float, default=1.0, help="облачность 0..1.5")
    ap.add_argument("--sway", type=float, default=1.0, help="сила качания и марева кромки леса, 1 = обычная")
    ap.add_argument("--empty", action="store_true", help="без целей: замер ложных тревог на пустом небе")
    a = ap.parse_args()
    W, H, fps = a.w, a.h, a.fps
    N = int(round(a.seconds * fps))
    rng = np.random.default_rng(a.seed)
    lo, hi = (float(v) for v in a.far_contrast.split(","))
    sc = W / 2304.0          # скорости заданы для кадра 2304 по ширине

    # --- статичные слои ---
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    sky = (150 + 60 * np.clip(yy / a.horizon, 0, 1) ** 1.5) * np.ones((1, W), np.float32)
    pad = int(400 * sc) + 64
    cl = fractal_noise(H + pad, W + pad, rng, octaves=7, base=6)
    cl = np.clip((cl - 0.12) * 3.0 * a.clouds, 0, 1)
    shade = fractal_noise(H + pad, W + pad, rng, octaves=6, base=12)
    clouds = cl * (30 + 22 * shade) - np.clip(cl - 0.65, 0, 1) * 50
    cloud_v = np.array([14.0, 2.0]) * sc     # пикс./с
    hy = a.horizon * H
    tree_top = (hy + H * 0.035 * fractal_noise(1, W, rng, octaves=7, base=12)[0]).astype(np.float32)
    leaves = fractal_noise(H, W, rng, octaves=3, base=300)
    X = np.arange(W, dtype=np.float32)[None, :]
    Y = np.arange(H, dtype=np.float32)[:, None]

    # --- цели (скорости в пикс./с, время в с) ---
    objs = []
    def add(**o):
        o["id"] = len(objs)
        o.setdefault("t0", 0.0)
        o.setdefault("bright", False)
        o.setdefault("flap", 0.0)
        o["k"] = blob(o["size"], rng)
        objs.append(o)
    for i in range(5):    # дальние: в полосе 1-12% высоты над лесом
        y = hy - H * rng.uniform(0.015, 0.12)
        add(group="far", kind="drone", size=rng.uniform(1.5, 2.5), c=rng.uniform(lo, hi),
            p=np.array([rng.uniform(.1, .9) * W, y]), v=np.array([rng.choice([-1, 1]) * rng.uniform(8, 40) * sc,
                                                                   rng.uniform(-2, 2) * sc]))
    for i in range(4):    # ближе и выше: контрастнее и быстрее, появляются в разное время
        add(group="high", kind="drone", size=rng.uniform(3, 6), c=rng.uniform(22, 40),
            p=np.array([rng.uniform(.1, .9) * W, rng.uniform(.06, .35) * H]),
            v=np.array([rng.uniform(-1, 1), rng.uniform(-0.3, 0.3)]) * rng.uniform(80, 220) * sc,
            t0=rng.uniform(0.5, 0.45 * a.seconds), bright=(i == 3))
    base = np.array([rng.uniform(.3, .7) * W, hy - H * 0.2])
    vel = np.array([rng.uniform(25, 45), rng.uniform(-3, 3)]) * sc
    add(group="pair", kind="drone", size=3.0, c=28.0, p=base.copy(), v=vel.copy())
    add(group="pair", kind="drone", size=2.5, c=22.0, p=base + np.array([20.0, 6.0]), v=vel.copy())
    for i in range(3):    # ласточки
        add(group="bird", kind="bird", size=rng.uniform(2.5, 4.5), c=rng.uniform(35, 55),
            p=np.array([rng.uniform(.1, .9) * W, rng.uniform(.15, .5) * H]),
            v=rng.normal(0, 150, 2) * sc, flap=rng.uniform(8, 12), erratic=True)
    for i in range(2):    # чайки: ровный полёт, медленные взмахи
        add(group="bird", kind="bird", size=rng.uniform(4, 6), c=rng.uniform(35, 50),
            p=np.array([rng.uniform(.1, .9) * W, rng.uniform(.15, .45) * H]),
            v=np.array([rng.choice([-1, 1]) * rng.uniform(60, 120), rng.uniform(-10, 10)]) * sc, flap=3.0,
            erratic=False)

    out = open(a.out, "wb")
    num, den = (int(round(fps * 1000)), 1000)
    out.write(f"YUV4MPEG2 W{W} H{H} F{num}:{den} Ip A1:1 Cmono\n".encode())
    srng = np.random.default_rng(a.seed + 7)   # шум и марево: свой генератор
    gt = []
    shimmer = np.zeros(W, np.float32)
    for f in range(N):
        t = f / fps
        d = cloud_v * t
        M = np.float32([[1, 0, -pad / 2 + d[0]], [0, 1, -pad / 2 + d[1]]])
        img = sky + cv2.warpAffine(clouds, M, (W, H), flags=cv2.INTER_LINEAR)
        # кромка леса: качание (0.4 Гц) и марево (быстрое дрожание по столбцам)
        shimmer = 0.6 * shimmer + 0.4 * cv2.GaussianBlur(srng.normal(0, 1.2 * a.sway, (1, W)).astype(np.float32),
                                                         (0, 0), 3)[0]
        top = tree_top + 2.0 * a.sway * np.sin(2 * np.pi * 0.4 * t + X[0] * 0.003) + shimmer
        tree = np.clip(Y - top[None, :] + 0.5, 0, 1)
        gaps = (leaves > 0.85).astype(np.float32) * (Y > top[None, :] + 12)
        img = img * (1 - tree) + tree * (60 + 25 * leaves)
        img = img * (1 - gaps) + gaps * 175
        rec = []
        for o in objs:
            if a.empty or t < o["t0"]:
                continue
            if o.get("erratic"):
                o["v"] = o["v"] * (1 - 2.5 / fps) + srng.normal(0, 600 * sc / np.sqrt(fps), 2)
            o["p"] = o["p"] + o["v"] / fps
            if o["group"] == "far":
                o["p"][1] = min(o["p"][1], hy - 0.01 * H)
            amp = o["c"]
            if o["flap"]:
                amp *= 0.55 + 0.45 * abs(np.sin(np.pi * o["flap"] * t + o["id"]))
            x, y = float(o["p"][0]), float(o["p"][1])
            if y < 20 or not (0 <= x < W) or y > top[int(x)] - 6:   # за лесом не видно
                continue
            if stamp(img, o["k"], x, y, amp if o["bright"] else -amp):
                rec.append(dict(kind=o["kind"], bright=o["bright"], x=round(x, 1), y=round(y, 1),
                                id=o["id"], group=o["group"]))
        gt.append(rec)
        img += srng.normal(0, a.noise, img.shape).astype(np.float32)
        out.write(b"FRAME\n")
        out.write(np.clip(img + 0.5, 0, 255).astype(np.uint8).tobytes())
    out.close()
    json.dump(dict(w=W, h=H, fps=fps, frames=gt), open(a.out + ".gt.json", "w"))


if __name__ == "__main__":
    main()
