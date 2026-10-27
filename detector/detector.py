"""Быстрый детектор точечных целей (дронов) на кадрах 16 Мп.

Идея та же, что в текущем алгоритме: точка, окружённая со всех сторон более
светлым (тёмная цель) или более тёмным (подсвеченная цель) фоном. Но вместо
попиксельного сравнения с соседями используются морфологические операции
OpenCV с линейными ядрами (O(1) на пиксель, SIMD, многопоточно):

  closing линейным ядром длины k "заливает" тёмные детали короче k вдоль
  этого направления. Минимум закрытий по 4 направлениям (—, |, \\, /)
  минус исходник = насколько точка темнее окружения ВО ВСЕХ направлениях.
  Провода (линии) и края облаков (ступеньки) дают ~0 хотя бы в одном
  направлении и поэтому подавляются; точки размером < k остаются.

Ускорение: перед поиском кадр сжимается 2x2 min-пулингом (для тёмных целей)
и max-пулингом (для ярких): 1-пиксельная точка при этом не теряется,
а объём работы падает в 4 раза.

Порог адаптивный: отклик сравнивается с локальным "уровнем клаттера"
(средний отклик в тайле), поэтому текстура облаков/леса даёт меньше ложных.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2


def _line_kernels(k):
    h = np.ones((1, k), np.uint8)
    v = np.ones((k, 1), np.uint8)
    d = np.eye(k, dtype=np.uint8)
    a = np.ascontiguousarray(np.fliplr(d))
    return [h, v, d, a]


class PointDetector:
    def __init__(self, k=7, pool=2, thr=10, clutter_k=3.0, tile=32, max_density=0.012,
                 dark=True, bright=True, max_cands=300, workers=1, use_umat=False):
        self.k = k                  # макс. размер цели (в пикселях после пулинга)
        self.pool = pool            # 1 = полный размер, 2 = пулинг 2x2
        self.thr = thr              # мин. контраст цели, уровни яркости
        self.clutter_k = clutter_k  # во сколько раз отклик выше локального клаттера
        self.tile = tile            # размер тайла для оценки клаттера и поиска максимумов
        self.max_density = max_density  # доля "откликнувшихся" пикселей в тайле, выше — это текстура
        self.dark, self.bright = dark, bright
        self.max_cands = max_cands
        self.kernels = _line_kernels(k)
        self.k2 = np.ones((2, 2), np.uint8)
        self.use_umat = use_umat    # OpenCL (T-API), если есть GPU
        # кадр режется на горизонтальные полосы, полосы считаются в потоках
        # (OpenCV отпускает GIL) — масштабируется по ядрам лучше, чем внутренний
        # параллелизм OpenCV на мелких операциях
        self.workers = workers
        self.pool_exec = ThreadPoolExecutor(workers) if workers > 1 else None
        self.timing = {}
        self.tile_mask = None       # тайлы, где ищем цели (None = везде)
        self.learn, self._tex_sum, self._tex_n = False, None, 0

    def learn_mask(self, frames, frac=0.3):
        """Камера стационарна: тайлы, которые постоянно "текстурные" (лес,
        здания, земля), исключаются из поиска. Учится по первым кадрам."""
        self.tile_mask = None
        h, w = frames[0].shape
        u = self.tile * self.pool
        self._tex_sum = np.zeros((h // u, w // u), np.float32)
        self._tex_n, self.learn = 0, True
        for f in frames:
            self.detect(f)
            self._tex_n += 1
        self.learn = False
        n = self._tex_n * (int(self.dark) + int(self.bright))
        bad = (self._tex_sum / n) > frac
        bad = cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        self.tile_mask = ~bad
        return self.tile_mask

    # --- пулинг 2x2 без потери точечных целей ---
    def _pool(self, img, op):
        if self.pool == 1:
            return img
        h, w = img.shape[:2]
        p = op(img, self.k2, anchor=(0, 0))
        return cv2.resize(p, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)

    def _dark_response(self, src):
        acc = None
        for K in self.kernels:
            c = cv2.morphologyEx(src, cv2.MORPH_CLOSE, K)
            acc = c if acc is None else cv2.min(acc, c)
        return cv2.subtract(acc, src)

    def _bright_response(self, src):
        acc = None
        for K in self.kernels:
            o = cv2.morphologyEx(src, cv2.MORPH_OPEN, K)
            acc = o if acc is None else cv2.max(acc, o)
        return cv2.subtract(src, acc)

    def _peaks(self, R, polarity, ty0):
        """Локальные максимумы отклика: по одному на тайл, с проверкой клаттера.
        R — отклик полосы, ty0 — номер первой строки тайлов полосы в кадре."""
        h, w = R.shape
        t = self.tile
        hs, ws = h // t, w // t
        Rc = R[:hs * t, :ws * t]
        # средний отклик в тайле = уровень клаттера (облака, листва)
        level = cv2.resize(Rc, (ws, hs), interpolation=cv2.INTER_AREA).astype(np.float32)
        # максимум в тайле: dilate + выборка (быстрее, чем findNonZero на 4-16 Мп)
        mx = cv2.dilate(Rc, np.ones((t, t), np.uint8), anchor=(0, 0))
        mx = mx[::t, ::t][:hs, :ws].astype(np.float32)
        # дрон — изолированная точка: в тайле мало пикселей выше порога.
        # Листва, рябь облаков, шум дают много откликов — такой тайл отбрасываем.
        hot = cv2.threshold(Rc, self.thr * 0.6, 255, cv2.THRESH_BINARY)[1]
        dens = cv2.resize(hot, (ws, hs), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        textured = dens > self.max_density
        if self.learn:   # статистика "текстурных" тайлов для авто-маски зоны неба
            self._tex_sum[ty0:ty0 + hs] += textured
        ok = (mx >= self.thr) & (mx >= self.clutter_k * level + self.thr * 0.5) & ~textured
        if self.tile_mask is not None:
            ok &= self.tile_mask[ty0:ty0 + hs]
        ys, xs = np.nonzero(ok)
        if len(ys) > self.max_cands:   # слишком много — берём самые контрастные
            idx = np.argsort(-mx[ys, xs])[:self.max_cands]
            ys, xs = ys[idx], xs[idx]
        out = []
        for ty, tx in zip(ys, xs):
            blk = Rc[ty * t:(ty + 1) * t, tx * t:(tx + 1) * t]
            _, v, _, (bx, by) = cv2.minMaxLoc(blk)
            out.append((tx * t + bx, (ty0 + ty) * t + by, float(v),
                        float(v / (level[ty, tx] + 1.0)), polarity))
        return out

    def _strip(self, gray, y0, y1):
        """Обработка полосы строк [y0, y1) кадра (y0, y1 кратны tile*pool)."""
        H, W = gray.shape
        s = self.pool
        m = 2 * self.k * s                       # перекрытие для морфологии
        a, b = max(0, y0 - m), min(H, y1 + m)
        sub = gray[a:b]
        r0 = (y0 - a) // s                       # строки ядра полосы в отклике
        r1 = r0 + (y1 - y0) // s
        e = self.k                               # морфология у края кадра даёт ложные отклики
        cands = []
        for on, op, resp, pol in ((self.dark, cv2.erode, self._dark_response, -1),
                                  (self.bright, cv2.dilate, self._bright_response, +1)):
            if not on:
                continue
            src = self._pool(cv2.UMat(sub) if self.use_umat else sub, op)
            R = resp(src)
            if self.use_umat:
                R = R.get()
            R = R[r0:r1]
            R[:, :e] = 0; R[:, -e:] = 0
            if y0 == 0:
                R[:e] = 0
            if y1 >= H - self.tile * s:
                R[-e:] = 0
            cands += self._peaks(R, pol, y0 // (self.tile * s))
        return cands

    def detect(self, gray):
        """gray: uint8 HxW. Возвращает [(x, y, contrast, snr, polarity)] в координатах кадра."""
        t0 = time.perf_counter()
        H = gray.shape[0]
        u = self.tile * self.pool
        rows = H // u
        n = self.workers
        cuts = [u * (rows * i // n) for i in range(n + 1)]
        cuts[-1] = H
        jobs = [(cuts[i], cuts[i + 1]) for i in range(n) if cuts[i + 1] > cuts[i]]
        if self.pool_exec:
            parts = list(self.pool_exec.map(lambda j: self._strip(gray, *j), jobs))
        else:
            parts = [self._strip(gray, *j) for j in jobs]
        s = self.pool
        cands = [(x * s + (s - 1) * 0.5, y * s + (s - 1) * 0.5, c, snr, p)
                 for part in parts for x, y, c, snr, p in part]
        self.timing = dict(detect_ms=(time.perf_counter() - t0) * 1000)
        return cands


class Tracker:
    """Простой трекер ближайшего соседа + карта статичного клаттера.

    Цель подтверждается, если найдена в >= confirm кадрах из последних window.
    Ячейки, где детекции появляются почти каждый кадр долгое время
    (просветы в листве, углы проводов), помечаются как клаттер.
    Ласточки остаются целями: их можно будет отсеять по паттерну движения
    (резкие развороты, высокая скорость) — скорость трека сохраняется.
    """

    def __init__(self, shape, gate=40.0, confirm=3, window=5, max_miss=5,
                 clutter_cell=8, clutter_decay=0.97, clutter_level=0.35):
        self.gate = gate
        self.confirm, self.window, self.max_miss = confirm, window, max_miss
        self.tracks = []
        self.next_id = 1
        self.cell = clutter_cell
        h, w = shape
        self.heat = np.zeros((h // clutter_cell + 1, w // clutter_cell + 1), np.float32)
        self.decay, self.clevel = clutter_decay, clutter_level

    def _hot(self, x, y):
        return self.heat[int(y) // self.cell, int(x) // self.cell] > self.clevel

    def reset_tracks(self):
        self.tracks = []

    def update(self, dets):
        # карта клаттера: экспоненциальная частота срабатываний в ячейке
        self.heat *= self.decay
        if dets:
            xy = np.array([(d[0], d[1]) for d in dets], np.float32)
            cy, cx = (xy[:, 1] // self.cell).astype(int), (xy[:, 0] // self.cell).astype(int)
            np.add.at(self.heat, (cy, cx), 1 - self.decay)
            hot = self.heat[cy, cx] > self.clevel
        else:
            xy, hot = np.zeros((0, 2), np.float32), np.zeros(0, bool)
        # детекции в "горячих" ячейках (статичный клаттер) в трекинг не идут
        self.clutter_dets = [d for d, h in zip(dets, hot) if h]
        dets = [d for d, h in zip(dets, hot) if not h]
        xy = xy[~hot]
        # ассоциация: матрица расстояний треки x детекции, жадно по возрастанию
        match = {}
        if self.tracks and len(dets):
            P = np.array([(t["x"] + t["vx"], t["y"] + t["vy"]) for t in self.tracks], np.float32)
            Dm = np.hypot(P[:, None, 0] - xy[None, :, 0], P[:, None, 1] - xy[None, :, 1])
            ti, di = np.nonzero(Dm < self.gate)
            used = set()
            for k in np.argsort(Dm[ti, di]):
                a, b = int(ti[k]), int(di[k])
                if a not in match and b not in used:
                    match[a] = b
                    used.add(b)
        used = set(match.values())
        for ti_, tr in enumerate(self.tracks):
            best = match.get(ti_)
            px, py = tr["x"] + tr["vx"], tr["y"] + tr["vy"]
            tr["hist"].append(best is not None)
            tr["hist"] = tr["hist"][-self.window:]
            if best is not None:
                used.add(best)
                d = dets[best]
                nvx, nvy = d[0] - tr["x"], d[1] - tr["y"]
                tr["vx"] = 0.5 * tr["vx"] + 0.5 * nvx
                tr["vy"] = 0.5 * tr["vy"] + 0.5 * nvy
                tr["x"], tr["y"], tr["pol"], tr["c"] = d[0], d[1], d[4], d[2]
                tr["miss"] = 0
                tr["path"].append((d[0], d[1]))
                tr["path"] = tr["path"][-30:]
            else:
                tr["miss"] += 1
                tr["x"], tr["y"] = px, py
        self.tracks = [t for t in self.tracks if t["miss"] <= self.max_miss]
        for i, d in enumerate(dets):
            if i not in used:
                self.tracks.append(dict(id=self.next_id, x=d[0], y=d[1], vx=0.0, vy=0.0,
                                        pol=d[4], c=d[2], miss=0, hist=[True],
                                        path=[(d[0], d[1])]))
                self.next_id += 1
        out = []
        for tr in self.tracks:
            if sum(tr["hist"]) < self.confirm or tr["miss"] > 0:
                continue
            out.append(dict(tr, clutter=self._hot(tr["x"], tr["y"])))
        return out
