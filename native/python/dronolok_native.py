"""Python-обёртка (ctypes) над нативным ядром: те же интерфейсы, что у
PointDetector и Tracker из detector/detector.py, но в 10+ раз быстрее.

    from dronolok_native import NativePointDetector, NativeTracker
    det = NativePointDetector(threads=4)          # isa="auto" | "scalar" | "sse2" | "avx2" | "neon"
    cands = det.detect(gray)                      # [(x, y, contrast, snr, polarity)]

Библиотека ищется в DRONOLOK_LIB, рядом с этим файлом и в native/build.
"""
import ctypes as C
import os
import numpy as np

ISA = {"auto": 0, "scalar": 1, "sse2": 2, "avx2": 3, "neon": 4}
CLASSES = ["unknown", "drone", "bird", "clutter"]


class _Config(C.Structure):
    _fields_ = [("k", C.c_int), ("pool", C.c_int), ("thr", C.c_float), ("clutter_k", C.c_float),
                ("tile", C.c_int), ("max_density", C.c_float), ("dark", C.c_int), ("bright", C.c_int),
                ("max_cands", C.c_int), ("threads", C.c_int), ("isa", C.c_int), ("job_rows", C.c_int),
                ("job_cols", C.c_int), ("ncpus", C.c_int), ("cpus", C.POINTER(C.c_int)),
                ("scales", C.c_int), ("focus_k", C.c_float)]


class _Cand(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("contrast", C.c_float), ("snr", C.c_float),
                ("polarity", C.c_int), ("level", C.c_int)]


class _Track(C.Structure):
    _fields_ = [("id", C.c_int), ("x", C.c_float), ("y", C.c_float), ("vx", C.c_float), ("vy", C.c_float),
                ("polarity", C.c_int), ("contrast", C.c_float), ("clutter", C.c_int), ("cls", C.c_int),
                ("bird_score", C.c_float), ("speed", C.c_float), ("resid", C.c_float),
                ("contrast_cv", C.c_float), ("hits", C.c_int), ("age", C.c_int), ("miss", C.c_int),
                ("sigma", C.c_float), ("score", C.c_float), ("locked", C.c_int)]


class _Timing(C.Structure):
    _fields_ = [("total_ms", C.c_double), ("response_ms", C.c_double), ("select_ms", C.c_double),
                ("jobs", C.c_int), ("jobs_skipped", C.c_int)]


def _load():
    here = os.path.dirname(os.path.abspath(__file__))
    names = ["dronolok.dll", "libdronolok.so", "libdronolok.dylib"]
    dirs = [os.environ.get("DRONOLOK_LIB", ""), here, os.path.join(here, "..", "build"),
            os.path.join(here, "..", "build", "Release")]
    for d in dirs:
        for n in names:
            p = os.path.join(d, n) if os.path.isdir(d) else d
            if p and os.path.isfile(p) and os.path.basename(p) in names:
                lib = C.CDLL(os.path.abspath(p))
                break
        else:
            continue
        break
    else:
        raise OSError("не найдена библиотека dronolok (соберите native/ или задайте DRONOLOK_LIB)")
    lib.dl_detector_create.restype = C.c_void_p
    lib.dl_detector_create.argtypes = [C.POINTER(_Config)]
    lib.dl_detector_destroy.argtypes = [C.c_void_p]
    lib.dl_detect.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.POINTER(_Cand), C.c_int]
    lib.dl_detector_timing.argtypes = [C.c_void_p, C.POINTER(_Timing)]
    lib.dl_detector_backend.restype = C.c_char_p
    lib.dl_detector_backend.argtypes = [C.c_void_p]
    lib.dl_detector_threads.argtypes = [C.c_void_p]
    lib.dl_mask_begin.argtypes = [C.c_void_p]
    lib.dl_mask_end.argtypes = [C.c_void_p, C.c_float]
    lib.dl_mask_get.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int)]
    lib.dl_tracker_create.restype = C.c_void_p
    lib.dl_tracker_create.argtypes = [C.c_int, C.c_int]
    lib.dl_tracker_destroy.argtypes = [C.c_void_p]
    lib.dl_tracker_update.argtypes = [C.c_void_p, C.POINTER(_Cand), C.c_int, C.POINTER(_Track), C.c_int]
    lib.dl_tracker_clutter.argtypes = [C.c_void_p, C.POINTER(_Cand), C.c_int]
    lib.dl_tracker_reset.argtypes = [C.c_void_p]
    lib.dl_config_default.argtypes = [C.POINTER(_Config)]
    lib.dl_motion_create.restype = C.c_void_p
    lib.dl_motion_destroy.argtypes = [C.c_void_p]
    lib.dl_motion_estimate.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_float),
                                       C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.dl_tracker_set_motion.argtypes = [C.c_void_p, C.c_float, C.c_float]
    lib.dl_tracker_focus.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int]
    lib.dl_detector_set_focus.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int]
    return lib


_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = _load()
    return _lib


class NativePointDetector:
    def __init__(self, k=7, pool=2, thr=10, clutter_k=3.0, tile=32, max_density=0.012, dark=True, bright=True,
                 max_cands=300, threads=0, isa="auto", cpus=None, job=(4, 0), scales=1, focus_k=0.6, **_ignored):
        L = lib()
        c = _Config()
        L.dl_config_default(C.byref(c))
        c.k, c.pool, c.thr, c.clutter_k, c.tile = k, pool, thr, clutter_k, tile
        c.max_density, c.dark, c.bright, c.max_cands = max_density, int(dark), int(bright), max_cands
        c.threads, c.isa, c.job_rows, c.job_cols = threads, ISA[isa], job[0], job[1]
        c.scales, c.focus_k = scales, focus_k
        self._cpus = (C.c_int * len(cpus))(*cpus) if cpus else None
        if cpus:
            c.ncpus, c.cpus = len(cpus), self._cpus
        self._h = L.dl_detector_create(C.byref(c))
        self._buf = (_Cand * 4096)()
        self.tile, self.pool, self.scales = tile, pool, scales
        self.timing = {}
        self.tile_mask = None

    def __del__(self):
        if getattr(self, "_h", None):
            lib().dl_detector_destroy(self._h)
            self._h = None

    @property
    def backend(self):
        return lib().dl_detector_backend(self._h).decode()

    @property
    def threads(self):
        return lib().dl_detector_threads(self._h)

    def detect(self, gray):
        g = np.ascontiguousarray(gray, dtype=np.uint8)
        h, w = g.shape
        n = lib().dl_detect(self._h, g.ctypes.data, w, h, g.strides[0], self._buf, len(self._buf))
        t = _Timing()
        lib().dl_detector_timing(self._h, C.byref(t))
        self.timing = dict(detect_ms=t.total_ms, response_ms=t.response_ms, select_ms=t.select_ms)
        b = self._buf
        return [(b[i].x, b[i].y, b[i].contrast, b[i].snr, b[i].polarity, b[i].level)[:6 if self.scales > 1 else 5]
                for i in range(min(n, len(b)))]

    def set_focus(self, areas):
        """areas: [(x, y, r)] — где ждём цели (NativeTracker.focus()), там порог ниже"""
        a = (C.c_float * (3 * max(1, len(areas))))(*[v for xyr in areas for v in xyr])
        lib().dl_detector_set_focus(self._h, a, len(areas))

    def learn_mask(self, frames, frac=0.3):
        L = lib()
        L.dl_mask_begin(self._h)
        for f in frames:
            self.detect(f)
        L.dl_mask_end(self._h, frac)
        tw, th = C.c_int(), C.c_int()
        L.dl_mask_get(self._h, None, 0, C.byref(tw), C.byref(th))
        m = np.zeros((th.value, tw.value), np.uint8)
        L.dl_mask_get(self._h, m.ctypes.data, m.size, None, None)
        self.tile_mask = m.astype(bool)
        return self.tile_mask


class NativeTracker:
    def __init__(self, shape):
        h, w = shape
        self._h = lib().dl_tracker_create(w, h)
        self._dets = (_Cand * 4096)()
        self._out = (_Track * 1024)()
        self._cl = (_Cand * 4096)()
        self.clutter_dets = []

    def __del__(self):
        if getattr(self, "_h", None):
            lib().dl_tracker_destroy(self._h)
            self._h = None

    def reset_tracks(self):
        lib().dl_tracker_reset(self._h)

    def set_motion(self, dx, dy):
        """сдвиг картинки к следующему кадру (камера движется)"""
        lib().dl_tracker_set_motion(self._h, dx, dy)

    def focus(self):
        buf = (C.c_float * (3 * 256))()
        n = lib().dl_tracker_focus(self._h, buf, 256)
        return [(buf[3 * i], buf[3 * i + 1], buf[3 * i + 2]) for i in range(min(n, 256))]

    def update(self, dets):
        n = min(len(dets), len(self._dets))
        for i in range(n):
            x, y, c, snr, p = dets[i][:5]
            lvl = dets[i][5] if len(dets[i]) > 5 else 0
            self._dets[i] = _Cand(x, y, c, snr, p, lvl)
        k = lib().dl_tracker_update(self._h, self._dets, n, self._out, len(self._out))
        m = lib().dl_tracker_clutter(self._h, self._cl, len(self._cl))
        self.clutter_dets = [(self._cl[i].x, self._cl[i].y, self._cl[i].contrast, self._cl[i].snr,
                              self._cl[i].polarity) for i in range(min(m, len(self._cl)))]
        out = []
        for i in range(min(k, len(self._out))):
            t = self._out[i]
            out.append(dict(id=t.id, x=t.x, y=t.y, vx=t.vx, vy=t.vy, pol=t.polarity, c=t.contrast,
                            clutter=bool(t.clutter), cls=CLASSES[t.cls], bird_score=t.bird_score,
                            speed=t.speed, resid=t.resid, contrast_cv=t.contrast_cv, hits=t.hits,
                            miss=t.miss, sigma=t.sigma, score=t.score, locked=bool(t.locked), path=[(t.x, t.y)]))
        return out


class NativeMotion:
    """Сдвиг картинки между кадрами (камера движется): estimate(gray) -> (dx, dy, quality, valid)"""

    def __init__(self):
        self._h = lib().dl_motion_create()

    def __del__(self):
        if getattr(self, "_h", None):
            lib().dl_motion_destroy(self._h)
            self._h = None

    def estimate(self, gray):
        g = np.ascontiguousarray(gray, dtype=np.uint8)
        dx, dy, q = C.c_float(), C.c_float(), C.c_float()
        ok = lib().dl_motion_estimate(self._h, g.ctypes.data, g.shape[1], g.shape[0], g.strides[0],
                                      C.byref(dx), C.byref(dy), C.byref(q))
        return dx.value, dy.value, q.value, bool(ok)
