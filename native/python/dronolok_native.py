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
                ("polarity", C.c_int), ("level", C.c_int), ("energy", C.c_float),
                ("margin", C.c_float)]


class _Track(C.Structure):
    _fields_ = [("id", C.c_int), ("x", C.c_float), ("y", C.c_float), ("vx", C.c_float), ("vy", C.c_float),
                ("polarity", C.c_int), ("contrast", C.c_float), ("clutter", C.c_int), ("cls", C.c_int),
                ("bird_score", C.c_float), ("speed", C.c_float), ("resid", C.c_float),
                ("contrast_cv", C.c_float), ("hits", C.c_int), ("age", C.c_int), ("miss", C.c_int),
                ("sigma", C.c_float), ("score", C.c_float), ("locked", C.c_int),
                ("flap_r2", C.c_float), ("flap_period", C.c_float)]


class _Camera(C.Structure):
    _fields_ = [("w", C.c_int), ("h", C.c_int), ("hfov_deg", C.c_float), ("lens", C.c_int), ("cx", C.c_float),
                ("cy", C.c_float), ("k1", C.c_float)]


class _Pose(C.Structure):
    _fields_ = [("az", C.c_float), ("el", C.c_float), ("roll", C.c_float)]


class _Target(C.Structure):
    _fields_ = [("id", C.c_int), ("t", C.c_double), ("az", C.c_float), ("el", C.c_float), ("vaz", C.c_float),
                ("vel", C.c_float), ("sigma", C.c_float), ("sigma_v", C.c_float), ("cls", C.c_int),
                ("confidence", C.c_float), ("hard", C.c_int), ("size_deg", C.c_float)]


class _Window(C.Structure):
    _fields_ = [("u", C.c_float), ("v", C.c_float), ("radius", C.c_float), ("visible", C.c_int)]


class _FieldLogConfig(C.Structure):
    _fields_ = [("root", C.c_char_p), ("session", C.c_char_p), ("max_mb", C.c_double), ("min_free_mb", C.c_double),
                ("pre_s", C.c_float), ("post_s", C.c_float), ("max_clip_s", C.c_float), ("crop_px", C.c_int),
                ("ring_tracks", C.c_int), ("jpeg_q", C.c_int), ("snap_px", C.c_int), ("overview_every_s", C.c_float),
                ("clips_per_min", C.c_int), ("all_tracks", C.c_int)]


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
    lib.dl_tracker_set_fps.argtypes = [C.c_void_p, C.c_float]
    lib.dl_tracker_update.argtypes = [C.c_void_p, C.POINTER(_Cand), C.c_int, C.POINTER(_Track), C.c_int]
    lib.dl_tracker_clutter.argtypes = [C.c_void_p, C.POINTER(_Cand), C.c_int]
    lib.dl_tracker_reset.argtypes = [C.c_void_p]
    lib.dl_tracker_queue.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_int]
    lib.dl_tracker_locked.argtypes = [C.c_void_p]
    lib.dl_tracker_select.argtypes = [C.c_void_p, C.c_int]
    lib.dl_tracker_next.argtypes = [C.c_void_p]
    lib.dl_tracker_set_static.argtypes = [C.c_void_p, C.c_int, C.c_float]
    lib.dl_tracker_static_save.argtypes = [C.c_void_p, C.c_char_p]
    lib.dl_tracker_static_load.argtypes = [C.c_void_p, C.c_char_p]
    lib.dl_tracker_static_clear.argtypes = [C.c_void_p]
    lib.dl_config_default.argtypes = [C.POINTER(_Config)]
    lib.dl_motion_create.restype = C.c_void_p
    lib.dl_motion_destroy.argtypes = [C.c_void_p]
    lib.dl_motion_estimate.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_float),
                                       C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.dl_motion_set_fps.argtypes = [C.c_void_p, C.c_float]
    lib.dl_tracker_set_motion.argtypes = [C.c_void_p, C.c_float, C.c_float]
    lib.dl_tracker_focus.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int]
    lib.dl_detector_set_focus.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int]
    lib.dl_tracker_hard.argtypes = [C.c_void_p]
    P, PC, PP, PF = C.POINTER, C.POINTER(_Camera), C.POINTER(_Pose), C.POINTER(C.c_float)
    lib.dl_project.argtypes = [PC, PP, C.c_float, C.c_float, PF, PF]
    lib.dl_unproject.argtypes = [PC, PP, C.c_float, C.c_float, PF, PF]
    lib.dl_pixel_deg.restype = C.c_float
    lib.dl_pixel_deg.argtypes = [PC, PP, C.c_float, C.c_float]
    lib.dl_target_from_track.argtypes = [PC, PP, C.c_double, C.c_float, P(_Track), C.c_int, P(_Target)]
    lib.dl_target_predict.argtypes = [P(_Target), C.c_double, PF, PF, PF]
    lib.dl_target_window.restype = _Window
    lib.dl_target_window.argtypes = [P(_Target), C.c_double, PC, PP, C.c_float]
    lib.dl_target_aim.restype = _Pose
    lib.dl_target_aim.argtypes = [P(_Target), C.c_double, C.c_double]
    lib.dl_pose_shift.argtypes = [PC, PP, PP, C.c_float, C.c_float, PF, PF]
    lib.dl_fieldlog_config_default.argtypes = [P(_FieldLogConfig)]
    lib.dl_fieldlog_open.restype = C.c_void_p
    lib.dl_fieldlog_open.argtypes = [P(_FieldLogConfig), C.c_int, C.c_int, C.c_double, C.c_char_p, C.c_char_p]
    lib.dl_fieldlog_error.restype = C.c_char_p
    lib.dl_fieldlog_frame.argtypes = [C.c_void_p, C.c_void_p, C.c_int64, C.c_double, C.c_void_p, C.c_int, C.c_int,
                                      C.c_int, C.c_float, C.c_float]
    lib.dl_fieldlog_note.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_char_p, C.c_int]
    lib.dl_fieldlog_dir.restype = C.c_char_p
    lib.dl_fieldlog_dir.argtypes = [C.c_void_p]
    lib.dl_fieldlog_close.argtypes = [C.c_void_p]
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
        # суммарный контраст кандидатов, в том же порядке: передать в NativeTracker.update(energy=...)
        self.energy = [b[i].energy for i in range(min(n, len(b)))]
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
    def __init__(self, shape, fps=None):
        h, w = shape
        self._h = lib().dl_tracker_create(w, h)
        if not self._h:
            raise ValueError(f"трекер не создан для кадра {w}x{h}")
        if fps:
            self.set_fps(fps)
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

    def set_fps(self, fps):
        """Частота кадров камеры (умолчания трекера — для 25 к/с), см. dl_tracker_set_fps."""
        lib().dl_tracker_set_fps(self._h, float(fps))

    def set_static(self, frames=3000, hits=8.0):
        """Долгая память статичных помех (неподвижная камера); frames=0 — выключить."""
        lib().dl_tracker_set_static(self._h, frames, hits)

    def save_static(self, path):
        return bool(lib().dl_tracker_static_save(self._h, os.fsencode(path)))

    def load_static(self, path):
        return bool(lib().dl_tracker_static_load(self._h, os.fsencode(path)))

    def clear_static(self):
        lib().dl_tracker_static_clear(self._h)

    def queue(self):
        """очередь целей: id по весомости для захвата, самая весомая первой"""
        buf = (C.c_int * 64)()
        n = lib().dl_tracker_queue(self._h, buf, 64)
        return [buf[i] for i in range(min(n, 64))]

    def locked(self):
        return lib().dl_tracker_locked(self._h)

    def hard(self):
        """состояние захвата: 0 нет, 1 захват, 2 жёсткий (подтверждён дрон)"""
        return lib().dl_tracker_hard(self._h)

    def select(self, track_id):
        """ручной выбор цели: захват держится на ней, пока она жива"""
        return bool(lib().dl_tracker_select(self._h, track_id))

    def next_target(self):
        """захват на следующую цель очереди; её id или 0"""
        return lib().dl_tracker_next(self._h)

    def set_motion(self, dx, dy):
        """сдвиг картинки к следующему кадру (камера движется)"""
        lib().dl_tracker_set_motion(self._h, dx, dy)

    def focus(self):
        buf = (C.c_float * (3 * 256))()
        n = lib().dl_tracker_focus(self._h, buf, 256)
        return [(buf[3 * i], buf[3 * i + 1], buf[3 * i + 2]) for i in range(min(n, 256))]

    def update(self, dets, energy=None):
        """dets — кандидаты NativePointDetector.detect; energy — его .energy (взмахи крыльев)"""
        n = min(len(dets), len(self._dets))
        for i in range(n):
            x, y, c, snr, p = dets[i][:5]
            lvl = dets[i][5] if len(dets[i]) > 5 else 0
            self._dets[i] = _Cand(x, y, c, snr, p, lvl, energy[i] if energy else 0.0)
        k = lib().dl_tracker_update(self._h, self._dets, n, self._out, len(self._out))
        m = lib().dl_tracker_clutter(self._h, self._cl, len(self._cl))
        self.clutter_dets = [(self._cl[i].x, self._cl[i].y, self._cl[i].contrast, self._cl[i].snr,
                              self._cl[i].polarity) for i in range(min(m, len(self._cl)))]
        out = []
        for i in range(min(k, len(self._out))):
            t = self._out[i]
            out.append(dict(id=t.id, x=t.x, y=t.y, vx=t.vx, vy=t.vy, pol=t.polarity, c=t.contrast,
                            clutter=bool(t.clutter), cls=CLASSES[t.cls], bird_score=t.bird_score,
                            speed=t.speed, resid=t.resid, contrast_cv=t.contrast_cv, flap_r2=t.flap_r2,
                            flap_period=t.flap_period, hits=t.hits,
                            miss=t.miss, sigma=t.sigma, score=t.score, locked=bool(t.locked), path=[(t.x, t.y)]))
        return out


class FieldLog:
    """Журнал полевых испытаний (native/README.md, «Журнал полевых испытаний»).

    log = FieldLog("/home/pi/dronolok-logs", (h, w), fps=14, source="cam0", max_mb=4096)
    ... каждый кадр после tracker.update(...):
    log.frame(tracker, index, t, gray, cam=(dx, dy))
    log.note("slew", track_id, "поворотка навелась", clip=True)
    log.close()

    Параметры как у dl_fieldlog_config: pre_s, post_s, crop_px, ring_tracks, jpeg_q, snap_px, overview_every_s,
    clips_per_min, all_tracks, min_free_mb, session. settings — словарь настроек, попадёт в session.json.
    """

    def __init__(self, root, shape, fps, source="", settings=None, **kw):
        import json
        h, w = shape
        cfg = _FieldLogConfig()
        lib().dl_fieldlog_config_default(C.byref(cfg))
        self._root = os.fsencode(root)
        cfg.root = self._root
        self._session = os.fsencode(kw.pop("session", "") or "")
        cfg.session = self._session
        for k, v in kw.items():
            if not hasattr(cfg, k):
                raise TypeError(f"неизвестный параметр журнала: {k}")
            setattr(cfg, k, int(v) if isinstance(v, bool) else v)
        sj = json.dumps(settings, ensure_ascii=False).encode() if settings else None
        self._h = lib().dl_fieldlog_open(C.byref(cfg), w, h, float(fps), source.encode(), sj)
        if not self._h:
            raise OSError("журнал не открыт: " + lib().dl_fieldlog_error().decode(errors="replace"))

    @property
    def dir(self):
        return lib().dl_fieldlog_dir(self._h).decode() if self._h else ""

    def frame(self, tracker, index, t, gray=None, cam=(0.0, 0.0)):
        """кадр после NativeTracker.update: треки, захват и очередь берутся из трекера"""
        if not self._h:
            return
        if gray is not None:
            import numpy as np
            g = np.ascontiguousarray(gray, dtype=np.uint8)
            lib().dl_fieldlog_frame(self._h, tracker._h, int(index), float(t), g.ctypes.data, g.shape[1], g.shape[0],
                                    g.strides[0], float(cam[0]), float(cam[1]))
        else:
            lib().dl_fieldlog_frame(self._h, tracker._h, int(index), float(t), None, 0, 0, 0, float(cam[0]),
                                    float(cam[1]))

    def note(self, kind, track_id=0, text="", clip=False):
        if self._h:
            lib().dl_fieldlog_note(self._h, kind.encode(), int(track_id), text.encode(), 1 if clip else 0)

    def close(self):
        if getattr(self, "_h", None):
            lib().dl_fieldlog_close(self._h)
            self._h = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class NativeMotion:
    """Сдвиг картинки между кадрами (камера движется): estimate(gray) -> (dx, dy, quality, valid)"""

    def __init__(self, fps=None):
        self._h = lib().dl_motion_create()
        if fps:
            self.set_fps(fps)

    def __del__(self):
        if getattr(self, "_h", None):
            lib().dl_motion_destroy(self._h)
            self._h = None

    def set_fps(self, fps):
        """Частота кадров камеры, см. dl_motion_set_fps."""
        lib().dl_motion_set_fps(self._h, float(fps))

    def estimate(self, gray):
        g = np.ascontiguousarray(gray, dtype=np.uint8)
        dx, dy, q = C.c_float(), C.c_float(), C.c_float()
        ok = lib().dl_motion_estimate(self._h, g.ctypes.data, g.shape[1], g.shape[0], g.strides[0],
                                      C.byref(dx), C.byref(dy), C.byref(q))
        return dx.value, dy.value, q.value, bool(ok)


# ---------------------------------------------------------------- геометрия и передача цели
LENS = {"pinhole": 0, "fisheye": 1}


class Camera:
    """Модель камеры (dl_camera): кадр w x h, поле по ширине hfov, объектив pinhole | fisheye."""

    def __init__(self, w, h, hfov, lens="pinhole", cx=-1.0, cy=-1.0, k1=0.0):
        self.w, self.h, self.hfov, self.lens = w, h, hfov, lens
        self.c = _Camera(w, h, hfov, LENS[lens], cx, cy, k1)

    def project(self, pose, az, el):
        u, v = C.c_float(), C.c_float()
        ok = lib().dl_project(C.byref(self.c), C.byref(_Pose(*pose)), az, el, C.byref(u), C.byref(v))
        return u.value, v.value, bool(ok)

    def unproject(self, pose, u, v):
        a, e = C.c_float(), C.c_float()
        lib().dl_unproject(C.byref(self.c), C.byref(_Pose(*pose)), u, v, C.byref(a), C.byref(e))
        return a.value, e.value

    def pixel_deg(self, pose, u, v):
        return lib().dl_pixel_deg(C.byref(self.c), C.byref(_Pose(*pose)), u, v)

    def pose_shift(self, pose_from, pose_to, u=None, v=None):
        """сдвиг картинки при повороте from -> to (для NativeTracker.set_motion по энкодерам)"""
        dx, dy = C.c_float(), C.c_float()
        lib().dl_pose_shift(C.byref(self.c), C.byref(_Pose(*pose_from)), C.byref(_Pose(*pose_to)),
                            self.w / 2 if u is None else u, self.h / 2 if v is None else v, C.byref(dx), C.byref(dy))
        return dx.value, dy.value


class Target:
    """Сообщение о цели (dl_target): направление, угловая скорость, время кадра, неопределённость."""

    def __init__(self, raw):
        self.raw = raw

    def __getattr__(self, k):
        return getattr(self.raw, k)

    @staticmethod
    def from_track(cam, pose, t, fps, track, hard):
        tr = _Track()
        tr.id, tr.x, tr.y, tr.vx, tr.vy = track["id"], track["x"], track["y"], track["vx"], track["vy"]
        tr.sigma, tr.cls = track.get("sigma", 1.0), CLASSES.index(track.get("cls", "unknown"))
        out = _Target()
        lib().dl_target_from_track(C.byref(cam.c), C.byref(_Pose(*pose)), t, fps, C.byref(tr), hard, C.byref(out))
        return Target(out)

    def predict(self, t):
        a, e, s = C.c_float(), C.c_float(), C.c_float()
        lib().dl_target_predict(C.byref(self.raw), t, C.byref(a), C.byref(e), C.byref(s))
        return a.value, e.value, s.value

    def window(self, t, cam, pose, k_sigma=3.0):
        w = lib().dl_target_window(C.byref(self.raw), t, C.byref(cam.c), C.byref(_Pose(*pose)), k_sigma)
        return w.u, w.v, w.radius, bool(w.visible)

    def aim(self, t, lead):
        p = lib().dl_target_aim(C.byref(self.raw), t, lead)
        return (p.az, p.el, p.roll)
