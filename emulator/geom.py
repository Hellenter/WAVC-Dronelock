"""Геометрия камер на numpy: то же, что native/include/dronolok/geometry.hpp, но сразу для массива пикселей.
Мир: x — восток, y — север, z — вверх; азимут от севера по часовой, угол места от горизонта, градусы."""
import numpy as np


def dir_from_azel(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.stack([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)], -1)


def azel_from_dir(d):
    d = np.asarray(d, dtype=np.float64)
    n = np.linalg.norm(d, axis=-1)
    el = np.degrees(np.arcsin(np.clip(d[..., 2] / np.where(n > 0, n, 1), -1, 1)))
    az = np.degrees(np.arctan2(d[..., 0], d[..., 1])) % 360
    return az, el


def cam_axes(pose):
    az, el, roll = pose
    f = dir_from_azel(az, el)
    a = np.radians(az)
    r = np.array([np.cos(a), -np.sin(a), 0.0])
    d = np.cross(f, r)
    c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    return c * r + s * d, -s * r + c * d, f


class CameraModel:
    """w x h, поле по ширине hfov, lens: 'pinhole' или 'fisheye' (r = f·θ)"""

    def __init__(self, w, h, hfov, lens="pinhole", k1=0.0):
        self.w, self.h, self.hfov, self.lens, self.k1 = w, h, hfov, lens, k1
        half = np.radians(hfov) / 2
        self.f = (w / 2) / half if lens == "fisheye" else (w / 2) / np.tan(half)
        self.cx, self.cy = w / 2, h / 2

    def native(self):
        """та же камера для dronolok_native (C API)"""
        from dronolok_native import Camera
        return Camera(self.w, self.h, self.hfov, self.lens, k1=self.k1)

    def project(self, pose, dirs):
        """направления (..., 3) -> u, v, ok"""
        R, D, F = cam_axes(pose)
        X, Y, Z = dirs @ R, dirs @ D, dirs @ F
        rho = np.hypot(X, Y)
        th = np.arctan2(rho, Z)
        if self.lens == "fisheye":
            r, ok = self.f * th, th < np.radians(100)
        else:
            r, ok = self.f * np.tan(np.minimum(th, np.radians(89.9))), Z > 1e-9
        if self.k1:
            q = r / self.f
            r = r * (1 + self.k1 * q * q)
        safe = np.where(rho > 1e-12, rho, 1)
        u = self.cx + r * np.where(rho > 1e-12, X / safe, 1)
        v = self.cy + r * np.where(rho > 1e-12, Y / safe, 0)
        return u, v, ok

    def unproject(self, pose, u, v):
        du, dv = np.asarray(u, np.float64) - self.cx, np.asarray(v, np.float64) - self.cy
        rr = np.hypot(du, dv)
        r = rr
        if self.k1:
            q = rr / self.f
            t = q.copy()
            for _ in range(6):
                q = q - (q * (1 + self.k1 * q * q) - t) / (1 + 3 * self.k1 * q * q)
            r = q * self.f
        th = r / self.f if self.lens == "fisheye" else np.arctan(r / self.f)
        safe = np.where(rr > 1e-12, rr, 1)
        cs, sn = np.where(rr > 1e-12, du / safe, 1), np.where(rr > 1e-12, dv / safe, 0)
        X, Y, Z = np.sin(th) * cs, np.sin(th) * sn, np.cos(th)
        R, D, F = cam_axes(pose)
        return X[..., None] * R + Y[..., None] * D + Z[..., None] * F

    def rays(self, pose, step=1):
        """направления всех пикселей (h/step, w/step, 3)"""
        v, u = np.mgrid[0:self.h:step, 0:self.w:step].astype(np.float64)
        return self.unproject(pose, u + 0.5 * (step - 1), v + 0.5 * (step - 1))

    def pixel_deg(self, pose, u, v):
        a, b = self.unproject(pose, u, v), self.unproject(pose, u + 1, v)
        return np.degrees(np.arccos(np.clip(np.sum(a * b, -1), -1, 1)))
