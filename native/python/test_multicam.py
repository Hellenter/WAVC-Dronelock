"""Проверка обёртки нескольких широких камер (Fusion, сообщения DLW1) на библиотеке ядра.

    DRONOLOK_LIB=native/build python native/python/test_multicam.py

Две широкие по 120° с осями 0° и 110° (перекрытие 10°); цель летит по азимуту от 30° до 90°: сначала её
видит первая, потом обе, потом вторая. Номер цели установки должен остаться одним, переход — засчитаться.
Цели второй камеры идут через кодирование в датаграмму и обратно (как по сети)."""
import ctypes as C
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dronolok_native as dn  # noqa: E402

cam = dn.Camera(4608, 2592, 120)
poses = [(0.0, 30.0, 0.0), (110.0, 30.0, 0.0)]
f = dn.Fusion()
ks = [f.add_camera(cam, p) for p in poses]
assert ks == [0, 1], ks

fps, speed = 14.0, 4.0   # к/с, град/с
ids, seq = set(), 0
for i in range(int(15 * fps)):
    t = i / fps
    az, el = 30.0 + speed * t, 25.0
    for k, p in enumerate(poses):
        u, v, ok = cam.project(p, az, el)
        if not ok:
            continue
        tg = dn._Target(id=7 + 100 * k, t=t, az=az, el=el, vaz=speed, vel=0.0, sigma=0.02, sigma_v=0.05, cls=1,
                        confidence=0.6, hard=0, size_deg=0.0)
        if k == 1:   # вторая камера — «по сети»
            data = dn.encode_targets(k, seq, t, t, [tg])
            seq += 1
            cam_k, seq_k, t_k, _, got = dn.decode_targets(data)
            assert cam_k == 1 and abs(t_k - t) < 1e-6 and len(got) == 1, (cam_k, t_k, got)
            assert abs(got[0].az - az) < 1e-4 and got[0].id == tg.id
            f.update(k, t_k, got)
        else:
            f.update(k, t, [tg])
    for g in f.targets():
        ids.add(g["id"])
tg = f.targets()
assert len(ids) == 1, ("у цели сменился номер", ids)
assert tg and tg[0]["cam"] == 1 and tg[0]["handoffs"] >= 1, tg
gid = tg[0]["id"]
assert f.global_of(1, 107) == gid and f.local_of(gid, 1) == 107
u, v, ok = f.project(1, 90.0, 25.0)
print("одна цель, номер %d; переходов %d; на кадре второй камеры (%.0f, %.0f)" % (gid, tg[0]["handoffs"], u, v))
s, n = f.calib(1)
print("калибровка второй камеры: неуверенность %.2f°, пар %d" % (s, n))
print("все проверки прошли")
