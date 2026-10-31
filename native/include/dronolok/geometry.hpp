// Геометрия камер и передача цели между камерами: пиксель <-> направление (азимут, угол места),
// сообщение о цели с временем кадра и угловой скоростью, прогноз и окно поиска для другой камеры,
// сдвиг картинки при повороте поворотки (вместо оценки движения по кадру).
//
// Мир: x — восток, y — север, z — вверх. Азимут от севера по часовой, угол места от горизонта, градусы.
// Камера: x — вправо, y — вниз, z — вдоль оптической оси. Положение камеры (Pose) — куда смотрит ось
// и поворот кадра вокруг оси (roll, по часовой, если смотреть из камеры).
#pragma once
#include <algorithm>
#include <cmath>

namespace dronolok {

enum class Lens : int { Pinhole = 0, Fisheye = 1 };   // fisheye — равноугловая модель r = f·θ

struct CameraModel {
    int w = 0, h = 0;
    float hfov_deg = 120.f;   // поле зрения по ширине кадра
    Lens lens = Lens::Pinhole;
    float cx = -1, cy = -1;   // центр кадра (оптическая ось); < 0 — середина
    float k1 = 0;             // радиальная дисторсия: r' = r·(1 + k1·(r/f)²), 0 — без неё
};

struct Pose {
    float az = 0, el = 0, roll = 0;   // градусы
};

struct Vec3 {
    double x = 0, y = 0, z = 0;
};

inline double deg2rad(double d) { return d * 0.017453292519943295; }
inline double rad2deg(double r) { return r * 57.29577951308232; }
// разность азимутов в -180..180
inline double az_diff(double a, double b) {
    double d = std::fmod(a - b, 360.0);
    if (d > 180) d -= 360;
    if (d < -180) d += 360;
    return d;
}
inline double az_norm(double a) {
    a = std::fmod(a, 360.0);
    return a < 0 ? a + 360 : a;
}

inline Vec3 dir_from_azel(double az_deg, double el_deg) {
    const double a = deg2rad(az_deg), e = deg2rad(el_deg);
    return {std::sin(a) * std::cos(e), std::cos(a) * std::cos(e), std::sin(e)};
}
inline void azel_from_dir(const Vec3& d, double& az_deg, double& el_deg) {
    const double n = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    el_deg = rad2deg(std::asin(std::max(-1.0, std::min(1.0, d.z / (n > 0 ? n : 1)))));
    az_deg = az_norm(rad2deg(std::atan2(d.x, d.y)));
}

// оси камеры в мире: right, down, fwd
struct CamAxes {
    Vec3 r, d, f;
};
inline CamAxes cam_axes(const Pose& p) {
    const Vec3 f = dir_from_azel(p.az, p.el);
    const double a = deg2rad(p.az);
    Vec3 r{std::cos(a), -std::sin(a), 0};   // вправо по горизонту
    // вниз = f x r
    Vec3 d{f.y * r.z - f.z * r.y, f.z * r.x - f.x * r.z, f.x * r.y - f.y * r.x};
    const double c = std::cos(deg2rad(p.roll)), s = std::sin(deg2rad(p.roll));
    const Vec3 r2{c * r.x + s * d.x, c * r.y + s * d.y, c * r.z + s * d.z};
    const Vec3 d2{-s * r.x + c * d.x, -s * r.y + c * d.y, -s * r.z + c * d.z};
    return {r2, d2, f};
}

inline double cam_focal(const CameraModel& m) {
    const double half = deg2rad(m.hfov_deg) * 0.5;
    return m.lens == Lens::Fisheye ? (m.w * 0.5) / half : (m.w * 0.5) / std::tan(half);
}
inline double cam_cx(const CameraModel& m) { return m.cx >= 0 ? m.cx : m.w * 0.5; }
inline double cam_cy(const CameraModel& m) { return m.cy >= 0 ? m.cy : m.h * 0.5; }

// направление в мире -> пиксель. false — позади камеры (или за 90° у обычного объектива)
inline bool project(const CameraModel& m, const Pose& p, const Vec3& dir, double& u, double& v) {
    const CamAxes A = cam_axes(p);
    const double X = dir.x * A.r.x + dir.y * A.r.y + dir.z * A.r.z;
    const double Y = dir.x * A.d.x + dir.y * A.d.y + dir.z * A.d.z;
    const double Z = dir.x * A.f.x + dir.y * A.f.y + dir.z * A.f.z;
    const double rho = std::sqrt(X * X + Y * Y);
    const double th = std::atan2(rho, Z);
    if (m.lens == Lens::Pinhole && Z <= 1e-9) return false;
    if (m.lens == Lens::Fisheye && th > deg2rad(100.0)) return false;
    const double f = cam_focal(m);
    double r = m.lens == Lens::Fisheye ? f * th : f * std::tan(th);
    if (m.k1 != 0) {
        const double q = r / f;
        r *= 1 + m.k1 * q * q;
    }
    const double cs = rho > 1e-12 ? X / rho : 1, sn = rho > 1e-12 ? Y / rho : 0;
    u = cam_cx(m) + r * cs;
    v = cam_cy(m) + r * sn;
    return true;
}
inline bool in_frame(const CameraModel& m, double u, double v, double margin = 0) {
    return u >= -margin && v >= -margin && u < m.w + margin && v < m.h + margin;
}

// пиксель -> направление в мире
inline Vec3 unproject(const CameraModel& m, const Pose& p, double u, double v) {
    const double f = cam_focal(m);
    const double du = u - cam_cx(m), dv = v - cam_cy(m);
    double r = std::sqrt(du * du + dv * dv);
    if (m.k1 != 0 && r > 0) {   // обратная дисторсия — несколько итераций Ньютона
        double q = r / f;
        const double target = q;
        for (int i = 0; i < 6; ++i) {
            const double g = q * (1 + m.k1 * q * q) - target, dg = 1 + 3 * m.k1 * q * q;
            q -= g / (dg != 0 ? dg : 1);
        }
        r = q * f;
    }
    const double th = m.lens == Lens::Fisheye ? r / f : std::atan(r / f);
    const double rr = std::sqrt(du * du + dv * dv);
    const double cs = rr > 1e-12 ? du / rr : 1, sn = rr > 1e-12 ? dv / rr : 0;
    const double X = std::sin(th) * cs, Y = std::sin(th) * sn, Z = std::cos(th);
    const CamAxes A = cam_axes(p);
    return {X * A.r.x + Y * A.d.x + Z * A.f.x, X * A.r.y + Y * A.d.y + Z * A.f.y, X * A.r.z + Y * A.d.z + Z * A.f.z};
}

inline double angle_between(const Vec3& a, const Vec3& b) {
    const double c = a.x * b.x + a.y * b.y + a.z * b.z;
    const double na = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z), nb = std::sqrt(b.x * b.x + b.y * b.y + b.z * b.z);
    return rad2deg(std::acos(std::max(-1.0, std::min(1.0, c / (na * nb)))));
}
// угловой размер пикселя в точке кадра, градусы
inline double pixel_deg(const CameraModel& m, const Pose& p, double u, double v) {
    return angle_between(unproject(m, p, u, v), unproject(m, p, u + 1, v));
}

// Сообщение о цели: всё, что нужно другой камере или поворотке, чтобы найти цель в своём кадре.
struct TargetReport {
    int id = 0;
    double t = 0;              // время кадра широкой камеры, с (часы общие для всех устройств)
    float az = 0, el = 0;      // направление, градусы
    float vaz = 0, vel = 0;    // угловая скорость, град/с (vaz — по азимуту, не умножена на cos el)
    float sigma = 0;           // СКО направления, градусы
    float sigma_v = 0;         // СКО скорости, град/с
    int cls = 0;               // 0 неизвестно, 1 дрон, 2 птица, 3 помеха
    float confidence = 0;      // 0..1
    int hard = 0;              // 1 — жёсткий захват (подтверждён дрон)
    float size_deg = 0;        // видимый размер, градусы; 0 — не измерен
};

// Из трека широкой камеры: x, y, vx, vy (пикс./кадр), sigma (пикс.) и частота кадров.
inline TargetReport make_report(const CameraModel& m, const Pose& p, double t, float fps, float x, float y, float vx,
                                float vy, float sigma_px, float size_px = 0) {
    TargetReport r;
    r.t = t;
    double az, el, az1, el1;
    azel_from_dir(unproject(m, p, x, y), az, el);
    azel_from_dir(unproject(m, p, x + vx, y + vy), az1, el1);
    r.az = (float)az;
    r.el = (float)el;
    r.vaz = (float)(az_diff(az1, az) * fps);
    r.vel = (float)((el1 - el) * fps);
    const double pd = pixel_deg(m, p, x, y);
    // ошибка положения детектора около пикселя + ошибка трека; скорость — по ошибке позиции за ~0.5 с
    r.sigma = (float)(pd * std::sqrt(1.0 + (double)sigma_px * sigma_px));
    r.sigma_v = (float)(r.sigma / 0.5 + 0.1 * std::hypot(r.vaz * std::cos(deg2rad(el)), r.vel));
    r.size_deg = (float)(size_px * pd);
    return r;
}

// Прогноз направления на момент t и его неопределённость. Поправка за cos(el) у азимутальной скорости
// не нужна: vaz уже в градусах азимута.
inline void predict(const TargetReport& r, double t, double& az, double& el, double& sigma) {
    const double dt = t - r.t;
    az = az_norm(r.az + r.vaz * dt);
    el = r.el + r.vel * dt;
    if (el > 90) el = 90;
    sigma = std::sqrt((double)r.sigma * r.sigma + (double)r.sigma_v * r.sigma_v * dt * dt);
}

// Окно поиска в кадре другой камеры: центр, радиус (пикс.), видна ли цель в кадре.
struct SearchWindow {
    float u = 0, v = 0, radius = 0;
    int visible = 0;
};
inline SearchWindow search_window(const TargetReport& r, double t, const CameraModel& m, const Pose& p,
                                  float k_sigma = 3.f, float min_radius_px = 8.f) {
    SearchWindow w;
    double az, el, s, u, v;
    predict(r, t, az, el, s);
    if (!project(m, p, dir_from_azel(az, el), u, v)) return w;
    const double pd = pixel_deg(m, p, u, v);
    w.u = (float)u;
    w.v = (float)v;
    w.radius = (float)std::max<double>(min_radius_px, k_sigma * s / (pd > 0 ? pd : 1));
    w.visible = in_frame(m, u, v) ? 1 : 0;
    return w;
}

// Куда повернуть ось, чтобы цель была в центре кадра через lead секунд (упреждение на задержку поворотки).
inline Pose aim(const TargetReport& r, double t, double lead) {
    double az, el, s;
    predict(r, t + lead, az, el, s);
    Pose p;
    p.az = (float)az;
    p.el = (float)el;
    return p;
}

// Сдвиг картинки между двумя положениями камеры (поворотка повернулась): куда ушла неподвижная точка
// мира из точки кадра (u, v). Это то, что трекеру даёт dl_tracker_set_motion, — по энкодерам, без
// оценки по картинке.
inline void pose_shift(const CameraModel& m, const Pose& from, const Pose& to, double u, double v, double& dx,
                       double& dy) {
    double u2, v2;
    if (!project(m, to, unproject(m, from, u, v), u2, v2)) {
        dx = dy = 0;
        return;
    }
    dx = u2 - u;
    dy = v2 - v;
}

}  // namespace dronolok
