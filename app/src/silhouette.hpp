// Силуэты целей эмулятора: квадрокоптер, летающее крыло (треугольник), самолёт, птица со взмахами.
// Цель — набор плоских треугольников в своей системе (x вправо, y вперёд, z вверх, метры), которые
// поворачиваются по курсу, проецируются камерой и закрашиваются с учётом доли пикселя (суперсэмплинг),
// затем размываются пятном объектива. Поэтому размер в пикселях честно падает с дальностью, а вид
// зависит от ракурса: крыло снизу — треугольник, сбоку — черта.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "dronolok/geometry.hpp"

namespace twocam {

enum class Shape : int { Bird = 0, Quad = 1, Delta = 2, Plane = 3 };

inline const char* shape_name(Shape s) {
    return s == Shape::Quad ? "квадрокоптер" : s == Shape::Delta ? "крыло" : s == Shape::Plane ? "самолёт" : "птица";
}

struct Tri {
    dronolok::Vec3 a, b, c;
    float w;   // непрозрачность: винты в движении — полупрозрачный диск
};

namespace sil {

using dronolok::Vec3;

constexpr double PI_ = 3.14159265358979323846;

inline void quad(std::vector<Tri>& out, Vec3 a, Vec3 b, Vec3 c, Vec3 d, float w = 1) {
    out.push_back({a, b, c, w});
    out.push_back({a, c, d, w});
}
// прямоугольник в горизонтальной плоскости: центр, полуширина по x, полудлина по y, поворот вокруг z
inline void hrect(std::vector<Tri>& out, double cx, double cy, double cz, double hx, double hy, double rot = 0, float w = 1) {
    const double c = std::cos(rot), s = std::sin(rot);
    auto P = [&](double x, double y) { return Vec3{cx + x * c - y * s, cy + x * s + y * c, cz}; };
    quad(out, P(-hx, -hy), P(hx, -hy), P(hx, hy), P(-hx, hy), w);
}
// прямоугольник в вертикальной плоскости вдоль y (фюзеляж, киль)
inline void vrect(std::vector<Tri>& out, double cx, double y0, double y1, double z0, double z1, float w = 1) {
    quad(out, {cx, y0, z0}, {cx, y1, z0}, {cx, y1, z1}, {cx, y0, z1}, w);
}
inline void disc(std::vector<Tri>& out, double cx, double cy, double cz, double r, float w) {
    const int n = 8;
    for (int k = 0; k < n; ++k) {
        const double a0 = 2 * sil::PI_ * k / n, a1 = 2 * sil::PI_ * (k + 1) / n;
        out.push_back({{cx, cy, cz}, {cx + r * std::cos(a0), cy + r * std::sin(a0), cz},
                       {cx + r * std::cos(a1), cy + r * std::sin(a1), cz}, w});
    }
}

}  // namespace sil

// Треугольники цели размаха s (м). flap — угол взмаха крыльев птицы, рад.
inline void build_silhouette(Shape shape, double s, double flap, std::vector<Tri>& out) {
    using namespace sil;
    out.clear();
    switch (shape) {
    case Shape::Quad: {
        const double arm = 0.36 * s;   // от центра до мотора
        hrect(out, 0, 0, 0, 0.11 * s, 0.17 * s);                 // корпус
        vrect(out, 0, -0.17 * s, 0.17 * s, -0.06 * s, 0.05 * s);  // корпус сбоку
        vrect(out, 0, 0.08 * s, 0.16 * s, -0.11 * s, -0.06 * s);  // подвес камеры
        for (int k = 0; k < 4; ++k) {
            const double a = sil::PI_ / 4 + k * sil::PI_ / 2;
            hrect(out, 0.5 * arm * std::cos(a), 0.5 * arm * std::sin(a), 0, 0.5 * arm, 0.022 * s, a - sil::PI_ / 2);
            disc(out, arm * std::cos(a), arm * std::sin(a), 0.03 * s, 0.03 * s, 1.f);   // мотор
            disc(out, arm * std::cos(a), arm * std::sin(a), 0.05 * s, 0.16 * s, 0.3f);  // винт
            vrect(out, arm * std::cos(a), arm * std::sin(a) - 0.03 * s, arm * std::sin(a) + 0.03 * s, -0.01 * s, 0.05 * s);
        }
        break;
    }
    case Shape::Delta: {   // летающее крыло: треугольник с вырезом сзади, кили на концах
        const double L = 0.62 * s;
        const Vec3 nose{0, 0.55 * L, 0}, lt{-0.5 * s, -0.45 * L, 0}, rt{0.5 * s, -0.45 * L, 0}, notch{0, -0.25 * L, 0};
        out.push_back({nose, lt, notch, 1});
        out.push_back({nose, notch, rt, 1});
        vrect(out, 0, -0.3 * L, 0.55 * L, -0.03 * s, 0.04 * s);   // центральная часть
        for (double x : {-0.47 * s, 0.47 * s}) vrect(out, x, -0.45 * L, -0.25 * L, 0, 0.1 * s);
        break;
    }
    case Shape::Plane: {   // самолётная схема: фюзеляж, прямое крыло, хвостовое оперение
        const double L = 0.75 * s;
        hrect(out, 0, 0, 0, 0.035 * s, 0.5 * L);
        vrect(out, 0, -0.5 * L, 0.5 * L, -0.035 * s, 0.035 * s);
        hrect(out, 0, 0.12 * L, 0, 0.5 * s, 0.065 * s);           // крыло
        hrect(out, 0, -0.45 * L, 0, 0.17 * s, 0.04 * s);          // стабилизатор
        vrect(out, 0, -0.5 * L, -0.38 * L, 0, 0.11 * s);          // киль
        break;
    }
    case Shape::Bird: {   // тело, хвост и два крыла, которые машут вокруг продольной оси
        hrect(out, 0, 0, 0, 0.04 * s, 0.17 * s);
        vrect(out, 0, -0.17 * s, 0.17 * s, -0.035 * s, 0.035 * s);
        out.push_back({{0, -0.15 * s, 0}, {-0.07 * s, -0.3 * s, 0}, {0.07 * s, -0.3 * s, 0}, 1});
        const double c = std::cos(flap), sn = std::sin(flap);
        for (double side : {-1.0, 1.0}) {
            auto W = [&](double r, double y) { return Vec3{side * (0.03 * s + r * c), y, r * sn}; };
            const double R = 0.47 * s;
            quad(out, W(0, 0.07 * s), W(0.55 * R, 0.08 * s), W(0.55 * R, -0.08 * s), W(0, -0.08 * s));
            quad(out, W(0.55 * R, 0.08 * s), W(R, 0.0), W(R, -0.05 * s), W(0.55 * R, -0.08 * s));
        }
        break;
    }
    }
}

// Курс по скорости: вперёд — по v (с набором высоты), вправо — горизонтально. pitch_extra: наклон носа, рад.
struct BodyFrame {
    dronolok::Vec3 r, f, u;
};
inline BodyFrame body_frame(const dronolok::Vec3& v, double pitch_extra, double bank) {
    const double h = std::sqrt(v.x * v.x + v.y * v.y);
    const double yaw = h > 1e-6 ? std::atan2(v.x, v.y) : 0;
    const double pitch = std::atan2(v.z, std::max(h, 1e-6)) + pitch_extra;
    dronolok::Vec3 f{std::sin(yaw) * std::cos(pitch), std::cos(yaw) * std::cos(pitch), std::sin(pitch)};
    dronolok::Vec3 r{std::cos(yaw), -std::sin(yaw), 0};
    dronolok::Vec3 u{r.y * f.z - r.z * f.y, r.z * f.x - r.x * f.z, r.x * f.y - r.y * f.x};
    const double cb = std::cos(bank), sb = std::sin(bank);   // крен вокруг f
    const dronolok::Vec3 r2{r.x * cb + u.x * sb, r.y * cb + u.y * sb, r.z * cb + u.z * sb};
    const dronolok::Vec3 u2{u.x * cb - r.x * sb, u.y * cb - r.y * sb, u.z * cb - r.z * sb};
    return {r2, f, u2};
}

// Нарисовать силуэт в 8-битный кадр: o += contrast * доля пикселя, размытая пятном psf (СКО, пикс.).
// q — центр цели в мире (камера в начале координат). Возвращает размер силуэта в пикселях (большая сторона).
template <class ProjectFn>
double draw_silhouette(const std::vector<Tri>& tris, const dronolok::Vec3& q, const BodyFrame& B, double contrast,
                       double psf, ProjectFn&& project, uint8_t* px, int w, int h) {
    struct P2 { double u, v; };
    std::vector<P2> pts(tris.size() * 3);
    double umin = 1e18, umax = -1e18, vmin = 1e18, vmax = -1e18;
    for (size_t i = 0; i < tris.size(); ++i) {
        const dronolok::Vec3* vs[3] = {&tris[i].a, &tris[i].b, &tris[i].c};
        for (int k = 0; k < 3; ++k) {
            const dronolok::Vec3& l = *vs[k];
            const dronolok::Vec3 p{q.x + l.x * B.r.x + l.y * B.f.x + l.z * B.u.x, q.y + l.x * B.r.y + l.y * B.f.y + l.z * B.u.y,
                                   q.z + l.x * B.r.z + l.y * B.f.z + l.z * B.u.z};
            double u, v;
            if (!project(p, u, v)) return 0;
            pts[i * 3 + k] = {u, v};
            umin = std::min(umin, u), umax = std::max(umax, u), vmin = std::min(vmin, v), vmax = std::max(vmax, v);
        }
    }
    const double ext = std::max(umax - umin, vmax - vmin);
    if (ext > 2000) return ext;   // вплотную к камере — не рисуем
    // доля пикселя под силуэтом: суперсэмплинг ss x ss (мелкую цель — гуще)
    const int ss = ext < 6 ? 8 : ext < 30 ? 4 : 2;
    const int m = (int)std::ceil(3 * psf) + 1;
    const int bx0 = (int)std::floor(umin) - m, by0 = (int)std::floor(vmin) - m;
    const int bw = (int)std::ceil(umax) + m - bx0 + 1, bh = (int)std::ceil(vmax) + m - by0 + 1;
    std::vector<float> sub((size_t)bw * ss * bh * ss, 0.f);
    const int SW = bw * ss;
    for (size_t i = 0; i < tris.size(); ++i) {
        const P2 a = pts[i * 3], b = pts[i * 3 + 1], c = pts[i * 3 + 2];
        const double area = (b.u - a.u) * (c.v - a.v) - (b.v - a.v) * (c.u - a.u);
        if (std::fabs(area) < 1e-12) continue;
        const int sx0 = std::max(0, (int)std::floor((std::min({a.u, b.u, c.u}) - bx0) * ss));
        const int sx1 = std::min(SW - 1, (int)std::ceil((std::max({a.u, b.u, c.u}) - bx0) * ss));
        const int sy0 = std::max(0, (int)std::floor((std::min({a.v, b.v, c.v}) - by0) * ss));
        const int sy1 = std::min(bh * ss - 1, (int)std::ceil((std::max({a.v, b.v, c.v}) - by0) * ss));
        const float wgt = tris[i].w;
        for (int sy = sy0; sy <= sy1; ++sy) {
            const double y = by0 + (sy + 0.5) / ss;
            for (int sx = sx0; sx <= sx1; ++sx) {
                const double x = bx0 + (sx + 0.5) / ss;
                const double e0 = (b.u - a.u) * (y - a.v) - (b.v - a.v) * (x - a.u);
                const double e1 = (c.u - b.u) * (y - b.v) - (c.v - b.v) * (x - b.u);
                const double e2 = (a.u - c.u) * (y - c.v) - (a.v - c.v) * (x - c.u);
                const bool in = area > 0 ? (e0 >= 0 && e1 >= 0 && e2 >= 0) : (e0 <= 0 && e1 <= 0 && e2 <= 0);
                if (in) {
                    float& o = sub[(size_t)sy * SW + sx];
                    o = std::max(o, wgt);
                }
            }
        }
    }
    std::vector<float> cov((size_t)bw * bh, 0.f);
    double total = 0;
    for (int y = 0; y < bh; ++y)
        for (int x = 0; x < bw; ++x) {
            float s = 0;
            for (int j = 0; j < ss; ++j)
                for (int i = 0; i < ss; ++i) s += sub[(size_t)(y * ss + j) * SW + x * ss + i];
            cov[(size_t)y * bw + x] = s / (ss * ss);
            total += s;
        }
    // тоньше суперсэмплинга (дальняя цель меньше 1/8 пикселя): закрашенная площадь по треугольникам
    if (total == 0) {
        double a = 0;
        for (size_t i = 0; i < tris.size(); ++i) {
            const P2 p0 = pts[i * 3], p1 = pts[i * 3 + 1], p2 = pts[i * 3 + 2];
            a += 0.5 * std::fabs((p1.u - p0.u) * (p2.v - p0.v) - (p1.v - p0.v) * (p2.u - p0.u)) * tris[i].w;
        }
        const double cu = 0.5 * (umin + umax), cv = 0.5 * (vmin + vmax);
        const int x = (int)std::floor(cu) - bx0, y = (int)std::floor(cv) - by0;
        if (x >= 0 && x < bw && y >= 0 && y < bh) cov[(size_t)y * bw + x] = (float)std::min(1.0, 0.5 * a);
    }
    // пятно объектива: разделимый гаусс
    if (psf > 0.05) {
        const int r = m;
        std::vector<float> k(2 * r + 1);
        float ks = 0;
        for (int i = -r; i <= r; ++i) ks += k[i + r] = (float)std::exp(-0.5 * i * i / (psf * psf));
        for (float& x : k) x /= ks;
        std::vector<float> tmp(cov.size(), 0.f);
        for (int y = 0; y < bh; ++y)
            for (int x = 0; x < bw; ++x) {
                float s = 0;
                for (int i = -r; i <= r; ++i) {
                    const int xx = x + i;
                    if (xx >= 0 && xx < bw) s += k[i + r] * cov[(size_t)y * bw + xx];
                }
                tmp[(size_t)y * bw + x] = s;
            }
        for (int y = 0; y < bh; ++y)
            for (int x = 0; x < bw; ++x) {
                float s = 0;
                for (int i = -r; i <= r; ++i) {
                    const int yy = y + i;
                    if (yy >= 0 && yy < bh) s += k[i + r] * tmp[(size_t)yy * bw + x];
                }
                cov[(size_t)y * bw + x] = s;
            }
    }
    for (int y = 0; y < bh; ++y) {
        const int Y = by0 + y;
        if (Y < 0 || Y >= h) continue;
        for (int x = 0; x < bw; ++x) {
            const int X = bx0 + x;
            const float c = cov[(size_t)y * bw + x];
            if (X < 0 || X >= w || c < 1e-4f) continue;
            uint8_t& o = px[(size_t)Y * w + X];
            o = (uint8_t)std::max(0.0, std::min(255.0, o + contrast * c));
        }
    }
    return ext;
}

}  // namespace twocam
