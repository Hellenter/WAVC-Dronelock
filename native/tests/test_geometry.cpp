// Геометрия камер и передача цели: пиксель <-> направление туда и обратно, прогноз, окно поиска,
// сдвиг картинки по положению поворотки.
#include <cmath>
#include <cstdio>

#include "dronolok/dronolok.h"
#include "dronolok/geometry.hpp"

using namespace dronolok;

static int failures = 0;
#define CHECK(cond, ...)                         \
    do {                                         \
        if (!(cond)) {                           \
            ++failures;                          \
            std::printf("FAIL: " __VA_ARGS__);   \
            std::printf("\n");                   \
        }                                        \
    } while (0)

static void roundtrip(const CameraModel& m, const Pose& p, const char* name) {
    double worst = 0;
    for (int j = 0; j <= 10; ++j)
        for (int i = 0; i <= 10; ++i) {
            const double u = (m.w - 1) * i / 10.0, v = (m.h - 1) * j / 10.0;
            double u2, v2;
            const bool ok = project(m, p, unproject(m, p, u, v), u2, v2);
            CHECK(ok, "%s: точка %.0f,%.0f не проецируется обратно", name, u, v);
            if (ok) worst = std::max(worst, std::hypot(u2 - u, v2 - v));
        }
    CHECK(worst < 1e-3, "%s: ошибка туда-обратно %.5f пикс.", name, worst);
}

int main() {
    CameraModel wide;   // широкая 16 Мп, 120°
    wide.w = 4608;
    wide.h = 3456;
    wide.hfov_deg = 120;
    CameraModel fish = wide;
    fish.lens = Lens::Fisheye;
    CameraModel narrow;   // быстрая 640x480, 10°
    narrow.w = 640;
    narrow.h = 480;
    narrow.hfov_deg = 10;
    const Pose p0{30, 15, 0}, p1{200, 40, 3};
    roundtrip(wide, p0, "обычный 120");
    roundtrip(fish, p1, "рыбий глаз 120");
    roundtrip(narrow, p1, "узкая 10");
    CameraModel dist = fish;
    dist.k1 = -0.05f;
    roundtrip(dist, p0, "рыбий глаз с дисторсией");

    // центр кадра смотрит вдоль оси, правый край — правее по азимуту, верх — выше по углу места
    double az, el;
    azel_from_dir(unproject(wide, p0, wide.w / 2.0, wide.h / 2.0), az, el);
    CHECK(std::fabs(az - 30) < 1e-6 && std::fabs(el - 15) < 1e-6, "центр: %.3f %.3f", az, el);
    azel_from_dir(unproject(wide, Pose{30, 0, 0}, wide.w - 1, wide.h / 2.0), az, el);
    CHECK(std::fabs(az - 90) < 0.1, "правый край обычного 120 по горизонту: аз %.2f", az);
    azel_from_dir(unproject(fish, Pose{30, 0, 0}, wide.w / 2.0, 0), az, el);
    CHECK(std::fabs(el - 45) < 0.1, "верх рыбьего глаза 120x90: угол места %.2f", el);

    // пиксель широкой около 0.026° (обычный объектив в центре — чуть меньше)
    const double pd = pixel_deg(fish, p0, wide.w / 2.0, wide.h / 2.0);
    CHECK(std::fabs(pd - 120.0 / 4608) < 1e-3, "пиксель рыбьего глаза %.4f°", pd);

    // цель: 2 пикс./кадр вправо на 14 к/с у центра -> около 0.73 °/с по азимуту
    TargetReport r = make_report(fish, Pose{30, 0, 0}, 10.0, 14.f, 2304, 1728, 2, 0, 1.5f);
    CHECK(std::fabs(r.vaz - 2 * 14 * pd) < 0.02 && std::fabs(r.vel) < 1e-3, "скорость %.3f %.3f", r.vaz, r.vel);
    double paz, pel, ps;
    predict(r, 12.0, paz, pel, ps);
    CHECK(std::fabs(az_diff(paz, 30 + 2 * r.vaz)) < 1e-4 && ps > r.sigma, "прогноз %.4f, СКО %.4f", paz, ps);

    // узкая камера, наведённая с упреждением, видит цель в центре окна
    const Pose aimp = aim(r, 12.0, 0.0);
    SearchWindow w = search_window(r, 12.0, narrow, aimp);
    CHECK(w.visible && std::fabs(w.u - 320) < 0.01 && std::fabs(w.v - 240) < 0.01, "окно %.2f %.2f", w.u, w.v);
    CHECK(w.radius >= 8, "радиус окна %.1f", w.radius);
    // переход азимута через север
    TargetReport r2 = r;
    r2.az = 359.5f;
    r2.vaz = 1.f;
    predict(r2, r2.t + 1, paz, pel, ps);
    CHECK(std::fabs(paz - 0.5) < 1e-4, "через север: %.4f", paz);

    // поворот поворотки на 1° вправо: картинка уходит влево примерно на 1°/пиксель
    double dx, dy;
    pose_shift(narrow, Pose{100, 20, 0}, Pose{101, 20, 0}, 320, 240, dx, dy);
    const double npd = pixel_deg(narrow, Pose{100, 20, 0}, 320, 240);
    CHECK(dx < 0 && std::fabs(-dx * npd - std::cos(deg2rad(20))) < 0.02 && std::fabs(dy) < 2, "сдвиг %.2f %.2f", dx, dy);
    pose_shift(narrow, Pose{100, 20, 0}, Pose{100, 21, 0}, 320, 240, dx, dy);
    CHECK(dy > 0 && std::fabs(dy * npd - 1) < 0.02, "сдвиг вверх: %.2f %.2f", dx, dy);

    // C API повторяет C++
    dl_camera c{4608, 3456, 120, 1, -1, -1, 0};
    dl_pose cp{30, 0, 0};
    dl_track tr{};
    tr.id = 7;
    tr.x = 2304;
    tr.y = 1728;
    tr.vx = 2;
    tr.sigma = 1.5f;
    tr.cls = 1;
    dl_target tg;
    dl_target_from_track(&c, &cp, 10.0, 14.f, &tr, 2, &tg);
    CHECK(tg.id == 7 && tg.hard == 1 && std::fabs(tg.vaz - r.vaz) < 1e-5, "C API цель");
    dl_camera nc{640, 480, 10, 0, -1, -1, 0};
    const dl_pose ap = dl_target_aim(&tg, 12.0, 0.0);
    const dl_window cw = dl_target_window(&tg, 12.0, &nc, &ap, 3);
    CHECK(cw.visible && std::fabs(cw.u - 320) < 0.01, "C API окно %.2f", cw.u);
    float u, v;
    CHECK(dl_project(&c, &cp, 30, 0, &u, &v) == 1 && std::fabs(u - 2304) < 1e-2, "C API проекция %.2f", u);

    std::printf(failures ? "geometry: %d FAIL\n" : "geometry: ok\n", failures);
    return failures ? 1 : 0;
}
