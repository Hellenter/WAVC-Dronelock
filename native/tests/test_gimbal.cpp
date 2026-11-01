// Наведение поворотки (dronolok/gimbal.hpp): профиль держит пределы скорости, ускорения и рывка, приходит к
// неподвижной и к летящей цели без перерегулирования; фильтр по отметкам с задержкой даёт прогноз на момент
// команды; история положения, поправка на люфт и уход нуля по широкой.
#include <cmath>
#include <cstdio>
#include <random>

#include "dronolok/gimbal.hpp"

using namespace dronolok;

static int fails = 0;
#define CHECK(c, ...)                         \
    do {                                      \
        if (!(c)) {                           \
            ++fails;                          \
            printf("НЕ ТАК: " __VA_ARGS__);   \
            printf("  (%s:%d)\n", __FILE__, __LINE__); \
        }                                     \
    } while (0)

// поворот к цели, которая стоит или летит с постоянной скоростью: время, перерегулирование, пределы
static void slew(double e0, double vt, double& tIn, double& over, double& maxV, double& maxA, double& maxJ) {
    GimbalConfig c;
    AxisProfile p;
    const double h = 0.001;
    tIn = -1, over = 0, maxV = 0, maxA = 0, maxJ = 0;
    double aPrev = 0;
    for (int i = 0; i < 6000; ++i) {
        const double t = i * h;
        const double sp = e0 + vt * t;
        p.track(sp, vt, 0, h, c, false);
        maxV = std::max(maxV, std::fabs(p.v));
        maxA = std::max(maxA, std::fabs(p.a));
        maxJ = std::max(maxJ, std::fabs(p.a - aPrev) / h);
        aPrev = p.a;
        const double e = sp + vt * h - p.p;   // ошибка после шага
        if (tIn < 0 && std::fabs(e) < 0.01) tIn = t;
        if (tIn >= 0) over = std::max(over, (e0 > 0 ? -e : e));
    }
}

static void test_profile() {
    double tIn, over, mv, ma, mj;
    slew(40, 0, tIn, over, mv, ma, mj);
    printf("поворот 40°: в 0.01° за %.3f с, перерегулирование %.4f°, скорость до %.1f, ускорение до %.0f, рывок до %.0f\n", tIn,
           over, mv, ma, mj);
    CHECK(tIn > 0 && tIn < 2.4, "поворот 40° за %.2f с\n", tIn);
    CHECK(over < 0.01, "перерегулирование %.4f°\n", over);
    CHECK(mv <= 30 + 1e-6 && ma <= 150 + 1e-6 && mj <= 3000 * 1.001 + 1, "пределы: %.2f %.1f %.0f\n", mv, ma, mj);
    slew(0.5, 0, tIn, over, mv, ma, mj);
    printf("поворот 0.5°: в 0.01° за %.3f с, перерегулирование %.4f°\n", tIn, over);
    CHECK(tIn > 0 && tIn < 0.6 && over < 0.005, "поворот 0.5°: %.3f с, %.4f°\n", tIn, over);
    // цель летит 10 °/с от поворотки — догоняет и идёт вместе с ней
    slew(20, 10, tIn, over, mv, ma, mj);
    printf("цель 20° впереди, 10 °/с: в 0.01° за %.3f с, перерегулирование %.4f°\n", tIn, over);
    CHECK(tIn > 0 && tIn < 2.5 && over < 0.01, "летящая цель: %.3f с, %.4f°\n", tIn, over);
    // цель навстречу
    slew(20, -10, tIn, over, mv, ma, mj);
    printf("цель 20° впереди, навстречу 10 °/с: в 0.01° за %.3f с, перерегулирование %.4f°\n", tIn, over);
    CHECK(tIn > 0 && tIn < 1.8 && over < 0.01, "встречная цель: %.3f с, %.4f°\n", tIn, over);
    // азимут через 0°
    {
        GimbalConfig c;
        AxisProfile p;
        p.p = 355;
        for (int i = 0; i < 3000; ++i) p.track(5, 0, 0, 0.001, c, true);
        CHECK(std::fabs(az_diff(az_norm(p.p), 5)) < 0.01, "азимут через 0: %.3f\n", p.p);
    }
}

// цель по кругу с манёвром, отметки 60 к/с с задержкой обработки 33 мс: прогноз на момент команды
static void test_filter() {
    GimbalConfig c;
    GimbalFilter f(c);
    std::mt19937 g(5);
    std::normal_distribution<double> N(0, 1);
    auto truth = [](double t, double& az, double& el) {
        az = 350 + 3 * t + 0.6 * std::sin(0.8 * t);   // через 0° по азимуту
        el = 20 + 1.0 * t - 0.3 * std::cos(1.1 * t);
    };
    const double sig = 0.004, lat = 0.033, lead = 0.010;
    double se = 0, sv = 0;
    int n = 0, acc = 0;
    TargetReport w;   // начало — по широкой: грубо
    {
        double az, el;
        truth(0, az, el);
        w.t = 0, w.az = (float)(az + 0.05), w.el = (float)(el - 0.04), w.vaz = 3.5f, w.vel = 0.7f, w.sigma = 0.06f, w.sigma_v = 0.5f;
    }
    f.reset(w);
    for (int k = 1; k < 600; ++k) {
        const double tc = k / 60.0;
        double az, el;
        truth(tc, az, el);
        if (f.measure(tc, az_norm(az + sig * N(g)), el + sig * N(g), sig)) ++acc;
        const double now = tc + lat;   // кадр обработан
        if (tc > 2) {
            double paz, pel, pva, pve, pa, pe;
            f.state(now + lead, paz, pel, pva, pve, pa, pe);
            double taz, tel, taz1, tel1;
            truth(now + lead, taz, tel);
            truth(now + lead + 1e-3, taz1, tel1);
            const double eaz = az_diff(paz, taz) * std::cos(deg2rad(tel)), eel = pel - tel;
            se += eaz * eaz + eel * eel;
            sv += std::pow(pva - (taz1 - taz) / 1e-3, 2) + std::pow(pve - (tel1 - tel) / 1e-3, 2);
            ++n;
        }
    }
    const double rmsP = std::sqrt(se / n), rmsV = std::sqrt(sv / n);
    printf("фильтр: прогноз на %.0f мс вперёд — ошибка %.4f° (отметки %.3f°), скорость %.3f °/с; принято %d из 599\n",
           (lat + lead) * 1000, rmsP, sig, rmsV, acc);
    CHECK(rmsP < 0.006, "ошибка прогноза %.4f°\n", rmsP);
    CHECK(rmsV < 0.1, "ошибка скорости %.3f\n", rmsV);
    CHECK(acc > 590, "принято %d\n", acc);
    // выброс не принимается, три подряд — принимается
    const double t1 = 10.0;
    double az, el;
    truth(t1, az, el);
    const bool o1 = f.measure(t1, az + 1, el, sig);
    CHECK(!o1, "выброс 1° принят\n");
    f.measure(t1 + 0.016, az + 1, el, sig);
    const bool o3 = f.measure(t1 + 0.033, az + 1, el, sig);
    CHECK(o3, "три выброса подряд не приняты\n");
    // отметка старше состояния
    GimbalFilter f2(c);
    TargetReport r;
    r.t = 1, r.az = 10, r.el = 10, r.vaz = 2, r.vel = 0, r.sigma = 0.01f, r.sigma_v = 0.05f;
    f2.reset(r);
    f2.measure(1.1, 10.2, 10, 0.004);
    CHECK(f2.measure(1.05, 10.1, 10, 0.02), "старая отметка не принята\n");
    CHECK(std::fabs(f2.time() - 1.1) < 1e-9, "старая отметка сдвинула время фильтра\n");
}

static void test_history() {
    PoseHistory h;
    for (int i = 0; i <= 100; ++i) h.push(i * 0.01, 10 + i * 0.01, 5);
    double az, el;
    h.at(0.505, az, el);
    CHECK(std::fabs(az - 10.505) < 1e-9, "история: %.4f\n", az);
    h.at(5, az, el);
    CHECK(std::fabs(az - 11) < 1e-9, "история за краем: %.4f\n", az);
    BacklashComp b;
    const double m1 = b.motor(10, 1, 0.1);
    CHECK(std::fabs(m1 - 10.05) < 1e-9 && std::fabs(b.output(m1, 0.1) - 10) < 1e-9, "люфт вперёд\n");
    const double m2 = b.motor(10, -1, 0.1);
    CHECK(std::fabs(m2 - 9.95) < 1e-9, "люфт назад\n");
    // уход нуля по широкой: невязка 0.1° с СКО широкой 0.02° — за ~3 с
    GimbalConfig c;
    WideBias wb;
    std::mt19937 g(1);
    std::normal_distribution<double> N(0, 0.02);
    for (int i = 0; i < 14 * 10; ++i) wb.update(0.1 - wb.az + N(g), -0.05 - wb.el + N(g), 0.02, 1 / 14.0, c);
    printf("уход нуля по широкой за 10 с: %.3f° / %.3f° (на самом деле 0.100 / -0.050)\n", wb.az, wb.el);
    CHECK(std::fabs(wb.az - 0.1) < 0.015 && std::fabs(wb.el + 0.05) < 0.015, "уход нуля %.3f %.3f\n", wb.az, wb.el);
}

static void test_eta() {
    GimbalConfig c;
    GimbalFilter f(c);
    TargetReport r;
    r.t = 0, r.az = 30, r.el = 20, r.vaz = 5, r.vel = 0, r.sigma = 0.02f, r.sigma_v = 0.1f;
    f.reset(r);
    AxisProfile az, el;
    az.p = 0, el.p = 20;
    const double eta = slew_eta(az, el, f, 0, 0.05);
    printf("время поворота к цели 30° в стороне, 5 °/с от поворотки: %.2f с\n", eta);
    CHECK(eta > 1.0 && eta < 2.5, "eta %.2f\n", eta);
    // цель потеряна: уставка по последнему сообщению — по прямой, через 0° азимута, с поправкой на уход нуля
    TargetReport q;
    q.t = 1, q.az = 359, q.el = 10, q.vaz = 2, q.vel = -1;
    const Setpoint s = report_setpoint(q, 1.5, 0.01, 0.1, 0.05);
    CHECK(!s.hold && std::fabs(az_diff(s.az, 0.02 - 0.1)) < 1e-6 && std::fabs(s.el - (10 - 0.51 - 0.05)) < 1e-6 && s.aaz == 0,
          "уставка по сообщению: %.4f %.4f\n", s.az, s.el);
}

int main() {
    test_profile();
    test_filter();
    test_history();
    test_eta();
    if (fails) {
        printf("ошибок: %d\n", fails);
        return 1;
    }
    printf("наведение поворотки: всё в порядке\n");
    return 0;
}
