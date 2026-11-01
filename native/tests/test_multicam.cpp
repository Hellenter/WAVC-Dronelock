// Несколько широких камер: одна цель при перелёте между секторами (перекрытие и промежуток), две
// цели рядом не путаются, слияние, калибровка позы камеры по точной, параллакс, одна камера —
// как раньше; сообщения по сети: кодек, сверка часов, UDP на 127.0.0.1.
#include <cmath>
#include <cstdio>
#include <random>
#include <set>

#include "dronolok/multicam.hpp"
#include "dronolok/netmsg.hpp"

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

namespace {

struct Body {
    Vec3 p0, v;
    Vec3 at(double t) const { return {p0.x + v.x * t, p0.y + v.y * t, p0.z + v.z * t}; }
};

CameraModel wideModel(float fov = 120) {
    CameraModel m;
    m.w = 4608;
    m.h = 3456;
    m.hfov_deg = fov;
    return m;
}

Vec3 from_azel_r(double az, double el, double R) {
    const Vec3 d = dir_from_azel(az, el);
    return {d.x * R, d.y * R, d.z * R};
}

// что сообщит камера (настоящая поза truePose, алгоритм думает — cam.pose) о теле в момент t
bool observe(const RigCamera& cam, const Pose& truePose, const Body& b, double t, int id, std::mt19937& g, TargetReport& out,
             double noisePx = 0.3) {
    const double fps = 14, dt = 1 / fps;
    auto px = [&](double tt, double& u, double& v) {
        const Vec3 q = b.at(tt);
        const Vec3 d{q.x - cam.pos.x, q.y - cam.pos.y, q.z - cam.pos.z};
        return project(cam.model, truePose, d, u, v) && in_frame(cam.model, u, v);
    };
    double u, v, u0, v0;
    if (!px(t, u, v) || !px(t - dt, u0, v0)) return false;
    std::normal_distribution<double> N(0, noisePx);
    out = make_report(cam.model, cam.pose, t, (float)fps, (float)(u + N(g)), (float)(v + N(g)), (float)(u - u0), (float)(v - v0), 0.5f);
    out.id = id;
    out.cls = 1;
    return true;
}

double errDeg(const TargetReport& r, const Vec3& q) {
    double az, el;
    azel_from_dir(q, az, el);
    return angle_between(dir_from_azel(az, el), dir_from_azel(r.az, r.el));
}

}  // namespace

// дрон летит через перекрытие (или промежуток) секторов двух камер: номер цели один всё время
static void crossing(double gapDeg, bool refInGap, const char* name) {
    std::mt19937 g(7);
    RigCamera a, b;
    a.model = b.model = wideModel();
    a.pose = {0, 20, 0};
    b.pose = {(float)(120 + gapDeg), 20, 0};
    FusionConfig fc;
    fc.calib = false;
    Fusion f({a, b}, fc);
    // 400 м, 25 м/с поперёк: ~3.6°/с по азимуту, пролёт от 40° до 100°+
    const double R = 400, az0 = 40, el = 20;
    const Vec3 p0 = from_azel_r(az0, el, R);
    const double hd = 90 + az0;   // курс поперёк луча, по часовой
    Body body{p0, {25 * std::sin(hd * M_PI / 180), 25 * std::cos(hd * M_PI / 180), 0}};
    std::set<int> ids;
    int localA = 5, localB = 9;
    int frames = 0, both = 0, dupFrames = 0, seenFrames = 0, none = 0;
    double worst = 0;
    for (int k = 0; k < 14 * 26; ++k) {
        const double t = k / 14.0;
        TargetReport ra, rb;
        std::vector<TargetReport> va, vb;
        if (observe(a, a.pose, body, t, localA, g, ra)) va.push_back(ra);
        if (observe(b, b.pose, body, t + 0.03, localB, g, rb)) vb.push_back(rb);   // камеры не в такт
        f.update(0, t, va);
        if (refInGap && va.empty() && vb.empty()) {   // точная ведёт цель в промежутке
            for (const auto& x : f.targets()) (void)x;
            TargetReport r;
            double az, e;
            azel_from_dir(body.at(t), az, e);
            r.t = t;
            r.az = (float)az;
            r.el = (float)e;
            r.vaz = 25 / R * 180 / M_PI / std::cos(el * M_PI / 180);
            r.sigma = 0.005f;
            r.sigma_v = 0.05f;
            for (int id : ids) f.observeReference(id, r, true);
        }
        f.update(1, t + 0.03, vb);
        const auto ts = f.targets();
        if (va.empty() && vb.empty() && frames > 0) ++none;
        if (!va.empty() || !vb.empty()) {
            ++frames;
            if (!va.empty() && !vb.empty()) ++both;
            if (ts.size() > 1) ++dupFrames;
            if (!ts.empty()) ++seenFrames;
        }
        for (const auto& x : ts) {
            ids.insert(x.id);
            worst = std::max(worst, errDeg(x.rep, body.at(x.rep.t)));
        }
    }
    CHECK(frames > 200, "%s: кадров с целью %d", name, frames);
    CHECK(ids.size() == 1, "%s: номеров у одной цели %zu", name, ids.size());
    CHECK(dupFrames <= 2, "%s: в перекрытии две цели вместо одной %d кадров (обе камеры видят %d)", name, dupFrames, both);
    CHECK(f.counters.handoffs == 1, "%s: переходов между камерами %d", name, f.counters.handoffs);
    CHECK(worst < 0.1, "%s: ошибка направления %.3f°", name, worst);
    if (gapDeg > 0) CHECK(none > 14 && f.counters.reacquired == 1, "%s: не видно %.1f с, найдена снова %d", name, none / 14.0, f.counters.reacquired);
    std::printf("%s: кадров %d, обе камеры %d, не видит никто %.1f с, номеров %zu, переходов %d, найдена снова %d, ошибка до %.3f°\n",
                name, frames, both, none / 14.0, ids.size(), f.counters.handoffs, f.counters.reacquired, worst);
}

// две цели летят рядом (1.5°) через шов: каждая сохраняет свой номер
static void twoClose() {
    std::mt19937 g(3);
    RigCamera a, b;
    a.model = b.model = wideModel();
    a.pose = {0, 20, 0};
    b.pose = {115, 20, 0};
    FusionConfig fc;
    fc.calib = false;
    Fusion f({a, b}, fc);
    const double R = 400;
    Body b1{from_azel_r(45, 20, R), {}}, b2{from_azel_r(45, 21.5, R), {}};
    const double hd = 135;
    b1.v = b2.v = {22 * std::sin(hd * M_PI / 180), 22 * std::cos(hd * M_PI / 180), 0};
    std::set<std::pair<int, int>> pairs;   // (тело, номер)
    for (int k = 0; k < 14 * 20; ++k) {
        const double t = k / 14.0;
        std::vector<TargetReport> va, vb;
        TargetReport r;
        if (observe(a, a.pose, b1, t, 11, g, r)) va.push_back(r);
        if (observe(a, a.pose, b2, t, 12, g, r)) va.push_back(r);
        if (observe(b, b.pose, b1, t, 21, g, r)) vb.push_back(r);
        if (observe(b, b.pose, b2, t, 22, g, r)) vb.push_back(r);
        f.update(0, t, va);
        f.update(1, t, vb);
        for (const auto& x : f.targets()) {
            const double e1 = errDeg(x.rep, b1.at(t)), e2 = errDeg(x.rep, b2.at(t));
            pairs.insert({e1 < e2 ? 1 : 2, x.id});
        }
    }
    std::set<int> ids;
    for (auto& p : pairs) ids.insert(p.second);
    CHECK(pairs.size() == 2 && ids.size() == 2, "две цели рядом: пар тело-номер %zu, номеров %zu", pairs.size(), ids.size());
    std::printf("две цели в 1.5° через шов: пар тело-номер %zu, переходов %d\n", pairs.size(), f.counters.handoffs);
}

// у второй камеры поза по паспорту неточна: калибровка по точной сводит ошибку к доле пикселя
static void calibration() {
    std::mt19937 g(5);
    RigCamera a, b;
    a.model = b.model = wideModel();
    a.pose = {0, 25, 0};
    b.pose = {110, 25, 0};
    const Pose bTrue{110.4f, 24.7f, 0.6f};   // на самом деле
    FusionConfig fc;
    Fusion f({a, b}, fc);
    std::uniform_real_distribution<double> U(0, 1);
    double before = 0, after = 0;
    int nb = 0, na = 0;
    // много целей в разных местах кадра второй камеры, точная по очереди ведёт каждую
    for (int s = 0; s < 40; ++s) {
        const double az = 110 + (U(g) * 2 - 1) * 50, el = 10 + U(g) * 35, R = 200 + U(g) * 500;
        Body body{from_azel_r(az, el, R), {(U(g) - 0.5) * 20, (U(g) - 0.5) * 20, 0}};
        for (int k = 0; k < 14 * 3; ++k) {
            const double t = s * 10 + k / 14.0;
            TargetReport r;
            std::vector<TargetReport> vb;
            if (observe(b, bTrue, body, t, 100 + s, g, r)) vb.push_back(r);
            f.update(1, t, vb);
            f.update(0, t, {});
            const auto ts = f.targets();
            if (ts.empty()) continue;
            const double e = errDeg(ts[0].rep, body.at(t));
            if (s == 0 && k <= 7) before += e, ++nb;   // пар ещё нет
            if (s >= 30) after += e, ++na;
            // точная: направление по правде (энкодеры), шум 0.003°
            double taz, tel;
            azel_from_dir(body.at(t), taz, tel);
            TargetReport ref;
            ref.t = t;
            ref.az = (float)taz;
            ref.el = (float)tel;
            ref.sigma = 0.003f;
            f.observeReference(ts[0].id, ref, k > 7);
        }
    }
    before /= std::max(1, nb);
    after /= std::max(1, na);
    const CameraCalib& c = f.calib(1);
    CHECK(before > 0.3, "калибровка: ошибка до неё %.3f° (ожидали > 0.3)", before);
    CHECK(after < 0.04, "калибровка: ошибка после %.3f°", after);
    CHECK(c.sigmaDeg < 0.05, "калибровка: неуверенность %.3f°", c.sigmaDeg);
    std::printf("калибровка второй камеры (ошибка установки 0.4° аз, 0.3° у.м., 0.6° поворот): ошибка %.3f° -> %.3f°, "
                "пар %d, отброшено %d, неуверенность %.3f°\n",
                before, after, c.used, c.rejected, c.sigmaDeg);
    // сохранить и загрузить
    Fusion f2({a, b}, fc);
    CHECK(f2.loadCalib(f.saveCalib()), "калибровка: не загрузилась");
    CHECK(std::fabs(f2.calib(1).w[2] - c.w[2]) < 1e-9 && f2.calib(1).sigmaDeg < 0.05, "калибровка: загрузилась не та");
}

// Калибровка не уходит, когда трек широкой сползает с цели на помеху (точная ведёт цель дальше), и учится
// заново, когда камеру сдвинули. drift: последние 1.5 с каждой цели трек широкой уезжает вниз на 2.5°/с;
// bump — после 20 целей камеру повернули на 0.8° по азимуту
static void robust(bool bumpCase) {
    std::mt19937 g(9);
    RigCamera a, b;
    a.model = b.model = wideModel();
    a.pose = {0, 25, 0};
    b.pose = {110, 25, 0};
    FusionConfig fc;
    Fusion f({a, b}, fc);
    std::uniform_real_distribution<double> U(0, 1);
    double after = 0;
    int na = 0;
    const int N = bumpCase ? 60 : 40;
    for (int s = 0; s < N; ++s) {
        const Pose bTrue = bumpCase && s >= 20 ? Pose{110.8f, 25, 0} : b.pose;
        const double az = 110 + (U(g) * 2 - 1) * 50, el = 10 + U(g) * 35, R = 200 + U(g) * 500;
        Body body{from_azel_r(az, el, R), {(U(g) - 0.5) * 20, (U(g) - 0.5) * 20, 0}};
        for (int k = 0; k < 14 * 4; ++k) {
            const double t = s * 10 + k / 14.0;
            TargetReport r;
            std::vector<TargetReport> vb;
            if (observe(b, bTrue, body, t, 100 + s, g, r)) {
                const double td = t - (s * 10 + 2.5);   // сползает на помеху
                if (!bumpCase && td > 0) {
                    r.el -= (float)(2.5 * td);
                    r.vel -= 2.5f;
                }
                vb.push_back(r);
            }
            f.update(1, t, vb);
            f.update(0, t, {});
            const auto ts = f.targets();
            if (ts.empty()) continue;
            if (s >= N - 10 && (bumpCase || k < 30)) after += errDeg(ts[0].rep, body.at(t)), ++na;
            double taz, tel;
            azel_from_dir(body.at(t), taz, tel);
            TargetReport ref;
            ref.t = t;
            ref.az = (float)taz;
            ref.el = (float)tel;
            ref.vaz = r.vaz;   // скорость цели у точной — по правде (у широкой она верная до сползания)
            ref.vel = (float)(r.vel + (!bumpCase && t - (s * 10 + 2.5) > 0 ? 2.5 : 0));
            ref.sigma = 0.003f;
            ref.sigma_v = 0.05f;
            f.observeReference(ts[0].id, ref, k > 7);
        }
    }
    after /= std::max(1, na);
    const CameraCalib& c = f.calib(1);
    if (bumpCase) {
        CHECK(after < 0.08, "камеру сдвинули на 0.8°: после переучивания ошибка %.3f°", after);
        CHECK(c.relearned >= 1, "камеру сдвинули: калибровка не началась заново");
        std::printf("камеру сдвинули на 0.8° после калибровки: заново %d раз, ошибка через 40 целей %.3f°, пар %d, отброшено %d\n",
                    c.relearned, after, c.used, c.rejected);
    } else {
        CHECK(after < 0.03, "трек широкой сползает на помеху: ошибка после калибровки %.3f°", after);
        std::printf("трек широкой сползает на помеху (2.5°/с): ошибка камеры после калибровки %.3f°, пар %d, отброшено %d\n", after,
                    c.used, c.rejected);
    }
}

// камера в 2 м от поворотки: направление пересчитывается из её положения
static void parallax() {
    std::mt19937 g(9);
    RigCamera a;
    a.model = wideModel();
    a.pose = {0, 20, 0};
    a.pos = {2, 0, 0};
    RigCamera b = a;
    b.pose = {120, 20, 0};
    FusionConfig fc;
    fc.calib = false;
    Fusion f({a, b}, fc);
    Body body{from_azel_r(10, 20, 300), {}};
    TargetReport r;
    observe(a, a.pose, body, 1, 3, g, r, 0);
    const TargetReport ref = f.toRef(0, r);
    const double raw = errDeg(r, body.at(1)), fixed = errDeg(ref, body.at(1));
    CHECK(raw > 0.3 && fixed < 0.01, "параллакс: без поправки %.3f°, с поправкой %.4f°", raw, fixed);
    CHECK(ref.sigma > r.sigma, "параллакс: СКО не выросло (%.4f -> %.4f)", r.sigma, ref.sigma);
    std::printf("камера в 2 м от поворотки, цель 300 м: без поправки %.3f°, с поправкой %.4f°, СКО %.3f° -> %.3f°\n", raw, fixed,
                r.sigma, ref.sigma);
}

// одна камера: номера и сообщения — как у её трекера
static void single() {
    RigCamera a;
    a.model = wideModel();
    Fusion f({a}, FusionConfig());
    TargetReport r;
    r.id = 42;
    r.az = 12.5f;
    r.el = 30;
    r.sigma = 0.02f;
    f.update(0, 1, {r});
    const auto ts = f.targets();
    CHECK(ts.size() == 1 && ts[0].id == 42 && ts[0].rep.az == r.az && ts[0].rep.sigma == r.sigma && f.localOf(42, 0) == 42,
          "одна камера: цель не та");
}

// слияние: вторая камера сначала завела цель отдельно (первое сообщение вне допуска), потом совпало
static void merge() {
    RigCamera a, b;
    a.model = b.model = wideModel();
    a.pose = {0, 20, 0};
    b.pose = {110, 20, 0};
    FusionConfig fc;
    fc.calib = false;
    Fusion f({a, b}, fc);
    for (int k = 0; k < 30; ++k) {
        const double t = k / 14.0;
        TargetReport ra, rb;
        ra.id = 1;
        ra.t = rb.t = t;
        ra.az = rb.az = (float)(55 + 2 * t);
        ra.el = rb.el = 20;
        ra.vaz = rb.vaz = 2;
        ra.sigma = rb.sigma = 0.02f;
        ra.sigma_v = rb.sigma_v = 0.1f;
        rb.id = 7;
        if (k < 3) rb.el += 2;   // первые кадры у второй — мимо (трек только родился)
        f.update(0, t, {ra});
        f.update(1, t, k >= 1 ? std::vector<TargetReport>{rb} : std::vector<TargetReport>{});
    }
    const auto ts = f.targets();
    CHECK(ts.size() == 1, "слияние: целей %zu", ts.size());
    CHECK(f.counters.merged == 1, "слияние: слияний %d", f.counters.merged);
    const auto m = f.takeMerges();
    CHECK(m.size() == 1 && f.resolve(m[0].first) == m[0].second && ts.size() == 1 && ts[0].id == m[0].second,
          "слияние: номер после слияния");
    CHECK(ts.size() == 1 && ts[0].seenBy == 3, "слияние: видят камеры %u", ts.empty() ? 0u : ts[0].seenBy);
}

// подсказка: цель в промежутке, по прогнозу уже в кадре второй камеры — у той область поиска
static void cue() {
    RigCamera a, b;
    a.model = b.model = wideModel();
    a.pose = {0, 20, 0};
    b.pose = {140, 20, 0};
    FusionConfig fc;
    fc.calib = false;
    Fusion f({a, b}, fc);
    TargetReport r;
    r.id = 3;
    r.el = 20;
    r.vaz = 6;
    r.sigma = 0.02f;
    r.sigma_v = 0.1f;
    for (int k = 0; k < 5; ++k) {
        r.t = k / 14.0;
        r.az = (float)(60 + 6 * r.t);
        f.update(0, r.t, {r});
        f.update(1, r.t, {});
    }
    // когда цель по прогнозу входит в кадр второй камеры
    double tIn = -1;
    for (double t = 0.3; t < 4 && tIn < 0; t += 0.05) {
        double u, v;
        if (f.projectTo(1, 60 + 6 * t, 20, u, v) && in_frame(b.model, u, v)) tIn = t;
    }
    f.update(0, 0.4, {});
    const auto c0 = f.cues(1, tIn - 0.1), c1 = f.cues(1, tIn + 0.1);
    CHECK(tIn > 0.5 && c0.empty() && c1.size() == 1, "подсказка: вход в кадр %.2f с, областей до %zu, после %zu", tIn, c0.size(),
          c1.size());
    if (c1.size() == 1) CHECK(c1[0].u < 200 && c1[0].r >= 12 && c1[0].id > 0,
                              "подсказка: у края кадра (u %.0f, r %.0f, цель %d)", c1[0].u, c1[0].r, c1[0].id);
    const auto c2 = f.cues(1, 0.4 + fc.coastS + 1);   // цель давно никто не видел
    CHECK(c2.empty(), "подсказка: по забытой цели");
}

static void codec() {
    net::WideMsg m;
    m.cam = 1;
    m.seq = 77;
    m.t = 12345.678901;
    m.sent = 12345.7;
    for (int i = 0; i < 25; ++i) {
        TargetReport r;
        r.id = 100 + i;
        r.az = 359.12345f - i;
        r.el = -3.5f + i;
        r.vaz = -1.25f;
        r.vel = 0.5f;
        r.sigma = 0.0234f;
        r.sigma_v = 0.12f;
        r.cls = i % 4;
        r.hard = i == 0;
        r.size_deg = 0.05f;
        r.confidence = 0.8f;
        m.reps.push_back(r);
    }
    const std::string s = net::encode(m);
    CHECK((int)s.size() <= net::MAX_DATAGRAM, "кодек: строка %zu байт", s.size());
    net::WideMsg d;
    CHECK(net::kind(s) == 'W' && net::decode(s, d), "кодек: не разобралось");
    CHECK(d.cam == 1 && d.seq == 77 && std::fabs(d.t - m.t) < 1e-6 && d.reps.size() == 18, "кодек: шапка (%zu целей)", d.reps.size());
    if (d.reps.size() == 18)
        CHECK(d.reps[0].id == 100 && std::fabs(d.reps[0].az - m.reps[0].az) < 1e-4 && d.reps[0].hard == 1 &&
                  d.reps[3].cls == 3 && std::fabs(d.reps[17].el - m.reps[17].el) < 1e-4 && d.reps[5].t == m.t,
              "кодек: цели");
    CHECK(!net::decode(std::string("DLW1 1 2 3.0 4.0 2;1 2 3"), d), "кодек: битое сообщение принято");
    CHECK(net::kind("hello") == 0, "кодек: чужое сообщение");
    net::CueMsg c;
    c.cam = 2;
    c.t = 5;
    c.areas.push_back({10.5f, 20.25f, 0.3f});
    net::CueMsg c2;
    CHECK(net::decode(net::encode(c), c2) && c2.cam == 2 && c2.areas.size() == 1 && c2.areas[0].el == 20.25f, "кодек: подсказка");
    net::PongMsg q{5, 1, 2, 3}, q2;
    CHECK(net::decode(net::encode(q), q2) && q2.seq == 5 && q2.t3 == 3, "кодек: сверка часов");
    std::printf("кодек: %zu байт на 18 целей\n", s.size());
}

static void clock() {
    std::mt19937 g(11);
    std::exponential_distribution<double> E(1 / 0.004);   // очереди в сети: 4 мс в среднем
    const double off = 0.1234;                              // часы устройства впереди
    net::ClockSync cs;
    for (int i = 0; i < 20; ++i) {
        const double t1 = i * 1.0;
        const double t2 = t1 + 0.0008 + E(g) + off;
        const double t3 = t2 + 0.0002;
        const double t4 = t3 - off + 0.0008 + E(g);
        cs.add(t1, t2, t3, t4);
    }
    CHECK(cs.valid() && std::fabs(cs.offset() - off) < 0.002, "сверка часов: смещение %.4f вместо %.4f", cs.offset(), off);
    std::printf("сверка часов: смещение %.4f с (на самом деле %.4f), задержка %.1f мс\n", cs.offset(), off, cs.delay() * 1000);
}

static void udp() {
    net::UdpSocket a, b;
    std::string err;
    CHECK(a.bind("127.0.0.1", 0, err) && b.bind("127.0.0.1", 0, err), "UDP: %s", err.c_str());
    if (!a.open() || !b.open()) return;
    CHECK(a.sendTo("127.0.0.1", b.port(), "DLB1 1 42"), "UDP: не отправилось");
    std::string got, host;
    int port = 0;
    CHECK(b.recv(got, 1000, &host, &port) && got == "DLB1 1 42" && port == a.port(), "UDP: пришло «%s»", got.c_str());
    CHECK(!b.recv(got, 10), "UDP: лишнее");
}

int main() {
    crossing(-10, false, "перекрытие 10°");
    crossing(16, false, "промежуток 16°");
    crossing(16, true, "промежуток 16°, точная ведёт");
    crossing(30, false, "промежуток 30° (в нём дольше coastS)");
    twoClose();
    calibration();
    robust(false);
    robust(true);
    parallax();
    single();
    merge();
    cue();
    codec();
    clock();
    udp();
    if (failures) std::printf("%d проверок не прошло\n", failures);
    else std::printf("все проверки прошли\n");
    return failures ? 1 : 0;
}
