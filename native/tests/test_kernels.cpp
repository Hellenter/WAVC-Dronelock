// Проверка: каждое SIMD-ядро совпадает со скалярным побайтно, а детектор даёт
// одинаковых кандидатов при любом наборе инструкций, числе потоков и размере заданий.
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "dronolok/detector.hpp"
#include "dronolok/scene.hpp"
#include "dronolok/tuning.hpp"
#include "kernels.hpp"

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

static std::vector<uint8_t> rnd(size_t n, std::mt19937& g) {
    std::vector<uint8_t> v(n);
    for (auto& x : v) x = (uint8_t)g();
    return v;
}

static void test_kernels(const Kernels* ref, const Kernels* K) {
    std::mt19937 g(42);
    for (int n : {1, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 257, 300}) {
        auto a = rnd(2 * n + 64, g), b = rnd(2 * n + 64, g);
        std::vector<uint8_t> d0(n), b0(n), d1(n), b1(n);
        ref->pool2(a.data(), b.data(), d0.data(), b0.data(), n);
        K->pool2(a.data(), b.data(), d1.data(), b1.data(), n);
        CHECK(d0 == d1 && b0 == b1, "%s pool2 n=%d", K->name, n);
        ref->pool1(a.data(), d0.data(), b0.data(), n);
        K->pool1(a.data(), d1.data(), b1.data(), n);
        CHECK(d0 == d1 && b0 == b1, "%s pool1 n=%d", K->name, n);
        {   // сглаживание 3x3: читает столбцы -1..n трёх строк; эталон — формула
            auto r0 = rnd(n + 2, g), r1 = rnd(n + 2, g), r2 = rnd(n + 2, g);
            std::vector<uint8_t> s0(n), s1(n);
            for (int x = 0; x < n; ++x) {
                const uint8_t *a0 = r0.data() + 1 + x, *a1 = r1.data() + 1 + x, *a2 = r2.data() + 1 + x;
                const unsigned v = a0[-1] + 2u * a0[0] + a0[1] + 2u * (a1[-1] + 2u * a1[0] + a1[1]) + a2[-1] + 2u * a2[0] + a2[1];
                s0[x] = (uint8_t)((v + 8) >> 4);
            }
            K->smooth3(r0.data() + 1, r1.data() + 1, r2.data() + 1, s1.data(), n);
            CHECK(s0 == s1, "%s smooth3 n=%d", K->name, n);
        }
        for (int k : {3, 5, 7, 9, 11}) {
            const int A = k / 2, os = n + 2 * A + 5;
            // k строк с полями по A слева и справа: ядро читает [-A, n + A)
            std::vector<std::vector<uint8_t>> rows;
            std::vector<const uint8_t*> rp;
            for (int i = 0; i < k; ++i) rows.push_back(rnd(n + 2 * A, g));
            for (auto& r : rows) rp.push_back(r.data() + A);
            std::vector<uint8_t> o0(4 * os), o1(4 * os);
            ref->dilate4(rp.data(), k, o0.data(), os, n);
            K->dilate4(rp.data(), k, o1.data(), os, n);
            bool ok = true;
            for (int d = 0; d < 4; ++d)
                for (int x = 0; x < n; ++x) ok &= o0[d * os + x] == o1[d * os + x];
            CHECK(ok, "%s dilate4 n=%d k=%d", K->name, n, k);
            // erode4_sub: k блоков по 4 строки
            const int ds = n + 2 * A + 3;
            std::vector<std::vector<uint8_t>> blocks;
            std::vector<const uint8_t*> bp;
            for (int i = 0; i < k; ++i) blocks.push_back(rnd(4 * ds + 2 * A, g));
            for (auto& b : blocks) bp.push_back(b.data() + A);
            auto base = rnd(n, g);
            ref->erode4_sub(bp.data(), k, ds, base.data(), d0.data(), n);
            K->erode4_sub(bp.data(), k, ds, base.data(), d1.data(), n);
            CHECK(d0 == d1, "%s erode4_sub n=%d k=%d", K->name, n, k);
        }
    }
    for (int T : {8, 16, 32, 64}) {
        const int nt = 5, stride = T * nt + 7;
        // разреженный отклик с повторяющимися максимумами — проверка "первого" максимума
        std::vector<uint8_t> band((size_t)T * stride);
        for (auto& x : band) x = (g() % 10 == 0) ? (uint8_t)(g() % 40) : 0;
        band[3 * stride + 5] = 200;
        band[7 * stride + 2] = 200;
        std::vector<uint32_t> s0(nt), s1(nt);
        std::vector<uint8_t> m0(nt), m1(nt);
        std::vector<uint16_t> h0(nt), h1(nt), a0(nt), a1(nt);
        ref->tile_stats(band.data(), stride, T, T, 0, nt, 6, s0.data(), m0.data(), h0.data(), a0.data());
        // тот же тайл кусками по 3 строки должен дать тот же результат
        for (int r0 = 0; r0 < T; r0 += 3)
            K->tile_stats(band.data() + (size_t)r0 * stride, stride, T, std::min(3, T - r0), r0, nt, 6, s1.data(),
                          m1.data(), h1.data(), a1.data());
        CHECK(s0 == s1 && m0 == m1 && h0 == h1 && a0 == a1, "%s tile_stats T=%d", K->name, T);
        CHECK(a0[0] == 3 * T + 5, "%s argmax first T=%d", K->name, T);
    }
}

static std::vector<uint8_t> synth_frame(int w, int h, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<uint8_t> f((size_t)w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            f[(size_t)y * w + x] = (uint8_t)(150 + 40 * y / h + (int)(g() % 5) + ((x / 97 + y / 61) % 7 == 0 ? 25 : 0));
    for (int i = 0; i < 60; ++i) {   // точки 1-3 px, тёмные и яркие
        const int x = 20 + g() % (w - 40), y = 20 + g() % (h - 40), r = 1 + g() % 3;
        const int dv = (i % 4 == 0) ? 40 : -40;
        for (int yy = y; yy < y + r; ++yy)
            for (int xx = x; xx < x + r; ++xx) {
                int v = f[(size_t)yy * w + xx] + dv;
                f[(size_t)yy * w + xx] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
            }
    }
    return f;
}

static bool same(const std::vector<Candidate>& a, const std::vector<Candidate>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].contrast != b[i].contrast || a[i].polarity != b[i].polarity ||
            a[i].level != b[i].level)
            return false;
    return true;
}

static void test_detector() {
    for (int smooth : {0, 1})
    for (int scales : {1, 3})
    for (int pool : {1, 2})
        for (auto wh : {std::pair<int, int>{1001, 777}, std::pair<int, int>{640, 480}}) {
            const int w = wh.first, h = wh.second;
            auto f = synth_frame(w, h, 7 + pool);
            DetectorConfig base;
            base.scales = scales;
            base.pool = pool;
            base.smooth = smooth != 0;
            base.threads = 1;
            base.isa = Isa::Scalar;
            base.job_rows = 1000;
            base.job_cols = 1000;
            Detector ref(base);
            std::vector<Candidate> r, c;
            ref.detect(f.data(), w, h, w, r);
            CHECK(!r.empty(), "no candidates pool=%d", pool);
            for (Isa isa : available_isas())
                for (int th : {1, 3})
                    for (auto j : {std::pair<int, int>{1, 1}, std::pair<int, int>{2, 3}, std::pair<int, int>{4, 8}}) {
                        DetectorConfig cfg = base;
                        cfg.isa = isa;
                        cfg.threads = th;
                        cfg.job_rows = j.first;
                        cfg.job_cols = j.second;
                        Detector d(cfg);
                        d.detect(f.data(), w, h, w, c);
                        CHECK(same(r, c), "detector %s threads=%d job=%dx%d pool=%d %dx%d: %zu vs %zu",
                              isa_name(isa), th, j.first, j.second, pool, w, h, r.size(), c.size());
                    }
            std::printf("detector smooth=%d scales=%d pool=%d %dx%d: %zu кандидатов, все варианты совпали\n", smooth,
                        scales, pool, w, h, r.size());
        }
}

// Крупная цель 12x12 на ровном фоне: на одном масштабе её нет (тайл выглядит текстурой), со вторым — есть,
// и прогноз трека (setFocus) снижает порог только возле себя.
static void test_scales_focus() {
    const int w = 640, h = 480;
    std::vector<uint8_t> f((size_t)w * h, 160);
    for (int y = 200; y < 212; ++y)
        for (int x = 300; x < 312; ++x) f[(size_t)y * w + x] = 100;
    for (int y = 100; y < 103; ++y)   // слабая точечная цель: контраст 8 < thr 10
        for (int x = 100; x < 103; ++x) f[(size_t)y * w + x] = 152;
    DetectorConfig c;
    c.threads = 1;
    c.compact_tex = false;   // иначе квадрат находится и на основном масштабе (см. test_compact_tex)
    std::vector<Candidate> out;
    Detector d1(c);
    d1.detect(f.data(), w, h, w, out);
    CHECK(out.empty(), "scales=1: %zu кандидатов, ждали 0", out.size());
    c.scales = 2;
    c.coarse_every = 1;   // грубый масштаб в каждом кадре: тут проверяется только фокус
    Detector d2(c);
    d2.detect(f.data(), w, h, w, out);
    CHECK(out.size() == 1 && out[0].level == 1 && std::fabs(out[0].x - 306) < 6 && std::fabs(out[0].y - 206) < 6,
          "scales=2: %zu кандидатов", out.size());
    d2.setFocus({{101, 101, 10}});
    d2.detect(f.data(), w, h, w, out);
    CHECK(out.size() == 2, "focus: %zu кандидатов, ждали 2", out.size());
    d2.setFocus({{400, 400, 10}});
    d2.detect(f.data(), w, h, w, out);
    CHECK(out.size() == 1, "focus мимо: %zu кандидатов, ждали 1", out.size());
    // экономия: в кадр между полными просмотрами ищем только вокруг треков
    c.scales = 1;
    c.full_every = 2;
    c.job_rows = 1;
    c.job_cols = 1;
    Detector d3(c);
    for (int y = 100; y < 103; ++y)
        for (int x = 100; x < 103; ++x) f[(size_t)y * w + x] = 120;   // точка, контраст 40
    d3.detect(f.data(), w, h, w, out);
    CHECK(out.size() == 1, "full_every: полный кадр %zu кандидатов", out.size());
    d3.setRoi({{400, 400, 20}});
    d3.detect(f.data(), w, h, w, out);
    CHECK(out.empty() && d3.timing().partial, "full_every: вне треков %zu кандидатов", out.size());
    d3.detect(f.data(), w, h, w, out);
    CHECK(out.size() == 1 && !d3.timing().partial, "full_every: снова полный кадр");
    d3.setRoi({{101, 101, 10}});
    d3.detect(f.data(), w, h, w, out);
    CHECK(out.size() == 1, "full_every: у трека %zu кандидатов", out.size());
    std::printf("scales + focus + roi: ok\n");
}

// Долгая память статичных помех: неподвижная точка выучивается и не даёт трека, летящая
// цель, проходящая через неё, не теряется, память переживает сохранение в файл
static void test_static_map() {
    TrackerConfig tc;
    tc.static_frames = 100;
    Tracker t(640, 480, tc);
    auto cand = [](float x, float y, float c) {
        Candidate d{};
        d.x = x; d.y = y; d.contrast = c; d.snr = 5; d.polarity = -1; d.margin = c / 10.f;   // как от Detector с порогом 10
        return d;
    };
    for (int i = 0; i < 200; ++i) {   // катушка: отклик в 4 кадрах из 5
        std::vector<Candidate> v;
        if (i % 5) v.push_back(cand(200, 200, 30));
        t.update(v);
    }
    std::vector<TrackPoint> cells;
    t.staticCells(cells);
    CHECK(!cells.empty() && cells.size() <= 2, "статичная точка: запомнено ячеек %zu", cells.size());
    // пауза 60 кадров: короткая карта клаттера остыла, долгая память — нет
    for (int i = 0; i < 60; ++i) t.update({});
    t.update({cand(200, 200, 30)});
    CHECK(t.clutterDets().size() == 1, "после паузы статичная точка не помеха");
    // дрон летит через катушку: трек один и тот же от начала до конца
    int id = 0;
    bool kept = true;
    for (int i = 0; i < 50; ++i) {
        std::vector<Candidate> v = {cand(100 + 4.f * i, 202, 40)};
        if (i % 5) v.push_back(cand(200, 200, 30));
        const auto& out = t.update(v);
        const Track* tr = nullptr;
        for (const auto& o : out)
            if (std::fabs(o.x - (100 + 4.f * i)) < 6) tr = &o;
        if (i >= 6) {
            if (!tr) kept = false;
            else if (!id) id = tr->id;
            else if (tr->id != id) kept = false;
        }
    }
    CHECK(kept && id, "летящая цель потерялась у статичной помехи");
    t.staticCells(cells);
    CHECK(cells.size() <= 3, "след летящей цели попал в память: %zu ячеек", cells.size());
    const std::string path = "test_static_map.bin";
    CHECK(t.saveStatic(path), "не сохранилась память");
    Tracker t2(640, 480, tc);
    CHECK(t2.loadStatic(path), "не загрузилась память");
    t2.update({cand(200, 200, 30)});
    CHECK(t2.clutterDets().size() == 1, "после загрузки статичная точка не помеха");
    Tracker t3(320, 240, tc);
    CHECK(!t3.loadStatic(path), "память загрузилась для другого размера кадра");
    std::remove(path.c_str());
    std::printf("static map: ok\n");
}

// Силуэт крупного БПЛА: от кончика крыла к центру всей цели; на чистом небе — ничего
static void test_silhouette() {
    const int W = 800, H = 600;
    std::mt19937 g(7);
    std::vector<uint8_t> img((size_t)W * H);
    for (auto& v : img) v = (uint8_t)(180 + (int)(g() % 7) - 3);
    auto rect = [&](int x0, int y0, int x1, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) img[(size_t)y * W + x] = 60;
    };
    rect(300, 290, 500, 304);   // крыло 200 пикс.
    rect(390, 260, 410, 350);   // фюзеляж
    rect(370, 340, 430, 346);   // хвост
    const Silhouette s = measure_silhouette(img.data(), W, H, W, 305, 297, 6, -1);
    CHECK(s.valid, "силуэт: не найден");
    CHECK(std::fabs(s.x - 400) < 12 && std::fabs(s.y - 300) < 15, "силуэт: центр %.0f,%.0f", s.x, s.y);
    CHECK(s.w > 180 && s.w < 230 && s.h > 70 && s.h < 120, "силуэт: размер %.0fx%.0f", s.w, s.h);
    CHECK(!s.attached, "силуэт: БПЛА принят за светильник на мачте");
    const Silhouette e = measure_silhouette(img.data(), W, H, W, 150, 150, 6, -1);
    CHECK(!e.valid, "силуэт на пустом небе");
    // прожектор на мачте: голова 20x14, мачта 3 пикс. вниз на 250 пикс. с наклоном
    rect(590, 100, 610, 114);
    for (int y = 114; y < 364; ++y) {
        const int x = 599 + (y - 114) / 25;
        for (int xx = x; xx < x + 3; ++xx) img[(size_t)y * W + xx] = 90;
    }
    const Silhouette m = measure_silhouette(img.data(), W, H, W, 600, 107, 6, -1);
    CHECK(m.valid && m.attached, "прожектор на мачте не распознан: valid %d attached %d", m.valid, m.attached);
    std::printf("silhouette: ok\n");
}

// Жёсткий захват: плавно летящий дрон проверяется verify_frames кадров, после этого захват
// не уходит на более яркую цель; без жёсткого захвата — уходит
static void test_hard_lock() {
    for (int hard = 0; hard <= 1; ++hard) {
        TrackerConfig tc;
        tc.hard_lock = hard != 0;
        tc.verify_frames = 10;
        Tracker t(1280, 720, tc);
        auto cand = [](float x, float y, float c) {
            Candidate d{};
            d.x = x; d.y = y; d.contrast = c; d.snr = 5; d.polarity = -1; d.margin = c / 10.f;
            return d;
        };
        int lock = 0;
        for (int i = 0; i < 80; ++i) {
            t.update({cand(100 + 3.f * i, 300 + 0.5f * i, 25)});
            lock = t.lockedId();
        }
        if (hard) CHECK(t.lockState() == LockState::Hard, "жёсткий захват не наступил");
        else CHECK(t.lockState() == LockState::Tracking, "захват без жёсткого: %d", (int)t.lockState());
        bool kept = true;
        for (int i = 80; i < 140; ++i) {   // рядом летит цель вдвое контрастнее
            t.update({cand(100 + 3.f * i, 300 + 0.5f * i, 25), cand(900 - 3.f * (i - 80), 500, 80)});
            if (t.lockedId() != lock) kept = false;
        }
        if (hard) CHECK(kept, "жёсткий захват ушёл на другую цель");
        else CHECK(!kept, "без жёсткого захвата лок не перешёл на сильную цель");
    }
    std::printf("hard lock: ok\n");
}

// Ожидание размера: цель приближается (силуэт растёт на 2% за кадр) — захват ведёт
// размер и скорость его изменения
static void test_size_expect() {
    TrackerConfig tc;
    Tracker t(1280, 720, tc);
    float size = 30, x = 200, y = 300, exp = 0, dls = 0;
    for (int i = 0; i < 90; ++i) {
        x += 4;
        size *= 1.02f;
        Silhouette s;
        s.valid = true;
        s.x = x; s.y = y; s.w = size; s.h = 0.4f * size; s.contrast = 30; s.area = (int)(0.2f * size * size);
        t.setLockSilhouette(s);
        Candidate d{};
        d.x = x; d.y = y; d.contrast = 30; d.snr = 5; d.polarity = -1; d.margin = 3;
        for (const auto& tr : t.update({d}))
            if (tr.id == t.lockedId()) { exp = tr.expectedSize(); dls = tr.dls; }
    }
    CHECK(exp > 0 && std::fabs(exp / (size * 1.02f) - 1) < 0.15f, "ожидание размера: %.0f при %.0f", exp, size * 1.02f);
    CHECK(dls > 0.01f, "ожидание размера: рост не замечен (%.3f за кадр)", dls);
    std::printf("size expectation: ok (%.0f / %.0f px)\n", exp, size * 1.02f);
}

// Камера на поворотке ведёт захваченную цель (setFollowing, aim): цель стоит в центре
// кадра, неподвижные точки сцены едут навстречу. Без setFollowing такую цель трекер счёл
// бы наложением на кадр (стоит в кадре, пока камера поворачивается) и снял захват
static void test_follow() {
    for (int follow = 0; follow <= 1; ++follow) {
        TrackerConfig tc;
        tc.verify_frames = 10;
        Tracker t(1280, 720, tc);
        t.setFollowing(follow != 0);
        auto cand = [](float x, float y, float c) {
            Candidate d{};
            d.x = x; d.y = y; d.contrast = c; d.snr = 5; d.polarity = -1; d.margin = c / 10.f;
            return d;
        };
        float ox = 0, oy = 0, step = 0, err = 0;   // поворот камеры: точка сцены (x, y) — в кадре (x - ox, y - oy), по вертикали стоит
        int lock = 0;
        bool kept = true;
        for (int i = 0; i < 200; ++i) {
            const float sx = 500 + 3.f * i, sy = 360;   // цель летит по сцене
            const std::vector<Candidate> dets = {cand(sx - ox, sy - oy, 30), cand(300 - ox, 200 - oy, 20),
                                                 cand(1100 - ox, 600 - oy, 20)};
            t.update(dets, -step, 0);
            float ax, ay, vx, vy;
            step = 0;
            if (t.aim(ax, ay, vx, vy)) {
                step = vx + 0.3f * (ax - 640.f);
                if (i >= 150) {
                    if (!lock) lock = t.lockedId();
                    kept &= t.lockedId() == lock;
                    err = std::max(err, std::fabs(ax - 640.f));
                }
            }
            ox += step;
        }
        if (follow) {
            CHECK(kept && lock && t.lockState() == LockState::Hard, "камера ведёт цель: захват %d, состояние %d", lock,
                  (int)t.lockState());
            CHECK(err < 30, "камера ведёт цель: цель ушла от центра на %.0f пикс.", err);
        } else {
            CHECK(!(kept && lock && t.lockState() == LockState::Hard), "без setFollowing цель не сочтена наложением");
        }
    }
    std::printf("follow: ok\n");
}

// Регулятор нагрузки: при перегрузке упрощает до предела, при запасе возвращает; профили
static void test_governor() {
    LoadGovernor g;
    int changes = 0;
    for (int i = 0; i < 200; ++i) changes += g.update(50, 33.3);   // кадр дольше периода камеры
    if (g.level() != LoadGovernor::MAX_LEVEL || changes != LoadGovernor::MAX_LEVEL) {
        std::printf("governor up: level %d, changes %d\n", g.level(), changes);
        ++failures;
    }
    for (int i = 0; i < 2000; ++i) g.update(5, 33.3);   // большой запас
    if (g.level() != 0) {
        std::printf("governor down: level %d\n", g.level());
        ++failures;
    }
    for (int i = 0; i < 300; ++i) g.update(25, 33.3);   // 75% бюджета — ступень не меняется
    if (g.level() != 0) {
        std::printf("governor steady: level %d\n", g.level());
        ++failures;
    }
    DetectorConfig d;
    d.scales = 3;
    LoadGovernor::apply(LoadGovernor::MAX_LEVEL, d);
    if (d.scales != 1 || d.full_every != 4 || d.coarse_every != 4) {
        std::printf("governor apply: scales %d full %d coarse %d\n", d.scales, d.full_every, d.coarse_every);
        ++failures;
    }
    TrackerConfig t;
    d = DetectorConfig();
    apply_blocks(profile_blocks(Profile::Minimal), d, t);
    if (d.scales != 1 || d.bright || !d.dark || d.full_every != 3) {
        std::printf("profile minimal\n");
        ++failures;
    }
    const double ms = calibrate_ms_per_mp(5);
    if (!(ms > 0 && ms < 1000)) {
        std::printf("calibrate: %f\n", ms);
        ++failures;
    }
    std::printf("governor, profiles, calibrate (%.2f ms/MP): ok\n", ms);
}

// Неподвижная камера: маска неба в файл и обратно, проверка на старте, второй кандидат в
// тайле, пересчёт трекера под частоту кадров
static std::vector<uint8_t> ground_frame(int w, int h, float horizon, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<uint8_t> f((size_t)w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            f[(size_t)y * w + x] = (uint8_t)(y < horizon * h ? 170 + (int)(g() % 3) : 40 + (int)(g() % 120));
    return f;
}

static void test_stationary() {
    const int w = 640, h = 480;
    const std::string path = "test_mask.bin";
    DetectorConfig cfg;
    cfg.threads = 1;
    std::vector<Candidate> c;
    Detector a(cfg);
    a.beginMaskLearning();
    for (int i = 0; i < 25; ++i) a.detect(ground_frame(w, h, 0.6f, i).data(), w, h, w, c);
    a.endMaskLearning();
    CHECK(a.saveMask(path), "saveMask");
    Detector b(cfg);
    CHECK(b.loadMask(path, 10) && b.tileMask() == a.tileMask(), "loadMask");
    for (int i = 0; i < 10; ++i) b.detect(ground_frame(w, h, 0.6f, 100 + i).data(), w, h, w, c);
    CHECK(!b.verifyingMask() && !b.maskReplaced() && b.tileMask() == a.tileMask(), "mask verify, same scene");
    Detector m(cfg);   // камеру повернули: земли стало больше — маска учится заново
    m.loadMask(path, 10);
    for (int i = 0; i < 10; ++i) m.detect(ground_frame(w, h, 0.3f, 200 + i).data(), w, h, w, c);
    CHECK(m.maskReplaced() && m.tileMask() != a.tileMask(), "mask verify, moved camera");
    std::remove(path.c_str());

    // две тёмные точки в одном тайле 64x64: при per_tile 2 обе
    std::vector<uint8_t> f((size_t)w * h, 170);
    for (int d = 0; d < 2; ++d)
        for (int y = 0; y < 3; ++y)
            for (int x = 0; x < 3; ++x) f[(size_t)(70 + y + d * 6) * w + 70 + x + d * 20] = 120;
    for (int pt : {1, 2}) {
        DetectorConfig k;
        k.threads = 1;
        k.per_tile = pt;
        k.bright = false;
        Detector d(k);
        d.detect(f.data(), w, h, w, c);
        CHECK((int)c.size() == pt, "per_tile %d: %zu candidates", pt, c.size());
    }

    TrackerConfig t;
    const TrackerConfig t25 = tracker_for_fps(t, 25.f), t10 = tracker_for_fps(t, 10.f);
    CHECK(t25.gate == t.gate && t25.max_miss == t.max_miss, "tracker_for_fps 25");
    CHECK(t10.gate > t.gate && t10.max_miss < t.max_miss && t10.confirm == t.confirm, "tracker_for_fps 10");
    std::printf("stationary camera (mask file, per_tile, fps): ok\n");
}

// Оценка сцены: небо сверху, рваная кромка леса, ночь, резкая смена света, сдвиг камеры
static void test_scene() {
    const int w = 640, h = 480;
    std::mt19937 g(5);
    std::normal_distribution<float> n(0.f, 2.f);
    auto frame = [&](float sky, float gain, int shift, std::vector<uint8_t>& f) {
        f.resize((size_t)w * h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const int xs = (x + shift) % w;
                const int top = 300 - shift + (int)(20 * std::sin(xs * 0.05f) + 15 * std::sin(xs * 0.31f));
                float v = y < top ? sky + 0.05f * y
                                 : 50.f + 30.f * (((xs / 3) * 7 + (y / 3) * 13) % 5 == 0) + 25.f * ((((xs / 37) * 73856093u) ^ ((y / 23) * 19349663u)) % 7 < 3);
                v = v * gain + n(g);
                f[(size_t)y * w + x] = (uint8_t)std::min(255.f, std::max(0.f, v));
            }
    };
    std::vector<uint8_t> f;
    SceneAnalyzer sa;
    for (int i = 0; i < 40; ++i) {
        frame(170, 1.f, 0, f);
        sa.update(f.data(), w, h, w, 25.f);
    }
    const SceneState& s = sa.state();
    CHECK(s.ready && s.light == Light::Day && s.skyFrac > 0.5f && s.skyFrac < 0.75f && s.noise > 1.f && s.noise < 3.f,
          "сцена днём: свет %d, неба %.2f, шум %.2f", (int)s.light, s.skyFrac, s.noise);
    DetectorConfig d0, d;
    TrackerConfig t0, t;
    scene_profile(s, d0, t0, d, t);
    CHECK(d.dark && d.bright && !d.auto_thr, "днём профиль не трогает порог и полярность");
    bool jump = false, moved = false;
    for (int i = 0; i < 40; ++i) {
        frame(170, 0.6f, 0, f);
        jump |= sa.update(f.data(), w, h, w, 25.f) && sa.state().lightJump;
    }
    CHECK(jump, "смена света не замечена");
    for (int i = 0; i < 40; ++i) {
        frame(170, 0.6f, 80, f);   // камеру подняли и повернули
        moved |= sa.update(f.data(), w, h, w, 25.f) && sa.state().moved;
    }
    CHECK(moved, "сдвиг камеры не замечен");
    SceneAnalyzer night;
    for (int i = 0; i < 40; ++i) {
        frame(25, 0.4f, 0, f);
        night.update(f.data(), w, h, w, 25.f);
    }
    scene_profile(night.state(), d0, t0, d, t);
    CHECK(night.state().light == Light::Night && !d.dark && d.bright && d.auto_thr, "ночь: свет %d",
          (int)night.state().light);
    std::printf("scene: ok\n");
}

// Крупная контрастная цель сама набирает "горячих" пикселей больше max_density: без compact_tex
// её тайл — текстура и цели нет, с ним — есть. Россыпь точек по тайлу (листва) остаётся текстурой.
static void test_compact_tex() {
    const int w = 640, h = 480;
    std::vector<uint8_t> f((size_t)w * h, 160);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float r2 = (float)((x - 330) * (x - 330) + (y - 210) * (y - 210));
            f[(size_t)y * w + x] = (uint8_t)std::lround(160.f - 40.f * std::exp(-r2 / (2.f * 2.5f * 2.5f)));
        }
    for (int y = 0; y < 64; y += 12)   // тайл (64..128, 320..384): точки с шагом 12 пикс.
        for (int x = 0; x < 64; x += 12) f[(size_t)(320 + y) * w + 64 + x] = f[(size_t)(321 + y) * w + 64 + x] = 120;
    DetectorConfig c;
    c.threads = 1;
    std::vector<Candidate> out;
    c.compact_tex = false;
    Detector d0(c);
    d0.detect(f.data(), w, h, w, out);
    CHECK(out.empty(), "compact_tex=0: %zu кандидатов, ждали 0", out.size());
    c.compact_tex = true;
    Detector d1(c);
    d1.detect(f.data(), w, h, w, out);
    CHECK(out.size() == 1 && std::fabs(out[0].x - 330) < 3 && std::fabs(out[0].y - 210) < 3,
          "compact_tex=1: %zu кандидатов, ждали 1 у (330, 210)", out.size());
    std::printf("compact_tex: ok\n");
}

int main() {
    const Kernels* ref = kernels_scalar();
    for (Isa isa : available_isas()) {
        if (isa == Isa::Scalar) continue;
        const Kernels* K = select_kernels(isa);
        test_kernels(ref, K);
        std::printf("kernels %s: ok\n", K->name);
    }
    test_detector();
    test_scales_focus();
    test_compact_tex();
    test_static_map();
    test_silhouette();
    test_hard_lock();
    test_size_expect();
    test_follow();
    test_governor();
    test_stationary();
    test_scene();
    std::printf(failures ? "FAILED: %d\n" : "all tests passed\n", failures);
    return failures ? 1 : 0;
}
