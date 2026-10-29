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
#include "gt_eval.hpp"
#include "kernels.hpp"
#include "mini_json.hpp"
#include "y4m_reader.hpp"

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

// SAD с маской против прямого счёта: маска 0 / 0xFF, nullptr, разные длины и смещения
static void test_sad(const Kernels* K) {
    std::mt19937 g(7);
    for (int n : {0, 1, 7, 15, 16, 17, 31, 32, 33, 64, 100, 236, 316, 1000}) {
        for (int off : {0, 1, 5}) {
            auto a = rnd(n + off, g), b = rnd(n + off, g), m = rnd(n + off, g);
            for (auto& x : m) x = (x & 3) == 0 ? 0xFF : 0;
            uint32_t es = 0, ek = 0, ea = 0;
            for (int x = off; x < n + off; ++x) {
                const uint32_t d = (uint32_t)std::abs(a[x] - b[x]);
                ea += d;
                if (m[x]) ++ek;
                else es += d;
            }
            uint32_t s = 1, k = 1;
            K->sad_masked(a.data() + off, b.data() + off, m.data() + off, n, &s, &k);
            CHECK(s == es && k == ek, "%s sad_masked n=%d off=%d: %u/%u против %u/%u", K->name, n, off, s, k, es, ek);
            K->sad_masked(a.data() + off, b.data() + off, nullptr, n, &s, &k);
            CHECK(s == ea && k == 0, "%s sad n=%d off=%d: %u против %u", K->name, n, off, s, ea);
        }
    }
    // худший случай: все разности 255
    std::vector<uint8_t> z(4096, 0), f(4096, 255), m0(4096, 0);
    uint32_t s = 0, k = 0;
    K->sad_masked(z.data(), f.data(), m0.data(), 4096, &s, &k);
    CHECK(s == 255u * 4096 && k == 0, "%s sad_masked max: %u", K->name, s);
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

// Очередь целей: две летящие цели — обе в очереди, весомая первой; "следующая" переводит
// захват на другую, и ручной выбор держится, хоть первая и весомее
static void test_lock_queue() {
    Tracker t(640, 480);
    auto cand = [](float x, float y, float c) {
        Candidate d{};
        d.x = x; d.y = y; d.contrast = c; d.snr = 5; d.polarity = -1; d.margin = c / 10.f;
        return d;
    };
    auto step = [&](int i) { t.update({cand(100 + 3.f * i, 150, 60), cand(100 + 3.f * i, 350, 25)}); };
    for (int i = 0; i < 40; ++i) step(i);
    std::vector<int> q;
    t.lockQueue(q);
    CHECK(q.size() == 2, "очередь целей: %zu вместо 2", q.size());
    CHECK(!q.empty() && t.lockedId() == q[0], "захват не на самой весомой цели");
    const int first = t.lockedId();
    const int second = t.nextLock();
    CHECK(second && second != first && t.lockedId() == second && t.manualLock(), "следующая цель не выбрана");
    for (int i = 40; i < 100; ++i) step(i);
    CHECK(t.lockedId() == second, "ручной выбор перебит более весомой целью");
    CHECK(t.nextLock() == first, "очередь не по кругу");
    CHECK(!t.selectLock(12345), "выбран несуществующий трек");
    std::printf("lock queue: ok\n");
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
// кадра, неподвижные точки сцены едут навстречу. Без setFollowing и без признака "камера
// ведёт цель" (follow_k = 0) такую цель трекер счёл бы наложением на кадр (стоит в кадре,
// пока камера поворачивается) и снял захват; с follow_k (по умолчанию) — держит и так
static void test_follow() {
    for (int follow = 0; follow <= 2; ++follow) {
        TrackerConfig tc;
        tc.verify_frames = 10;
        if (follow == 0) tc.follow_k = 0;
        Tracker t(1280, 720, tc);
        t.setFollowing(follow == 1);
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
            CHECK(kept && lock && t.lockState() == LockState::Hard, "камера ведёт цель (%s): захват %d, состояние %d",
                  follow == 1 ? "setFollowing" : "follow_k", lock, (int)t.lockState());
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
    // с маской неба задание сужается до своих тайлов маски: кандидаты при любом размере
    // заданий те же (на полном кадре и между полными просмотрами, full_every 2)
    for (int fe : {1, 2}) {
        std::vector<Candidate> ref;
        int run = 0;
        for (auto jj : {std::pair<int, int>{1000, 1000}, std::pair<int, int>{1, 1}, std::pair<int, int>{4, 0},
                        std::pair<int, int>{3, 5}}) {
            DetectorConfig k = cfg;
            k.job_rows = jj.first;
            k.job_cols = jj.second;
            k.full_every = fe;
            Detector d(k);
            d.loadMask(path, 0);
            std::vector<Candidate> all;
            for (int i = 0; i < 6; ++i) {
                auto f = ground_frame(w, h, 0.6f, 300 + i);
                for (int y = 0; y < 3; ++y)   // точка в небе, по кадру сдвигается
                    for (int x = 0; x < 3; ++x) f[(size_t)(60 + y) * w + 100 + 9 * i + x] = 40;
                d.detect(f.data(), w, h, w, c);
                all.insert(all.end(), c.begin(), c.end());
            }
            if (run++ == 0) ref = all;
            CHECK(!all.empty() && same(ref, all), "mask + jobs %dx%d full_every %d: %zu vs %zu candidates", jj.first,
                  jj.second, fe, all.size(), ref.size());
        }
    }
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

// Маска дорастает по помехам: дыра в земле, где копятся помехи, закрывается; ряд неба над
// кромкой земли с теми же помехами — нет; вокруг цели в захвате — тоже нет
static void test_grow_mask() {
    const int w = 640, h = 480, T = 64;   // тайлы 10 x 7
    auto frame = [&](unsigned seed) {
        auto f = ground_frame(w, h, 0.6f, seed);
        for (int y = 6 * T; y < 7 * T; ++y)   // гладкая "поляна" внизу кадра — дыра в маске
            for (int x = 5 * T; x < 6 * T; ++x) f[(size_t)y * w + x] = 170;
        return f;
    };
    auto points = [&](int tx, int ty) {   // помехи в трёх ячейках тайла
        std::vector<Candidate> v;
        for (int k = 0; k < 3; ++k) v.push_back(Candidate{tx * T + 8.f + 20 * k, ty * T + 8.f + 16 * k, 20, 3, -1});
        return v;
    };
    for (int variant = 0; variant < 3; ++variant) {
        DetectorConfig cfg;
        cfg.threads = 1;
        cfg.skyline = false;   // иначе дыру внизу закрыла бы линия горизонта
        cfg.grow_mask = variant != 2;
        Detector d(cfg);
        std::vector<Candidate> c;
        d.beginMaskLearning();
        for (int i = 0; i < 25; ++i) d.detect(frame(i).data(), w, h, w, c);
        d.endMaskLearning();
        const auto m0 = d.tileMask();
        const int tw = d.tilesX();
        CHECK(m0[6 * tw + 5] && !m0[5 * tw + 5] && !m0[6 * tw + 4] && !m0[6 * tw + 6] && m0[3 * tw + 2] && !m0[5 * tw + 2],
              "grow mask: learned mask (hole, edge)");
        // помехи в дыре и в ряду неба над кромкой земли (над ним чистое небо)
        std::vector<Candidate> cl = points(5, 6), top = points(2, 3);
        cl.insert(cl.end(), top.begin(), top.end());
        const std::vector<FocusArea> keep = variant == 1 ? std::vector<FocusArea>{FocusArea{5.5f * T, 6.5f * T, 10}}
                                                         : std::vector<FocusArea>{};
        for (int i = 0; i < 90; ++i) {
            d.detect(frame(100 + i).data(), w, h, w, c);
            d.noteClutter(cl, keep);
        }
        const auto& m = d.tileMask();
        const bool hole = !m[6 * tw + 5], edge = !m[3 * tw + 2];
        CHECK(hole == (variant == 0) && !edge && d.maskGrown() == (variant == 0 ? 1 : 0),
              "grow mask variant %d: hole %d, edge %d, grown %d", variant, (int)hole, (int)edge, d.maskGrown());
        size_t diff = 0;
        for (size_t i = 0; i < m.size(); ++i) diff += m[i] != m0[i];
        CHECK(diff == (variant == 0 ? 1u : 0u), "grow mask variant %d: %zu tiles changed", variant, diff);
    }
    std::printf("mask grows where clutter piles up: ok\n");

    // перепроверка маски: облако (крупная рябь: горячих пикселей много — тайл закрыт при обучении,
    // но отклик ниже порога) потом открывается; контрастная "крона" (отклик выше порога) и земля
    // под ними остаются закрытыми
    for (int variant = 0; variant < 3; ++variant) {
        const int A = variant == 2 ? 30 : 8;   // размах ряби; порог 10, горячий пиксель 6
        auto cloudy = [&](unsigned seed) {
            auto f = ground_frame(w, h, 0.6f, seed);
            std::mt19937 g(seed + 7);
            for (int y = T; y < 2 * T; ++y)
                for (int x = 2 * T; x < 4 * T; ++x)
                    f[(size_t)y * w + x] = (uint8_t)(160 + (((x + seed) / 8 + y / 8) % 2) * A + g() % 3);
            return f;
        };
        DetectorConfig cfg;
        cfg.threads = 1;
        cfg.mask_recheck = variant != 0;
        Detector d(cfg);
        std::vector<Candidate> c;
        d.beginMaskLearning();
        for (int i = 0; i < 25; ++i) d.detect(cloudy(i).data(), w, h, w, c);
        d.endMaskLearning();
        const int tw = d.tilesX();
        const auto m0 = d.tileMask();
        CHECK(!m0[tw + 2] && !m0[tw + 3] && !m0[6 * tw + 2], "recheck %d: cloud and ground masked after learning",
              variant);
        for (int i = 0; i < 200; ++i) d.detect(cloudy(500 + i).data(), w, h, w, c);
        const auto& m = d.tileMask();
        const bool open = m[tw + 2] && m[tw + 3], want = variant == 1;
        CHECK(open == want && !m[6 * tw + 2] && !m[5 * tw + 2] && d.maskOpened() == (want ? 2 : 0),
              "recheck %d: cloud open %d, opened %d", variant, (int)open, d.maskOpened());
    }
    std::printf("mask recheck opens clouds: ok\n");
}

// Сдвиг камеры: текстура (пятна 24x24) сдвинута на известное число пикселей; с надписью
// поверх кадра (неподвижные точки исключаются из оценки) и с повтором кадра
static void test_motion() {
    const int w = 1280, h = 720, B = 24;
    std::mt19937 g(7);
    const int bw = w / B + 8, bh = h / B + 8;
    std::vector<uint8_t> blocks((size_t)bw * bh);
    for (auto& b : blocks) b = (uint8_t)(60 + g() % 140);
    auto frame = [&](int ox, int oy, bool osd) {
        std::vector<uint8_t> f((size_t)w * h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) f[(size_t)y * w + x] = blocks[(size_t)((y + oy + 4 * B) / B) * bw + (x + ox + 4 * B) / B];
        if (osd)   // надпись в углу не двигается вместе с картинкой
            for (int y = 20; y < 120; ++y)
                for (int x = 20; x < 400; ++x) f[(size_t)y * w + x] = ((x / 6 + y / 6) & 1) ? 250 : 10;
        return f;
    };
    for (bool osd : {false, true}) {
        GlobalMotion gm;
        auto f0 = frame(0, 0, osd), f1 = frame(-13, 7, osd);   // картинка ушла вправо на 13 и вверх на 7
        gm.estimate(f0.data(), w, h, w);
        const MotionEstimate e = gm.estimate(f1.data(), w, h, w);
        CHECK(e.valid && std::fabs(e.dx - 13) <= 1.5f && std::fabs(e.dy + 7) <= 1.5f, "motion%s: %.2f %.2f (13, -7)",
              osd ? " with overlay" : "", e.dx, e.dy);
        const MotionEstimate z = gm.estimate(f1.data(), w, h, w);   // повтор кадра
        CHECK(z.valid && z.dx == 0 && z.dy == 0, "motion: repeated frame %.2f %.2f", z.dx, z.dy);
    }
    std::printf("motion: ok\n");
}

// Испорченные файлы (маска, разметка, y4m, память помех): отказ, без падения и без гигабайтных выделений
static void put_file(const std::string& path, const std::string& data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
}

static void test_bad_files() {
    const std::string path = "test_bad.bin";
    {   // маска: заголовок согласован, но кадр 2e9 x 2e9 — было выделение ~1e15 байт
        const char magic[8] = {'D', 'L', 'M', 'A', 'S', 'K', '1', 0};
        const int32_t hdr[6] = {2000000000, 2000000000, 2, 32, 2000000000 / 64, 2000000000 / 64};
        std::string s(magic, 8);
        s.append((const char*)hdr, sizeof(hdr));
        put_file(path, s);
        Detector d;
        CHECK(!d.loadMask(path), "loadMask: huge header accepted");
    }
    {   // JSON: глубокая вложенность (было переполнение стека), nan/inf, обычный файл
        minijson::Value v;
        CHECK(!minijson::parse("{\"frames\": " + std::string(200000, '[') + "}", v), "json: deep nesting accepted");
        CHECK(!minijson::parse("{\"x\": nan}", v) && !minijson::parse("[inf]", v), "json: nan/inf accepted");
        CHECK(minijson::parse("{\"a\": [1, 2.5, {\"b\": \"c\"}], \"d\": true}", v) && v["a"].a.size() == 3 &&
                  v["a"].a[1].num() == 2.5 && v["a"].a[2]["b"].s == "c" && v["d"].num() == 1,
              "json: valid file");
        put_file(path, "{\"w\": 1e300, \"h\": -1e300, \"frames\": [[{\"x\": 1e300, \"y\": 5, \"id\": 1e300}]]}");
        GroundTruth g;
        CHECK(g.load(path) && g.frames.size() == 1 && g.frames[0].size() == 1 && g.frames[0][0].id == 1000000000,
              "gt: huge numbers");
    }
    {   // y4m: огромный кадр в заголовке, заголовок без конца строки
        put_file(path, "YUV4MPEG2 W1000000 H1000000 F25:1 Cmono\nFRAME\n");
        FrameReader r;
        CHECK(!r.open(path), "y4m: huge frame accepted");
        put_file(path, "YUV4MPEG2 W64 H48 F25:1 Cmono\nFRAME\n" + std::string(64 * 48, '\x80'));
        FrameReader ok;
        std::vector<uint8_t> y;
        CHECK(ok.open(path) && ok.width() == 64 && ok.next(y) && y.size() == 64u * 48 && !ok.next(y), "y4m: valid file");
    }
    {   // трекер: ячейка 0 из C API/Python (было деление на ноль), память помех с nan
        TrackerConfig tc;
        tc.clutter_cell = 0;
        Tracker t(320, 240, tc);
        std::vector<Candidate> dets(1);
        dets[0].x = 100;
        dets[0].y = 100;
        dets[0].contrast = 20;
        for (int i = 0; i < 3; ++i) t.update(dets);
        CHECK(t.staticCellPx() >= 1, "tracker: clutter_cell 0");
        TrackerConfig sc;
        sc.static_frames = 1000;
        Tracker a(320, 240, sc);
        CHECK(a.saveStatic(path), "saveStatic");
        FILE* f = std::fopen(path.c_str(), "r+b");
        const float bad[2] = {NAN, -INFINITY};
        if (f) {
            std::fseek(f, 5 * 4, SEEK_SET);
            std::fwrite(bad, sizeof(bad), 1, f);
            std::fclose(f);
        }
        std::vector<float> v;
        CHECK(a.loadStatic(path), "loadStatic");
        a.getStatic(v);
        CHECK(!v.empty() && v[0] == 0.f && v[1] == 0.f, "loadStatic: nan/inf kept");
    }
    std::remove(path.c_str());
    std::printf("bad files (mask, json, y4m, tracker): ok\n");
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
    test_sad(ref);
    for (Isa isa : available_isas()) {
        if (isa == Isa::Scalar) continue;
        const Kernels* K = select_kernels(isa);
        test_kernels(ref, K);
        test_sad(K);
        std::printf("kernels %s: ok\n", K->name);
    }
    test_detector();
    test_scales_focus();
    test_compact_tex();
    test_static_map();
    test_lock_queue();
    test_silhouette();
    test_hard_lock();
    test_size_expect();
    test_follow();
    test_governor();
    test_stationary();
    test_grow_mask();
    test_scene();
    test_motion();
    test_bad_files();
    std::printf(failures ? "FAILED: %d\n" : "all tests passed\n", failures);
    return failures ? 1 : 0;
}
