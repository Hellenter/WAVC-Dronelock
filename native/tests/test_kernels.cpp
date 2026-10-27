// Проверка: каждое SIMD-ядро совпадает со скалярным побайтно, а детектор даёт
// одинаковых кандидатов при любом наборе инструкций, числе потоков и размере заданий.
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

#include "dronolok/detector.hpp"
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
    for (int scales : {1, 3})
    for (int pool : {1, 2})
        for (auto wh : {std::pair<int, int>{1001, 777}, std::pair<int, int>{640, 480}}) {
            const int w = wh.first, h = wh.second;
            auto f = synth_frame(w, h, 7 + pool);
            DetectorConfig base;
            base.scales = scales;
            base.pool = pool;
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
            std::printf("detector scales=%d pool=%d %dx%d: %zu кандидатов, все варианты совпали\n", scales, pool, w,
                        h, r.size());
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
    std::vector<Candidate> out;
    Detector d1(c);
    d1.detect(f.data(), w, h, w, out);
    CHECK(out.empty(), "scales=1: %zu кандидатов, ждали 0", out.size());
    c.scales = 2;
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
    std::printf("scales + focus: ok\n");
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
    std::printf(failures ? "FAILED: %d\n" : "all tests passed\n", failures);
    return failures ? 1 : 0;
}
