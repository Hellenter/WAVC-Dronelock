// CPU-бэкенд: потоковый проход по блокам кадра.
//
// Кадр (после пулинга) режется на задания: блок job_rows x job_cols тайлов.
// Поток идёт по строкам блока сверху вниз и держит только несколько последних
// строк каждой стадии в кольцевых буферах:
//
//   P[y]   пулинг 2x2 двух строк кадра (тёмная полярность = min, яркая = 255 - max)
//   D_d[y] дилатация P линейным ядром k вдоль направления d (—, |, \, /)
//   R[y]   min_d эрозии D_d тем же ядром (= closing) минус P[y]
//
// и сразу считает статистику тайлов по полосе R из tile строк. Промежуточные
// кадры в память не пишутся; на блок 256 px буферы обеих полярностей занимают
// ~40 КБ и живут в L1/L2. Кадр из памяти читается ровно один раз.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>

#include "backend.hpp"
#include "kernels.hpp"
#include "thread_pool.hpp"

namespace dronolok {
namespace {

constexpr int MAX_K = 31;
// статистика тайлов считается кусками по BAND строк: буфер отклика остаётся в L1
constexpr int BAND = 8;

// Кольцо последних cap строк; слот строки — `per` подряд идущих строк шириной stride
// (для D это 4 направления одной строки y, чтобы ядру хватало одного указателя на строку).
struct Ring {
    std::vector<uint8_t> buf;
    std::vector<uint8_t*> slot;
    int stride = 0, mask = 0;
    void init(int cap, int width, int per) {
        stride = (width + 63) & ~63;
        mask = cap - 1;
        buf.assign((size_t)cap * per * stride + 64, 0);
        // начало буфера выравнено на 64 байта, чтобы строки не делили кэш-линии
        uint8_t* base = buf.data() + ((64 - (uintptr_t)buf.data() % 64) % 64);
        slot.resize(cap);
        for (int i = 0; i < cap; ++i) slot[i] = base + (size_t)i * per * stride;
    }
    // y может быть отрицательным: & даёт правильный слот в дополнительном коде
    uint8_t* row(int y) { return slot[y & mask]; }
};

struct Scratch {
    Ring P[2];
    Ring D[2];
    std::vector<uint8_t> band[2];
    std::vector<uint8_t> colmax;
    std::vector<uint8_t> sm;   // две сглаженные строки кадра (DetectorConfig::smooth)
    int bandStride = 0, smStride = 0;
    int cap = 0, cwp = 0, tile = 0;

    void ensure(int cap_, int cwp_, int tile_) {
        if (cap_ == cap && cwp_ <= cwp && tile_ == tile) return;
        cap = cap_; cwp = cwp_; tile = tile_;
        smStride = (2 * cwp + 64) & ~63;
        sm.assign((size_t)2 * smStride, 0);
        for (int p = 0; p < 2; ++p) {
            P[p].init(cap, cwp, 1);
            D[p].init(cap, cwp, 4);
            bandStride = (cwp + 63) & ~63;
            band[p].assign((size_t)BAND * bandStride, 0);
        }
    }
};

class CpuBackend final : public Backend {
public:
    CpuBackend(const Kernels* k, int threads, const std::vector<int>& cpus)
        : K_(k), pool_(threads, cpus), scratch_(pool_.size()) {}

    std::string name() const override { return std::string("cpu-") + K_->name; }
    int threads() const override { return pool_.size(); }

    void compute(const FrameView& f, const ResponseParams& p, const uint8_t* mask,
                 TileStats& dk, TileStats& br, int& jobs, int& skipped) override {
        const int T = p.tile, k = std::min(p.k | 1, MAX_K), a = k / 2;
        const int Wp = f.w / p.pool, Hp = f.h / p.pool, tw = Wp / T, th = Hp / T;
        const size_t nt = (size_t)tw * th;
        // выключенная полярность — пустая статистика, а не прошлая (её читают перепроверка маски и доращивание)
        dk.resize(p.dark ? nt : 0);
        br.resize(p.bright ? nt : 0);
        const bool cells = p.cells && T % CELL == 0;
        const size_t nc = cells ? (size_t)(tw * T / CELL) * (th * T / CELL) : 0;
        dk.resizeCells(p.dark ? nc : 0);
        br.resizeCells(p.bright ? nc : 0);
        jobs = skipped = 0;
        if (!tw || !th || (!p.dark && !p.bright)) return;
        const int jr = std::max(1, p.job_rows);
#if defined(DL_HAVE_X86)
        const int autoCols = 16;   // замеры на x86: блок 512 px быстрее всего
#else
        const int autoCols = 8;    // L1 у Cortex-A53/A76 32-64 КБ: блок 256 px с буферами влезает целиком
#endif
        const int jc = p.job_cols > 0 ? p.job_cols : autoCols;
        const int nbr = (th + jr - 1) / jr, nbc = (tw + jc - 1) / jc;
        int cap = 1;
        while (cap < k) cap <<= 1;
        const int cwp = std::min(jc, tw) * T + 4 * a;
        for (auto& s : scratch_) s.ensure(cap, cwp, T);
        std::atomic<int> nskip{0};
        ResponseParams q = p;
        q.k = k;
        pool_.run(nbr * nbc, [&](int j, int wi) {
            int tr0 = (j / nbc) * jr, tr1 = std::min(th, tr0 + jr);
            int tc0 = (j % nbc) * jc, tc1 = std::min(tw, tc0 + jc);
            if (mask && p.trim) {
                // блок сужается до рамки своих тайлов из маски: у кромки леса это часть
                // строк и столбцов блока; результат по тайлу от границ блока не зависит
                int r0 = tr1, r1 = tr0, c0 = tc1, c1 = tc0;
                for (int ty = tr0; ty < tr1; ++ty)
                    for (int tx = tc0; tx < tc1; ++tx)
                        if (mask[(size_t)ty * tw + tx]) {
                            r0 = std::min(r0, ty);
                            r1 = std::max(r1, ty + 1);
                            c0 = std::min(c0, tx);
                            c1 = std::max(c1, tx + 1);
                        }
                if (r0 >= r1) {   // весь блок вне зоны неба — статистика остаётся нулевой
                    nskip.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                tr0 = r0, tr1 = r1, tc0 = c0, tc1 = c1;
            } else if (mask) {
                bool any = false;
                for (int ty = tr0; ty < tr1 && !any; ++ty)
                    for (int tx = tc0; tx < tc1 && !any; ++tx) any = mask[(size_t)ty * tw + tx] != 0;
                if (!any) {   // весь блок вне зоны неба — статистика остаётся нулевой
                    nskip.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
            }
            runJob(f, q, Wp, Hp, tw, tr0, tr1, tc0, tc1, scratch_[wi], dk, br);
        });
        jobs = nbr * nbc;
        skipped = nskip.load();
    }

private:
    void runJob(const FrameView& f, const ResponseParams& p, int Wp, int Hp, int tw,
                int tr0, int tr1, int tc0, int tc1, Scratch& S, TileStats& dk, TileStats& br) {
        const int T = p.tile, k = p.k, a = k / 2;
        const int e = k;   // у края кадра морфология даёт ложные отклики (как в Python-версии)
        const int y0 = tr0 * T, y1 = tr1 * T, cx0 = tc0 * T, cx1 = tc1 * T, cw = cx1 - cx0;
        // локальные столбцы буферов: 0 соответствует столбцу gx0 пулингованного кадра
        const int gx0 = cx0 - 2 * a, cwp = cw + 4 * a;
        const int vx0 = std::max(0, gx0), vx1 = std::min(Wp, gx0 + cwp);
        const int off = vx0 - gx0, nv = vx1 - vx0, rpad = cwp - off - nv;
        const bool on[2] = {p.dark, p.bright};
        const int ntiles = tc1 - tc0;
        const uint8_t* rows[MAX_K];

        for (int y = y0 - 2 * a; y < y1 + 2 * a; ++y) {
            // 1) строка пулинга; вне кадра — повтор крайних пикселей
            //    (влияет только на зону e у края, которая всё равно обнуляется)
            {
                const int yy = std::min(std::max(y, 0), Hp - 1);
                uint8_t* pd = S.P[0].row(y);
                uint8_t* pb = S.P[1].row(y);
                if (p.pool == 2) {
                    const uint8_t* r0 = f.data + (size_t)(2 * yy) * f.stride + 2 * vx0;
                    size_t r1 = f.stride;
                    if (p.smooth) {
                        smoothRow(f, 2 * yy, 2 * vx0, 2 * nv, S.sm.data());
                        smoothRow(f, 2 * yy + 1, 2 * vx0, 2 * nv, S.sm.data() + S.smStride);
                        r0 = S.sm.data();
                        r1 = S.smStride;
                    }
                    K_->pool2(r0, r0 + r1, pd + off, pb + off, nv);
                } else if (p.smooth) {
                    smoothRow(f, yy, vx0, nv, S.sm.data());
                    K_->pool1(S.sm.data(), pd + off, pb + off, nv);
                } else {
                    K_->pool1(f.data + (size_t)yy * f.stride + vx0, pd + off, pb + off, nv);
                }
                if (off > 0) { memset(pd, pd[off], off); memset(pb, pb[off], off); }
                if (rpad > 0) { memset(pd + off + nv, pd[off + nv - 1], rpad); memset(pb + off + nv, pb[off + nv - 1], rpad); }
            }
            // 2) дилатация по 4 направлениям для строки yd: нужны P[yd-a .. yd+a]
            const int yd = y - a;
            if (yd >= y0 - a) {
                for (int pol = 0; pol < 2; ++pol) {
                    if (!on[pol]) continue;
                    for (int i = 0; i < k; ++i) rows[i] = S.P[pol].row(yd + i - a) + a;
                    K_->dilate4(rows, k, S.D[pol].row(yd) + a, S.D[pol].stride, cwp - 2 * a);
                }
            }
            // 3) эрозия + минимум по направлениям - P для строки yr: нужны D[yr-a .. yr+a]
            const int yr = yd - a;
            if (yr < y0) continue;
            const int rr = (yr - y0) % T;
            for (int pol = 0; pol < 2; ++pol) {
                if (!on[pol]) continue;
                for (int i = 0; i < k; ++i) rows[i] = S.D[pol].row(yr + i - a) + 2 * a;
                uint8_t* R = S.band[pol].data() + (size_t)(rr % BAND) * S.bandStride;
                K_->erode4_sub(rows, k, S.D[pol].stride, S.P[pol].row(yr) + 2 * a, R, cw);
                if (yr < e || yr >= Hp - e) {
                    memset(R, 0, cw);
                } else {
                    if (cx0 < e) memset(R, 0, std::min(cw, e - cx0));
                    const int xr = std::max(0, Wp - e - cx0);
                    if (xr < cw) memset(R + xr, 0, cw - xr);
                }
                if (rr % BAND == BAND - 1 || rr == T - 1) {
                    TileStats& ts = pol ? br : dk;
                    const size_t i0 = (size_t)(yr / T) * tw + tc0;
                    const int nrows = rr % BAND + 1;
                    K_->tile_stats(S.band[pol].data(), S.bandStride, T, nrows, rr + 1 - nrows, ntiles, p.hot_thr,
                                   &ts.sum[i0], &ts.maxv[i0], &ts.hot[i0], &ts.argmax[i0]);
                    if (!ts.cmax.empty() && nrows == CELL) {
                        const int ctw = tw * T / CELL;
                        cell_stats(S.band[pol].data(), S.bandStride, cw, p.cell_thr, S.colmax,
                                   &ts.cmax[(size_t)(yr / CELL) * ctw + cx0 / CELL],
                                   &ts.carg[(size_t)(yr / CELL) * ctw + cx0 / CELL]);
                    }
                }
            }
        }
    }

    // Строка y кадра, сглаженная 3x3, столбцы x0..x0+n: у краёв кадра — повтор крайних пикселей
    void smoothRow(const FrameView& f, int y, int x0, int n, uint8_t* out) const {
        const uint8_t* b = f.data + (size_t)y * f.stride;
        const uint8_t* a = y > 0 ? b - f.stride : b;
        const uint8_t* c = y + 1 < f.h ? b + f.stride : b;
        const int xs = std::max(x0, 1), xe = std::min(x0 + n, f.w - 1);
        if (xe > xs) K_->smooth3(a + xs, b + xs, c + xs, out + (xs - x0), xe - xs);
        auto px = [&](int x) {
            const int l = std::max(x - 1, 0), r = std::min(x + 1, f.w - 1);
            const unsigned s = (unsigned)a[l] + 2u * a[x] + a[r] + 2u * ((unsigned)b[l] + 2u * b[x] + b[r]) + c[l] + 2u * c[x] + c[r];
            out[x - x0] = (uint8_t)((s + 8) >> 4);
        };
        for (int x = x0; x < std::min(xs, x0 + n); ++x) px(x);
        for (int x = std::max(xe, x0); x < x0 + n; ++x) px(x);
    }

    // Ячейки по полосе из CELL строк отклика: максимум по столбцам (векторизуется
    // компилятором), потом по группам из CELL столбцов. Место максимума ищется только у
    // ячеек выше thr — на чистом небе таких единицы.
    static void cell_stats(const uint8_t* band, int stride, int cw, uint8_t thr, std::vector<uint8_t>& colmax,
                           uint8_t* cmax, uint8_t* carg) {
        colmax.resize(cw);
        uint8_t* cm = colmax.data();
        std::memcpy(cm, band, cw);
        for (int r = 1; r < CELL; ++r) {
            const uint8_t* row = band + (size_t)r * stride;
            for (int x = 0; x < cw; ++x) cm[x] = cm[x] > row[x] ? cm[x] : row[x];
        }
        const int ncell = cw / CELL;
        for (int c = 0; c < ncell; ++c) {
            const uint8_t* g = cm + c * CELL;
            uint8_t m = g[0];
            for (int x = 1; x < CELL; ++x) m = g[x] > m ? g[x] : m;
            cmax[c] = m;
            if (m <= thr) continue;
            for (int r = 0, found = 0; r < CELL && !found; ++r) {
                const uint8_t* row = band + (size_t)r * stride + c * CELL;
                for (int x = 0; x < CELL; ++x)
                    if (row[x] == m) { carg[c] = (uint8_t)(r * CELL + x); found = 1; break; }
            }
        }
    }

    void pool2(const FrameView& f, uint8_t* dk, uint8_t* br, int ds) override {
        const int w2 = f.w / 2, h2 = f.h / 2;
        if (w2 <= 0 || h2 <= 0) return;
        const int rowsPerJob = 32, njobs = (h2 + rowsPerJob - 1) / rowsPerJob;
        pool_.run(njobs, [&](int job, int) {
            const int y1 = std::min(h2, (job + 1) * rowsPerJob);
            for (int y = job * rowsPerJob; y < y1; ++y) {
                const uint8_t* r0 = f.data + (size_t)(2 * y) * f.stride;
                K_->pool2(r0, r0 + f.stride, dk + (size_t)y * ds, br + (size_t)y * ds, w2);
            }
        });
    }

    const Kernels* K_;
    ThreadPool pool_;
    std::vector<Scratch> scratch_;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(Isa isa, int threads, const std::vector<int>& cpus) {
    if (threads <= 0) threads = hardware_threads();
    return std::unique_ptr<Backend>(new CpuBackend(select_kernels(isa), threads, cpus));
}

}  // namespace dronolok
