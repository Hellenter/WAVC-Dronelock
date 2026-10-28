// Реализация ядер поверх simd.hpp. Включается в каждый kernels_<isa>.cpp внутри
// анонимного пространства имён, после определения ровно одного DL_SIMD_*.
//
// Хвост строки: если n >= W, последний вектор берётся с перекрытием (x = n - W),
// поэтому скалярный код нужен только для совсем коротких строк n < W.
// Перекрытие безопасно, потому что ядра не пишут на место своих входов.

static inline uint8_t smax(uint8_t a, uint8_t b) { return a > b ? a : b; }
static inline uint8_t smin(uint8_t a, uint8_t b) { return a < b ? a : b; }

template <class VecBody, class ScalarBody>
static inline void for_cols(int n, VecBody vb, ScalarBody sb) {
    if (n < W) {
        for (int x = 0; x < n; ++x) sb(x);
        return;
    }
    int x = 0;
    for (; x + W <= n; x += W) vb(x);
    if (x < n) vb(n - W);
}

static void k_pool2(const uint8_t* r0, const uint8_t* r1, uint8_t* dk, uint8_t* br, int n) {
    for_cols(n,
        [&](int x) { pool2_vec(r0 + 2 * x, r1 + 2 * x, dk + x, br + x); },
        [&](int x) {
            uint8_t lo = smin(smin(r0[2 * x], r0[2 * x + 1]), smin(r1[2 * x], r1[2 * x + 1]));
            uint8_t hi = smax(smax(r0[2 * x], r0[2 * x + 1]), smax(r1[2 * x], r1[2 * x + 1]));
            dk[x] = lo;
            br[x] = (uint8_t)(255 - hi);
        });
}

static void k_pool1(const uint8_t* r0, uint8_t* dk, uint8_t* br, int n) {
    const V ff = vdup(255);
    for_cols(n,
        [&](int x) { V a = ld(r0 + x); st(dk + x, a); st(br + x, vsubs(ff, a)); },
        [&](int x) { dk[x] = r0[x]; br[x] = (uint8_t)(255 - r0[x]); });
}

// Сглаживание 3x3 [1 2 1]x[1 2 1]/16 (с округлением) строки b по строкам a, b, c: n выходов,
// читает столбцы -1..n. Перед пулингом оно работает как согласованный фильтр для цели в 1-4
// пикс.: шум падает втрое, а тусклая точка — меньше чем вдвое.
static void k_smooth3(const uint8_t* a, const uint8_t* b, const uint8_t* c, uint8_t* out, int n) {
    for_cols(n,
        [&](int x) { smooth3_vec(a + x, b + x, c + x, out + x); },
        [&](int x) {
            const unsigned s = (unsigned)a[x - 1] + 2u * a[x] + a[x + 1] + 2u * ((unsigned)b[x - 1] + 2u * b[x] + b[x + 1]) +
                               c[x - 1] + 2u * c[x] + c[x + 1];
            out[x] = (uint8_t)((s + 8) >> 4);
        });
}

// Длина ядра известна при компиляции для частых k: указатели строк живут в регистрах,
// а 4 направления считаются независимыми цепочками (без длинной зависимости по min/max).
template <int K>
static void dilate4_k(const uint8_t* const* rows, uint8_t* out, int os, int n) {
    constexpr int A = K / 2;
    const uint8_t* p[K];
    for (int i = 0; i < K; ++i) p[i] = rows[i];
    uint8_t* oh = out;
    uint8_t* ov = out + os;
    uint8_t* o1 = out + 2 * os;
    uint8_t* o2 = out + 3 * os;
    for_cols(n,
        [&](int x) {
            const uint8_t* c = p[A] + x;
            const V mid = ld(c);
            V h = mid, v = mid, d1 = mid, d2 = mid;
            for (int i = 0; i < K; ++i) {
                if (i == A) continue;
                const int o = i - A;
                h = vmax(h, ld(c + o));
                v = vmax(v, ld(p[i] + x));
                d1 = vmax(d1, ld(p[i] + x + o));
                d2 = vmax(d2, ld(p[i] + x - o));
            }
            st(oh + x, h); st(ov + x, v); st(o1 + x, d1); st(o2 + x, d2);
        },
        [&](int x) {
            const uint8_t* c = p[A] + x;
            uint8_t h = *c, v = *c, d1 = *c, d2 = *c;
            for (int i = 0; i < K; ++i) {
                const int o = i - A;
                h = smax(h, c[o]);
                v = smax(v, p[i][x]);
                d1 = smax(d1, p[i][x + o]);
                d2 = smax(d2, p[i][x - o]);
            }
            oh[x] = h; ov[x] = v; o1[x] = d1; o2[x] = d2;
        });
}

static void dilate4_any(const uint8_t* const* p, int k, uint8_t* out, int os, int n) {
    const int A = k / 2;
    for (int x = 0; x < n; ++x) {
        const uint8_t* c = p[A] + x;
        uint8_t h = *c, v = *c, d1 = *c, d2 = *c;
        for (int i = 0; i < k; ++i) {
            const int o = i - A;
            h = smax(h, c[o]);
            v = smax(v, p[i][x]);
            d1 = smax(d1, p[i][x + o]);
            d2 = smax(d2, p[i][x - o]);
        }
        out[x] = h; out[os + x] = v; out[2 * os + x] = d1; out[3 * os + x] = d2;
    }
}

static void k_dilate4(const uint8_t* const* rows, int k, uint8_t* out, int os, int n) {
    switch (k) {
        case 3: dilate4_k<3>(rows, out, os, n); break;
        case 5: dilate4_k<5>(rows, out, os, n); break;
        case 7: dilate4_k<7>(rows, out, os, n); break;
        case 9: dilate4_k<9>(rows, out, os, n); break;
        default: dilate4_any(rows, k, out, os, n);
    }
}

template <int K>
static void erode4_sub_k(const uint8_t* const* rows, int ds, const uint8_t* p, uint8_t* out, int n) {
    constexpr int A = K / 2;
    const uint8_t* q[K];
    for (int i = 0; i < K; ++i) q[i] = rows[i];
    for_cols(n,
        [&](int x) {
            const uint8_t* c = q[A] + x;   // D_h строки y
            V h = ld(c), v = ld(c + ds), d1 = ld(c + 2 * ds), d2 = ld(c + 3 * ds);
            for (int i = 0; i < K; ++i) {
                if (i == A) continue;
                const int o = i - A;
                const uint8_t* r = q[i] + x;
                h = vmin(h, ld(c + o));
                v = vmin(v, ld(r + ds));
                d1 = vmin(d1, ld(r + 2 * ds + o));
                d2 = vmin(d2, ld(r + 3 * ds - o));
            }
            st(out + x, vsubs(vmin(vmin(h, v), vmin(d1, d2)), ld(p + x)));
        },
        [&](int x) {
            const uint8_t* c = q[A] + x;
            uint8_t m = *c;
            for (int i = 0; i < K; ++i) {
                const int o = i - A;
                const uint8_t* r = q[i] + x;
                m = smin(m, smin(smin(c[o], r[ds]), smin(r[2 * ds + o], r[3 * ds - o])));
            }
            out[x] = (uint8_t)(m > p[x] ? m - p[x] : 0);
        });
}

static void erode4_sub_any(const uint8_t* const* q, int k, int ds, const uint8_t* p, uint8_t* out, int n) {
    const int A = k / 2;
    for (int x = 0; x < n; ++x) {
        const uint8_t* c = q[A] + x;
        uint8_t m = *c;
        for (int i = 0; i < k; ++i) {
            const int o = i - A;
            const uint8_t* r = q[i] + x;
            m = smin(m, smin(smin(c[o], r[ds]), smin(r[2 * ds + o], r[3 * ds - o])));
        }
        out[x] = (uint8_t)(m > p[x] ? m - p[x] : 0);
    }
}

static void k_erode4_sub(const uint8_t* const* rows, int k, int ds, const uint8_t* p, uint8_t* out, int n) {
    switch (k) {
        case 3: erode4_sub_k<3>(rows, ds, p, out, n); break;
        case 5: erode4_sub_k<5>(rows, ds, p, out, n); break;
        case 7: erode4_sub_k<7>(rows, ds, p, out, n); break;
        case 9: erode4_sub_k<9>(rows, ds, p, out, n); break;
        default: erode4_sub_any(rows, k, ds, p, out, n);
    }
}

// Статистика по тайлам для части полосы: rows строк, начиная со строки r0 тайла.
// Результат накапливается: сумма и счётчик складываются, максимум с позицией
// обновляется только при строгом превышении — так позиция остаётся первой в
// построчном порядке, как у cv2.minMaxLoc, даже если полоса считается кусками.
static void tile_stats_scalar(const uint8_t* band, int stride, int T, int rows, int r0, int ntiles, uint8_t thr,
                              uint32_t* sum, uint8_t* maxv, uint16_t* hot, uint16_t* argmax) {
    for (int t = 0; t < ntiles; ++t) {
        const uint8_t* b = band + t * T;
        uint32_t s = 0, h = 0;
        uint8_t m = maxv[t];
        int am = argmax[t];
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < T; ++c) {
                uint8_t v = b[r * stride + c];
                s += v;
                h += v > thr;
                if (v > m) { m = v; am = (r0 + r) * T + c; }
            }
        sum[t] += s; hot[t] = (uint16_t)(hot[t] + h); maxv[t] = m; argmax[t] = (uint16_t)am;
    }
}

static void k_tile_stats(const uint8_t* band, int stride, int T, int rows, int r0, int ntiles, uint8_t thr,
                         uint32_t* sum, uint8_t* maxv, uint16_t* hot, uint16_t* argmax) {
    if (W == 1 || T % W != 0) {
        tile_stats_scalar(band, stride, T, rows, r0, ntiles, thr, sum, maxv, hot, argmax);
        return;
    }
    const V vthr = vdup(thr), one = vdup(1);
    const int per_row = T / W;
    const int flush_rows = 255 / per_row;   // счётчик в u8-лейнах не должен переполниться
    for (int t = 0; t < ntiles; ++t) {
        const uint8_t* b = band + t * T;
        S s = szero(), hs = szero();
        V mx = vdup(0), cnt = vdup(0);
        int since = 0;
        for (int r = 0; r < rows; ++r) {
            const uint8_t* row = b + r * stride;
            for (int c = 0; c < T; c += W) {
                V v = ld(row + c);
                s = sacc(s, v);
                mx = vmax(mx, v);
                cnt = vadd(cnt, vmin(vsubs(v, vthr), one));   // +1, где v > thr
            }
            if (++since == flush_rows) {
                hs = sacc(hs, cnt);
                cnt = vdup(0);
                since = 0;
            }
        }
        hs = sacc(hs, cnt);
        sum[t] += shsum(s);
        hot[t] = (uint16_t)(hot[t] + shsum(hs));
        const uint8_t m = vhmax(mx);
        if (m <= maxv[t]) continue;   // максимум не превышен — позиция остаётся прежней
        for (int r = 0; r < rows; ++r) {
            const uint8_t* row = b + r * stride;
            int i = -1, c = 0;
            for (; c < T && i < 0; c += W) i = vfirst_eq(ld(row + c), m);
            if (i >= 0) {
                maxv[t] = m;
                argmax[t] = (uint16_t)((r0 + r) * T + c - W + i);
                break;
            }
        }
    }
}

static const dronolok::Kernels* make_table(const char* name) {
    static const dronolok::Kernels k = {name, W, k_pool2, k_pool1, k_dilate4, k_erode4_sub, k_tile_stats, k_smooth3};
    return &k;
}
