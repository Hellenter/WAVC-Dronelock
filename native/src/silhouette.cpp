// Силуэт захваченной цели: связная область темнее (светлее) фона вокруг точки, где
// ждём цель. Детектор настроен на точечные цели и у крупного БПЛА находит части —
// кончик крыла, хвост, подвес; силуэт собирает их в один объект и даёт его центр и
// габарит. Окно растёт, пока силуэт упирается в его край (до maxR), и считается на
// уменьшенной сетке не больше ~97x97 ячеек: доли миллисекунды и для цели в 800 пикс.
#include <algorithm>
#include <cmath>
#include <vector>

#include "dronolok/detector.hpp"

namespace dronolok {

static float median_of(std::vector<float>& v) {
    if (v.empty()) return 0.f;
    const size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + (std::ptrdiff_t)m, v.end());
    return v[m];
}

// Тонкая линия от края силуэта: шаг за шагом по направлению (dx, dy) ищем самую тёмную
// (светлую) точку в ±1 пикс. поперёк от прошлой (наклон линии до 45°) и сравниваем с фоном по бокам (±7..10 пикс.):
// небо с градиентом не мешает. Линия есть, если тянется len шагов почти без разрывов.
static bool stem(const uint8_t* gray, int w, int h, int stride, int x, int y, int dx, int dy, int len, float pol,
                 float thr) {
    int gaps = 0, run = 0;
    const int px = dy != 0, py = dx != 0;   // поперечное направление
    for (int k = 0; k < len; ++k) {
        x += dx;
        y += dy;
        // линия ушла за край кадра (столб снизу, из-за кадра): своё продолжение она уже показала
        if (x < 12 || y < 12 || x >= w - 12 || y >= h - 12) return k >= 12 && run >= k * 3 / 4;
        float best = -1e9f;
        int bo = 0;
        for (int o = -1; o <= 1; ++o) {
            const float v = pol * (float)gray[(size_t)(y + o * py) * stride + x + o * px];
            if (v > best) { best = v; bo = o; }
        }
        float bg = 0;
        for (int o = 7; o <= 10; ++o)
            bg += (float)gray[(size_t)(y + o * py) * stride + x + o * px] +
                  (float)gray[(size_t)(y - o * py) * stride + x - o * px];
        bg *= 0.125f;
        if (best - pol * bg > thr) {
            x += bo * px;
            y += bo * py;
            ++run;
        } else if (++gaps > len / 8) {
            return false;
        }
    }
    return run >= len * 3 / 4;
}

Silhouette measure_silhouette(const uint8_t* gray, int w, int h, int stride, float x, float y, float r, int polarity,
                              float minContrast, float maxR) {
    Silhouette out;
    if (!gray || w <= 0 || h <= 0 || x < 0 || y < 0 || x >= w || y >= h) return out;
    const float pol = polarity < 0 ? -1.f : 1.f;
    float H = std::min(maxR, std::max(24.f, 2.5f * std::max(r, 8.f)));
    for (int attempt = 0; attempt < 8; ++attempt) {   // 24 -> maxR удвоениями
        const int s = std::max(1, (int)std::ceil(H / 48.f));   // ячейка сетки, пикс.
        const int half = (int)std::ceil(H / s);
        const int n = 2 * half + 1;
        const int ox = (int)std::lround(x) - half * s, oy = (int)std::lround(y) - half * s;
        // сетка: среднее по ячейке; ячейки за краем кадра — NaN
        std::vector<float> g((size_t)n * n, NAN);
        for (int cy = 0; cy < n; ++cy) {
            const int py = oy + cy * s;
            if (py < 0 || py + s > h) continue;
            for (int cx = 0; cx < n; ++cx) {
                const int px = ox + cx * s;
                if (px < 0 || px + s > w) continue;
                int sum = 0;
                for (int yy = 0; yy < s; ++yy) {
                    const uint8_t* row = gray + (size_t)(py + yy) * stride + px;
                    for (int xx = 0; xx < s; ++xx) sum += row[xx];
                }
                g[(size_t)cy * n + cx] = (float)sum / (float)(s * s);
            }
        }
        // фон и шум — по рамке окна (две внешние ячейки)
        std::vector<float> ring;
        for (int cy = 0; cy < n; ++cy)
            for (int cx = 0; cx < n; ++cx)
                if ((cy < 2 || cy >= n - 2 || cx < 2 || cx >= n - 2) && !std::isnan(g[(size_t)cy * n + cx]))
                    ring.push_back(g[(size_t)cy * n + cx]);
        if (ring.size() < 16) return out;
        const float bg = median_of(ring);
        for (auto& v : ring) v = std::fabs(v - bg);
        const float noise = 1.4826f * median_of(ring);
        const float thr = std::max(minContrast, 4.f * noise);
        std::vector<uint8_t> m((size_t)n * n, 0), md((size_t)n * n, 0);
        for (size_t i = 0; i < m.size(); ++i) m[i] = !std::isnan(g[i]) && pol * (g[i] - bg) > thr;
        // расширение на ячейку: тонкие перемычки (крыло, лучи винтов) не рвут силуэт
        for (int cy = 0; cy < n; ++cy)
            for (int cx = 0; cx < n; ++cx) {
                bool v = false;
                for (int dy = -1; dy <= 1 && !v; ++dy)
                    for (int dx = -1; dx <= 1 && !v; ++dx) {
                        const int yy = cy + dy, xx = cx + dx;
                        v = yy >= 0 && yy < n && xx >= 0 && xx < n && m[(size_t)yy * n + xx];
                    }
                md[(size_t)cy * n + cx] = v;
            }
        // затравка: ближайшая к центру ячейка цели, не дальше ожидаемого радиуса
        const float seedR = std::max(2.f, (r > 0 ? 0.6f * r : 0.25f * H) / s + 1.f);
        int seed = -1;
        float bd = seedR * seedR + 1e-3f;
        for (int cy = 0; cy < n; ++cy)
            for (int cx = 0; cx < n; ++cx) {
                if (!m[(size_t)cy * n + cx]) continue;
                const float d = (float)((cx - half) * (cx - half) + (cy - half) * (cy - half));
                if (d < bd) { bd = d; seed = cy * n + cx; }
            }
        if (seed < 0) return out;
        // связная область по расширенной маске (8-связность)
        std::vector<int> stack{seed};
        std::vector<uint8_t> comp((size_t)n * n, 0);
        comp[(size_t)seed] = 1;
        int x0 = n, x1 = -1, y0 = n, y1 = -1, cells = 0;
        while (!stack.empty()) {
            const int i = stack.back();
            stack.pop_back();
            const int cx = i % n, cy = i / n;
            ++cells;
            x0 = std::min(x0, cx); x1 = std::max(x1, cx);
            y0 = std::min(y0, cy); y1 = std::max(y1, cy);
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int yy = cy + dy, xx = cx + dx;
                    if (yy < 0 || yy >= n || xx < 0 || xx >= n) continue;
                    const size_t j = (size_t)yy * n + xx;
                    if (!md[j] || comp[j]) continue;
                    comp[j] = 1;
                    stack.push_back((int)j);
                }
        }
        // упирается в край окна (не кадра) — цель крупнее окна: окно вдвое больше
        auto inside = [&](int cx, int cy) {
            return cx >= 0 && cy >= 0 && cx < n && cy < n && !std::isnan(g[(size_t)cy * n + cx]);
        };
        const bool touches = (x0 <= 1 && inside(0, (y0 + y1) / 2)) || (x1 >= n - 2 && inside(n - 1, (y0 + y1) / 2)) ||
                             (y0 <= 1 && inside((x0 + x1) / 2, 0)) || (y1 >= n - 2 && inside((x0 + x1) / 2, n - 1));
        if (touches) {
            if (H >= maxR) return out;   // крупнее предела — это не цель, а земля, крона, облако
            H = std::min(maxR, H * 2.f);
            continue;
        }
        if (cells > n * n / 3) return out;   // пол-окна "цели" — это фон
        double sw = 0, sx = 0, sy = 0;
        int area = 0;
        for (int cy = y0; cy <= y1; ++cy)
            for (int cx = x0; cx <= x1; ++cx) {
                const size_t i = (size_t)cy * n + cx;
                if (!comp[i] || !m[i]) continue;
                const double v = pol * (g[i] - bg);
                sw += v;
                sx += v * cx;
                sy += v * cy;
                ++area;
            }
        if (sw <= 0 || area == 0) return out;
        out.valid = true;
        out.x = (float)(ox + (sx / sw + 0.5) * s);
        out.y = (float)(oy + (sy / sw + 0.5) * s);
        out.w = (float)((x1 - x0 + 1) * s);
        out.h = (float)((y1 - y0 + 1) * s);
        out.area = area * s * s;
        out.contrast = (float)(sw / area);
        // мачта, столб, ствол: из верха или низа силуэта уходит линия длиннее цели в
        // несколько раз. Только вверх и вниз: вбок тянутся крылья крупного БПЛА, если
        // силуэт собрал не всё крыло. Начинаем с самой контрастной точки стороны
        {
            const int bx0 = ox + x0 * s, bx1 = ox + (x1 + 1) * s - 1, by0 = oy + y0 * s, by1 = oy + (y1 + 1) * s - 1;
            const int len = (int)std::max(30.f, 2.5f * std::max(out.w, out.h));
            const float lthr = std::max(0.5f * minContrast, 0.25f * out.contrast);
            const int cx = (int)std::lround(out.x), cy = (int)std::lround(out.y);
            auto from = [&](int sx, int sy, int dx, int dy) {
                // по стороне силуэта ищем, где вдоль неё темнее всего (основание мачты)
                const int span = dx ? (by1 - by0) / 2 : (bx1 - bx0) / 2;
                int bxs = sx, bys = sy;
                float bv = -1e9f;
                for (int o = -span; o <= span; ++o) {
                    const int xx = dx ? sx : cx + o, yy = dx ? cy + o : sy;
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                    const float v = pol * (float)gray[(size_t)yy * stride + xx];
                    if (v > bv) { bv = v; bxs = xx; bys = yy; }
                }
                return stem(gray, w, h, stride, bxs, bys, dx, dy, len, pol, lthr);
            };
            // силуэт обрезан низом кадра: стоит на том, что ниже кадра (верхушка столба или
            // кроны над самым краем), — летящий дрон в нижнюю кромку кадра не упирается
            out.attached = oy + (y1 + 2) * s > h || from(cx, by1, 0, 1) || from(cx, by0, 0, -1);
            // контрастная мачта попала в сам силуэт: по строкам габарита ширина тёмного
            // (светлого) — длинный участок узкий, как мачта, а над ним голова прожектора
            if (!out.attached && by1 - by0 >= 30) {
                const int x0c = std::max(0, bx0), x1c = std::min(w - 1, bx1);
                const int y0c = std::max(0, by0), y1c = std::min(h - 1, by1);
                std::vector<int> wid((size_t)(y1c - y0c + 1), 0);
                int maxW = 0;
                for (int y = y0c; y <= y1c; ++y) {
                    const uint8_t* row = gray + (size_t)y * stride;
                    int n = 0;
                    for (int x = x0c; x <= x1c; ++x) n += pol * ((float)row[x] - bg) > thr;
                    wid[(size_t)(y - y0c)] = n;
                    maxW = std::max(maxW, n);
                }
                const int narrow = std::max(4, (int)(0.3f * (float)maxW));
                int run = 0, bestRun = 0, wide = 0;
                for (int v : wid) {
                    wide += v > narrow;
                    run = v > 0 && v <= narrow ? run + 1 : 0;
                    bestRun = std::max(bestRun, run);
                }
                // мачта длиннее и высоты, и ширины головы (фюзеляж БПЛА короче размаха крыла)
                out.attached = wide > 0 && bestRun >= std::max({30, (int)(2.5f * (float)wide), maxW});
            }
        }
        return out;
    }
    return out;
}

}  // namespace dronolok
