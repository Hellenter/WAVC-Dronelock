// Сдвиг картинки между кадрами (камера в руке, поворотное устройство).
//
// Кадр уменьшается до ~300 пикселей по большей стороне (среднее нескольких точек
// блока, а не всех: на 16 Мп это доли миллисекунды), строится пирамида, на каждом
// уровне берётся модуль градиента (не зависит от автоэкспозиции) и ищется сдвиг с
// минимальной суммой модулей разностей: на верхнем уровне перебором ±R, ниже —
// уточнение ±2 вокруг удвоенного ответа, в конце субпиксель по параболе.
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>

#include "dronolok/detector.hpp"
#include "kernels.hpp"

namespace dronolok {
namespace {

constexpr int MAX_SIDE = 320;   // размер уменьшенной копии
constexpr int MIN_SIDE = 40;    // верхний уровень пирамиды не меньше
constexpr int TOP_RANGE = 6;    // перебор на верхнем уровне, пикс. этого уровня
constexpr int REFINE = 2;

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// модуль градиента |dx| + |dy| (центральные разности, /2), края — 0
void gradient(const std::vector<uint8_t>& in, int w, int h, std::vector<uint8_t>& out) {
    out.assign((size_t)w * h, 0);
    for (int y = 1; y + 1 < h; ++y) {
        const uint8_t* r = &in[(size_t)y * w];
        const uint8_t* u = r - w;
        const uint8_t* d = r + w;
        uint8_t* o = &out[(size_t)y * w];
        for (int x = 1; x + 1 < w; ++x) {
            const int g = std::abs(r[x + 1] - r[x - 1]) + std::abs(d[x] - u[x]);
            o[x] = (uint8_t)std::min(255, g >> 1);
        }
    }
}

// средняя разность при сдвиге (u, v): cur(x) против prev(x - u); край margin не считаем.
// skip — точки, которые не учитываются (неподвижные наложения на кадр), может быть пустым.
// Строка — векторное ядро (psadbw / vpadal): с ветвлением на каждой точке оценка стоила
// 7 мс на кадр 16 Мп, автовекторизованным циклом 1.5 мс.
double sad(const Kernels* K, const std::vector<uint8_t>& cur, const std::vector<uint8_t>& prev, int w, int h,
           int u, int v, int margin, const std::vector<uint8_t>& skip) {
    const int x0 = std::max(margin, margin + u), x1 = std::min(w - margin, w - margin + u);
    const int y0 = std::max(margin, margin + v), y1 = std::min(h - margin, h - margin + v);
    if (x1 - x0 < 8 || y1 - y0 < 8) return 1e9;
    long long s = 0, n = 0;
    for (int y = y0; y < y1; ++y) {
        const uint8_t* c = &cur[(size_t)y * w];
        const uint8_t* p = prev.data() + (std::ptrdiff_t)(y - v) * w - u;
        const uint8_t* k = skip.empty() ? nullptr : &skip[(size_t)y * w + x0];
        uint32_t rs = 0, rk = 0;
        K->sad_masked(c + x0, p + x0, k, x1 - x0, &rs, &rk);
        s += rs;
        n += (x1 - x0) - (long long)rk;
    }
    return n < 64 ? 1e9 : (double)s / (double)n;
}

// Неподвижные наложения на кадр (надписи, логотипы, OSD при записи экрана) и повтор
// кадра: текстура, которая не изменилась ни на уровень. Такие точки тянут оценку к нулю,
// когда камера на самом деле поворачивается, — их не учитываем. Возвращает долю таких
// точек среди текстурных; 1 — кадр повторился или камера стоит. В маске 0xFF — пропустить.
double static_mask(const std::vector<uint8_t>& cur, const std::vector<uint8_t>& prev, std::vector<uint8_t>& skip) {
    skip.resize(cur.size());
    size_t tex = 0, st = 0;
    for (size_t i = 0; i < cur.size(); ++i) {   // без ветвлений — векторизуется
        const unsigned t = cur[i] >= 4, still = t & (std::abs(cur[i] - prev[i]) <= 1);
        skip[i] = (uint8_t)(0u - still);
        tex += t;
        st += still;
    }
    return tex ? (double)st / (double)tex : 1.0;
}

}  // namespace

void GlobalMotion::setRegion(const uint8_t* tiles, int tw, int th, int tilePx) {
    region_.clear();
    if (!tiles || tw <= 0 || th <= 0 || tilePx <= 0) return;
    size_t ground = 0;
    for (size_t i = 0; i < (size_t)tw * th; ++i) ground += tiles[i] == 0;
    if (ground * 20 < (size_t)tw * th) return;
    region_.resize((size_t)tw * th);
    for (size_t i = 0; i < region_.size(); ++i) region_[i] = tiles[i] == 0;
    rtw_ = tw;
    rth_ = th;
    rpx_ = tilePx;
}

void GlobalMotion::setAnchored(bool on) {
    if (on == anchored_) return;
    anchored_ = on;
    prev_.clear();
    last_.clear();
    accX_ = accY_ = 0;
}

bool GlobalMotion::inRegion(int x, int y, int scale) const {
    if (region_.empty()) return true;
    const int tx = std::min(rtw_ - 1, x * scale / rpx_), ty = std::min(rth_ - 1, y * scale / rpx_);
    return region_[(size_t)ty * rtw_ + tx] != 0;
}

MotionEstimate GlobalMotion::estimate(const uint8_t* gray, int w, int h, int stride) {
    const double t0 = now_ms();
    MotionEstimate res;
    if (w != w_ || h != h_) {
        w_ = w;
        h_ = h;
        prev_.clear();
        last_.clear();
        f_ = 1;
        while (w / f_ > MAX_SIDE || h / f_ > MAX_SIDE) f_ *= 2;
    }
    // уменьшенная копия: среднее 4 точек блока f x f (для f <= 2 — всех)
    std::vector<Img> pyr(1);
    Img& b = pyr[0];
    b.w = std::max(1, w / f_);
    b.h = std::max(1, h / f_);
    b.px.resize((size_t)b.w * b.h);
    const int o1 = f_ / 4, o2 = f_ - 1 - f_ / 4;
    for (int y = 0; y < b.h; ++y) {
        const uint8_t* r1 = gray + (size_t)(y * f_ + o1) * stride;
        const uint8_t* r2 = gray + (size_t)(y * f_ + o2) * stride;
        uint8_t* o = &b.px[(size_t)y * b.w];
        for (int x = 0; x < b.w; ++x) {
            const int xs = x * f_;
            o[x] = (uint8_t)((r1[xs + o1] + r1[xs + o2] + r2[xs + o1] + r2[xs + o2] + 2) >> 2);
        }
    }
    while (std::min(pyr.back().w, pyr.back().h) / 2 >= MIN_SIDE) {
        const Img& s = pyr.back();
        Img n;
        n.w = s.w / 2;
        n.h = s.h / 2;
        n.px.resize((size_t)n.w * n.h);
        for (int y = 0; y < n.h; ++y)
            for (int x = 0; x < n.w; ++x) {
                const uint8_t* p = &s.px[(size_t)(2 * y) * s.w + 2 * x];
                n.px[(size_t)y * n.w + x] = (uint8_t)((p[0] + p[1] + p[s.w] + p[s.w + 1] + 2) >> 2);
            }
        pyr.push_back(std::move(n));
    }
    // на каждом уровне храним градиент
    for (auto& lv : pyr) {
        std::vector<uint8_t> g;
        gradient(lv.px, lv.w, lv.h, g);
        lv.px.swap(g);
    }
    if (!anchored_) {
        if (prev_.size() == pyr.size()) res = match(pyr, prev_, 0.f, 0.f, true);
        prev_ = std::move(pyr);
        ms_ = now_ms() - t0;
        return res;
    }
    // От опорного кадра (prev_): сдвиг за кадр — разность с прошлым итогом. Прогноз для
    // поиска и проверка — обычный сдвиг от прошлого кадра (last_): при повторе кадра или
    // рывке камеры (запись экрана) опорный совмещается плохо — тогда он и берётся.
    if (prev_.size() != pyr.size() || last_.size() != pyr.size()) {
        prev_ = pyr;
        last_ = std::move(pyr);
        accX_ = accY_ = 0;
        ms_ = now_ms() - t0;
        return res;
    }
    const MotionEstimate step = match(pyr, last_, 0.f, 0.f, false);
    MotionEstimate key = match(pyr, prev_, accX_ + step.dx, accY_ + step.dy, false);
    const float ix = key.dx - accX_, iy = key.dy - accY_;
    const float tol = std::max(4.f, 0.5f * std::hypot(step.dx, step.dy));
    const bool keyOk = key.valid && step.valid && std::hypot(ix - step.dx, iy - step.dy) <= tol;
    const bool far = std::max(std::fabs(key.dx), std::fabs(key.dy)) > std::min(w, h) / 6.f;
    if (keyOk) {
        res = key;
        res.dx = ix;
        res.dy = iy;
        accX_ = key.dx;
        accY_ = key.dy;
    } else {
        res = step;
    }
    if (!keyOk || far) {   // новый опорный кадр
        prev_ = pyr;
        accX_ = accY_ = 0;
    }
    last_ = std::move(pyr);
    ms_ = now_ms() - t0;
    return res;
}

// Сдвиг cur относительно ref (пирамиды градиента), поиск вокруг (px, py) пикс. кадра.
// partCheck — проверка по частям кадра (ветки на ветру при неподвижной камере)
MotionEstimate GlobalMotion::match(const std::vector<Img>& pyr, const std::vector<Img>& ref, float px, float py,
                                   bool partCheck) const {
    MotionEstimate res;
    // кадр повторился (запись 25 к/с в ролике 60 к/с) или вся текстура на месте — сдвига нет
    std::vector<std::vector<uint8_t>> skip(pyr.size());
    if (static_mask(pyr[0].px, ref[0].px, skip[0]) > 0.9) {
        res.valid = true;
        res.quality = 1.f;
        return res;
    }
    for (size_t L = 1; L < pyr.size(); ++L) static_mask(pyr[L].px, ref[L].px, skip[L]);
    if (!region_.empty())   // только земля: небо — как неподвижные точки, не учитываются
        for (size_t L = 0; L < pyr.size(); ++L)
            for (int y = 0; y < pyr[L].h; ++y)
                for (int x = 0; x < pyr[L].w; ++x)
                    if (!inRegion(x, y, f_ << L)) skip[L][(size_t)y * pyr[L].w + x] = 0xFF;
    static const Kernels* const K = select_kernels(Isa::Auto);
    const int topS = f_ << (pyr.size() - 1);
    int u = (int)std::lround(px / topS), v = (int)std::lround(py / topS);
    double best = 0;
    double sum = 0;
    int cnt = 0;
    for (int L = (int)pyr.size() - 1; L >= 0; --L) {
        const Img& c = pyr[L];
        const Img& p = ref[L];
        const int range = L == (int)pyr.size() - 1 ? TOP_RANGE : REFINE;
        const int margin = 2;
        if (L != (int)pyr.size() - 1) {
            u *= 2;
            v *= 2;
        }
        int bu = u, bv = v;
        best = 1e18;
        for (int dv = -range; dv <= range; ++dv)
            for (int du = -range; du <= range; ++du) {
                const double e = sad(K, c.px, p.px, c.w, c.h, u + du, v + dv, margin, skip[L]);
                if (L == 0) { sum += e; ++cnt; }
                if (e < best) { best = e; bu = u + du; bv = v + dv; }
            }
        u = bu;
        v = bv;
    }
    // субпиксель: парабола по трём точкам вдоль каждой оси на нижнем уровне
    const Img& c = pyr[0];
    const Img& p = ref[0];
    auto sub = [&](double em, double e0, double ep) {
        const double den = em - 2 * e0 + ep;
        return den > 1e-9 ? std::max(-0.5, std::min(0.5, 0.5 * (em - ep) / den)) : 0.0;
    };
    const double e0 = best;
    const auto& k0 = skip[0];
    auto at = [&](int du, int dv) { return sad(K, c.px, p.px, c.w, c.h, u + du, v + dv, 2, k0); };
    const double fx = sub(at(-1, 0), e0, at(1, 0));
    const double fy = sub(at(0, -1), e0, at(0, 1));
    // уверенность: насколько лучший сдвиг выделяется среди соседних и есть ли вообще текстура
    uint64_t tsum = 0;
    for (uint8_t g : c.px) tsum += g;
    const double texture = (double)tsum / (double)c.px.size();
    const double mean = cnt ? sum / cnt : best;
    res.quality = (float)std::max(0.0, std::min(1.0, mean > 1e-9 ? 1.0 - best / mean : 0.0));
    res.valid = texture > 1.0 && res.quality > 0.05;
    // Проверка по частям кадра: при повороте камеры сдвиг совмещает почти все текстурные
    // куски кадра лучше, чем "сдвига нет". Если за сдвиг только часть кадра, а другая
    // часть стоит на месте, — это шевелится что-то в кадре (ветки на ветру, облака) при
    // неподвижной камере: неподвижный фон мы сами исключили маской, и оценка ушла за
    // ветками. Считаем по всем точкам, с неподвижными; наложения на кадр при настоящем
    // повороте — малая часть кусков.
    if (partCheck && res.valid && (u != 0 || v != 0)) {
        int forShift = 0, forZero = 0;
        const int G = 4, m = 2 + std::max(std::abs(u), std::abs(v));
        for (int gy = 0; gy < G; ++gy)
            for (int gx = 0; gx < G; ++gx) {
                const int x0 = std::max(m, c.w * gx / G), x1 = std::min(c.w - m, c.w * (gx + 1) / G);
                const int y0 = std::max(m, c.h * gy / G), y1 = std::min(c.h - m, c.h * (gy + 1) / G);
                if (x1 - x0 < 4 || y1 - y0 < 4) continue;
                if (!inRegion((x0 + x1) / 2, (y0 + y1) / 2, f_)) continue;   // кусок неба: облака не в счёт
                long long es = 0, ez = 0, tx = 0;
                for (int y = y0; y < y1; ++y) {
                    const uint8_t* cr = &c.px[(size_t)y * c.w];
                    const uint8_t* pz = &p.px[(size_t)y * c.w];
                    const uint8_t* ps = p.px.data() + (std::ptrdiff_t)(y - v) * c.w - u;
                    for (int x = x0; x < x1; ++x) {
                        es += std::abs(cr[x] - ps[x]);
                        ez += std::abs(cr[x] - pz[x]);
                        tx += cr[x];
                    }
                }
                if (tx < 2LL * (x1 - x0) * (y1 - y0)) continue;   // кусок без текстуры (чистое небо)
                if (es * 5 < ez * 4) ++forShift;
                else if (ez * 5 < es * 4) ++forZero;
            }
        if (forShift < 2 * forZero || forShift == 0) {
            res.quality = 1.f;
            return res;   // valid, сдвиг 0
        }
    }
    if (res.valid) {
        res.dx = (float)((u + fx) * f_);
        res.dy = (float)((v + fy) * f_);
    }
    return res;
}

}  // namespace dronolok
