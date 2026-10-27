// Выбор кандидатов по статистике тайлов, авто-маска неба, замер времени.
// Логика один в один с PointDetector._peaks из detector/detector.py, включая
// округление OpenCV (INTER_AREA даёт u8 с округлением к ближайшему чётному).
#include <algorithm>
#include <chrono>
#include <cmath>

#include "backend.hpp"
#include "dronolok/detector.hpp"

namespace dronolok {

// Уровень грубого масштаба: кадр, уменьшенный в 2^l раз (min и 255-max), и его статистика
struct Level {
    std::vector<uint8_t> d, b, junk;
    int w = 0, h = 0, tw = 0, th = 0, tile = 0;
    TileStats sd, sb;
};

struct Detector::Stats {
    TileStats dark, bright, unused;
    std::vector<Level> lv;   // lv[0] — уровень 1 (x2), lv[1] — уровень 2 (x4)
};

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// num / den с округлением к ближайшему чётному — как saturate_cast<uchar>(float) в OpenCV
// для INTER_AREA (там сумма умножается на 1/den; при den = 2^n это точно то же самое).
// Целочисленно, потому что nearbyint в MinGW-рантайме медленный.
static inline uint32_t div_round_even(uint32_t num, uint32_t den) {
    const uint32_t q = num / den, r = num % den;
    return q + (2 * r > den || (2 * r == den && (q & 1)));
}

static DetectorConfig sanitize(DetectorConfig c) {
    c.k = std::max(3, std::min(31, c.k | 1));
    c.pool = c.pool == 1 ? 1 : 2;
    c.tile = std::max(8, std::min(128, c.tile));
    c.max_cands = std::max(1, c.max_cands);
    c.scales = std::max(1, std::min(4, c.scales));
    c.focus_k = std::max(0.2f, std::min(1.f, c.focus_k));
    c.full_every = std::max(1, std::min(16, c.full_every));
    c.coarse_every = std::max(1, std::min(16, c.coarse_every));
    c.mixed_k = std::max(1.f, std::min(5.f, c.mixed_k));
    return c;
}

Detector::Detector(const DetectorConfig& cfg) : cfg_(sanitize(cfg)), stats_(new Stats) {
    backend_ = make_cpu_backend(cfg_.isa, cfg_.threads, cfg_.cpus);
}

Detector::~Detector() = default;

std::string Detector::backendName() const { return backend_->name(); }
int Detector::threads() const { return backend_->threads(); }

void Detector::reconfigure(const DetectorConfig& c0) {
    const DetectorConfig c = sanitize(c0);
    const bool rebuild = c.threads != cfg_.threads || c.cpus != cfg_.cpus || c.isa != cfg_.isa;
    const bool geom = c.k != cfg_.k || c.pool != cfg_.pool || c.tile != cfg_.tile;
    cfg_ = c;
    if (rebuild) backend_ = make_cpu_backend(cfg_.isa, cfg_.threads, cfg_.cpus);
    if (geom) {   // сетка тайлов поменялась — старая маска недействительна
        mask_.clear();
        learning_ = false;
        w_ = h_ = 0;
    }
}

void Detector::beginMaskLearning() {
    mask_.clear();
    learning_ = true;
    texSum_.assign((size_t)tw_ * th_, 0.f);
    texN_ = 0;
}

void Detector::endMaskLearning(float frac) {
    learning_ = false;
    const int n = texN_ * ((int)cfg_.dark + (int)cfg_.bright);
    if (n == 0 || texSum_.size() != (size_t)tw_ * th_) return;
    std::vector<uint8_t> bad(texSum_.size());
    for (size_t i = 0; i < bad.size(); ++i) bad[i] = texSum_[i] / n > frac;
    // расширяем "плохую" зону на 1 тайл (dilate 3x3)
    mask_.assign(bad.size(), 1);
    for (int y = 0; y < th_; ++y)
        for (int x = 0; x < tw_; ++x) {
            bool b = false;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int yy = y + dy, xx = x + dx;
                    if (yy >= 0 && yy < th_ && xx >= 0 && xx < tw_) b |= bad[(size_t)yy * tw_ + xx] != 0;
                }
            mask_[(size_t)y * tw_ + x] = !b;
        }
}

void Detector::clearMask() {
    mask_.clear();
    learning_ = false;
}

void Detector::detect(const uint8_t* gray, int w, int h, int stride, std::vector<Candidate>& out) {
    const double t0 = now_ms();
    out.clear();
    const int s = cfg_.pool, T = cfg_.tile;
    if (w != w_ || h != h_) {
        w_ = w;
        h_ = h;
        tw_ = (w / s) / T;
        th_ = (h / s) / T;
        mask_.clear();
        if (learning_) texSum_.assign((size_t)tw_ * th_, 0.f);
    }
    // как cv2.threshold(R, thr*0.6) на u8: горячий, если R > floor(thr*0.6)
    const float ht = std::floor(cfg_.thr * 0.6f);
    ResponseParams p;
    p.k = cfg_.k;
    p.pool = s;
    p.tile = T;
    p.hot_thr = (uint8_t)std::max(0.f, std::min(255.f, ht));
    p.dark = cfg_.dark;
    p.bright = cfg_.bright;
    p.job_rows = cfg_.job_rows;
    p.job_cols = cfg_.job_cols;
    const uint8_t* m = (!learning_ && !mask_.empty()) ? mask_.data() : nullptr;
    // между полными просмотрами — только тайлы вокруг треков (маска неба тоже действует)
    const long fn = frameNo_++;
    timing_.partial = !learning_ && cfg_.full_every > 1 && fn % cfg_.full_every != 0;
    timing_.coarsePartial = cfg_.coarse_every > 1 && fn % cfg_.coarse_every != 0;
    if (timing_.partial) {
        roiMask(tw_, th_, T * s, m);
        m = roiMask_.data();
    }
    backend_->compute(FrameView{gray, w, h, stride}, p, m, stats_->dark, stats_->bright,
                      timing_.jobs, timing_.jobs_skipped);

    // грубые масштабы: кадр уменьшается min/max-пулингом, обе полярности считаются
    // как "тёмные" (яркая — на инвертированном 255-max кадре)
    auto& lv = stats_->lv;
    lv.resize(cfg_.scales - 1);
    // между полными просмотрами грубые масштабы нужны, только если по ним ведётся цель
    bool coarseRun = !lv.empty();
    if (coarseRun && timing_.coarsePartial) {
        coarseRun = false;
        for (const FocusArea& a : roi_) coarseRun |= a.level > 0;
    }
    for (size_t l = 0; coarseRun && l < lv.size(); ++l) {
        Level& L = lv[l];
        const int sw = l ? lv[l - 1].w : w, sh = l ? lv[l - 1].h : h;
        L.w = sw / 2;
        L.h = sh / 2;
        // на мелком кадре тайл уменьшается, чтобы по каждой оси было хотя бы 2 тайла
        L.tile = T;
        while (L.tile > 8 && (std::min(L.w, L.h) / s) / L.tile < 2) L.tile /= 2;
        L.tw = (L.w / s) / L.tile;
        L.th = (L.h / s) / L.tile;
        const size_t n = (size_t)L.w * L.h;
        L.d.resize(n);
        L.b.resize(n);
        L.junk.resize(n);
        if (L.tw == 0 || L.th == 0) { L.sd.resize(0); L.sb.resize(0); continue; }
        if (l == 0) {
            backend_->pool2(FrameView{gray, w, h, stride}, L.d.data(), L.b.data(), L.w);
        } else {
            const Level& P = lv[l - 1];
            if (cfg_.dark) backend_->pool2(FrameView{P.d.data(), P.w, P.h, P.w}, L.d.data(), L.junk.data(), L.w);
            if (cfg_.bright) backend_->pool2(FrameView{P.b.data(), P.w, P.h, P.w}, L.b.data(), L.junk.data(), L.w);
        }
        ResponseParams q = p;
        q.tile = L.tile;
        q.dark = true;
        q.bright = false;
        int jobs = 0, skipped = 0;
        const uint8_t* cm = nullptr;
        if (timing_.coarsePartial) {
            roiMask(L.tw, L.th, L.tile * (s << (l + 1)), nullptr);
            cm = roiMask_.data();
        }
        if (cfg_.dark) backend_->compute(FrameView{L.d.data(), L.w, L.h, L.w}, q, cm, L.sd, stats_->unused, jobs, skipped);
        timing_.jobs += jobs;
        timing_.jobs_skipped += skipped;
        if (cfg_.bright) backend_->compute(FrameView{L.b.data(), L.w, L.h, L.w}, q, cm, L.sb, stats_->unused, jobs, skipped);
        timing_.jobs += jobs;
        timing_.jobs_skipped += skipped;
    }
    const double t1 = now_ms();
    if (cfg_.zones) computeZones();
    else zones_.clear();
    markFocus(tw_, th_, T * s);
    if (cfg_.dark) selectTiles(stats_->dark, tw_, th_, T, s, 0, 0, true, out);
    if (cfg_.bright) selectTiles(stats_->bright, tw_, th_, T, s, 1, 0, true, out);
    if (coarseRun) {
        for (size_t l = 0; l < lv.size(); ++l) {
            const Level& L = lv[l];
            if (L.tw == 0 || L.th == 0) continue;
            const int f = s << (l + 1);
            markFocus(L.tw, L.th, L.tile * f);
            const size_t first = out.size();
            if (cfg_.dark) selectTiles(L.sd, L.tw, L.th, L.tile, f, 0, (int)l + 1, false, out);
            if (cfg_.bright) selectTiles(L.sb, L.tw, L.th, L.tile, f, 1, (int)l + 1, false, out);
            // цель, которую уже нашёл более мелкий масштаб, второй раз не отдаём
            const float R = 0.5f * cfg_.k * f;
            size_t keep = first;
            for (size_t i = first; i < out.size(); ++i) {
                const Candidate& c = out[i];
                bool dup = false;
                for (size_t j = 0; j < first && !dup; ++j)
                    dup = out[j].polarity == c.polarity && std::fabs(out[j].x - c.x) < R && std::fabs(out[j].y - c.y) < R;
                // и не ищем там, где маска неба запрещает поиск, и на земле
                const int mx = (int)c.x / (T * s), my = (int)c.y / (T * s);
                const size_t mi = (size_t)my * tw_ + mx;
                if (!dup && !mask_.empty() && !learning_) dup = mx >= tw_ || my >= th_ || !mask_[mi];
                if (!dup && !zones_.empty()) dup = mx >= tw_ || my >= th_ || zones_[mi] == 0;
                if (!dup) {
                    out[keep] = c;
                    out[keep++].edge = !zones_.empty() && zones_[mi] == 1;
                }
            }
            out.resize(keep);
        }
    }
    if (learning_) ++texN_;
    const double t2 = now_ms();
    timing_.response_ms = t1 - t0;
    timing_.select_ms = t2 - t1;
    timing_.total_ms = t2 - t0;
}

// Зоны по тайлам основного масштаба (см. DetectorConfig::zones)
void Detector::computeZones() {
    const size_t n = (size_t)tw_ * th_;
    zones_.assign(n, 0);
    if (!n) return;
    const uint32_t area = (uint32_t)(cfg_.tile * cfg_.tile);
    auto textured = [&](const TileStats& ts, size_t i) {
        return ts.hot.size() == n && (float)div_round_even(ts.hot[i] * 255u, area) / 255.f > cfg_.max_density;
    };
    const bool useMask = !learning_ && !mask_.empty();
    std::vector<uint8_t>& smooth = roiMask_;   // временный буфер
    smooth.assign(n, 0);
    for (size_t i = 0; i < n; ++i)
        smooth[i] = !(useMask && !mask_[i]) && !(cfg_.dark && textured(stats_->dark, i)) &&
                    !(cfg_.bright && textured(stats_->bright, i));
    // земля: крупные связные области текстуры (кроны, дома, земля, край виньетки);
    // мелкие (сама цель, птица, фонарь) — не земля. Остальное — небо.
    const int minGround = std::max(4, (int)(n / 40));
    std::fill(zones_.begin(), zones_.end(), 2);
    std::vector<uint8_t>& seen = focusTiles_;   // тоже временный буфер, markFocus перезапишет
    seen.assign(n, 0);
    for (size_t s0 = 0; s0 < n; ++s0) {
        if (smooth[s0] || seen[s0]) continue;
        fill_.clear();
        fill_.push_back((int)s0);
        seen[s0] = 1;
        for (size_t k = 0; k < fill_.size(); ++k) {
            const int i = fill_[k], x = i % tw_, y = i / tw_;
            for (int yy = std::max(0, y - 1); yy <= std::min(th_ - 1, y + 1); ++yy)
                for (int xx = std::max(0, x - 1); xx <= std::min(tw_ - 1, x + 1); ++xx) {
                    const int j = yy * tw_ + xx;
                    if (!smooth[j] && !seen[j]) { seen[j] = 1; fill_.push_back(j); }
                }
        }
        if ((int)fill_.size() >= minGround)
            for (int i : fill_) zones_[i] = 0;
    }
    // смешанные: небо, у которого в 8-окрестности есть земля
    for (int y = 0; y < th_; ++y)
        for (int x = 0; x < tw_; ++x) {
            const size_t i = (size_t)y * tw_ + x;
            if (zones_[i] != 2) continue;
            bool g = false;
            for (int yy = std::max(0, y - 1); yy <= std::min(th_ - 1, y + 1) && !g; ++yy)
                for (int xx = std::max(0, x - 1); xx <= std::min(tw_ - 1, x + 1) && !g; ++xx)
                    g = zones_[(size_t)yy * tw_ + xx] == 0;
            if (g) zones_[i] = 3;   // пометка, чтобы не размножать смешанные внутри прохода
        }
    for (auto& z : zones_)
        if (z == 3) z = 1;
}

// маска тайлов вокруг треков (и внутри маски неба base, если она есть)
void Detector::roiMask(int tw, int th, int tilePx, const uint8_t* base) {
    roiMask_.assign((size_t)tw * th, 0);
    for (const FocusArea& a : roi_) {
        const int x0 = std::max(0, (int)std::floor((a.x - a.r) / tilePx));
        const int x1 = std::min(tw - 1, (int)std::floor((a.x + a.r) / tilePx));
        const int y0 = std::max(0, (int)std::floor((a.y - a.r) / tilePx));
        const int y1 = std::min(th - 1, (int)std::floor((a.y + a.r) / tilePx));
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const size_t i = (size_t)y * tw + x;
                roiMask_[i] = base ? base[i] : 1;
            }
    }
}

// тайлы, задетые областями прогноза треков
void Detector::markFocus(int tw, int th, int tilePx) {
    focusTiles_.assign((size_t)tw * th, 0);
    for (const FocusArea& a : focus_) {
        const int x0 = std::max(0, (int)std::floor((a.x - a.r) / tilePx));
        const int x1 = std::min(tw - 1, (int)std::floor((a.x + a.r) / tilePx));
        const int y0 = std::max(0, (int)std::floor((a.y - a.r) / tilePx));
        const int y1 = std::min(th - 1, (int)std::floor((a.y + a.r) / tilePx));
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) focusTiles_[(size_t)y * tw + x] = 1;
    }
}

void Detector::selectTiles(const TileStats& ts, int tw, int th, int T, int f, int pol, int lvl, bool fineGrid,
                           std::vector<Candidate>& out) {
    if (ts.sum.size() != (size_t)tw * th) return;
    const uint32_t area = (uint32_t)(T * T);
    // на грубых масштабах крупная цель сама занимает заметную долю тайла: ей прощаем
    // до k*k/2 "горячих" пикселей. Основной масштаб — строго как в detector.py.
    const float maxDens = fineGrid ? cfg_.max_density : cfg_.max_density + 0.5f * cfg_.k * cfg_.k / (float)area;
    const float ck = fineGrid ? 1.f : cfg_.coarse_thr_k;
    const float thr0 = cfg_.thr * ck, thrF = cfg_.thr * ck * cfg_.focus_k;
    const size_t first = out.size();
    const bool useMask = fineGrid && !learning_ && !mask_.empty();
    for (int ty = 0; ty < th; ++ty)
        for (int tx = 0; tx < tw; ++tx) {
            const size_t i = (size_t)ty * tw + tx;
            const float level = (float)std::min<uint32_t>(255, div_round_even(ts.sum[i], area));
            const float dens = (float)div_round_even(ts.hot[i] * 255u, area) / 255.f;
            const bool textured = dens > maxDens;
            if (fineGrid && learning_) texSum_[i] += textured;
            if (useMask && !mask_[i]) continue;
            const uint8_t zone = fineGrid && !zones_.empty() ? zones_[i] : 2;
            if (zone == 0) continue;
            const float zk = zone == 1 ? cfg_.mixed_k : 1.f;
            const float mx = ts.maxv[i];
            const float thr = (focusTiles_[i] ? thrF : thr0) * zk;
            if (textured || !(mx >= thr) || !(mx >= cfg_.clutter_k * level + thr * 0.5f)) continue;
            const int bx = ts.argmax[i] % T, by = ts.argmax[i] / T;
            Candidate c;
            c.x = (float)((tx * T + bx) * f) + (f - 1) * 0.5f;
            c.y = (float)((ty * T + by) * f) + (f - 1) * 0.5f;
            c.contrast = mx;
            c.snr = mx / (level + 1.f);
            c.polarity = pol ? 1 : -1;
            c.level = lvl;
            c.weak = focusTiles_[i] && !(mx >= thr0 * zk && mx >= cfg_.clutter_k * level + thr0 * zk * 0.5f);
            c.edge = zone == 1;
            out.push_back(c);
        }
    if (out.size() - first > (size_t)cfg_.max_cands) {   // слишком много — самые контрастные
        std::stable_sort(out.begin() + first, out.end(),
                         [](const Candidate& a, const Candidate& b) { return a.contrast > b.contrast; });
        out.resize(first + cfg_.max_cands);
    }
}

}  // namespace dronolok
