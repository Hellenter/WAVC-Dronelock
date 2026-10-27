// Выбор кандидатов по статистике тайлов, авто-маска неба, замер времени.
// Логика один в один с PointDetector._peaks из detector/detector.py, включая
// округление OpenCV (INTER_AREA даёт u8 с округлением к ближайшему чётному).
#include <algorithm>
#include <chrono>
#include <cmath>

#include "backend.hpp"
#include "dronolok/detector.hpp"

namespace dronolok {

struct Detector::Stats {
    TileStats dark, bright;
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
    backend_->compute(FrameView{gray, w, h, stride}, p, m, stats_->dark, stats_->bright,
                      timing_.jobs, timing_.jobs_skipped);
    const double t1 = now_ms();
    if (cfg_.dark) selectPolarity(0, out);
    if (cfg_.bright) selectPolarity(1, out);
    if (learning_) ++texN_;
    const double t2 = now_ms();
    timing_.response_ms = t1 - t0;
    timing_.select_ms = t2 - t1;
    timing_.total_ms = t2 - t0;
}

void Detector::selectPolarity(int pol, std::vector<Candidate>& out) {
    const TileStats& ts = pol ? stats_->bright : stats_->dark;
    if (ts.sum.size() != (size_t)tw_ * th_) return;
    const int T = cfg_.tile, s = cfg_.pool;
    const uint32_t area = (uint32_t)(T * T);
    const float thr = cfg_.thr;
    const size_t first = out.size();
    const bool useMask = !learning_ && !mask_.empty();
    for (int ty = 0; ty < th_; ++ty)
        for (int tx = 0; tx < tw_; ++tx) {
            const size_t i = (size_t)ty * tw_ + tx;
            const float level = (float)std::min<uint32_t>(255, div_round_even(ts.sum[i], area));
            const float dens = (float)div_round_even(ts.hot[i] * 255u, area) / 255.f;
            const bool textured = dens > cfg_.max_density;
            if (learning_) texSum_[i] += textured;
            if (useMask && !mask_[i]) continue;
            const float mx = ts.maxv[i];
            if (textured || !(mx >= thr) || !(mx >= cfg_.clutter_k * level + thr * 0.5f)) continue;
            const int bx = ts.argmax[i] % T, by = ts.argmax[i] / T;
            Candidate c;
            c.x = (float)((tx * T + bx) * s) + (s - 1) * 0.5f;
            c.y = (float)((ty * T + by) * s) + (s - 1) * 0.5f;
            c.contrast = mx;
            c.snr = mx / (level + 1.f);
            c.polarity = pol ? 1 : -1;
            out.push_back(c);
        }
    if (out.size() - first > (size_t)cfg_.max_cands) {   // слишком много — самые контрастные
        std::stable_sort(out.begin() + first, out.end(),
                         [](const Candidate& a, const Candidate& b) { return a.contrast > b.contrast; });
        out.resize(first + cfg_.max_cands);
    }
}

}  // namespace dronolok
