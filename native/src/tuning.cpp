#include "dronolok/tuning.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

namespace dronolok {

const char* profile_name(Profile p) {
    switch (p) {
        case Profile::Full: return "Полный";
        case Profile::Balanced: return "Сбалансированный";
        case Profile::Economy: return "Экономный";
        case Profile::Minimal: return "Минимальный";
    }
    return "?";
}

AlgoBlocks profile_blocks(Profile p) {
    AlgoBlocks b;
    switch (p) {
        case Profile::Full:   // всё включено, каждый кадр целиком
            b.scales = 3;
            b.coarseEvery = 1;
            break;
        case Profile::Balanced:   // по умолчанию: крупные масштабы через кадр
            break;
        case Profile::Economy:   // Pi 4-5: весь кадр через кадр, 2 масштаба
            b.scales = 2;
            b.fullEvery = 2;
            b.coarseEvery = 4;
            break;
        case Profile::Minimal:   // Pi 3: только точки, только тёмные цели, кадр вдвое меньше
            b.scales = 1;
            b.zones = false;
            b.bothPolarities = false;
            b.fullEvery = 3;
            b.coarseEvery = 4;
            b.resolution = 0.5f;
            break;
    }
    return b;
}

void apply_blocks(const AlgoBlocks& b, DetectorConfig& d, TrackerConfig& t) {
    if (b.scales > 0) d.scales = std::min(4, b.scales);
    d.zones = b.zones;
    d.dark = true;
    d.bright = b.bothPolarities;
    d.full_every = std::max(1, b.fullEvery);
    d.coarse_every = std::max(1, b.coarseEvery);
    t.static_frames = b.staticMemory ? (t.static_frames > 0 ? t.static_frames : 3000) : 0;
}

void LoadGovernor::reset() {
    level_ = since_ = 0;
    avg_ = 0;
    init_ = false;
}

bool LoadGovernor::update(double ms, double budgetMs) {
    if (!(budgetMs > 0) || !(ms >= 0)) return false;
    avg_ = init_ ? avg_ + 0.15 * (ms - avg_) : ms;
    init_ = true;
    ++since_;
    // после смены ступени ждём, пока среднее догонит новое время кадра;
    // вверх по сложности — осторожнее, чтобы не качаться
    int to = level_;
    if (avg_ > 0.9 * budgetMs && since_ >= 8 && level_ < MAX_LEVEL) to = level_ + 1;
    else if (avg_ < 0.5 * budgetMs && since_ >= 45 && level_ > 0) to = level_ - 1;
    if (to == level_) return false;
    level_ = to;
    since_ = 0;
    return true;
}

void LoadGovernor::apply(int level, DetectorConfig& d) {
    if (level >= 1) d.coarse_every = std::max(d.coarse_every, 2);
    if (level >= 2) d.full_every = std::max(d.full_every, 2);
    if (level >= 3) d.coarse_every = std::max(d.coarse_every, 4);
    if (level >= 4) {
        d.full_every = std::max(d.full_every, 3);
        d.scales = std::min(d.scales, 2);
    }
    if (level >= 5) {
        d.full_every = std::max(d.full_every, 4);
        d.scales = 1;
    }
}

const char* LoadGovernor::describe(int level) {
    static const char* t[] = {"без упрощений",
                              "крупные масштабы раз в 2 кадра",
                              "весь кадр раз в 2 кадра",
                              "весь кадр раз в 2, крупные раз в 4",
                              "весь кадр раз в 3, не больше 2 масштабов",
                              "весь кадр раз в 4, 1 масштаб"};
    return t[std::max(0, std::min(MAX_LEVEL, level))];
}

double calibrate_ms_per_mp(int runs) {
    // синтетическое небо 2048x1024: градиент, шум, текстура внизу (кроны) и тёмные точки-цели
    const int w = 2048, h = 1024;
    std::vector<uint8_t> img((size_t)w * h);
    uint32_t r = 12345;
    auto rnd = [&] {
        r = r * 1664525u + 1013904223u;
        return (int)(r >> 24);   // 0..255
    };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int v = 150 + 50 * y / h + (rnd() % 13) - 6;
            if (y > h * 4 / 5) v += (rnd() % 81) - 40;
            img[(size_t)y * w + x] = (uint8_t)std::max(0, std::min(255, v));
        }
    for (int i = 0; i < 40; ++i) {
        const int cx = 8 + rnd() * (w - 16) / 256, cy = 8 + rnd() * (h * 3 / 4) / 256;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) img[(size_t)(cy + dy) * w + cx + dx] -= 30;
    }
    DetectorConfig c;
    c.threads = 1;
    c.scales = 1;
    c.isa = Isa::Scalar;
    for (Isa i : available_isas())
        if (i == Isa::SSE2 || i == Isa::NEON) c.isa = i;
    Detector det(c);
    std::vector<Candidate> out;
    std::vector<double> t;
    runs = std::max(3, runs);
    for (int i = 0; i < runs + 2; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        det.detect(img.data(), w, h, w, out);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (i >= 2) t.push_back(ms);   // первые два — прогрев
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2] / (w * (double)h / 1e6);
}

}  // namespace dronolok
