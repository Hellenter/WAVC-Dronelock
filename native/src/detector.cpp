// Выбор кандидатов по статистике тайлов, авто-маска неба, замер времени.
// Логика один в один с PointDetector._peaks из detector/detector.py, включая
// округление OpenCV (INTER_AREA даёт u8 с округлением к ближайшему чётному).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "backend.hpp"
#include "dronolok/detector.hpp"
#include "dronolok/fileutil.hpp"

namespace dronolok {

// Уровень грубого масштаба: кадр, уменьшенный в 2^l раз (min и 255-max), и его статистика
struct Level {
    std::vector<uint8_t> d, b, junk;
    int w = 0, h = 0, tw = 0, th = 0, tile = 0;
    TileStats sd, sb;
};

// Фон и накопление по ячейкам основного масштаба (одна полярность)
// Фон ячейки — медиана и 90-й перцентиль её максимального отклика, оценённые по шагам
// (стохастическая оценка квантилей): пролетевшая цель за свои 1-2 с в ячейке сдвигает их
// на доли разброса, а кромка леса, мерцающая постоянно, — до своего настоящего уровня.
// Среднее и дисперсия так не умели: медленный дальний дрон раздувал СКО своей же ячейки.
struct CellBg {
    std::vector<float> q50, q90, A, tmp;
    std::vector<uint8_t> nupd;   // сколько раз ячейка училась (до 255): фон надёжен с 8
    void reset(size_t n) {
        q50.assign(n, 0.f);
        q90.assign(n, 0.f);
        A.assign(n, 0.f);
        tmp.assign(n, 0.f);
        nupd.assign(n, 0);
    }
};

static inline float bg_sd(const CellBg& c, size_t ci) { return (c.q90[ci] - c.q50[ci]) * (1.f / 1.28f); }

struct Detector::Stats {
    TileStats dark, bright, unused;
    TileStats stripD, stripB;    // ряд тайлов по низу кадра для зон (DetectorConfig::zone_strip)
    bool strip = false;          // посчитан в этом кадре
    std::vector<Level> lv;   // lv[0] — уровень 1 (x2), lv[1] — уровень 2 (x4)
    CellBg cb[2];
    std::vector<uint8_t> comp;   // тайлы, посчитанные бэкендом в этом кадре (пусто — все)
    std::vector<uint8_t> excl;   // ячейки вокруг треков: фон не учат
    std::vector<uint8_t> tex[2]; // тайлы, текстурные в этом кадре (облака, кромки), по полярностям
    std::vector<float> e;        // превышение ячеек над фоном в этом кадре (накопление)
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

// Суммарный контраст: центр — самый тёмный (яркий) пиксель возле кандидата (координаты
// кандидата квантованы пулингом), окно (2r+1)^2, фон — среднее по рамке на 2 пикс. дальше
static void measure_energy(const uint8_t* g, int w, int h, int stride, Candidate& c) {
    const int r = 2 << std::min(2, c.level), R = r + 2, srch = 1 << std::min(2, c.level + 1);
    int cx = (int)std::lround(c.x), cy = (int)std::lround(c.y);
    if (cx - R - srch < 0 || cy - R - srch < 0 || cx + R + srch >= w || cy + R + srch >= h) return;
    int best = -1, bx = cx, by = cy;
    for (int y = cy - srch; y <= cy + srch; ++y)
        for (int x = cx - srch; x <= cx + srch; ++x) {
            const int v = c.polarity > 0 ? g[(size_t)y * stride + x] : 255 - g[(size_t)y * stride + x];
            if (v > best) { best = v; bx = x; by = y; }
        }
    cx = bx;
    cy = by;
    int64_t ring = 0;
    for (int x = cx - R; x <= cx + R; ++x) ring += g[(size_t)(cy - R) * stride + x] + g[(size_t)(cy + R) * stride + x];
    for (int y = cy - R + 1; y < cy + R; ++y) ring += g[(size_t)y * stride + cx - R] + g[(size_t)y * stride + cx + R];
    const double bg = (double)ring / (8.0 * R);
    int64_t in = 0;
    for (int y = cy - r; y <= cy + r; ++y) {
        const uint8_t* row = g + (size_t)y * stride;
        for (int x = cx - r; x <= cx + r; ++x) in += row[x];
    }
    const double n = (double)(2 * r + 1) * (2 * r + 1);
    c.energy = (float)std::max(0.0, c.polarity * ((double)in - bg * n));
}

// Уточнение положения цели основного масштаба: в окне 5x5 вокруг кандидата — экстремум
// сглаженного 3x3 кадра, затем вершина по трём точкам гаусса (логарифм амплитуды над фоном
// рамки) вдоль x и y. Амплитуды не положительны (шум, плоское пятно) — по параболе.
static void refine_position(const uint8_t* g, int w, int h, int stride, Candidate& c) {
    if (c.level != 0) return;
    const int R = 5;   // рамка фона
    int cx = (int)std::lround(c.x), cy = (int)std::lround(c.y);
    if (cx - R - 3 < 0 || cy - R - 3 < 0 || cx + R + 3 >= w || cy + R + 3 >= h) return;
    auto sm = [&](int x, int y) {   // сглаженное значение x16
        const uint8_t* a = g + (size_t)(y - 1) * stride + x;
        const uint8_t* b = a + stride;
        const uint8_t* d = b + stride;
        return (int)a[-1] + 2 * a[0] + a[1] + 2 * (b[-1] + 2 * b[0] + b[1]) + d[-1] + 2 * d[0] + d[1];
    };
    int best = c.polarity > 0 ? -1 : 1 << 30, bx = cx, by = cy;
    for (int y = cy - 2; y <= cy + 2; ++y)
        for (int x = cx - 2; x <= cx + 2; ++x) {
            const int v = sm(x, y);
            if (c.polarity > 0 ? v > best : v < best) { best = v; bx = x; by = y; }
        }
    int64_t ring = 0;
    for (int x = bx - R; x <= bx + R; ++x) ring += g[(size_t)(by - R) * stride + x] + g[(size_t)(by + R) * stride + x];
    for (int y = by - R + 1; y < by + R; ++y) ring += g[(size_t)y * stride + bx - R] + g[(size_t)y * stride + bx + R];
    const float bg16 = 16.f * (float)ring / (8.f * R);
    auto amp = [&](int x, int y) { return (float)c.polarity * ((float)sm(x, y) - bg16); };
    auto vertex = [](float m, float z, float p) {   // амплитуды слева, в центре, справа
        if (m > 0 && z > 0 && p > 0) {
            const float lm = std::log(m), lz = std::log(z), lp = std::log(p), den = lm - 2 * lz + lp;
            if (den < -1e-6f) return std::max(-0.75f, std::min(0.75f, 0.5f * (lm - lp) / den));
        }
        const float den = m - 2 * z + p;
        return den < -1e-6f ? std::max(-0.75f, std::min(0.75f, 0.5f * (m - p) / den)) : 0.f;
    };
    const float a0 = amp(bx, by);
    if (a0 <= 0) return;
    c.x = (float)bx + vertex(amp(bx - 1, by), a0, amp(bx + 1, by));
    c.y = (float)by + vertex(amp(bx, by - 1), a0, amp(bx, by + 1));
}

// Ширина пятна кандидата основного масштаба: амплитуда сглаженного кадра в 2 пикс. от
// пика (среднее по 4 сторонам) к амплитуде пика. Точка с размытием оптики (FWHM до ~3
// пикс.) — 0.2-0.4, пятно облака шире 5 пикс. — 0.6 и выше. -1 — не измерить.
static float spot_width(const uint8_t* g, int w, int h, int stride, const Candidate& c) {
    if (c.level != 0) return -1.f;
    const int R = 6;   // рамка фона
    const int cx = (int)std::lround(c.x), cy = (int)std::lround(c.y);
    if (cx - R - 4 < 0 || cy - R - 4 < 0 || cx + R + 4 >= w || cy + R + 4 >= h) return -1.f;
    auto sm = [&](int x, int y) {   // сглаженное значение x16
        const uint8_t* a = g + (size_t)(y - 1) * stride + x;
        const uint8_t* b = a + stride;
        const uint8_t* d = b + stride;
        return (int)a[-1] + 2 * a[0] + a[1] + 2 * (b[-1] + 2 * b[0] + b[1]) + d[-1] + 2 * d[0] + d[1];
    };
    int best = c.polarity > 0 ? -1 : 1 << 30, bx = cx, by = cy;
    for (int y = cy - 2; y <= cy + 2; ++y)
        for (int x = cx - 2; x <= cx + 2; ++x) {
            const int v = sm(x, y);
            if (c.polarity > 0 ? v > best : v < best) { best = v; bx = x; by = y; }
        }
    // фон — медиана рамки (облако или кромка на краю рамки не сдвигают его)
    int ring[8 * R], n = 0;
    for (int x = bx - R; x < bx + R; ++x) { ring[n++] = sm(x, by - R); ring[n++] = sm(x + 1, by + R); }
    for (int y = by - R; y < by + R; ++y) { ring[n++] = sm(bx - R, y + 1); ring[n++] = sm(bx + R, y); }
    std::nth_element(ring, ring + n / 2, ring + n);
    const float bg = (float)ring[n / 2];
    auto amp = [&](int x, int y) { return (float)c.polarity * ((float)sm(x, y) - bg); };
    const float a0 = amp(bx, by);
    if (a0 <= 0) return -1.f;
    return 0.25f * (amp(bx - 2, by) + amp(bx + 2, by) + amp(bx, by - 2) + amp(bx, by + 2)) / a0;
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
    c.noise_k = std::max(1.f, std::min(20.f, c.noise_k));
    c.thr_min = std::max(1.f, c.thr_min);
    c.thr_max = std::max(c.thr_min, c.thr_max);
    c.per_tile = std::max(1, std::min(4, c.per_tile));
    c.bg_k = std::max(0.5f, c.bg_k);
    c.bg_frames = std::max(2, c.bg_frames);
    c.tbd_decay = std::max(0.f, std::min(0.95f, c.tbd_decay));
    c.band_px = std::max(0, c.band_px);
    c.band_k = std::max(0.3f, std::min(2.f, c.band_k));
    c.upper_every = std::max(1, std::min(16, c.upper_every));
    c.smooth_k = std::max(0.1f, std::min(1.f, c.smooth_k));
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
    for (auto& t : texSumP_) t.assign((size_t)tw_ * th_, 0.f);
    texN_ = 0;
    resetGrow();
}

void Detector::endMaskLearning(float frac) {
    learning_ = false;
    resetGrow();
    std::vector<uint8_t> m;
    buildMask(frac, m);
    if (!m.empty()) mask_.swap(m);
    markTexBad(frac);
}

// тайлы, закрытые маской по своей текстуре (не линией горизонта): их маска перепроверяет
void Detector::markTexBad(float frac) {
    texBad_.clear();
    recheckTex_.clear();
    recheckN_.clear();
    const int n = texN_ * ((int)cfg_.dark + (int)cfg_.bright);
    if (n == 0 || texSum_.size() != mask_.size()) return;
    texBad_.resize(mask_.size());
    for (size_t i = 0; i < texBad_.size(); ++i) texBad_[i] = !mask_[i] && texSum_[i] / n > frac;
    recheckTex_.assign(mask_.size(), 0.f);
    recheckN_.assign(mask_.size(), 0);
    // маска по полярности: тайл закрыт своей текстурой, а этой полярности в нём почти не было
    for (auto& v : polOpen_) v.clear();
    if (cfg_.pol_mask <= 0 || !cfg_.dark || !cfg_.bright) return;
    for (int p = 0; p < (cfg_.pol_mask >= 2 ? 2 : 1); ++p) {
        const auto& tp = texSumP_[p];
        if (tp.size() != mask_.size()) continue;
        polOpen_[p].assign(mask_.size(), 0);
        for (size_t i = 0; i < tp.size(); ++i) polOpen_[p][i] = texBad_[i] && tp[i] / texN_ <= cfg_.pol_tex;
    }
}

// Маска по накопленной текстуре тайлов (texSum_ за texN_ кадров): тайл, текстурный дольше
// доли frac кадров, — земля; вокруг неё ещё тайл запаса. Пусто — копить было нечего.
void Detector::buildMask(float frac, std::vector<uint8_t>& out) const {
    out.clear();
    const int n = texN_ * ((int)cfg_.dark + (int)cfg_.bright);
    if (n == 0 || texSum_.size() != (size_t)tw_ * th_) return;
    std::vector<uint8_t> bad(texSum_.size());
    for (size_t i = 0; i < bad.size(); ++i) bad[i] = texSum_[i] / n > frac;
    // кромка земли — по тайлам с тёмной текстурой: тайл только со светлой (облако, pol_mask)
    // сам закрыт, но столбец под ним землёй не делает — иначе облачное небо до самого леса
    // закрывалось целиком, вместе с гладкими просветами между облаками
    std::vector<uint8_t> edgeBad(bad);
    if (cfg_.pol_mask > 0 && cfg_.dark && cfg_.bright && texN_ > 0 && texSumP_[0].size() == bad.size())
        for (size_t i = 0; i < bad.size(); ++i) edgeBad[i] = bad[i] && texSumP_[0][i] / texN_ > cfg_.pol_tex;
    if (cfg_.skyline) {
        std::vector<int> edge(tw_, th_);
        for (int x = 0; x < tw_; ++x) {
            int below = 0;   // текстурных тайлов от y до низа кадра
            int gap = 0;     // гладких тайлов подряд над землёй
            for (int y = th_ - 1; y >= 0; --y) {
                const bool b = edgeBad[(size_t)y * tw_ + x] != 0;
                below += b;
                gap = b ? 0 : gap + 1;
                // кромка не выше просвета в 3 тайла: облако над лесом (тёмное небо, яркие
                // облака — wt7) — не продолжение крон, и столбец под ним не земля
                if (below && gap > 2) break;
                if (b && 2 * below > th_ - y) edge[x] = y;
            }
        }
        // столбец, где кромки не нашлось или она много ниже, чем у обоих соседей (гладкая
        // поляна в просвете между деревьями), берёт кромку пониже из соседних
        for (int x = 0; x < tw_; ++x) {
            const int l = x > 0 ? edge[x - 1] : edge[x + (tw_ > 1)], r = x + 1 < tw_ ? edge[x + 1] : edge[x - (tw_ > 1)];
            const int e = std::min(edge[x], std::max(l, r));
            for (int y = e; y < th_; ++y) bad[(size_t)y * tw_ + x] = 1;
        }
    }
    // расширяем "плохую" зону на 1 тайл (dilate 3x3). С фоном по ячейкам запас не нужен:
    // тайлы у кромки смотрятся по ячейкам, и мерцающие ячейки сами поднимают себе порог
    out.assign(bad.size(), 1);
    if (cfg_.bg_map) {
        for (size_t i = 0; i < bad.size(); ++i) out[i] = !bad[i];
        return;
    }
    for (int y = 0; y < th_; ++y)
        for (int x = 0; x < tw_; ++x) {
            bool b = false;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int yy = y + dy, xx = x + dx;
                    if (yy >= 0 && yy < th_ && xx >= 0 && xx < tw_) b |= bad[(size_t)yy * tw_ + xx] != 0;
                }
            out[(size_t)y * tw_ + x] = !b;
        }
}

namespace {
const char kMaskMagic[8] = {'D', 'L', 'M', 'A', 'S', 'K', '1', 0};
const int32_t kMaxSide = 1 << 16;   // сторона кадра в файле маски, пикс.
}

bool Detector::saveMask(const std::string& path) const {
    if (mask_.empty() || mask_.size() != (size_t)tw_ * th_) return false;
    FILE* f = open_file(path, "wb");
    if (!f) return false;
    const int32_t hdr[6] = {w_, h_, cfg_.pool, cfg_.tile, tw_, th_};
    bool ok = std::fwrite(kMaskMagic, 1, 8, f) == 8 && std::fwrite(hdr, sizeof(hdr), 1, f) == 1 &&
              std::fwrite(mask_.data(), 1, mask_.size(), f) == mask_.size();
    ok = std::fclose(f) == 0 && ok;
    return ok;
}

bool Detector::loadMask(const std::string& path, int verify) {
    FILE* f = open_file(path, "rb");
    if (!f) return false;
    char magic[8];
    int32_t hdr[6];
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, kMaskMagic, 8) == 0 &&
              std::fread(hdr, sizeof(hdr), 1, f) == 1 && hdr[2] == cfg_.pool && hdr[3] == cfg_.tile && hdr[4] > 0 &&
              hdr[5] > 0 && hdr[4] == (hdr[0] / cfg_.pool) / cfg_.tile && hdr[5] == (hdr[1] / cfg_.pool) / cfg_.tile &&
              // размер кадра из файла ограничен: испорченный заголовок не выделяет гигабайты под маску
              hdr[0] <= kMaxSide && hdr[1] <= kMaxSide;
    std::vector<uint8_t> m;
    if (ok) {
        m.resize((size_t)hdr[4] * hdr[5]);
        ok = std::fread(m.data(), 1, m.size(), f) == m.size();
    }
    std::fclose(f);
    if (!ok) return false;
    w_ = hdr[0];
    h_ = hdr[1];
    tw_ = hdr[4];
    th_ = hdr[5];
    mask_.swap(m);
    learning_ = false;
    maskReplaced_ = false;
    verifyLeft_ = std::max(0, verify);
    texSum_.assign((size_t)tw_ * th_, 0.f);
    for (auto& t : texSumP_) t.assign((size_t)tw_ * th_, 0.f);
    texN_ = 0;
    resetGrow();
    return true;
}

void Detector::resetBackground() {
    stats_->cb[0].reset(0);
    stats_->cb[1].reset(0);
    thrEff_ = 0;
}

void Detector::clearMask() {
    mask_.clear();
    learning_ = false;
    verifyLeft_ = 0;
    resetGrow();
}

void Detector::resetGrow() {
    growCell_.clear();
    growCellD_.clear();
    growTex_.clear();
    growN_ = 0;
    grown_ = 0;
    opened_ = 0;
    cloud_.clear();
    texBad_.clear();
    recheckTex_.clear();
    recheckN_.clear();
    for (auto& v : polOpen_) v.clear();
    grownT_.clear();
    growQuiet_.clear();
    growQN_.clear();
    reopened_ = 0;
}

// Перепроверка маски (DetectorConfig::mask_recheck): тайлы, закрытые по текстуре, раз в
// recheck_every кадров считаются; тайл, где за ~recheck_frames кадров сильный отклик бывал реже
// recheck_tex, открывается, если выше него в столбце нет земли — это облако
void Detector::updateRecheck() {
    const size_t n = mask_.size();
    const int cpt = cfg_.tile / CELL, ctw = tw_ * cpt;
    const auto& dk = stats_->dark.cmax;
    const auto& br = stats_->bright.cmax;
    const size_t nc = (size_t)ctw * th_ * cpt;
    if (dk.size() != nc && br.size() != nc) return;
    const int need = std::max(1, cfg_.recheck_frames / std::max(1, cfg_.recheck_every));
    const float a = 1.f / (float)need, thr = threshold();
    // закрытые доращиванием: если помехам там больше неоткуда взяться — ни одна ячейка закрытой
    // полярности не дотягивает до порога почти во всех замерах за вдвое большее окно, — открываем
    if (grownT_.size() == n) {
        const int needG = 2 * need;
        const float ag = 1.f / (float)needG;
        if (growQuiet_.size() != n) growQuiet_.assign(n, 0.f), growQN_.assign(n, 0);
        for (size_t i = 0; i < n; ++i) {
            if (!grownT_[i] || mask_[i]) {
                grownT_[i] = 0;
                continue;
            }
            const bool brightOnly = polOpenAt(0, i);   // закрыт только для светлых
            const int ty = (int)(i / tw_), tx = (int)(i % tw_);
            bool loud = false;
            for (int cy = ty * cpt; cy < (ty + 1) * cpt && !loud; ++cy)
                for (int cx = tx * cpt; cx < (tx + 1) * cpt && !loud; ++cx) {
                    const size_t ci = (size_t)cy * ctw + cx;
                    loud = (br.size() == nc && br[ci] >= thr) || (!brightOnly && dk.size() == nc && dk[ci] >= thr);
                }
            if (growQN_[i] < 65535) ++growQN_[i];
            growQuiet_[i] += std::max(ag, 1.f / (float)growQN_[i]) * ((loud ? 0.f : 1.f) - growQuiet_[i]);
            if (growQN_[i] < needG || growQuiet_[i] < cfg_.reopen_quiet) continue;
            mask_[i] = 1;
            if (fineBase_.size() == n) fineBase_[i] = 1;
            for (auto& o : polOpen_)
                if (o.size() == n) o[i] = 0;
            grownT_[i] = 0;
            growQN_[i] = 0;
            growQuiet_[i] = 0;
            ++reopened_;
        }
    }
    if (texBad_.size() != n) return;
    uint8_t v[64];
    for (size_t i = 0; i < n; ++i) {
        if (!texBad_[i] || mask_[i]) continue;
        // медиана максимумов ячеек тайла: у кромки облака отклик слабый почти везде, у крон,
        // домов и травы — сильный в большинстве ячеек
        const int ty = (int)(i / tw_), tx = (int)(i % tw_);
        int k = 0;
        for (int cy = ty * cpt; cy < (ty + 1) * cpt; ++cy)
            for (int cx = tx * cpt; cx < (tx + 1) * cpt && k < 64; ++cx) {
                const size_t ci = (size_t)cy * ctw + cx;
                v[k++] = std::max(dk.size() == nc ? dk[ci] : (uint8_t)0, br.size() == nc ? br[ci] : (uint8_t)0);
            }
        std::nth_element(v, v + k / 2, v + k);
        const bool strong = v[k / 2] >= thr;
        if (recheckN_[i] < 65535) ++recheckN_[i];
        recheckTex_[i] += std::max(a, 1.f / (float)recheckN_[i]) * ((strong ? 1.f : 0.f) - recheckTex_[i]);
    }
    // открываем только выше самой верхней "сильной" земли столбца: гладкая трава под кромкой
    // леса тоже слаба, но она земля (её закрыла линия горизонта или своя текстура)
    for (int x = 0; x < tw_; ++x) {
        int top = th_;
        for (int y = 0; y < th_ && top == th_; ++y) {
            const size_t i = (size_t)y * tw_ + x;
            if (!mask_[i] && (!texBad_[i] || recheckN_[i] < need || recheckTex_[i] >= cfg_.recheck_tex)) top = y;
        }
        for (int y = 0; y < top; ++y) {
            const size_t i = (size_t)y * tw_ + x;
            if (mask_[i] || !texBad_[i]) continue;
            mask_[i] = 1;
            if (fineBase_.size() == n) fineBase_[i] = 1;
            texBad_[i] = 0;
            if (cloud_.size() != n) cloud_.assign(n, 0);
            cloud_[i] = 1;
            ++opened_;
        }
    }
}

// Доращивание маски по помехам (DetectorConfig::grow_mask): помехи копятся по ячейкам с
// забыванием за ~grow_frames кадров, раз в четверть окна тайлы неба перепроверяются
void Detector::noteClutter(const std::vector<Candidate>& clutter, const std::vector<FocusArea>& keep) {
    const int cpt = cfg_.tile / CELL, ctw = tw_ * cpt, cth = th_ * cpt;
    if (!cfg_.grow_mask || learning_ || verifyLeft_ || cfg_.tile % CELL || mask_.size() != (size_t)tw_ * th_) {
        if (!growCell_.empty()) growCell_.clear(), growCellD_.clear(), growTex_.clear(), growN_ = 0;
        return;
    }
    const size_t nc = (size_t)ctw * cth, n = (size_t)tw_ * th_;
    if (growCell_.size() != nc || growCellD_.size() != nc || growTex_.size() != n) {
        growCell_.assign(nc, 0.f);
        growCellD_.assign(nc, 0.f);
        growTex_.assign(n, 0.f);
        growN_ = 0;
    }
    const int W = std::max(10, cfg_.grow_frames);
    const float k = 1.f - 1.f / (float)W;
    for (float& v : growCell_) v *= k;
    for (float& v : growCellD_) v *= k;
    // как часто тайл текстурный: дальняя трава и поле — иногда, чистое небо — почти никогда
    const auto& t0 = stats_->tex[0];
    const auto& t1 = stats_->tex[1];
    for (size_t i = 0; i < n; ++i) {
        const bool t = (t0.size() == n && t0[i]) || (t1.size() == n && t1[i]);
        growTex_[i] = k * growTex_[i] + (t ? 1.f / (float)W : 0.f);
    }
    const float cpx = (float)(CELL * cfg_.pool);
    for (const Candidate& c : clutter) {
        const int cx = (int)(c.x / cpx), cy = (int)(c.y / cpx);
        if (cx >= 0 && cx < ctw && cy >= 0 && cy < cth) {
            growCell_[(size_t)cy * ctw + cx] += 1.f;
            if (c.polarity < 0) growCellD_[(size_t)cy * ctw + cx] += 1.f;
        }
    }
    ++growN_;
    if (growN_ >= W && growN_ % std::max(1, W / 4) == 0) growMask(keep);
}

void Detector::growMask(const std::vector<FocusArea>& keep) {
    const int cpt = cfg_.tile / CELL, ctw = tw_ * cpt;
    const size_t n = (size_t)tw_ * th_;
    const float W = (float)std::max(10, cfg_.grow_frames);
    std::vector<float> rate(n, 0.f), rateD(n, 0.f);   // ...и только тёмных
    std::vector<uint8_t> cells(n, 0), cand(n, 0);
    double sum = 0;
    int sky = 0;
    for (int ty = 0; ty < th_; ++ty)
        for (int tx = 0; tx < tw_; ++tx) {
            const size_t i = (size_t)ty * tw_ + tx;
            if (!mask_[i] && !polOpenAt(0, i) && !polOpenAt(1, i)) continue;
            float r = 0, rd = 0;
            int c = 0;
            for (int cy = ty * cpt; cy < (ty + 1) * cpt; ++cy)
                for (int cx = tx * cpt; cx < (tx + 1) * cpt; ++cx) {
                    const float v = growCell_[(size_t)cy * ctw + cx];
                    r += v;
                    rd += growCellD_[(size_t)cy * ctw + cx];
                    c += v > 0.5f;
                }
            rate[i] = r / W;
            rateD[i] = rd / W;
            cells[i] = (uint8_t)std::min(255, c);
            sum += rate[i];
            ++sky;
        }
    if (!sky) return;
    // помех на тайл неба в среднем по сцене: в чистом небе их почти нет, в дыре маски — десятки
    const float lim = std::max(cfg_.grow_k * (float)(sum / sky), cfg_.grow_min);
    const int tilePx = cfg_.tile * cfg_.pool;
    for (size_t i = 0; i < n; ++i)
        cand[i] = (mask_[i] || polOpenAt(0, i) || polOpenAt(1, i)) && rate[i] >= lim && cells[i] >= cfg_.grow_cells;
    for (const FocusArea& a : keep) {   // вокруг цели маска не растёт
        const int x0 = std::max(0, (int)std::floor((a.x - a.r) / tilePx)), x1 = std::min(tw_ - 1, (int)std::floor((a.x + a.r) / tilePx));
        const int y0 = std::max(0, (int)std::floor((a.y - a.r) / tilePx)), y1 = std::min(th_ - 1, (int)std::floor((a.y + a.r) / tilePx));
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) cand[(size_t)y * tw_ + x] = 0;
    }
    // открытый для одной полярности тайл уже внутри маски: закрывается без условий
    // (светлых кандидатов там нет, помехи в нём — тёмные)
    for (size_t i = 0; i < n; ++i) {
        if (mask_[i] || !cand[i]) continue;
        for (auto& v : polOpen_)
            if (v.size() == n) v[i] = 0;
        cand[i] = 0;
        ++grown_;
        const int x = (int)(i % tw_), y = (int)(i / tw_);
        for (int cy = y * cpt; cy < (y + 1) * cpt; ++cy) {
            std::fill_n(&growCell_[(size_t)cy * ctw + x * cpt], cpt, 0.f);
            std::fill_n(&growCellD_[(size_t)cy * ctw + x * cpt], cpt, 0.f);
        }
    }
    // прирастает к маске: земля хотя бы у двух соседей (край кадра слева, справа и снизу —
    // тоже), и над тайлом не чистое небо (земля или такой же тайл с помехами). Ряд тайлов над
    // ровной кромкой крон соседа-землю имеет одного — снизу, и не закрывается
    // земля — только связанная с низом кадра: к островку маски в небе (облако) не прирастаем
    std::vector<uint8_t> base(n, 0);
    fill_.clear();
    for (int x = 0; x < tw_; ++x) {
        const size_t i = (size_t)(th_ - 1) * tw_ + x;
        if (!mask_[i]) base[i] = 1, fill_.push_back((int)i);
    }
    for (size_t k = 0; k < fill_.size(); ++k) {
        const int i = fill_[k], x = i % tw_, y = i / tw_;
        const int nb[4][2] = {{x - 1, y}, {x + 1, y}, {x, y - 1}, {x, y + 1}};
        for (const auto& q : nb) {
            if (q[0] < 0 || q[0] >= tw_ || q[1] < 0 || q[1] >= th_) continue;
            const int j = q[1] * tw_ + q[0];
            if (!mask_[j] && !base[j]) base[j] = 1, fill_.push_back(j);
        }
    }
    auto ground = [&](int x, int y) {
        if (y < 0) return false;
        if (x < 0 || x >= tw_ || y >= th_) return true;
        return base[(size_t)y * tw_ + x] != 0;
    };
    // чистое небо: помех мало и текстуры почти не бывает
    auto clean = [&](size_t j) { return mask_[j] && !cand[j] && rate[j] < lim && growTex_[j] < cfg_.grow_sky_tex; };
    const bool polGrow = cfg_.pol_mask > 0 && cfg_.dark && cfg_.bright;
    bool changed = true;
    while (changed) {
        changed = false;
        for (int y = 1; y < th_; ++y)
            for (int x = 0; x < tw_; ++x) {
                const size_t i = (size_t)y * tw_ + x;
                if (!cand[i]) continue;
                const int g = ground(x - 1, y) + ground(x + 1, y) + ground(x, y + 1) + ground(x, y - 1);
                if (g < 2 || (g == 2 && clean(i - tw_))) continue;
                mask_[i] = 0;
                if (grownT_.size() != n) grownT_.assign(n, 0);
                grownT_[i] = 1;
                if (growQN_.size() == n) growQN_[i] = 0, growQuiet_[i] = 0;
                // помехи светлые (клочья облака): тайл закрыт только для светлых, тёмные там ищутся
                if (polGrow && rateD[i] < lim) {
                    if (polOpen_[0].size() != n) polOpen_[0].assign(n, 0);
                    polOpen_[0][i] = 1;
                }
                base[i] = 1;
                if (fineBase_.size() == n) fineBase_[i] = 0;   // уточнение по ячейкам у кромки не сбрасываем
                cand[i] = 0;
                ++grown_;
                changed = true;
                for (int cy = y * cpt; cy < (y + 1) * cpt; ++cy) {
                    std::fill_n(&growCell_[(size_t)cy * ctw + x * cpt], cpt, 0.f);
                    std::fill_n(&growCellD_[(size_t)cy * ctw + x * cpt], cpt, 0.f);
                }
            }
    }
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
        verifyLeft_ = 0;
        if (learning_) {
            texSum_.assign((size_t)tw_ * th_, 0.f);
            for (auto& t : texSumP_) t.assign((size_t)tw_ * th_, 0.f);
        }
    }
    // как cv2.threshold(R, thr*0.6) на u8: горячий, если R > floor(thr*0.6)
    const float ht = std::floor(hotLevel());
    ResponseParams p;
    p.k = cfg_.k;
    p.pool = s;
    p.tile = T;
    p.hot_thr = (uint8_t)std::max(0.f, std::min(255.f, ht));
    p.dark = cfg_.dark;
    p.bright = cfg_.bright;
    p.job_rows = cfg_.job_rows;
    p.job_cols = cfg_.job_cols;
    p.cells = (cfg_.per_tile > 1 || cfg_.bg_map || cfg_.tbd) && T % CELL == 0;
    p.smooth = cfg_.smooth;
    {   // место максимума ячейки нужно, если она может дать кандидата: от пониженных порогов
        const float lo = threshold() * std::min({1.f, cfg_.tbd ? cfg_.tbd_min : 1.f, cfg_.focus_k, cfg_.band_k});
        p.cell_thr = (uint8_t)std::max(1.f, std::min(254.f, std::floor(lo) - 1.f));
    }
    // при проверке загруженной маски текстуру считаем по всему кадру, как при обучении
    const uint8_t* m = (!learning_ && !verifyLeft_ && !mask_.empty()) ? mask_.data() : nullptr;
    updateBand();
    const bool polOn = polOpen_[0].size() == mask_.size() || polOpen_[1].size() == mask_.size();
    const bool lockOn = cfg_.lock_through && lockOn_ && m;
    if (m && (edgeG_.size() == mask_.size() || polOn || lockOn)) {
        // тайлы земли у кромки тоже считаем: маска уточняется по ячейкам; и открытые для одной
        // полярности; и под кругом цели в захвате
        compMask_.resize(mask_.size());
        for (size_t i = 0; i < mask_.size(); ++i)
            compMask_[i] = mask_[i] | (edgeG_.size() == mask_.size() && edgeG_[i]) | polOpenAt(0, i) | polOpenAt(1, i) |
                           (lockOn && lockTile((int)(i % tw_), (int)(i / tw_)));
        m = compMask_.data();
    }
    // между полными просмотрами — только тайлы вокруг треков (маска неба тоже действует);
    // полоса у горизонта — всегда, небо выше неё — раз в upper_every кадров
    const long fn = frameNo_++;
    timing_.partial = !learning_ && cfg_.full_every > 1 && fn % cfg_.full_every != 0;
    timing_.upperSkipped = !band_.empty() && m && cfg_.upper_every > 1 && fn % cfg_.upper_every != 0;
    timing_.coarsePartial = cfg_.coarse_every > 1 && fn % cfg_.coarse_every != 0;
    if (timing_.partial || timing_.upperSkipped) {
        roiMask(tw_, th_, T * s, m);
        if (!band_.empty())
            for (size_t i = 0; i < roiMask_.size(); ++i)
                if (band_[i] && (!m || m[i])) roiMask_[i] = 1;
        m = roiMask_.data();
        timing_.partial = true;
    }
    // перепроверка маски: в этот кадр считаются и тайлы, закрытые по текстуре
    const bool recheck = m && !timing_.partial && cfg_.mask_recheck && cfg_.recheck_every > 0 &&
                         (texBad_.size() == mask_.size() || grownT_.size() == mask_.size()) && fn % cfg_.recheck_every == 0;
    if (recheck) {
        recheckMask_.assign(m, m + mask_.size());
        for (size_t i = 0; i < mask_.size(); ++i)
            recheckMask_[i] |= (texBad_.size() == mask_.size() && texBad_[i]) || (grownT_.size() == mask_.size() && grownT_[i]);
        m = recheckMask_.data();
    }
    if (m) stats_->comp.assign(m, m + (size_t)tw_ * th_);
    else stats_->comp.clear();
    // полный кадр по маске неба: закрытые тайлы не читаются (кроме кромки, она в compMask_)
    p.trim = m && !timing_.partial;
    backend_->compute(FrameView{gray, w, h, stride}, p, m, stats_->dark, stats_->bright,
                      timing_.jobs, timing_.jobs_skipped);
    if (cfg_.auto_thr && !timing_.partial) {
        const bool first = thrEff_ <= 0;
        updateNoise();
        // первый кадр: порога по шуму ещё не было — пересчитать с ним, иначе маска неба
        // и текстура этого кадра посчитаны со случайным порогом
        const uint8_t ht2 = (uint8_t)std::max(0.f, std::min(255.f, std::floor(hotLevel())));
        if (first && ht2 != p.hot_thr) {
            p.hot_thr = ht2;
            backend_->compute(FrameView{gray, w, h, stride}, p, m, stats_->dark, stats_->bright,
                              timing_.jobs, timing_.jobs_skipped);
        }
    }

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
        q.cells = false;   // ячейки грубых масштабов никто не читает (selectTiles берёт их только с основного)
        q.trim = false;
        if (q.smooth) {   // грубые масштабы — без сглаживания, "горячий" пиксель в уровнях яркости
            q.smooth = false;
            q.hot_thr = (uint8_t)std::max(0.f, std::min(255.f, std::floor(hotLevel() / smoothK())));
        }
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
    // полоса под сеткой тайлов: ряд тайлов, выровненный по низу кадра, — только для зон
    stats_->strip = false;
    if (cfg_.zones && cfg_.zone_strip && !timing_.partial && tw_ > 0 && th_ > 0 && h - th_ * T * s >= 2 * CELL * s) {
        ResponseParams q = p;
        q.cells = false;
        q.trim = false;
        q.hot_thr = (uint8_t)std::max(0.f, std::min(255.f, std::floor(hotLevel())));
        int jobs = 0, skipped = 0;
        backend_->compute(FrameView{gray + (size_t)(h - T * s) * stride, w, T * s, stride}, q, nullptr, stats_->stripD,
                          stats_->stripB, jobs, skipped);
        timing_.jobs += jobs;
        timing_.jobs_skipped += skipped;
        stats_->strip = stats_->stripD.hot.size() == (size_t)tw_ || stats_->stripB.hot.size() == (size_t)tw_;
    }
    const double t1 = now_ms();
    if (cfg_.zones) computeZones();
    else zones_.clear(), zoneHist_.clear();
    markFocus(tw_, th_, T * s);
    if (cfg_.dark) selectTiles(stats_->dark, tw_, th_, T, s, 0, 0, true, out);
    if (cfg_.bright) selectTiles(stats_->bright, tw_, th_, T, s, 1, 0, true, out);
    if (cfg_.lock_through && lockOn_ && !learning_ && !mask_.empty() && (lockPol_ > 0 ? cfg_.bright : cfg_.dark))
        selectThrough(lockPol_ > 0 ? stats_->bright : stats_->dark, lockPol_ > 0 ? 1 : 0, out);
    // фон ячеек читают кромка маски неба и накопление (tbd). Без маски и без накопления (камера
    // движется, поворотка) его не учим — это заметная доля последовательной части кадра; когда
    // маска появится, фон учится заново: выученный на движущейся картинке ей всё равно чужой
    const bool bgUsed = cfg_.tbd || learning_ || verifyLeft_ || !mask_.empty();
    if (p.cells && cfg_.bg_map && !bgUsed) {
        bgIdle_ = true;
    } else if (p.cells && (cfg_.bg_map || cfg_.tbd)) {
        if (bgIdle_) {
            stats_->cb[0].reset(0);
            stats_->cb[1].reset(0);
            bgIdle_ = false;
        }
        markExcl();
        for (int pol = 0; pol < 2; ++pol) {
            if (!(pol ? cfg_.bright : cfg_.dark)) continue;
            const TileStats& ts = pol ? stats_->bright : stats_->dark;
            if (cfg_.tbd && !learning_) selectTbd(ts, pol, out);
            updateBg(ts, pol);
        }
        if (!timing_.partial) updateFine();
    }
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
                if (!dup && !mask_.empty() && !learning_)
                    dup = mx >= tw_ || my >= th_ || !(mask_[mi] || polOpenAt(c.polarity > 0, mi));
                if (!dup && !zones_.empty()) dup = mx >= tw_ || my >= th_ || zones_[mi] == 0;
                if (!dup) {
                    out[keep] = c;
                    out[keep++].edge = !zones_.empty() && zones_[mi] == 1;
                }
            }
            out.resize(keep);
        }
    }
    if (!learning_) {
        // Сглаживание выгоднее всего широким пятнам: облако или блик шириной 5+ пикс. с
        // контрастом ниже обычного порога проходит половинный. Бонус сглаживания — только
        // компактным (точка с размытием оптики), широкое пятно должно пройти полный порог
        // (как без сглаживания), между ними — плавно. Слабым у прогноза трека не мешаем.
        if (cfg_.smooth && cfg_.smooth_compact) {
            const float k = smoothK();
            size_t keep = 0;
            for (size_t i = 0; i < out.size(); ++i) {
                Candidate c = out[i];
                const float r = c.weak ? -1.f : spot_width(gray, w, h, stride, c);
                if (r > cfg_.compact_r0) {
                    const float t = std::min(1.f, (r - cfg_.compact_r0) / std::max(0.01f, cfg_.compact_r1 - cfg_.compact_r0));
                    c.margin *= 1.f / (1.f + t * (1.f / k - 1.f));
                    if (c.margin < 1.f) continue;
                }
                out[keep++] = c;
            }
            out.resize(keep);
        }
        for (Candidate& c : out) {
            measure_energy(gray, w, h, stride, c);
            if (cfg_.subpixel) refine_position(gray, w, h, stride, c);
        }
    }
    if (learning_ || verifyLeft_) ++texN_;
    if (recheck) updateRecheck();
    if (verifyLeft_ && --verifyLeft_ == 0) {   // сверка загруженной маски с тем, что видно сейчас
        std::vector<uint8_t> fresh;
        buildMask(0.3f, fresh);
        if (fresh.size() == mask_.size()) {
            size_t diff = 0;
            for (size_t i = 0; i < fresh.size(); ++i) diff += fresh[i] != mask_[i];
            if (diff * 10 > fresh.size()) {
                mask_.swap(fresh);
                maskReplaced_ = true;
            }
            markTexBad(0.3f);   // и для загруженной маски: перепроверка и маска по полярности
        }
    }
    const double t2 = now_ms();
    timing_.response_ms = t1 - t0;
    timing_.select_ms = t2 - t1;
    timing_.total_ms = t2 - t0;
}

// Шум отклика: медиана среднего отклика по посчитанным тайлам основного масштаба.
// Лес, облака и цели занимают меньшую часть кадра, медиана их не видит. Сглаживание
// по кадрам — чтобы порог не дёргался от кадра к кадру.
void Detector::updateNoise() {
    const TileStats& a = cfg_.dark ? stats_->dark : stats_->bright;
    const TileStats* b = cfg_.dark && cfg_.bright ? &stats_->bright : nullptr;
    const size_t n = a.sum.size();
    noiseTmp_.clear();
    const float inv = 1.f / (float)(cfg_.tile * cfg_.tile * (b ? 2 : 1));
    // с фоном по ячейкам маска не расширяется, тайлы у кромки леса считаются, но в оценку
    // шума неба их не берём, иначе кромка поднимает порог всему небу. Бэкенд считает и
    // закрытые маской тайлы внутри своих полос, их тоже пропускаем
    const bool skipEdge = nearGround_.size() == n && mask_.size() == n;
    for (size_t i = 0; i < n; ++i) {
        if (skipEdge && (nearGround_[i] || !mask_[i])) continue;
        const uint32_t s = a.sum[i] + (b ? b->sum[i] : 0);
        if (s) noiseTmp_.push_back((float)s * inv);   // 0 — тайл не считался (маска)
    }
    if (noiseTmp_.size() < 8) return;
    auto mid = noiseTmp_.begin() + noiseTmp_.size() / 2;
    std::nth_element(noiseTmp_.begin(), mid, noiseTmp_.end());
    noise_ = thrEff_ > 0 ? 0.9f * noise_ + 0.1f * *mid : *mid;
    thrEff_ = std::max(cfg_.thr_min, std::min(cfg_.thr_max, cfg_.noise_k * noise_));
}

// Зоны по тайлам основного масштаба (см. DetectorConfig::zones)
void Detector::computeZones() {
    const size_t n = (size_t)tw_ * th_;
    // ряд полосы под сеткой (zone_strip) — лишний ряд снизу: в нём не ищем, но он земля
    const bool strip = stats_->strip;
    const int Th = th_ + (strip ? 1 : 0);
    const size_t na = (size_t)tw_ * Th;
    // кадр между полными просмотрами: посчитаны только тайлы вокруг треков, остальные выглядели
    // бы гладкими — небом. Зоны остаются с последнего полного кадра
    if (timing_.partial && zones_.size() == n && zoneHist_.size() >= n) return;
    zones_.assign(n, 0);
    if (!n) return;
    std::vector<uint8_t>& zz = zoneAll_;
    zz.assign(na, 2);
    const uint32_t area = (uint32_t)(cfg_.tile * cfg_.tile);
    auto textured = [&](const TileStats& ts, size_t i) {
        return ts.hot.size() == n && (float)div_round_even(ts.hot[i] * 255u, area) / 255.f > cfg_.max_density;
    };
    auto texStrip = [&](const TileStats& ts, size_t x) {
        return ts.hot.size() == (size_t)tw_ && (float)div_round_even(ts.hot[x] * 255u, area) / 255.f > cfg_.max_density;
    };
    const bool useMask = !learning_ && !mask_.empty();
    std::vector<uint8_t>& smooth = roiMask_;   // временный буфер
    smooth.assign(na, 0);
    for (size_t i = 0; i < n; ++i)
        smooth[i] = !(useMask && !mask_[i]) && !(cfg_.dark && textured(stats_->dark, i)) &&
                    !(cfg_.bright && textured(stats_->bright, i));
    for (size_t x = 0; strip && x < (size_t)tw_; ++x)
        smooth[n + x] = !(cfg_.dark && texStrip(stats_->stripD, x)) && !(cfg_.bright && texStrip(stats_->stripB, x));
    // земля: крупные связные области текстуры (кроны, дома, земля, край виньетки);
    // мелкие (сама цель, птица, фонарь) — не земля. Остальное — небо.
    const int minGround = std::max(4, (int)(n / 40));
    std::vector<uint8_t>& seen = focusTiles_;   // тоже временный буфер, markFocus перезапишет
    seen.assign(na, 0);
    for (size_t s0 = 0; s0 < na; ++s0) {
        if (smooth[s0] || seen[s0]) continue;
        fill_.clear();
        fill_.push_back((int)s0);
        seen[s0] = 1;
        for (size_t k = 0; k < fill_.size(); ++k) {
            const int i = fill_[k], x = i % tw_, y = i / tw_;
            for (int yy = std::max(0, y - 1); yy <= std::min(Th - 1, y + 1); ++yy)
                for (int xx = std::max(0, x - 1); xx <= std::min(tw_ - 1, x + 1); ++xx) {
                    const int j = yy * tw_ + xx;
                    if (!smooth[j] && !seen[j]) { seen[j] = 1; fill_.push_back(j); }
                }
        }
        // полоса — земля и малым куском: всё, что упирается в низ кадра, растёт из-за него
        bool low = false;
        for (int i : fill_) low |= i >= (int)n;
        if ((int)fill_.size() >= minGround || low)
            for (int i : fill_) zz[i] = 0;
    }
    // земля — по большинству из трёх последних полных кадров: тайл на краю кроны или облака,
    // который через кадр то земля, то небо, не мигает (cfg_.zone_vote). Новая сетка — с этого кадра
    if (zoneHist_.size() != na) zoneHist_.assign(na, 0x80);
    for (size_t i = 0; i < na; ++i) {
        const uint8_t g = zz[i] == 0, h = zoneHist_[i];
        if (!cfg_.zone_vote || (h & 0x80)) {   // истории ещё нет: как в этом кадре
            zoneHist_[i] = (uint8_t)(g * 7);
            continue;
        }
        const uint8_t nh = (uint8_t)(((h << 1) | g) & 7);
        zoneHist_[i] = nh;
        const int votes = cfg_.zone_vote == 1 ? ((nh & 1) + ((nh >> 1) & 1) + ((nh >> 2) & 1)) / 2 : (nh & 3) != 0;
        zz[i] = votes ? 0 : 2;
    }
    // тонкое над землёй (столб ЛЭП, мачта, антенна, узкая верхушка ели): сверху вниз ряд земли
    // не шире thin_tiles, над которым земли нет, остаётся землёй, но смешанных тайлов вокруг
    // себя не даёт. Иначе столб тянет за собой полосу смешанных шириной в три тайла до самого верха
    if (cfg_.thin_tiles > 0)
        for (int y = 0; y < Th; ++y)
            for (int x = 0; x < tw_;) {
                const size_t i = (size_t)y * tw_ + x;
                if (zz[i] != 0) {
                    ++x;
                    continue;
                }
                int e = x;
                while (e < tw_ && zz[(size_t)y * tw_ + e] == 0) ++e;
                bool open = e - x <= cfg_.thin_tiles;
                for (int xx = x; xx < e && open && y > 0; ++xx) open = zz[(size_t)(y - 1) * tw_ + xx] != 0;
                if (open)
                    for (int xx = x; xx < e; ++xx) zz[(size_t)y * tw_ + xx] = 4;   // земля без кольца
                x = e;
            }
    // смешанные: небо, у которого в 8-окрестности есть земля
    for (int y = 0; y < th_; ++y)
        for (int x = 0; x < tw_; ++x) {
            const size_t i = (size_t)y * tw_ + x;
            if (zz[i] != 2) continue;
            bool g = false;
            for (int yy = std::max(0, y - 1); yy <= std::min(Th - 1, y + 1) && !g; ++yy)
                for (int xx = std::max(0, x - 1); xx <= std::min(tw_ - 1, x + 1) && !g; ++xx)
                    g = zz[(size_t)yy * tw_ + xx] == 0;
            if (g) zz[i] = 3;   // пометка, чтобы не размножать смешанные внутри прохода
        }
    for (size_t i = 0; i < n; ++i) zones_[i] = zz[i] == 3 ? 1 : zz[i] == 4 ? 0 : zz[i];
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
    // грубые масштабы не сглаживаются: их порог — в уровнях яркости, без smooth_k
    const float ck = fineGrid ? 1.f : cfg_.coarse_thr_k / smoothK();
    const float thr0 = threshold() * ck, thrF = threshold() * ck * cfg_.focus_k;
    // отклик сглаженного кадра меньше, контраст кандидата — в уровнях, как без сглаживания
    const float cScale = fineGrid ? 1.f / smoothK() : 1.f;
    const size_t first = out.size();
    const bool useMask = fineGrid && !learning_ && !mask_.empty();
    // ячейки основного масштаба: второй кандидат, фон по ячейкам
    const int cpt = T / CELL, ctw = tw * cpt;
    const bool cells = fineGrid && T % CELL == 0 && ts.cmax.size() == (size_t)ctw * th * cpt;
    const bool bgOn = cells && cfg_.bg_map && !learning_;
    const bool bandOn = fineGrid && band_.size() == (size_t)tw * th;
    const bool nearOn = bgOn && nearGround_.size() == (size_t)tw * th;
    const int sep = cfg_.k + 1;   // кандидаты одного тайла не ближе, пикс. после пулинга
    const float hotB = std::floor(hotLevel());   // "горячий" отклик основного масштаба
    uint8_t* texOut = nullptr;
    if (fineGrid) {
        auto& t = stats_->tex[pol ? 1 : 0];
        t.assign((size_t)tw * th, 0);
        texOut = t.data();
    }
    for (int ty = 0; ty < th; ++ty)
        for (int tx = 0; tx < tw; ++tx) {
            const size_t i = (size_t)ty * tw + tx;
            const float level = (float)std::min<uint32_t>(255, div_round_even(ts.sum[i], area));
            const float dens = (float)div_round_even(ts.hot[i] * 255u, area) / 255.f;
            const bool texRaw = dens > maxDens;
            if (fineGrid && (learning_ || verifyLeft_)) {
                texSum_[i] += texRaw;
                auto& tp = texSumP_[pol ? 1 : 0];
                if (tp.size() == texSum_.size()) tp[i] += texRaw;
            }
            if (texOut) texOut[i] = texRaw;
            bool textured = texRaw;
            // Крупная контрастная цель (FWHM 5+ пикс.) сама набирает "горячих" пикселей больше
            // max_density, и тайл с ней считался текстурой — цель терялась. Если все горячие
            // ячейки тайла — вокруг его максимума (3x3 ячейки), это одно компактное пятно, а не
            // текстура: листва и кромка облака рассыпаны по тайлу. У земли не проверяем — там
            // одиночная верхушка дерева тоже компактное пятно. Маска неба учится по исходному
            // признаку текстуры: пятно на земле не должно открывать её для поиска
            if (textured && cells && cfg_.compact_tex && !(nearOn && nearGround_[i]) && (zones_.empty() || zones_[i] == 2)) {
                const int ax = tx * cpt + (ts.argmax[i] % T) / CELL, ay = ty * cpt + (ts.argmax[i] / T) / CELL;
                bool spread = false;
                for (int cy = ty * cpt; cy < (ty + 1) * cpt && !spread; ++cy)
                    for (int cx = tx * cpt; cx < (tx + 1) * cpt && !spread; ++cx)
                        spread = (std::abs(cx - ax) > 1 || std::abs(cy - ay) > 1) && ts.cmax[(size_t)cy * ctw + cx] > hotB;
                textured = spread;
            }
            // тайл земли у кромки, где уточнение нашло ячейки неба: только эти ячейки, по их фону
            bool fineTile = false;
            // закрытый только текстурой другой полярности (облако для тёмных целей)
            const bool polTile = useMask && !mask_[i] && polOpenAt(pol, i);
            if (useMask && !mask_[i] && !polTile) {
                if (!(cells && bgOn && fineSky_.size() == (size_t)ctw * th * cpt && edgeG_.size() == mask_.size() && edgeG_[i]))
                    continue;
                for (int cy = ty * cpt; cy < (ty + 1) * cpt && !fineTile; ++cy)
                    for (int cx = tx * cpt; cx < (tx + 1) * cpt && !fineTile; ++cx) fineTile = fineSky_[(size_t)cy * ctw + cx] != 0;
                if (!fineTile) continue;
            }
            const uint8_t zone = fineGrid && !zones_.empty() ? zones_[i] : 2;
            if (zone == 0) continue;
            // текстурный тайл у самой земли (кромка леса): с фоном по ячейкам смотрим его
            // по ячейкам, иначе выбрасываем целиком, как раньше
            // облако, открытое для этой полярности, — тоже: застывшие просветы между клочьями
            // ячейки поднимают себе порог сами
            const bool edgeTex = fineTile || polTile || (textured && nearOn && nearGround_[i]);
            if (textured && !edgeTex) continue;
            const float zk = zone == 1 ? cfg_.mixed_k : 1.f;
            const float bk = bandOn && band_[i] ? cfg_.band_k : 1.f;
            const float thrB = thr0 * bk;   // порог тайла без фокуса
            const float thr = (focusTiles_[i] ? thrF : thr0) * zk * bk;
            int taken = 0, px[4], py[4];
            auto push = [&](int gx, int gy, float mx, bool second) {   // gx, gy — пикс. после пулинга
                Candidate c;
                c.x = (float)(gx * f) + (f - 1) * 0.5f;
                c.y = (float)(gy * f) + (f - 1) * 0.5f;
                c.contrast = mx * cScale;
                c.margin = mx / thrB;
                c.snr = mx / (level + 1.f);
                c.polarity = pol ? 1 : -1;
                c.level = lvl;
                c.weak = focusTiles_[i] && !(mx >= thrB * zk && (edgeTex || mx >= cfg_.clutter_k * level + thrB * zk * 0.5f));
                // открытое перепроверкой облако: цель там подтверждается, только когда полетела
                c.edge = zone == 1 || edgeTex || polTile || (fineGrid && !cloud_.empty() && cloud_.size() == mask_.size() && cloud_[i]);
                c.second = second;
                out.push_back(c);
                px[taken] = gx;
                py[taken] = gy;
                ++taken;
            };
            const float mx = ts.maxv[i];
            if (!fineTile && !polTile && !textured && mx >= thr && mx >= cfg_.clutter_k * level + thr * 0.5f) {
                const int gx = tx * T + ts.argmax[i] % T, gy = ty * T + ts.argmax[i] / T;
                push(gx, gy, mx, false);
            } else if (!edgeTex) {
                continue;   // самый сильный отклик тайла не прошёл — остальные тем более
            }
            if (!cells || cfg_.per_tile <= taken) continue;
            // ещё кандидаты из ячеек тайла: самые сильные, вдали от уже взятых
            const int want = std::max(1, cfg_.per_tile);
            for (int iter = taken; iter < want; ++iter) {
                int best = -1, bgx = 0, bgy = 0;
                float bm = 0;
                for (int cy = ty * cpt; cy < (ty + 1) * cpt; ++cy)
                    for (int cx = tx * cpt; cx < (tx + 1) * cpt; ++cx) {
                        const size_t ci = (size_t)cy * ctw + cx;
                        if (fineTile && !fineSky_[ci]) continue;
                        const float v = ts.cmax[ci];
                        if (v <= bm || ts.carg[ci] == 255 || !(v >= thr)) continue;
                        if (!edgeTex && !(v >= cfg_.clutter_k * level + thr * 0.5f)) continue;
                        // у кромки леса — порог по своей ячейке (шум листвы и марева), в чистом
                        // небе хватает общего порога по шуму: там фон ячейки его только завышает
                        if (edgeTex && !(bgWarm(pol, ci) && v >= bgThr(pol, ci))) continue;
                        const int gx = cx * CELL + ts.carg[ci] % CELL, gy = cy * CELL + ts.carg[ci] / CELL;
                        bool far = true;
                        for (int k = 0; k < taken && far; ++k) far = std::max(std::abs(gx - px[k]), std::abs(gy - py[k])) >= sep;
                        if (!far) continue;
                        best = (int)ci;
                        bm = v;
                        bgx = gx;
                        bgy = gy;
                    }
                if (best < 0) break;
                push(bgx, bgy, bm, taken > 0);
            }
        }
    if (out.size() - first > (size_t)cfg_.max_cands) {   // слишком много — самые контрастные
        std::stable_sort(out.begin() + first, out.end(),
                         [](const Candidate& a, const Candidate& b) { return a.contrast > b.contrast; });
        out.resize(first + cfg_.max_cands);
    }
}

bool Detector::lockTile(int tx, int ty) const {
    const float tp = (float)(cfg_.tile * cfg_.pool);
    const float nx = std::max((float)tx * tp, std::min(lockX_, (float)(tx + 1) * tp));
    const float ny = std::max((float)ty * tp, std::min(lockY_, (float)(ty + 1) * tp));
    return std::hypot(nx - lockX_, ny - lockY_) <= lockR_;
}

// Цель в захвате под маской (DetectorConfig::lock_through): одно пятно той же полярности и
// похожего контраста в круге прогноза, в ячейках тайлов, закрытых для этой полярности
void Detector::selectThrough(const TileStats& ts, int pol, std::vector<Candidate>& out) {
    const int cpt = cfg_.tile / CELL, ctw = tw_ * cpt, cth = th_ * cpt, f = cfg_.pool;
    const size_t nc = (size_t)ctw * cth;
    if (cfg_.tile % CELL || ts.cmax.size() != nc || ts.carg.size() != nc || mask_.size() != (size_t)tw_ * th_) return;
    for (const Candidate& c : out)   // цель видна и так
        if ((c.polarity > 0) == (pol == 1) && std::hypot(c.x - lockX_, c.y - lockY_) <= lockR_) return;
    const float cpx = (float)(CELL * f), thr = threshold() * cfg_.lock_through_k, cs = 1.f / smoothK();
    const int x0 = std::max(0, (int)std::floor((lockX_ - lockR_) / cpx)), x1 = std::min(ctw - 1, (int)std::floor((lockX_ + lockR_) / cpx));
    const int y0 = std::max(0, (int)std::floor((lockY_ - lockR_) / cpx)), y1 = std::min(cth - 1, (int)std::floor((lockY_ + lockR_) / cpx));
    int best = -1, n = 0, cx[8], cy[8];
    float bm = 0, bx = 0, by = 0;
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x) {
            const size_t ci = (size_t)y * ctw + x, ti = (size_t)(y / cpt) * tw_ + x / cpt;
            if (mask_[ti] || polOpenAt(pol, ti)) continue;   // открытое смотрится и так
            const float v = ts.cmax[ci];
            if (v < thr || ts.carg[ci] == 255) continue;
            if (lockC_ > 0 && (v * cs < 0.5f * lockC_ || v * cs > 2.f * lockC_)) continue;
            const float px = (float)((x * CELL + ts.carg[ci] % CELL) * f) + (f - 1) * 0.5f;
            const float py = (float)((y * CELL + ts.carg[ci] / CELL) * f) + (f - 1) * 0.5f;
            if (std::hypot(px - lockX_, py - lockY_) > lockR_) continue;
            if (n == 8) return;   // много пятен — листва, не цель
            cx[n] = x, cy[n] = y, ++n;
            if (v > bm) bm = v, best = n - 1, bx = px, by = py;
        }
    if (best < 0) return;
    for (int k = 0; k < n; ++k)   // пятно одно: все сработавшие ячейки — соседи самой сильной
        if (std::abs(cx[k] - cx[best]) > 1 || std::abs(cy[k] - cy[best]) > 1) return;
    const size_t ti = (size_t)(cy[best] / cpt) * tw_ + cx[best] / cpt;
    const uint32_t area = (uint32_t)(cfg_.tile * cfg_.tile);
    const float level = (float)std::min<uint32_t>(255, ts.sum.size() > ti ? div_round_even(ts.sum[ti], area) : 0);
    Candidate c;
    c.x = bx;
    c.y = by;
    c.contrast = bm * cs;
    c.margin = bm / threshold();
    c.snr = bm / (level + 1.f);
    c.polarity = pol ? 1 : -1;
    c.weak = true;
    c.edge = true;
    out.push_back(c);
}

const std::vector<uint8_t>& Detector::texturedTiles(int pol) const { return stats_->tex[pol ? 1 : 0]; }

bool Detector::cellInfo(int pol, float x, float y, float& cmax, float& mu, float& sd, float& A) const {
    const TileStats& ts = pol ? stats_->bright : stats_->dark;
    const CellBg& c = stats_->cb[pol];
    const int ctw = tw_ * (cfg_.tile / CELL), cth = th_ * (cfg_.tile / CELL), cpx = CELL * cfg_.pool;
    const int cx = (int)x / cpx, cy = (int)y / cpx;
    if (cx < 0 || cy < 0 || cx >= ctw || cy >= cth || ts.cmax.size() != (size_t)ctw * cth) return false;
    const size_t ci = (size_t)cy * ctw + cx;
    cmax = ts.cmax[ci];
    mu = ci < c.q50.size() ? c.q50[ci] : 0;
    sd = ci < c.q90.size() ? bg_sd(c, ci) : 0;
    A = ci < c.A.size() ? c.A[ci] : 0;
    return true;
}

// Порог ячейки по её фону: медиана + bg_k "СКО" (разброс 50..90 перцентиля / 1.28, как у
// нормального шума); 0 — фон ещё не выучен
float Detector::bgThr(int pol, size_t ci) const {
    const CellBg& c = stats_->cb[pol];
    if (ci >= c.q50.size() || c.nupd[ci] < 8) return 0.f;
    return c.q50[ci] + cfg_.bg_k * bg_sd(c, ci);
}

bool Detector::bgWarm(int pol, size_t ci) const {
    const CellBg& c = stats_->cb[pol];
    return ci < c.nupd.size() && c.nupd[ci] >= 8;
}

// ячейки вокруг треков (прогнозы setRoi и setFocus): там фон не учится, иначе медленная
// или зависшая цель за bg_frames кадров подняла бы себе порог
void Detector::markExcl() {
    const int ctw = tw_ * (cfg_.tile / CELL), cth = th_ * (cfg_.tile / CELL), cpx = CELL * cfg_.pool;
    auto& ex = stats_->excl;
    ex.assign((size_t)ctw * cth, 0);
    auto mark = [&](const FocusArea& a) {
        if (a.level > 0) return;
        const float r = a.r + cpx;
        const int x0 = std::max(0, (int)std::floor((a.x - r) / cpx)), x1 = std::min(ctw - 1, (int)std::floor((a.x + r) / cpx));
        const int y0 = std::max(0, (int)std::floor((a.y - r) / cpx)), y1 = std::min(cth - 1, (int)std::floor((a.y + r) / cpx));
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) ex[(size_t)y * ctw + x] = 1;
    };
    for (const FocusArea& a : roi_) mark(a);
    for (const FocusArea& a : focus_) mark(a);
}

// Фон ячеек: шаг квантилей к текущему максимуму отклика (см. CellBg)
void Detector::updateBg(const TileStats& ts, int pol) {
    const size_t n = ts.cmax.size();
    if (!n) return;
    CellBg& c = stats_->cb[pol];
    if (c.q50.size() != n) c.reset(n);
    const int cpt = cfg_.tile / CELL, ctw = tw_ * cpt;
    const auto& comp = stats_->comp;
    const auto& ex = stats_->excl;
    const bool exOn = ex.size() == n;
    // без накопления фон нужен только у кромки леса (порог ячеек там) — остальное небо не трогаем
    const bool edgeOnly = !cfg_.tbd && nearGround_.size() == (size_t)tw_ * th_;
    const float rate = 4.f / (float)cfg_.bg_frames, k = cfg_.bg_k;
    for (int ty = 0; ty < th_; ++ty)
        for (int tx = 0; tx < tw_; ++tx) {
            const size_t ti = (size_t)ty * tw_ + tx;
            if (!comp.empty() && !comp[ti]) continue;
            if (edgeOnly && !nearGround_[ti] && !(edgeG_.size() == nearGround_.size() && edgeG_[ti]) && !polOpenAt(pol, ti))
                continue;
            for (int cy = ty * cpt; cy < (ty + 1) * cpt; ++cy)
                for (int cx = tx * cpt; cx < (tx + 1) * cpt; ++cx) {
                    const size_t ci = (size_t)cy * ctw + cx;
                    if (exOn && ex[ci] && c.nupd[ci] >= 8) continue;
                    const float x = ts.cmax[ci];
                    float& q50 = c.q50[ci];
                    float& q90 = c.q90[ci];
                    if (c.nupd[ci] == 0) {
                        q50 = x;
                        q90 = x + 0.5f * x + 2.f;   // поначалу осторожно: разброс велик
                        c.nupd[ci] = 1;
                        continue;
                    }
                    // шаг — доля разброса; первые кадры учимся быстрее
                    const float spread = std::max(0.5f, q90 - q50);
                    float eta = spread * (c.nupd[ci] < 32 ? std::max(rate, 0.25f) : rate);
                    // выброс выше порога (медленная цель, которую не поймали) почти не тянет фон
                    // вверх, иначе дрон, висящий в ячейке, сам себе поднимает порог; настоящая
                    // перемена фона (облако) всё равно принимается, только в 10 раз медленнее
                    if (c.nupd[ci] >= 32 && x > q50 + k * (q90 - q50) * (1.f / 1.28f)) eta *= 0.1f;
                    q50 += eta * (x > q50 ? 0.5f : -0.5f);
                    q90 += eta * (x > q90 ? 0.9f : -0.1f);
                    if (q90 < q50 + 0.25f) q90 = q50 + 0.25f;
                    if (c.nupd[ci] < 255) ++c.nupd[ci];
                }
        }
}

// Накопление по кадрам на сетке ячеек (см. DetectorConfig::tbd)
void Detector::selectTbd(const TileStats& ts, int pol, std::vector<Candidate>& out) {
    const size_t n = ts.cmax.size();
    CellBg& c = stats_->cb[pol];
    if (!n || c.q50.size() != n) return;
    const int cpt = cfg_.tile / CELL, ctw = tw_ * cpt, cth = th_ * cpt, f = cfg_.pool;
    const auto& comp = stats_->comp;
    const bool useMask = !mask_.empty() && !learning_;
    const bool bandOn = band_.size() == (size_t)tw_ * th_;
    // max3x3 прошлого A: цель за кадр переходит не дальше соседней ячейки
    std::vector<float>& M = c.tmp;
    for (int y = 0; y < cth; ++y)
        for (int x = 0; x < ctw; ++x) {
            float v = 0;
            for (int yy = std::max(0, y - 1); yy <= std::min(cth - 1, y + 1); ++yy)
                for (int xx = std::max(0, x - 1); xx <= std::min(ctw - 1, x + 1); ++xx)
                    v = std::max(v, c.A[(size_t)yy * ctw + xx]);
            M[(size_t)y * ctw + x] = v;
        }
    const size_t first = out.size();
    const float R = 1.5f * CELL * f;   // уже есть кандидат ближе — не дублируем
    const auto& tex = stats_->tex[pol];
    // превышение над фоном; текстурные тайлы (края облаков, листва) не копят: облако
    // ползёт медленно и копилось бы не хуже цели
    std::vector<float>& E = stats_->e;
    E.assign(n, -1.f);   // -1 — ячейку в этот кадр не смотрели или она не небо
    for (size_t ci = 0; ci < n; ++ci) {
        const int cx = (int)(ci % ctw), cy = (int)(ci / ctw);
        const size_t ti = (size_t)(cy / cpt) * tw_ + cx / cpt;
        if ((useMask && !mask_[ti] && !polOpenAt(pol, ti)) || (!zones_.empty() && zones_[ti] == 0)) continue;
        if (!comp.empty() && !comp[ti]) continue;
        if (c.nupd[ci] < 8 || (tex.size() == (size_t)tw_ * th_ && tex[ti])) { E[ci] = 0; continue; }
        const float sd = std::max(1.f, bg_sd(c, ci));
        // сверху ограничено: накопление — для слабых целей; яркая цель и так видна, а её
        // большое превышение расползалось бы через max3x3 ореолом на соседние ячейки
        E[ci] = std::min(2.5f, std::max(0.f, ((float)ts.cmax[ci] - c.q50[ci]) / sd - cfg_.tbd_z0));
    }
    for (size_t ci = 0; ci < n; ++ci) {
        const int cx = (int)(ci % ctw), cy = (int)(ci / ctw);
        const size_t ti = (size_t)(cy / cpt) * tw_ + cx / cpt;
        if (E[ci] < 0) {   // не смотрели — затухает; не небо — ноль
            const bool seen = comp.empty() || comp[ti];
            c.A[ci] = seen ? 0.f : c.A[ci] * cfg_.tbd_decay;
            continue;
        }
        // точечная цель поднимает 1-2 ячейки, край облака или марево — целую гряду
        int lit = 0;
        for (int yy = std::max(0, cy - 1); yy <= std::min(cth - 1, cy + 1); ++yy)
            for (int xx = std::max(0, cx - 1); xx <= std::min(ctw - 1, cx + 1); ++xx)
                lit += E[(size_t)yy * ctw + xx] > 0;
        const float e = lit <= 3 ? E[ci] : 0.f;
        const float A = cfg_.tbd_decay * M[ci] + e;
        c.A[ci] = A;
        const float bk = bandOn && band_[ti] ? cfg_.band_k : 1.f;
        const float thrB = threshold() * bk;
        if (!(A >= cfg_.tbd_thr) || e <= 0 || ts.cmax[ci] < thrB * cfg_.tbd_min || ts.carg[ci] == 255) continue;
        const float gx = (float)((cx * CELL + ts.carg[ci] % CELL) * f) + (f - 1) * 0.5f;
        const float gy = (float)((cy * CELL + ts.carg[ci] / CELL) * f) + (f - 1) * 0.5f;
        const int pl = pol ? 1 : -1;
        bool dup = false;
        for (size_t k = 0; k < first && !dup; ++k)
            dup = out[k].polarity == pl && std::fabs(out[k].x - gx) < R && std::fabs(out[k].y - gy) < R;
        if (dup) continue;
        Candidate cd;
        cd.x = gx;
        cd.y = gy;
        cd.contrast = ts.cmax[ci] / smoothK();
        cd.margin = std::sqrt(1.f + A / cfg_.tbd_thr);
        cd.snr = E[ci] + cfg_.tbd_z0;
        cd.polarity = pl;
        cd.tbd = true;
        cd.edge = (!zones_.empty() && zones_[ti] == 1) || (nearGround_.size() == (size_t)tw_ * th_ && nearGround_[ti]);
        out.push_back(cd);
    }
}

// Полоса у горизонта и тайлы у кромки земли — по маске неба (неподвижная камера)
void Detector::updateBand() {
    band_.clear();
    nearGround_.clear();
    if (mask_.empty() || learning_ || mask_.size() != (size_t)tw_ * th_) {
        fineBase_.clear();
        edgeG_.clear();
        fineSky_.clear();
        fineTex_.clear();
        fineN_.clear();
        return;
    }
    if (cfg_.bg_map) {
        nearGround_.assign(mask_.size(), 0);
        for (int y = 0; y < th_; ++y)
            for (int x = 0; x < tw_; ++x) {
                if (!mask_[(size_t)y * tw_ + x]) continue;
                bool g = false;
                for (int yy = std::max(0, y - 1); yy <= std::min(th_ - 1, y + 1) && !g; ++yy)
                    for (int xx = std::max(0, x - 1); xx <= std::min(tw_ - 1, x + 1) && !g; ++xx)
                        g = !mask_[(size_t)yy * tw_ + xx];
                nearGround_[(size_t)y * tw_ + x] = g;
            }
    }
    // тайлы земли рядом с небом — там уточняется маска по ячейкам. Тайл, где уже нашлись
    // ячейки неба, сам считается небом для соседей: уточнение со временем спускается ниже,
    // пока не упрётся в кроны
    if (mask_ != fineBase_) {   // маска тайлов сменилась — уточнение копится заново
        fineBase_ = mask_;
        edgeG_.clear();
        fineSky_.clear();
        fineTex_.clear();
        fineN_.clear();
    }
    if (cfg_.bg_map && cfg_.fine_mask && cfg_.tile % CELL == 0) {
        const int cpt = cfg_.tile / CELL, ctw = tw_ * cpt;
        const bool fs = fineSky_.size() == (size_t)ctw * th_ * cpt;
        std::vector<uint8_t> open(mask_.size(), 0);   // небо или тайл земли с найденным небом
        for (int y = 0; y < th_; ++y)
            for (int x = 0; x < tw_; ++x) {
                const size_t i = (size_t)y * tw_ + x;
                bool o = mask_[i] != 0;
                for (int cy = y * cpt; fs && !o && cy < (y + 1) * cpt; ++cy)
                    for (int cx = x * cpt; !o && cx < (x + 1) * cpt; ++cx) o = fineSky_[(size_t)cy * ctw + cx] != 0;
                open[i] = o;
            }
        edgeG_.resize(mask_.size(), 0);
        for (int y = 0; y < th_; ++y)
            for (int x = 0; x < tw_; ++x) {
                if (mask_[(size_t)y * tw_ + x]) continue;
                bool g = false;
                for (int yy = std::max(0, y - 1); yy <= std::min(th_ - 1, y + 1) && !g; ++yy)
                    for (int xx = std::max(0, x - 1); xx <= std::min(tw_ - 1, x + 1) && !g; ++xx)
                        g = open[(size_t)yy * tw_ + xx] != 0;
                edgeG_[(size_t)y * tw_ + x] |= g;   // раз начав, тайл не бросаем
            }
    } else {
        edgeG_.clear();
    }
    if (cfg_.band_px <= 0) return;
    const int tilePx = cfg_.tile * cfg_.pool, bt = (cfg_.band_px + tilePx - 1) / tilePx;
    band_.assign(mask_.size(), 0);
    for (int x = 0; x < tw_; ++x) {
        int y = th_ - 1;   // снизу вверх по земле до первого тайла неба: над ним — горизонт
        while (y >= 0 && !mask_[(size_t)y * tw_ + x]) --y;
        if (y < 0) continue;
        for (int yy = std::max(0, y + 1 - bt); yy <= y; ++yy) band_[(size_t)yy * tw_ + x] = 1;
    }
}

// Уточнение маски по ячейкам: в тайлах земли у кромки — доля кадров, когда в ячейке есть
// текстура (максимум отклика выше уровня "горячих" пикселей хоть одной полярности). Небо — ячейка,
// где текстуры почти не бывает и над которой небо (тайл неба или такая же ячейка): одинокая
// тихая ячейка внутри кроны небом не станет
void Detector::updateFine() {
    const int cpt = cfg_.tile / CELL, ctw = tw_ * cpt, cth = th_ * cpt;
    const size_t n = (size_t)ctw * cth;
    if (edgeG_.size() != (size_t)tw_ * th_ || stats_->dark.cmax.size() != n) return;
    if (fineTex_.size() != n) {
        fineTex_.assign(n, 0.f);
        fineN_.assign(n, 0);
        fineSky_.assign(n, 0);
    }
    const auto& comp = stats_->comp;
    const auto& ex = stats_->excl;
    const bool exOn = ex.size() == n;
    const float hot = hotLevel();
    const float rate = 1.f / (float)std::max(10, cfg_.fine_frames);
    // тайлы земли у кромки — малая часть кадра: обходим только их ячейки (на 16 Мп
    // проход по всем 62 тыс. ячеек стоил больше, чем сами ячейки кромки)
    for (int ty = 0; ty < th_; ++ty)
        for (int tx = 0; tx < tw_; ++tx) {
            const size_t ti = (size_t)ty * tw_ + tx;
            if (!edgeG_[ti] || (!comp.empty() && !comp[ti])) continue;
            for (int cy = ty * cpt; cy < (ty + 1) * cpt; ++cy)
                for (int cx = tx * cpt; cx < (tx + 1) * cpt; ++cx) {
                    const size_t ci = (size_t)cy * ctw + cx;
                    if (exOn && ex[ci]) continue;   // вокруг треков: цель — не текстура
                    float v = cfg_.dark ? stats_->dark.cmax[ci] : 0.f;
                    if (cfg_.bright && stats_->bright.cmax.size() == n) v = std::max(v, (float)stats_->bright.cmax[ci]);
                    const float t = v > hot ? 1.f : 0.f;
                    if (fineN_[ci] < 65535) ++fineN_[ci];
                    // первые кадры — среднее, дальше — забывание за fine_frames кадров
                    const float a = std::max(rate, 1.f / (float)fineN_[ci]);
                    fineTex_[ci] += a * (t - fineTex_[ci]);
                }
        }
    // сверху вниз: небо, если тихо давно и рядом сверху (или сбоку в той же строке) небо —
    // так небо обтекает облачко или ветку, но в одинокий просвет внутри кроны не попадает
    const int need = std::max(10, cfg_.fine_frames / 2);
    auto sky = [&](int cx, int cy) {   // ячейка — небо: в тайле неба или уже найдена
        if (cx < 0 || cx >= ctw) return false;
        if (cy < 0) return true;
        const size_t ti = (size_t)(cy / cpt) * tw_ + cx / cpt;
        return mask_[ti] != 0 || (edgeG_[ti] && fineSky_[(size_t)cy * ctw + cx] != 0);
    };
    // вне тайлов кромки ячейка не тихая и небом не станет: строки тайлов без кромки — нули,
    // в остальных проходы идут по прежнему порядку, но только по ячейкам тайлов кромки
    for (int ty = 0; ty < th_; ++ty) {
        const uint8_t* eg = &edgeG_[(size_t)ty * tw_];
        const bool any = std::find(eg, eg + tw_, 1) != eg + tw_;
        for (int cy = ty * cpt; cy < (ty + 1) * cpt; ++cy) {
            uint8_t* fs = &fineSky_[(size_t)cy * ctw];
            std::fill(fs, fs + ctw, 0);
            if (!any) continue;
            auto quiet = [&](int cx) {
                const size_t ci = (size_t)cy * ctw + cx;
                return fineN_[ci] >= need && fineTex_[ci] < cfg_.fine_tex;
            };
            for (int tx = 0; tx < tw_; ++tx)
                if (eg[tx])
                    for (int cx = tx * cpt; cx < (tx + 1) * cpt; ++cx)
                        fs[cx] = quiet(cx) && (sky(cx - 1, cy - 1) || sky(cx, cy - 1) || sky(cx + 1, cy - 1));
            // в строке — от найденного неба вбок по тихим ячейкам
            for (int tx = 0; tx < tw_; ++tx)
                if (eg[tx])
                    for (int cx = std::max(1, tx * cpt); cx < (tx + 1) * cpt; ++cx)
                        if (!fs[cx] && quiet(cx) && sky(cx - 1, cy)) fs[cx] = 1;
            for (int tx = tw_ - 1; tx >= 0; --tx)
                if (eg[tx])
                    for (int cx = std::min(ctw - 2, (tx + 1) * cpt - 1); cx >= tx * cpt; --cx)
                        if (!fs[cx] && quiet(cx) && sky(cx + 1, cy)) fs[cx] = 1;
        }
    }
}

}  // namespace dronolok
