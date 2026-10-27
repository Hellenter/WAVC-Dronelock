// Трекер: фильтр Калмана с постоянной скоростью на каждую цель, ассоциация по
// прогнозу, удержание трека на прогнозе при пропусках, компенсация движения камеры,
// карта статичного клаттера и классификатор треков по характеру движения.
//
// Отличия от прототипа (detector/detector.py, ближайший сосед): ворота сужаются,
// когда трек захвачен, и расширяются при пропусках; подтверждённый трек не пропадает
// при одном-двух пропущенных кадрах; цель, которая уже пролетела по сцене, не
// записывается в помехи, даже если зависла; карта клаттера едет вместе с камерой.
//
// Помехи помнятся двояко: карта клаттера — на секунды (блик, рябь), долгая память
// статичных помех — на минуты (катушки ЛЭП, фонари) и сохраняется в файл, чтобы
// неподвижная камера после перезапуска не училась заново.
//
// Классификатор: по последним cls_window попаданиям трека строится плавная
// траектория (квадратичная по времени). Дрон летит плавно и отклоняется от неё
// на 1-2 пикселя (квантование пулинга), ласточка дёргается и разворачивается,
// а взмахи крыльев меняют её контраст. Птицу, которая летит ровно, выдают взмахи:
// суммарный контраст пятна пульсирует с их частотой (flap_features). Махавшая недавно
// птица и её попутчики по стае остаются птицами; зависший после полёта — дрон.
// Пороги — в TrackerConfig.
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "dronolok/detector.hpp"
#include "dronolok/fileutil.hpp"

namespace dronolok {

const char* class_name(TargetClass c) {
    switch (c) {
        case TargetClass::Unknown: return "unknown";
        case TargetClass::Drone: return "drone";
        case TargetClass::Bird: return "bird";
        case TargetClass::Clutter: return "clutter";
    }
    return "?";
}

Tracker::Tracker(int w, int h, const TrackerConfig& cfg) : cfg_(cfg) {
    gw_ = w / cfg_.clutter_cell + 1;
    gh_ = h / cfg_.clutter_cell + 1;
    heat_.assign((size_t)gw_ * gh_, 0.f);
    stat_.assign(heat_.size(), 0.f);
    statT_.assign(heat_.size(), 0);
}

void Tracker::resetTracks() {
    tracks_.clear();
    out_.clear();
    lockId_ = 0;
}

bool Tracker::hot(float x, float y) const {
    const int cx = std::min(gw_ - 1, std::max(0, (int)x / cfg_.clutter_cell));
    const int cy = std::min(gh_ - 1, std::max(0, (int)y / cfg_.clutter_cell));
    return heat_[(size_t)cy * gw_ + cx] > cfg_.clutter_level;
}

// ---------------------------------------------------------------- память статичных помех

// значение — число "неподвижных" откликов в ячейке, затухает с постоянной static_frames
float Tracker::staticAt(size_t i) const {
    if (cfg_.static_frames <= 0 || stat_[i] <= 0) return 0.f;
    return stat_[i] * std::pow(1.f - 1.f / (float)cfg_.static_frames, (float)(frame_ - statT_[i]));
}

void Tracker::staticAdd(size_t i) {
    stat_[i] = staticAt(i) + 1.f;
    statT_[i] = frame_;
}

// Ячейка и соседи: провод качается на ветру, катушка дрожит в пределах пары пикселей
bool Tracker::staticHot(float x, float y) const {
    if (cfg_.static_frames <= 0) return false;
    const int cx = std::min(gw_ - 1, std::max(0, (int)x / cfg_.clutter_cell));
    const int cy = std::min(gh_ - 1, std::max(0, (int)y / cfg_.clutter_cell));
    for (int yy = std::max(0, cy - 1); yy <= std::min(gh_ - 1, cy + 1); ++yy)
        for (int xx = std::max(0, cx - 1); xx <= std::min(gw_ - 1, cx + 1); ++xx)
            if (staticAt((size_t)yy * gw_ + xx) > cfg_.static_hits) return true;
    return false;
}

void Tracker::clearStatic() {
    std::fill(stat_.begin(), stat_.end(), 0.f);
    std::fill(statT_.begin(), statT_.end(), 0);
}

void Tracker::staticCells(std::vector<TrackPoint>& out) const {
    out.clear();
    const float c = (float)cfg_.clutter_cell;
    for (int y = 0; y < gh_; ++y)
        for (int x = 0; x < gw_; ++x)
            if (staticAt((size_t)y * gw_ + x) > cfg_.static_hits) out.push_back({(x + 0.5f) * c, (y + 0.5f) * c});
}

// Файл: "DLSM", версия, ширина и высота сетки, ячейка, затем значения по ячейкам
bool Tracker::saveStatic(const std::string& path) const {
    FILE* f = open_file(path, "wb");
    if (!f) return false;
    const int32_t hdr[5] = {0x4D534C44, 2, gw_, gh_, cfg_.clutter_cell};
    std::vector<float> v(stat_.size());
    for (size_t i = 0; i < v.size(); ++i) v[i] = staticAt(i);
    const bool ok = fwrite(hdr, sizeof(hdr), 1, f) == 1 && fwrite(v.data(), sizeof(float), v.size(), f) == v.size();
    return fclose(f) == 0 && ok;
}

bool Tracker::loadStatic(const std::string& path) {
    FILE* f = open_file(path, "rb");
    if (!f) return false;
    int32_t hdr[5];
    std::vector<float> v(stat_.size());
    const bool ok = fread(hdr, sizeof(hdr), 1, f) == 1 && hdr[0] == 0x4D534C44 && hdr[1] == 2 && hdr[2] == gw_ &&
                    hdr[3] == gh_ && hdr[4] == cfg_.clutter_cell && fread(v.data(), sizeof(float), v.size(), f) == v.size();
    fclose(f);
    if (!ok) return false;
    stat_ = std::move(v);
    std::fill(statT_.begin(), statT_.end(), frame_);
    return true;
}

// МНК-парабола p(t) = a + b t + c t^2; возвращает сумму квадратов остатков, b — скорость в середине окна
static double fit_quad(const std::deque<Track::Obs>& obs, size_t from, bool useX, double& slope) {
    const auto o0 = obs.begin() + (std::ptrdiff_t)from;
    const int n = (int)(obs.size() - from);
    const double tm = 0.5 * (o0->t + obs.back().t);
    double S[5] = {0, 0, 0, 0, 0}, B[3] = {0, 0, 0};
    for (auto it = o0; it != obs.end(); ++it) {
        const auto& q = *it;
        const double t = q.t - tm, v = useX ? q.x : q.y;
        double tp = 1;
        for (int i = 0; i < 5; ++i) { S[i] += tp; if (i < 3) B[i] += tp * v; tp *= t; }
    }
    // нормальные уравнения 3x3 (при n < 3 или вырожденности — прямая)
    double A[3][4] = {{S[0], S[1], S[2], B[0]}, {S[1], S[2], S[3], B[1]}, {S[2], S[3], S[4], B[2]}};
    int m = n >= 4 ? 3 : 2;
    for (int c = 0; c < m; ++c) {
        int piv = c;
        for (int r = c + 1; r < m; ++r) if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
        if (std::fabs(A[piv][c]) < 1e-9) { m = c; break; }
        std::swap(A[c], A[piv]);
        for (int r = 0; r < m; ++r) {
            if (r == c) continue;
            const double f = A[r][c] / A[c][c];
            for (int j = c; j < 4; ++j) A[r][j] -= f * A[c][j];
        }
    }
    double coef[3] = {0, 0, 0};
    for (int c = 0; c < m; ++c) coef[c] = A[c][3] / A[c][c];
    slope = coef[1];
    double ss = 0;
    for (auto it = o0; it != obs.end(); ++it) {
        const auto& q = *it;
        const double t = q.t - tm, v = useX ? q.x : q.y;
        const double r = v - (coef[0] + coef[1] * t + coef[2] * t * t);
        ss += r * r;
    }
    return ss;
}

// Взмахи: суммарный контраст за последние W кадров (пропуски не мешают) без линейного
// тренда (цель приближается, уходит в облако) приближается синусоидой; перебор частот
// от 1/16 до 1/2 цикла на кадр, косинус и синус — поворотом, без тригонометрии в цикле.
// ~40 частот x 48 точек — пара тысяч умножений на трек.
struct FlapFit { float r2 = 0, period = 0, depth = 0; };

static FlapFit flap_fit(const std::deque<Track::Obs>& obs, int W) {
    FlapFit f;
    W = std::min(W, 64);
    const int tEnd = obs.back().t;
    double ee[64];
    bool ok[64] = {};
    int n = 0;
    for (auto it = obs.rbegin(); it != obs.rend() && tEnd - it->t < W; ++it, ++n) {
        ok[tEnd - it->t] = true;
        ee[tEnd - it->t] = it->e;
    }
    if (n < std::max(12, W * 2 / 3)) return f;
    double st = 0, se = 0, stt = 0, ste = 0;
    for (int i = 0; i < W; ++i)
        if (ok[i]) { st += i; se += ee[i]; stt += (double)i * i; ste += i * ee[i]; }
    const double mean = se / n;
    if (mean <= 0) return f;
    const double den = n * stt - st * st;
    const double b = den > 0 ? (n * ste - st * se) / den : 0, a = (se - b * st) / n;
    double var = 0;
    for (int i = 0; i < W; ++i)
        if (ok[i]) { ee[i] -= a + b * i; var += ee[i] * ee[i]; }
    if (var <= 0) return f;
    const int steps = 2 * W;   // шаг по частоте 1/(2W) цикла на кадр
    for (int k = std::max(1, steps / 16); k <= steps / 2; ++k) {
        const double w = 2 * 3.14159265358979323846 * k / steps, rc = std::cos(w), rs = std::sin(w);
        double c = 1, s = 0, cc = 0, ss = 0, cs = 0, ec = 0, es = 0;
        for (int i = 0; i < W; ++i) {
            if (ok[i]) { cc += c * c; ss += s * s; cs += c * s; ec += ee[i] * c; es += ee[i] * s; }
            const double c2 = c * rc - s * rs;
            s = s * rc + c * rs;
            c = c2;
        }
        // МНК e = p cos + q sin; на частоте 1/2 синус вырожден — только косинус
        double p, q;
        const double d = cc * ss - cs * cs;
        if (d > 1e-6 * cc * ss) { p = (ec * ss - es * cs) / d; q = (es * cc - ec * cs) / d; }
        else { p = cc > 0 ? ec / cc : 0; q = 0; }
        const double r2 = (p * ec + q * es) / var;   // доля объяснённой дисперсии
        if (r2 > f.r2) {
            f.r2 = (float)r2;
            f.period = (float)steps / k;
            f.depth = (float)(2 * std::sqrt(p * p + q * q) / mean);
        }
    }
    return f;
}

void Tracker::classify(Track& t) const {
    const auto& o = t.obs;
    const size_t from = o.size() > (size_t)cfg_.cls_window ? o.size() - cfg_.cls_window : 0;
    const int n = (int)(o.size() - from);
    if (n >= 3) {
        double sx, sy;
        const double ss = fit_quad(o, from, true, sx) + fit_quad(o, from, false, sy);
        t.resid = (float)std::sqrt(ss / n);
        t.speed = (float)std::hypot(sx, sy);
        double mc = 0, vc = 0;
        for (size_t i = from; i < o.size(); ++i) mc += o[i].c;
        mc /= n;
        for (size_t i = from; i < o.size(); ++i) vc += (o[i].c - mc) * (o[i].c - mc);
        t.contrastCv = mc > 0 ? (float)(std::sqrt(vc / n) / mc) : 0.f;
    }
    if (cfg_.flap_window > 0 && !o.empty()) {
        const FlapFit ff = flap_fit(o, cfg_.flap_window);
        t.flapR2 = ff.r2;
        t.flapPeriod = ff.period;
        t.flapDepth = ff.depth;
        if (ff.r2 >= cfg_.flap_r2 && ff.depth >= cfg_.flap_depth) t.flapT = frame_;
    }
    // стоит в горячей ячейке или за cls_min_hits попаданий так и не сдвинулся по сцене
    // (фонарь, угол дома, когда камера ведёт панораму) — помеха
    if (t.clutter || (n >= cfg_.cls_min_hits && t.travel < std::max(3.f, 0.25f * (float)t.age))) {
        t.cls = TargetClass::Clutter;
        return;
    }
    if (n < cfg_.cls_min_hits) {
        if (t.cls == TargetClass::Clutter) t.cls = TargetClass::Unknown;
        return;
    }
    // крупная цель найдена на грубом масштабе: позиция квантуется сильнее, порог шире
    double lf = 0;
    for (size_t i = from; i < o.size(); ++i) lf += (double)(1 << std::min(4, o[i].level));
    lf /= n;
    float e = std::max(t.resid / (cfg_.bird_resid * (float)lf), t.contrastCv / cfg_.bird_contrast_cv);
    if (cfg_.bird_speed > 0) e = std::max(e, t.speed / cfg_.bird_speed);
    const bool flapped = frame_ - t.flapT <= cfg_.flap_hold;
    t.hovering = cfg_.hover_speed > 0 && t.everMover && t.speed < cfg_.hover_speed;
    float inst = e > 1.f ? 1.f : 0.f;
    if (flapped || t.flock) inst = 1.f;
    else if (t.hovering) inst = 0.f;
    // сглаживание, чтобы класс не мигал от кадра к кадру
    if (t.cls == TargetClass::Unknown || t.cls == TargetClass::Clutter) t.birdScore = inst;
    else t.birdScore = 0.8f * t.birdScore + 0.2f * inst;
    t.cls = t.birdScore > 0.5f ? TargetClass::Bird : TargetClass::Drone;
}

// Сколько кадров подтверждённый трек держится на прогнозе: короткий (возможно, шум) —
// 1 кадр, устоявшийся и летящий по сцене — столько, сколько было попаданий, но не больше max_miss
int Tracker::coastLimit(const Track& t) const {
    return t.hits < cfg_.cls_min_hits || t.travel < cfg_.mover_px ? 1 : std::min(cfg_.max_miss, t.hits);
}

float Tracker::measVar(int level) const {
    const float r = cfg_.r_meas * (float)(1 << std::max(0, std::min(4, level)));
    return r * r;
}

// Карта клаттера в координатах кадра: при движении камеры сдвигаем её на целые ячейки
void Tracker::shiftHeat(float dx, float dy) {
    heatAccX_ += dx;
    heatAccY_ += dy;
    const int cell = cfg_.clutter_cell;
    const int sx = (int)std::trunc(heatAccX_ / cell), sy = (int)std::trunc(heatAccY_ / cell);
    if (!sx && !sy) return;
    heatAccX_ -= (float)sx * cell;
    heatAccY_ -= (float)sy * cell;
    heatTmp_.assign(heat_.size(), 0.f);
    for (int y = 0; y < gh_; ++y) {
        const int ys = y - sy;
        if (ys < 0 || ys >= gh_) continue;
        for (int x = 0; x < gw_; ++x) {
            const int xs = x - sx;
            if (xs >= 0 && xs < gw_) heatTmp_[(size_t)y * gw_ + x] = heat_[(size_t)ys * gw_ + xs];
        }
    }
    heat_.swap(heatTmp_);
}

// Прогноз на один кадр вперёд: позиция += скорость по сцене + сдвиг камеры
static inline void kf_predict(Track& t, float camDx, float camDy, float q) {
    t.x += t.vsx + camDx;
    t.y += t.vsy + camDy;
    // F = [[1,1],[0,1]], Q — дискретный белый шум ускорения
    const float p00 = t.p00 + 2 * t.p01 + t.p11 + 0.25f * q;
    const float p01 = t.p01 + t.p11 + 0.5f * q;
    const float p11 = t.p11 + q;
    t.p00 = p00;
    t.p01 = p01;
    t.p11 = p11;
}

void Tracker::focus(float camDx, float camDy, std::vector<FocusArea>& out) const {
    out.clear();
    const float q = cfg_.q_accel * cfg_.q_accel;
    for (const Track& t0 : tracks_) {
        // только устоявшиеся летящие цели: иначе пониженный порог подкармливает шумовые треки
        if (!t0.solid || t0.cls == TargetClass::Clutter) continue;
        if (t0.id != lockId_ && (t0.hits < cfg_.cls_min_hits || t0.travel < cfg_.mover_px)) continue;
        Track t = t0;
        kf_predict(t, camDx, camDy, q);
        const float S = t.p00 + measVar(t.level);
        const float r = std::min(cfg_.gate, std::max(cfg_.gate_min, cfg_.gate_sigma * std::sqrt(S)));
        out.push_back({t.x, t.y, r});
    }
}

void Tracker::roi(float camDx, float camDy, std::vector<FocusArea>& out) const {
    out.clear();
    const float q = cfg_.q_accel * cfg_.q_accel;
    for (const Track& t0 : tracks_) {
        Track t = t0;
        kf_predict(t, camDx, camDy, q);
        const float S = t.p00 + measVar(t.level);
        const float gmin = cfg_.gate_min * (float)(1 << std::min(4, t.level));
        // запас на размер цели: на грубом масштабе её отклик гуляет по корпусу
        out.push_back({t.x, t.y, std::min(cfg_.gate, std::max(gmin, cfg_.gate_sigma * std::sqrt(S))) + 4.f * (1 << t.level),
                       t.level});
    }
}

const std::vector<Track>& Tracker::update(const std::vector<Candidate>& dets, float camDx, float camDy) {
    ++frame_;
    gx_ += camDx;
    gy_ += camDy;
    const float q = cfg_.q_accel * cfg_.q_accel;
    const float decay = cfg_.clutter_decay;
    shiftHeat(camDx, camDy);
    if (!partial_)
        for (auto& v : heat_) v *= decay;
    partial_ = false;
    const int cell = cfg_.clutter_cell;
    auto cellOf = [&](const Candidate& d) {
        const int cx = std::min(gw_ - 1, std::max(0, (int)d.x / cell));
        const int cy = std::min(gh_ - 1, std::max(0, (int)d.y / cell));
        return (size_t)cy * gw_ + cx;
    };
    for (auto& t : tracks_) {
        kf_predict(t, camDx, camDy, q);
        for (auto& p : t.path) { p.x += camDx; p.y += camDy; }
    }

    // "тени": слабый отклик рядом с более сильным (светлый ореол у тёмной цели от резкости
    // камеры, цель на границе тайлов, на двух масштабах) — это та же цель, в трекинг не идёт
    std::vector<char> shadow(dets.size(), 0);
    for (size_t i = 0; i < dets.size(); ++i) {
        const float dup = cfg_.dup_px * (float)(1 << std::min(4, dets[i].level));
        for (size_t j = 0; j < dets.size() && !shadow[i]; ++j)
            shadow[i] = j != i && (dets[j].contrast > dets[i].contrast || (dets[j].contrast == dets[i].contrast && j < i)) &&
                        std::fabs(dets[j].x - dets[i].x) < dup && std::fabs(dets[j].y - dets[i].y) < dup;
    }

    // ассоциация по прогнозу: нормированное расстояние d^2/S, жадно по возрастанию
    struct Pair { float d; int t, k; };
    std::vector<Pair> pairs;
    for (int ti = 0; ti < (int)tracks_.size(); ++ti) {
        const Track& t = tracks_[ti];
        for (int di = 0; di < (int)dets.size(); ++di) {
            if (shadow[di]) continue;
            const float S = t.p00 + measVar(dets[di].level);
            // крупная цель (грубый масштаб) — протяжённая, её отклик гуляет по корпусу
            const float gmin = cfg_.gate_min * (float)(1 << std::min(4, std::max(dets[di].level, t.level)));
            const float r = std::min(cfg_.gate, std::max(gmin, cfg_.gate_sigma * std::sqrt(S)));
            const float ex = dets[di].x - t.x, ey = dets[di].y - t.y, d2 = ex * ex + ey * ey;
            if (d2 < r * r) pairs.push_back({d2 / S, ti, di});
        }
    }
    std::stable_sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.d < b.d; });
    std::vector<int> match(tracks_.size(), -1), owner(dets.size(), -1);
    for (const auto& p : pairs)
        if (match[p.t] < 0 && owner[p.k] < 0) { match[p.t] = p.k; owner[p.k] = p.t; }

    // клаттер: всё, что не продолжает движущуюся цель, греет карту; в горячих ячейках —
    // помеха (её не отдаём ни трекам, ни в новые треки)
    auto mover = [&](const Track& t) { return t.confirmed && t.travel >= cfg_.mover_px; };
    std::vector<char> isMover(dets.size(), 0);
    for (size_t di = 0; di < dets.size(); ++di) {
        isMover[di] = owner[di] >= 0 && mover(tracks_[owner[di]]);
        if (!isMover[di] && !shadow[di] && !dets[di].weak) heat_[cellOf(dets[di])] += 1.f - decay;
    }
    // долгая память учится только при неподвижной камере: координаты ячеек — кадра
    const bool learn = cfg_.static_frames > 0 && std::fabs(camDx) < 0.5f && std::fabs(camDy) < 0.5f;
    clutterDets_.clear();
    std::vector<char> clut(dets.size(), 0);
    for (size_t di = 0; di < dets.size(); ++di) {
        if (isMover[di] || (heat_[cellOf(dets[di])] <= cfg_.clutter_level && !staticHot(dets[di].x, dets[di].y)))
            continue;
        clut[di] = 1;
        clutterDets_.push_back(dets[di]);
        if (learn && !shadow[di] && !dets[di].weak) staticAdd(cellOf(dets[di]));
        if (owner[di] >= 0) {
            match[owner[di]] = -1;
            owner[di] = -1;
        }
    }

    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        Track& tr = tracks_[ti];
        const int best = match[ti];
        tr.det = best;
        tr.hist.push_back(best >= 0);
        while ((int)tr.hist.size() > cfg_.window) tr.hist.pop_front();
        ++tr.age;
        if (best < 0) {
            ++tr.miss;
            continue;
        }
        const Candidate& d = dets[best];
        // обновление Калмана, измеряется только позиция
        const float R = measVar(d.level), S = tr.p00 + R;
        const float k0 = tr.p00 / S, k1 = tr.p01 / S;
        const float ex = d.x - tr.x, ey = d.y - tr.y;
        tr.x += k0 * ex;
        tr.y += k0 * ey;
        tr.vsx += k1 * ex;
        tr.vsy += k1 * ey;
        const float p00 = (1 - k0) * tr.p00, p01 = (1 - k0) * tr.p01, p11 = tr.p11 - k1 * tr.p01;
        tr.p00 = p00;
        tr.p01 = p01;
        tr.p11 = p11;
        tr.polarity = d.polarity;
        tr.contrast = d.contrast;
        tr.level = d.level;
        tr.miss = 0;
        ++tr.hits;
        tr.path.push_back({tr.x, tr.y});
        while (tr.path.size() > 30) tr.path.pop_front();
        const float sx = d.x - gx_, sy = d.y - gy_;
        tr.travel = std::max(tr.travel, std::hypot(sx - tr.bx, sy - tr.by));
        tr.obs.push_back({frame_, sx, sy, d.contrast, d.level, d.energy});
        while ((int)tr.obs.size() > std::max(cfg_.cls_window, cfg_.flap_window)) tr.obs.pop_front();
        // полетел — его отклики не помеха; стоит долго — уже ясно, что помеха
        if (mover(tr)) {
            tr.everMover = true;
            tr.pend.clear();
        } else if (learn && !tr.everMover) {
            tr.pend.push_back((uint32_t)cellOf(d));
            if (tr.pend.size() >= 256 && tr.id != lockId_) {
                for (uint32_t c : tr.pend) staticAdd(c);
                tr.pend.clear();
            }
        }
    }
    // трек умер, так и не полетев: его отклики — в долгую память
    if (cfg_.static_frames > 0)
        for (const Track& t : tracks_)
            if (t.miss > (t.confirmed ? coastLimit(t) : cfg_.tentative_miss) && !t.everMover && t.id != lockId_)
                for (uint32_t c : t.pend) staticAdd(c);
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Track& t) {
                                     return t.miss > (t.confirmed ? coastLimit(t) : cfg_.tentative_miss);
                                 }),
                  tracks_.end());
    // новые треки из свободных детекций; начальная скорость неизвестна — широкие ворота
    const float v0 = cfg_.gate / 3.f;
    for (int di = 0; di < (int)dets.size(); ++di) {
        // слабый отклик (пониженный порог у прогноза) только продолжает трек, новый не рождает
        if (owner[di] >= 0 || clut[di] || shadow[di] || dets[di].weak) continue;
        const Candidate& d = dets[di];
        // второй отклик той же цели (она легла на границу тайлов или найдена на двух
        // масштабах) нового трека не рождает
        const float dup = cfg_.dup_px * (float)(1 << std::min(4, d.level));
        bool near = false;
        for (const Track& o : tracks_)
            if (std::fabs(o.x - d.x) < dup && std::fabs(o.y - d.y) < dup) { near = true; break; }
        if (near) continue;
        Track t;
        t.id = nextId_++;
        t.x = d.x; t.y = d.y;
        t.polarity = d.polarity; t.contrast = d.contrast; t.level = d.level;
        t.p00 = measVar(d.level);
        t.p01 = 0;
        t.p11 = v0 * v0;
        t.hits = 1;
        t.det = di;
        t.bx = d.x - gx_;
        t.by = d.y - gy_;
        t.hist.push_back(true);
        t.path.push_back({t.x, t.y});
        t.obs.push_back({frame_, t.bx, t.by, t.contrast, d.level, d.energy});
        t.edgeBorn = d.edge;
        if (learn) t.pend.push_back((uint32_t)cellOf(d));
        tracks_.push_back(std::move(t));
    }
    // стая: по взмахам соседей с прошлого кадра
    for (auto& tr : tracks_) {
        tr.flock = false;
        const float v = std::hypot(tr.vsx, tr.vsy);
        if (cfg_.flock_px <= 0 || !tr.confirmed || v < 0.5f) continue;
        for (const auto& o : tracks_)
            if (&o != &tr && o.confirmed && frame_ - o.flapT <= cfg_.flap_hold &&
                std::fabs(o.x - tr.x) < cfg_.flock_px && std::fabs(o.y - tr.y) < cfg_.flock_px &&
                std::hypot(o.vsx - tr.vsx, o.vsy - tr.vsy) < cfg_.flock_dv * v) { tr.flock = true; break; }
    }
    out_.clear();
    std::vector<char> inTrack(dets.size(), 0);
    for (auto& tr : tracks_) {
        int nh = 0;
        for (bool b : tr.hist) nh += b;
        // "надёжный" трек: confirm попаданий из window; на пропусках признак сохраняется,
        // чтобы цель держалась на прогнозе, а не исчезала
        if (tr.miss == 0) tr.solid = nh >= cfg_.confirm;
        if (tr.solid) tr.confirmed = true;
        tr.clutter = tr.miss == 0 && !mover(tr) && (hot(tr.x, tr.y) || staticHot(tr.x, tr.y));
        tr.vx = tr.vsx + camDx;
        tr.vy = tr.vsy + camDy;
        tr.sigma = std::sqrt(std::max(0.f, tr.p00));
        classify(tr);
        // весомость: средний контраст попаданий, растёт со стажем до 10 попаданий
        double mc = 0;
        const size_t from = tr.obs.size() > (size_t)cfg_.cls_window ? tr.obs.size() - cfg_.cls_window : 0;
        for (size_t i = from; i < tr.obs.size(); ++i) mc += tr.obs[i].c;
        mc /= std::max<size_t>(1, tr.obs.size() - from);
        tr.score = tr.cls == TargetClass::Clutter ? 0.f
                                                  : (float)(mc * std::min(1.0, tr.hits / 10.0) /
                                                            (1.0 + 0.25 * tr.miss));
        // родился у кромки неба (кроны, крыши): цель, только если полетела
        if (!tr.solid || tr.miss > coastLimit(tr) || (tr.edgeBorn && !tr.everMover)) continue;
        out_.push_back(tr);
        if (tr.det >= 0) inTrack[tr.det] = 1;
    }
    // захват: держим текущую цель, пока она жива и не стала помехой
    const Track* cur = nullptr;
    const Track* best = nullptr;
    for (const Track& t : out_) {
        if (t.id == lockId_) cur = &t;
        if (t.score > 0 && (!best || t.score > best->score)) best = &t;
    }
    if (cur && cur->score <= 0) cur = nullptr;
    // ближайший живой трек к точке, где ждём захваченную цель (её же подхватил другой трек)
    auto nearest = [&](float x, float y, float r) {
        const Track* n = nullptr;
        float bd = r * r;
        for (const Track& t : out_) {
            if (t.score <= 0 || t.miss > 0 || &t == cur) continue;
            const float d = (t.x - x) * (t.x - x) + (t.y - y) * (t.y - y);
            if (d < bd) { bd = d; n = &t; }
        }
        return n;
    };
    if (cur) {
        lostFor_ = 0;
        lastX_ = cur->x; lastY_ = cur->y; lastVx_ = cur->vsx; lastVy_ = cur->vsy;
        if (cur->miss >= 2) {   // цель на прогнозе, а рядом её уже ведёт свежий трек
            if (const Track* n = nearest(cur->x, cur->y, std::max(cfg_.gate, 4 * cur->sigma))) cur = n;
        } else if (best && best != cur && best->score > cfg_.lock_switch * cur->score) {
            cur = best;
        }
        lockId_ = cur->id;
    } else if (lockId_ && lostFor_ < cfg_.max_miss) {
        // захваченный трек пропал: ищем цель там, где она должна быть, и там, где её видели
        // последний раз (если скорость была оценена неверно)
        ++lostFor_;
        lastX_ += camDx;
        lastY_ += camDy;
        const float r = cfg_.gate + 4.f * lostFor_;
        const Track* n = nearest(lastX_ + lastVx_ * lostFor_, lastY_ + lastVy_ * lostFor_, r);
        if (!n) n = nearest(lastX_, lastY_, r);
        if (n) { lockId_ = n->id; lostFor_ = 0; }
    } else {
        lockId_ = best ? best->id : 0;
        lostFor_ = 0;
        if (best) { lastX_ = best->x; lastY_ = best->y; lastVx_ = best->vsx; lastVy_ = best->vsy; }
    }

    // шум: детекции, которые не помеха и не продолжают подтверждённый трек
    noiseDets_.clear();
    for (size_t di = 0; di < dets.size(); ++di)
        if (!inTrack[di] && !clut[di] && !dets[di].weak) noiseDets_.push_back(dets[di]);
    return out_;
}

}  // namespace dronolok
