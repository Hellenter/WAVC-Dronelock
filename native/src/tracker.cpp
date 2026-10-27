// Трекер: фильтр Калмана с постоянной скоростью на каждую цель, ассоциация по
// прогнозу, удержание трека на прогнозе при пропусках, компенсация движения камеры,
// карта статичного клаттера и классификатор треков по характеру движения.
//
// Отличия от прототипа (detector/detector.py, ближайший сосед): ворота сужаются,
// когда трек захвачен, и расширяются при пропусках; подтверждённый трек не пропадает
// при одном-двух пропущенных кадрах; цель, которая уже пролетела по сцене, не
// записывается в помехи, даже если зависла; карта клаттера едет вместе с камерой.
//
// Классификатор: по последним cls_window попаданиям трека строится плавная
// траектория (квадратичная по времени). Дрон летит плавно и отклоняется от неё
// на 1-2 пикселя (квантование пулинга), ласточка дёргается и разворачивается,
// а взмахи крыльев меняют её контраст. Пороги — в TrackerConfig.
#include <algorithm>
#include <cmath>

#include "dronolok/detector.hpp"

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

// МНК-парабола p(t) = a + b t + c t^2; возвращает сумму квадратов остатков, b — скорость в середине окна
static double fit_quad(const std::deque<Track::Obs>& o, bool useX, double& slope) {
    const int n = (int)o.size();
    const double tm = 0.5 * (o.front().t + o.back().t);
    double S[5] = {0, 0, 0, 0, 0}, B[3] = {0, 0, 0};
    for (const auto& q : o) {
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
    for (const auto& q : o) {
        const double t = q.t - tm, v = useX ? q.x : q.y;
        const double r = v - (coef[0] + coef[1] * t + coef[2] * t * t);
        ss += r * r;
    }
    return ss;
}

void Tracker::classify(Track& t) const {
    const auto& o = t.obs;
    const int n = (int)o.size();
    if (n >= 3) {
        double sx, sy;
        const double ss = fit_quad(o, true, sx) + fit_quad(o, false, sy);
        t.resid = (float)std::sqrt(ss / n);
        t.speed = (float)std::hypot(sx, sy);
        double mc = 0, vc = 0;
        for (const auto& q : o) mc += q.c;
        mc /= n;
        for (const auto& q : o) vc += (q.c - mc) * (q.c - mc);
        t.contrastCv = mc > 0 ? (float)(std::sqrt(vc / n) / mc) : 0.f;
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
    for (const auto& q : o) lf += (double)(1 << std::min(4, q.level));
    lf /= n;
    float e = std::max(t.resid / (cfg_.bird_resid * (float)lf), t.contrastCv / cfg_.bird_contrast_cv);
    if (cfg_.bird_speed > 0) e = std::max(e, t.speed / cfg_.bird_speed);
    const float inst = e > 1.f ? 1.f : 0.f;
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

const std::vector<Track>& Tracker::update(const std::vector<Candidate>& dets, float camDx, float camDy) {
    ++frame_;
    gx_ += camDx;
    gy_ += camDy;
    const float q = cfg_.q_accel * cfg_.q_accel;
    const float decay = cfg_.clutter_decay;
    shiftHeat(camDx, camDy);
    for (auto& v : heat_) v *= decay;
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
    clutterDets_.clear();
    std::vector<char> clut(dets.size(), 0);
    for (size_t di = 0; di < dets.size(); ++di) {
        if (isMover[di] || heat_[cellOf(dets[di])] <= cfg_.clutter_level) continue;
        clut[di] = 1;
        clutterDets_.push_back(dets[di]);
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
        tr.obs.push_back({frame_, sx, sy, d.contrast, d.level});
        while ((int)tr.obs.size() > cfg_.cls_window) tr.obs.pop_front();
    }
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
        t.obs.push_back({frame_, t.bx, t.by, t.contrast, d.level});
        tracks_.push_back(std::move(t));
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
        tr.clutter = tr.miss == 0 && !mover(tr) && hot(tr.x, tr.y);
        tr.vx = tr.vsx + camDx;
        tr.vy = tr.vsy + camDy;
        tr.sigma = std::sqrt(std::max(0.f, tr.p00));
        classify(tr);
        // весомость: средний контраст попаданий, растёт со стажем до 10 попаданий
        double mc = 0;
        for (const auto& q : tr.obs) mc += q.c;
        mc /= std::max<size_t>(1, tr.obs.size());
        tr.score = tr.cls == TargetClass::Clutter ? 0.f
                                                  : (float)(mc * std::min(1.0, tr.hits / 10.0) /
                                                            (1.0 + 0.25 * tr.miss));
        if (!tr.solid || tr.miss > coastLimit(tr)) continue;
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
