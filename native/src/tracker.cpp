// Трекер ближайшего соседа + карта статичного клаттера (порт Tracker из
// detector/detector.py) и классификатор треков по характеру движения.
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

void Tracker::resetTracks() { tracks_.clear(); }

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
    if (t.clutter) {
        t.cls = TargetClass::Clutter;
        return;
    }
    if (n < cfg_.cls_min_hits) {
        if (t.cls == TargetClass::Clutter) t.cls = TargetClass::Unknown;
        return;
    }
    float e = std::max(t.resid / cfg_.bird_resid, t.contrastCv / cfg_.bird_contrast_cv);
    if (cfg_.bird_speed > 0) e = std::max(e, t.speed / cfg_.bird_speed);
    const float inst = e > 1.f ? 1.f : 0.f;
    // сглаживание, чтобы класс не мигал от кадра к кадру
    if (t.cls == TargetClass::Unknown || t.cls == TargetClass::Clutter) t.birdScore = inst;
    else t.birdScore = 0.8f * t.birdScore + 0.2f * inst;
    t.cls = t.birdScore > 0.5f ? TargetClass::Bird : TargetClass::Drone;
}

const std::vector<Track>& Tracker::update(const std::vector<Candidate>& dets) {
    ++frame_;
    const float decay = cfg_.clutter_decay;
    for (auto& v : heat_) v *= decay;
    const int cell = cfg_.clutter_cell;
    auto cellOf = [&](const Candidate& d) {
        const int cx = std::min(gw_ - 1, std::max(0, (int)d.x / cell));
        const int cy = std::min(gh_ - 1, std::max(0, (int)d.y / cell));
        return (size_t)cy * gw_ + cx;
    };
    for (const auto& d : dets) heat_[cellOf(d)] += 1.f - decay;
    // детекции в "горячих" ячейках (статичный клаттер) в трекинг не идут
    clutterDets_.clear();
    std::vector<Candidate> live;
    live.reserve(dets.size());
    for (const auto& d : dets) {
        if (heat_[cellOf(d)] > cfg_.clutter_level) clutterDets_.push_back(d);
        else live.push_back(d);
    }
    // ассоциация: пары трек-детекция ближе gate, жадно по возрастанию расстояния
    struct Pair { float d; int t, k; };
    std::vector<Pair> pairs;
    for (int ti = 0; ti < (int)tracks_.size(); ++ti) {
        const float px = tracks_[ti].x + tracks_[ti].vx, py = tracks_[ti].y + tracks_[ti].vy;
        for (int di = 0; di < (int)live.size(); ++di) {
            const float dd = std::hypot(px - live[di].x, py - live[di].y);
            if (dd < cfg_.gate) pairs.push_back({dd, ti, di});
        }
    }
    std::stable_sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.d < b.d; });
    std::vector<int> match(tracks_.size(), -1);
    std::vector<char> used(live.size(), 0);
    for (const auto& p : pairs)
        if (match[p.t] < 0 && !used[p.k]) { match[p.t] = p.k; used[p.k] = 1; }

    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        Track& tr = tracks_[ti];
        const int best = match[ti];
        tr.det = best;
        tr.hist.push_back(best >= 0);
        while ((int)tr.hist.size() > cfg_.window) tr.hist.pop_front();
        ++tr.age;
        if (best >= 0) {
            const Candidate& d = live[best];
            tr.vx = 0.5f * tr.vx + 0.5f * (d.x - tr.x);
            tr.vy = 0.5f * tr.vy + 0.5f * (d.y - tr.y);
            tr.x = d.x; tr.y = d.y; tr.polarity = d.polarity; tr.contrast = d.contrast;
            tr.miss = 0;
            ++tr.hits;
            tr.path.push_back({d.x, d.y});
            while (tr.path.size() > 30) tr.path.pop_front();
            tr.obs.push_back({frame_, d.x, d.y, d.contrast});
            while ((int)tr.obs.size() > cfg_.cls_window) tr.obs.pop_front();
        } else {
            ++tr.miss;
            tr.x += tr.vx;
            tr.y += tr.vy;
        }
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Track& t) { return t.miss > cfg_.max_miss; }),
                  tracks_.end());
    for (int di = 0; di < (int)live.size(); ++di) {
        if (used[di]) continue;
        Track t;
        t.id = nextId_++;
        t.x = live[di].x; t.y = live[di].y;
        t.polarity = live[di].polarity; t.contrast = live[di].contrast;
        t.hits = 1;
        t.det = di;
        t.hist.push_back(true);
        t.path.push_back({t.x, t.y});
        t.obs.push_back({frame_, t.x, t.y, t.contrast});
        tracks_.push_back(std::move(t));
    }
    out_.clear();
    std::vector<char> inTrack(live.size(), 0);
    for (auto& tr : tracks_) {
        tr.clutter = tr.miss == 0 && hot(tr.x, tr.y);
        classify(tr);
        int nh = 0;
        for (bool b : tr.hist) nh += b;
        if (nh < cfg_.confirm || tr.miss > 0) continue;
        out_.push_back(tr);
    }
    // шум: живые детекции, которые не продолжают подтверждённый трек
    noiseDets_.clear();
    for (const auto& t : out_) inTrack[t.det] = 1;
    for (int di = 0; di < (int)live.size(); ++di)
        if (!inTrack[di]) noiseDets_.push_back(live[di]);
    return out_;
}

}  // namespace dronolok
