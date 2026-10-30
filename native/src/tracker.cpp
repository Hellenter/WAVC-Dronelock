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

TrackerConfig tracker_for_fps(const TrackerConfig& c0, float fps, float ref, float ufps) {
    TrackerConfig c = c0;
    // медленная камера: новый захват — только на летавший трек (см. lock_mover_fps)
    if (fps > 0 && fps <= c.lock_mover_fps) c.lock_mover = true;
    if (!(fps > 0) || !(ref > 0)) return c;
    // предел ворот — не меньше пути цели за кадр при gate_pps пикс./с (на 10 к/с — 36 пикс.)
    if (c.gate_pps > 0) c.gate = std::max(c.gate, c.gate_pps / fps);
    // окно ведомой цели: заданное — от 30 к/с и выше, ниже — во времени, но не меньше 9
    if (c.follow_min_obs > 0)
        c.follow_min_obs = std::min(c.follow_min_obs, std::max(9, (int)std::lround(c.follow_min_obs * fps / 30.f)));
    // правила на кадр жизни трека — по частоте вызовов update
    const float vu = ref / (ufps > 0 ? ufps : fps);
    c.still_speed *= vu;
    c.flock_speed *= vu;
    c.lost_grow *= vu;
    if (c.creep_frames > 0) c.creep_frames = std::max(4, (int)std::lround((float)c.creep_frames / vu));
    if (std::fabs(fps - ref) < 0.01f * ref) return c;
    const float t = fps / ref, v = ref / fps;   // t — кадров в той же секунде, v — пикс. за кадр
    auto frames = [&](int n, int lo) { return n > 0 ? std::max(lo, (int)std::lround(n * t)) : n; };
    // по физике ворота растут как v, а СКО ускорения в пикс./кадр^2 как v^2. Замер на 10 к/с
    // (30 с синтетики): так птиц находится больше всех (88%), но 12% попаданий по дронам
    // уходят в "птицу" — трек слишком вольно гнётся. С v^0.5 и v^1.5 птиц 77% (без пересчёта
    // 51%), дронов-"птиц" 0, ложных тревог столько же.
    // Предел ворот gate — в пикселях кадра без пересчёта по v (ночной тест 30.10; ниже 14 к/с — путь за кадр,
    // gate_pps): ворота по ошибке прогноза растут и так (q_accel), а предел 40·√v на 14 к/с (53 пикс.)
    // цеплял чужие отклики — без дронов кадров с захватом 19 -> 3%, на 60 к/с 20·√v (13 пикс.) рвал
    // ведомую цель (wt17, wt16)
    c.q_accel *= v * std::sqrt(v);
    c.bird_speed *= v;
    c.hover_speed *= v;
    c.lock_slow *= v;
    c.drift_min *= v;
    c.size_rate_max = std::min(0.5f, c.size_rate_max * v);
    c.clutter_decay = std::pow(c.clutter_decay, v);
    c.max_miss = frames(c.max_miss, 3);
    c.static_frames = frames(c.static_frames, 100);
    c.quiet_frames = frames(c.quiet_frames, 10);
    c.quiet_max = frames(c.quiet_max, 100);
    c.flap_hold = frames(c.flap_hold, 5);
    c.flap_max_period = c.flap_max_period * t;
    // окно взмахов: не короче 32 кадров — на 10 к/с частые взмахи приходят сложенными в
    // низкие частоты, на их период нужно несколько секунд
    c.flap_window = c.flap_window > 0 ? std::min(64, std::max(32, (int)std::lround(c.flap_window * t))) : 0;
    c.verify_frames = frames(c.verify_frames, std::max(1, c.verify_min));
    c.drift_obs = std::min(40, frames(c.drift_obs, 8));
    c.lock_dwell = frames(c.lock_dwell, 3);
    c.hard_miss = frames(c.hard_miss, 6);
    c.hard_full_every = frames(c.hard_full_every, 1);
    return c;
}

Tracker::Tracker(int w, int h, const TrackerConfig& cfg) : cfg_(cfg) {
    cfg_.clutter_cell = std::max(1, cfg_.clutter_cell);   // из C API/Python может прийти 0
    frameW_ = w;
    frameH_ = h;
    gw_ = w / cfg_.clutter_cell + 1;
    gh_ = h / cfg_.clutter_cell + 1;
    heat_.assign((size_t)gw_ * gh_, 0.f);
    stat_.assign(heat_.size(), 0.f);
    statT_.assign(heat_.size(), 0);
    lastDet_.assign(heat_.size(), 0);
}

void Tracker::resetTracks() {
    tracks_.clear();
    out_.clear();
    lockId_ = 0;
    beat_.clear();
    lostFor_ = 0;
    lastLsN_ = 0;
    verify_ = 0;
    hard_ = false;
    manual_ = false;
    lockSilh_ = Silhouette();
    driftX_ = driftY_ = 0;
    driftOn_ = false;
}

void Tracker::resetTracksKeepFollowed() {
    std::vector<Track> keep;
    for (Track& t : tracks_)
        if (t.everFollowed) keep.push_back(std::move(t));
    const int lock = lockId_;
    const bool hard = hard_;
    const int verify = verify_;
    const bool manual = manual_;
    resetTracks();
    tracks_.swap(keep);
    for (const Track& t : tracks_)
        if (t.id == lock) lockId_ = lock, hard_ = hard, verify_ = verify, manual_ = manual;
}

void Tracker::dropMasked(const std::vector<uint8_t>& mask, int tw, int th, int tilePx) {
    if (mask.size() != (size_t)tw * th || tilePx <= 0) return;
    auto masked = [&](const Track& t) {
        const int tx = (int)t.x / tilePx, ty = (int)t.y / tilePx;
        return t.id != lockId_ && tx >= 0 && ty >= 0 && tx < tw && ty < th && !mask[(size_t)ty * tw + tx];
    };
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), masked), tracks_.end());
    out_.erase(std::remove_if(out_.begin(), out_.end(), masked), out_.end());
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

void Tracker::getStatic(std::vector<float>& out) const {
    out.resize(stat_.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = staticAt(i);
}

void Tracker::setStatic(const std::vector<float>& v) {
    if (v.size() != stat_.size()) return;
    stat_ = v;
    std::fill(statT_.begin(), statT_.end(), frame_);
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

// Файл: "DLSM", версия, ширина и высота сетки, ячейка, затем значения по ячейкам. Версия 3:
// файлы версии 2 могли запомнить сам дрон (камера вела его, и он стоял в кадре) — не грузятся
bool Tracker::saveStatic(const std::string& path) const {
    FILE* f = open_file(path, "wb");
    if (!f) return false;
    const int32_t hdr[5] = {0x4D534C44, 3, gw_, gh_, cfg_.clutter_cell};
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
    const bool ok = fread(hdr, sizeof(hdr), 1, f) == 1 && hdr[0] == 0x4D534C44 && hdr[1] == 3 && hdr[2] == gw_ &&
                    hdr[3] == gh_ && hdr[4] == cfg_.clutter_cell && fread(v.data(), sizeof(float), v.size(), f) == v.size();
    fclose(f);
    if (!ok) return false;
    for (float& x : v)   // испорченный файл: nan/inf/минус не должны попадать в память помех
        if (!(x > 0 && x < 1e30f)) x = 0;
    stat_ = std::move(v);
    std::fill(statT_.begin(), statT_.end(), frame_);
    return true;
}

// МНК-парабола p(t) = a + b t + c t^2; возвращает сумму квадратов остатков, b — скорость в середине окна
static double fit_quad(const std::deque<Track::Obs>& obs, size_t from, bool useX, double& slope, bool frame = false) {
    const auto o0 = obs.begin() + (std::ptrdiff_t)from;
    const int n = (int)(obs.size() - from);
    const double tm = 0.5 * (o0->t + obs.back().t);
    double S[5] = {0, 0, 0, 0, 0}, B[3] = {0, 0, 0};
    for (auto it = o0; it != obs.end(); ++it) {
        const auto& q = *it;
        const double t = q.t - tm, v = frame ? (useX ? q.fx : q.fy) : (useX ? q.x : q.y);
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
        const double t = q.t - tm, v = frame ? (useX ? q.fx : q.fy) : (useX ? q.x : q.y);
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

static FlapFit flap_fit(const std::deque<Track::Obs>& obs, int W, float maxPeriod) {
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
    const float pmax = std::max(2.5f, std::min(16.f, maxPeriod));
    const int k0 = pmax >= 16.f ? steps / 16 : (int)std::ceil(steps / pmax);
    for (int k = std::max(1, k0); k <= steps / 2; ++k) {
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

// середина кадра: камеру за целью ведут к центру, а надписи и логотипы записи экрана — у краёв
bool Tracker::midFrame(float x, float y) const {
    return std::fabs(x - 0.5f * frameW_) < 0.25f * frameW_ && std::fabs(y - 0.5f * frameH_) < 0.3f * frameH_;
}

// Камера ведёт цель: за последние до 48 попаданий камера (разность координат кадра и сцены)
// прошла заметный путь, а цель по кадру — меньше follow_frac этого пути. Путь — сумма 8 отрезков:
// игра и оператор водят камеру за целью туда-обратно (по сцене цель "летает" на месте, пробег
// мал), а дрожь замера по кадру в путь не копится — с рук неподвижное ведомым не становится
bool Tracker::followedNow(const Track& t) const {
    if (cfg_.follow_k <= 0 || !movingCam_ || (int)t.obs.size() < std::max(9, cfg_.follow_min_obs)) return false;
    const int n = (int)std::min<size_t>(48, t.obs.size()), a = (int)t.obs.size() - n;
    float cam = 0, frame = 0;
    for (int s = 0; s < 8; ++s) {
        const auto& p = t.obs[a + (n - 1) * s / 8];
        const auto& q = t.obs[a + (n - 1) * (s + 1) / 8];
        cam += std::hypot((q.fx - q.x) - (p.fx - p.x), (q.fy - q.y) - (p.fy - p.y));
        frame += std::hypot(q.fx - p.fx, q.fy - p.fy);
    }
    // и держит её в середине кадра: у краёв стоят надписи и логотипы (запись экрана)
    return cam > cfg_.mover_px && frame < cfg_.follow_frac * cam && !t.overlay && midFrame(t.obs.back().fx, t.obs.back().fy);
}

void Tracker::lockQueue(std::vector<int>& ids) const {
    std::vector<const Track*> q;
    for (const Track& t : out_)
        if (t.score > 0 && t.x >= 0 && t.y >= 0 && t.x < (float)frameW_ && t.y < (float)frameH_) q.push_back(&t);
    std::stable_sort(q.begin(), q.end(), [](const Track* a, const Track* b) { return a->lockW > b->lockW; });
    ids.clear();
    for (const Track* t : q) ids.push_back(t->id);
}

bool Tracker::selectLock(int id) {
    for (const Track& t : out_) {
        if (t.id != id || t.score <= 0) continue;
        if (id != lockId_) hard_ = false, verify_ = 0, lastLsN_ = 0;
        lockId_ = id;
        manual_ = true;
        lostFor_ = 0;
        beat_.clear();
        lastX_ = t.x; lastY_ = t.y; lastVx_ = t.vsx; lastVy_ = t.vsy;
        return true;
    }
    return false;
}

int Tracker::nextLock() {
    std::vector<int> q;
    lockQueue(q);
    if (q.empty()) return 0;
    const auto it = std::find(q.begin(), q.end(), lockId_);
    const int id = it == q.end() || it + 1 == q.end() ? q.front() : *(it + 1);
    return selectLock(id) ? id : 0;
}

void Tracker::classify(Track& t) const {
    const auto& o = t.obs;
    const size_t from = o.size() > (size_t)cfg_.cls_window ? o.size() - cfg_.cls_window : 0;
    const int n = (int)(o.size() - from);
    if (n >= 3) {
        double sx, sy;
        const double ss = fit_quad(o, from, true, sx) + fit_quad(o, from, false, sy);
        t.resid = (float)std::sqrt(ss / n);
        if (cfg_.resid_frame) {   // по кадру ровнее (камера ведёт цель) — шум поворота не в счёт
            double fx, fy;
            const double sf = fit_quad(o, from, true, fx, true) + fit_quad(o, from, false, fy, true);
            t.resid = std::min(t.resid, (float)std::sqrt(sf / n));
        }
        t.speed = (float)std::hypot(sx, sy);
        double mc = 0, vc = 0;
        for (size_t i = from; i < o.size(); ++i) mc += o[i].c;
        mc /= n;
        for (size_t i = from; i < o.size(); ++i) vc += (o[i].c - mc) * (o[i].c - mc);
        t.contrastCv = mc > 0 ? (float)(std::sqrt(vc / n) / mc) : 0.f;
    }
    if (cfg_.flap_window > 0 && !o.empty()) {
        const FlapFit ff = flap_fit(o, cfg_.flap_window, cfg_.flap_max_period);
        t.flapR2 = ff.r2;
        t.flapPeriod = ff.period;
        t.flapDepth = ff.depth;
        if (ff.r2 >= cfg_.flap_r2 && ff.depth >= cfg_.flap_depth) t.flapT = frame_;
    }
    // стоит в горячей ячейке или за cls_min_hits попаданий так и не сдвинулся по сцене
    // (фонарь, угол дома, когда камера ведёт панораму) — помеха; наложение на кадр — тоже.
    // Цель жёсткого захвата уже признана дроном: зависшая, она помехой не становится
    // раз камера вела цель, она остаётся "ведомой", пока камера в движении: игра и оператор
    // останавливают поворот, когда цель зависла, — по сцене она тогда стоит
    // Потерянный (на прогнозе дольше 4 кадров) или ушедший из середины кадра — уже не ведомый:
    // с рук камера движется всегда, и прогноз потерянной цели уплывает с её сдвигом
    t.followed = movingCam_ && t.miss <= 4 && midFrame(t.obs.back().fx, t.obs.back().fy) &&
                 (t.followed || followedNow(t));
    t.everFollowed |= t.followed;
    // цель, которую ведёт или вела камера, — не помеха: остановив поворот, камера держит
    // зависшую цель, и по сцене она стоит
    const bool held = (hard_ && t.id == lockId_) || t.everFollowed;
    const bool qh = quietHold(t);
    if (t.overlay || t.anchored ||
        (!held && (t.clutter || (!qh && !creeper(t) && n >= cfg_.cls_min_hits &&
                                 t.travel < std::max(3.f, cfg_.still_speed * (float)t.age))))) {
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
    // зависшая после полёта (hover_first): дрон, даже если контраст качнулся похоже на взмахи
    const bool hoverNow = cfg_.hover_first && cfg_.hover_speed > 0 && t.everMover && t.speed < cfg_.hover_speed;
    const bool flapped = frame_ - t.flapT <= cfg_.flap_hold && !hoverNow;
    t.hovering = cfg_.hover_speed > 0 && (t.everMover || qh) && t.speed < cfg_.hover_speed;
    float inst = e > 1.f ? 1.f : 0.f;
    if (flapped || t.flock) inst = 1.f;
    else if (t.hovering) inst = 0.f;
    // сглаживание, чтобы класс не мигал от кадра к кадру
    if (t.cls == TargetClass::Unknown || t.cls == TargetClass::Clutter) t.birdScore = inst;
    else t.birdScore = 0.8f * t.birdScore + 0.2f * inst;
    t.cls = t.birdScore > 0.5f ? TargetClass::Bird : TargetClass::Drone;
}

// Скорость цели для захвата, пикс./кадр по сцене; пока есть дрейф фона — относительно него
float Tracker::relSpeed(const Track& t) const {
    return driftOn_ ? std::hypot(t.vsx - driftX_, t.vsy - driftY_) : t.speed;
}

// Сколько кадров подтверждённый трек держится на прогнозе: короткий (возможно, шум) —
// 1 кадр, устоявшийся и летящий по сцене — столько, сколько было попаданий, но не больше max_miss
int Tracker::coastLimit(const Track& t) const {
    if (hard_ && t.id == lockId_) return std::max(1, cfg_.hard_miss);
    const int lim = t.hits < cfg_.cls_min_hits || (t.travel < cfg_.mover_px && !quietHold(t)) ? 1 : std::min(cfg_.max_miss, t.hits);
    // захваченную цель не бросаем после одного пропуска: у крупного дрона детектор
    // ловит то одну, то другую часть, и молодой трек иначе рвётся (а при 25 к/с —
    // на каждом втором захвате)
    return t.id == lockId_ ? std::max(lim, std::min(cfg_.max_miss, 4)) : lim;
}

// Родился у кромки крон и сейчас ходит туда-сюда, а не летит (см. edge_straight)
bool Tracker::edgeJitter(const Track& t) const {
    return t.edgeBorn && cfg_.edge_straight > 0 && t.obs.size() >= 16 && t.straight < cfg_.edge_straight &&
           !(hard_ && t.id == lockId_);
}

// Трек спокойного неба (см. quiet_frames) ещё не простоял quiet_max кадров
bool Tracker::quietHold(const Track& t) const {
    return t.quiet && t.age < cfg_.quiet_max && !t.overlay && !t.anchored;
}

// Ползёт при неподвижной камере: ушёл по сцене дальше creep_px (и дальше пути камеры за то же
// время) — медленно летящий дрон, а не помеха, хоть и медленнее still_speed. Пятно на земле,
// плывущее от неточной оценки поворота (из-за него still_speed), уходит на 10-20% пути камеры
bool Tracker::creeper(const Track& t) const {
    return cfg_.creep_frames > 0 && t.travel >= cfg_.creep_px && t.travel >= t.camPath && !t.overlay && !t.anchored &&
           !edgeJitter(t);
}

// Медленная цель в горячей ячейке: трек, начатый до того, как ячейка стала помехой, ведёт её
// отклики дальше (класс — помеха, захвата нет), пока не станет ясно, ползёт она или стоит.
// Иначе почти зависший дрон греет карту, трек на нём умирает и пробег не копится никогда:
// карта и долгая память ползут за ним следом (wt12, дрон 4: 0.2 пикс./кадр, 7 с без трека).
// Только при неподвижной камере: при повороте в горячих ячейках — земля, и её трек, оставленный
// жить, "летит" от ошибки оценки сдвига. Отклик — рядом с прошлой отметкой (ползёт, а не прыгает):
// иначе трек помехи перескакивает на соседнее пятно и за пару скачков "пролетает" mover_px (n1)
bool Tracker::creepKeep(const Track& t, const Candidate& d) const {
    if (cfg_.creep_frames <= 0 || !t.confirmed || t.overlay || t.anchored || t.edgeBorn || t.camPath >= cfg_.mover_px ||
        t.obs.empty() || (t.age >= cfg_.creep_frames && t.travel < cfg_.creep_px))
        return false;
    const auto& o = t.obs.back();
    const float gap = (float)std::max(1, frame_ - o.t);
    return std::hypot(d.x - o.fx, d.y - o.fy) <= cfg_.creep_step * gap;
}

// вклад попадания в уверенность: ~ логарифм отношения правдоподобия "цель / шум" для
// отклика в margin раз выше порога. Кандидат не от Detector (margin неизвестен) —
// подтверждается как раньше, по одному счёту попаданий.
float Tracker::evidenceOf(const Candidate& d) const {
    if (d.margin <= 0) return cfg_.ev_confirm;
    return std::min(9.f, d.margin * d.margin) - 1.f + cfg_.ev_hit;
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

// Прогноз позиции и её дисперсии без изменения трека (те же формулы, что kf_predict): трек
// не копируется ради прогноза — в нём очереди точек, копия выделяет память на каждый трек
struct KfPeek { float x, y, p00; };
static inline KfPeek kf_peek(const Track& t, float camDx, float camDy, float q) {
    return {t.x + (t.vsx + camDx), t.y + (t.vsy + camDy), t.p00 + 2 * t.p01 + t.p11 + 0.25f * q};
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
    for (const Track& t : tracks_) {
        // только устоявшиеся летящие цели: иначе пониженный порог подкармливает шумовые треки
        if (!t.solid || t.cls == TargetClass::Clutter) continue;
        if (t.id != lockId_ && (t.hits < cfg_.cls_min_hits || (t.travel < cfg_.mover_px && !quietHold(t)))) continue;
        const KfPeek k = kf_peek(t, camDx, camDy, q);
        const float S = k.p00 + measVar(t.level);
        const float r = std::min(gateCap(t), std::max(cfg_.gate_min, cfg_.gate_sigma * std::sqrt(S)));
        out.push_back({k.x, k.y, r});
    }
}

void Tracker::roi(float camDx, float camDy, std::vector<FocusArea>& out) const {
    out.clear();
    const float q = cfg_.q_accel * cfg_.q_accel;
    for (const Track& t : tracks_) {
        // жёсткий захват: весь кадр смотрим редко, но вокруг других живых треков — каждый кадр,
        // иначе вторая цель (дальний дрон рядом с захваченным самолётом) не набирает попаданий
        // на подтверждение и не находится никогда. Помехи и наложения пропускаем.
        if (hard_ && t.id != lockId_ && (t.clutter || t.overlay || t.cls == TargetClass::Clutter)) continue;
        const KfPeek k = kf_peek(t, camDx, camDy, q);
        const float S = k.p00 + measVar(t.level);
        const float gmin = cfg_.gate_min * (float)(1 << std::min(4, t.level));
        // захват с ожиданием размера: смотрим на масштабе, подходящем размеру на этом кадре
        const float e = cfg_.size_predict && t.id == lockId_ ? t.expectedSize() : 0.f;
        int lv = t.level;
        if (e > 0) {
            lv = 0;
            while (lv < 4 && e > 14.f * (float)(1 << lv)) ++lv;
        }
        // запас на размер цели: на грубом масштабе её отклик гуляет по корпусу
        out.push_back({k.x, k.y,
                       std::min(gateCap(t), std::max(gmin, cfg_.gate_sigma * std::sqrt(S))) + 4.f * (1 << lv) +
                           0.5f * std::max(e, std::max(t.bw, t.bh)),
                       lv});
    }
}

// Силуэт ищем у цели, которая уже похожа на цель: устоялась и не помеха. Иначе силуэт
// надписи или кроны "ползёт" по ней, и неподвижная помеха выглядит летящей
bool Tracker::silhouetteFor(const Track& t) const {
    return (hard_ && t.id == lockId_) ||
           (t.hits >= cfg_.cls_min_hits && !t.clutter && !t.overlay && t.cls != TargetClass::Clutter && t.everMover);
}

bool Tracker::lockSearch(float camDx, float camDy, float& x, float& y, float& r, int& polarity) const {
    if (!lockId_ || !(cfg_.silhouette || cfg_.anchor_check)) return false;
    for (const Track& t : tracks_) {
        if (t.id != lockId_ || t.anchored || (!silhouetteFor(t) && !cfg_.anchor_check)) continue;
        const KfPeek k = kf_peek(t, camDx, camDy, cfg_.q_accel * cfg_.q_accel);
        x = k.x;
        y = k.y;
        // ожидаемый радиус: прошлый силуэт или размер по масштабу, плюс ошибка прогноза
        const float size = t.bw > 0 ? 0.5f * std::max(t.bw, t.bh) : 4.f * (float)(1 << std::min(4, t.level));
        r = size + 2.f * std::sqrt(std::max(0.f, k.p00));
        polarity = t.polarity;
        return true;
    }
    return false;
}

bool Tracker::aim(float& x, float& y, float& vx, float& vy) const {
    for (const Track& t : out_) {
        if (t.id != lockId_) continue;
        x = t.x;
        y = t.y;
        vx = t.vsx;
        vy = t.vsy;
        return true;
    }
    return false;
}

const std::vector<Track>& Tracker::update(const std::vector<Candidate>& dets0, float camDx, float camDy) {
    ++frame_;
    // сдвиг камеры известен (учёт движения, панорама): дрейф фона не оцениваем — его уже вычли
    const bool camKnown = camDx != 0 || camDy != 0 || sceneDx_ != 0 || sceneDy_ != 0;
    gx_ += camDx + sceneDx_;
    gy_ += camDy + sceneDy_;
    const float q = cfg_.q_accel * cfg_.q_accel;
    const float decay = cfg_.clutter_decay;
    // карта помех привязана к сцене: цель, которую ведёт камера, не "греет" одну ячейку
    shiftHeat(camDx + sceneDx_, camDy + sceneDy_);
    sceneDx_ = sceneDy_ = 0;
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

    // Силуэт захваченной цели: его центр — измерение для захваченного трека (добавляется
    // к детекциям), отклики внутри силуэта — части той же цели, треки на них не нужны
    std::vector<Candidate> dets = dets0;
    std::vector<char> part(dets.size(), 0);
    int silhDet = -1, lockIdx = -1;
    float bx0 = 0, bx1 = 0, by0 = 0, by1 = 0;
    for (int i = 0; i < (int)tracks_.size(); ++i)
        if (tracks_[i].id == lockId_) lockIdx = i;
    const Silhouette sil = lockSilh_;
    lockSilh_ = Silhouette();
    // силуэт, резко сменивший размер, — захватил соседнее (облако, крону): не верим
    // (с ожиданием размера — не похожий по размеру на ожидаемый, с учётом приближения)
    bool silOk = sil.valid && (lockIdx < 0 || tracks_[lockIdx].bw <= 0 ||
                               std::max(sil.w, sil.h) < 2.5f * std::max(tracks_[lockIdx].bw, tracks_[lockIdx].bh) + 16.f);
    // С ожиданием размера верхняя граница — от ожидаемого размера (сглаженного, с учётом
    // приближения), а не от прошлого замера: силуэт тонкого крыла скачет от кадра к кадру.
    // Меньший силуэт принимается — это часть цели (крыло слилось с небом)
    if (sil.valid && lockIdx >= 0 && cfg_.size_predict && tracks_[lockIdx].lsN >= 3)
        silOk = std::max(sil.w, sil.h) < cfg_.size_tol * tracks_[lockIdx].expectedSize() + 16.f;
    // захваченное на мачте или столбе: большинство замеров с линией от силуэта
    if (cfg_.anchor_check && lockIdx >= 0 && (sil.valid || sil.attached)) {
        Track& L = tracks_[lockIdx];
        if (L.silN >= 30) { L.silN /= 2; L.attN /= 2; }
        ++L.silN;
        L.attN += sil.attached;
        if (L.silN >= 4 && L.attN * 10 >= L.silN * 6) L.anchored = true;
    }
    if (cfg_.silhouette && lockIdx >= 0 && silOk && silhouetteFor(tracks_[lockIdx]) && !tracks_[lockIdx].anchored) {
        Track& L = tracks_[lockIdx];
        const float size = std::max(sil.w, sil.h);
        const float m = std::max(6.f, 0.15f * size);
        bx0 = sil.x - 0.5f * sil.w - m; bx1 = sil.x + 0.5f * sil.w + m;
        by0 = sil.y - 0.5f * sil.h - m; by1 = sil.y + 0.5f * sil.h + m;
        for (size_t i = 0; i < dets.size(); ++i)
            part[i] = dets[i].x >= bx0 && dets[i].x <= bx1 && dets[i].y >= by0 && dets[i].y <= by1;
        Candidate c{};
        c.x = sil.x;
        c.y = sil.y;
        c.contrast = sil.contrast;
        c.snr = 0;
        c.polarity = L.polarity;
        int lv = 0;
        while (lv < 4 && size > 14.f * (float)(1 << lv)) ++lv;
        c.level = lv;
        c.energy = sil.contrast * (float)sil.area;
        c.margin = 0;
        dets.push_back(c);
        part.push_back(0);
        silhDet = (int)dets.size() - 1;
        // трек сидел на кончике крыла или хвосте — переносим на центр силуэта
        if (std::hypot(sil.x - L.x, sil.y - L.y) > std::max(cfg_.gate_min, 0.25f * size)) {
            // точка рождения переезжает вместе с треком: перенос — не полёт
            if (L.hits >= 5) {
                L.bx += sil.x - L.x;
                L.by += sil.y - L.y;
                L.rbx += sil.x - L.x;
                L.rby += sil.y - L.y;
            }
            L.x = sil.x;
            L.y = sil.y;
            L.p00 = std::max(L.p00, measVar(lv));
        }
        L.bw = sil.w;
        L.bh = sil.h;
        if (cfg_.size_predict) {   // ожидание размера: альфа-бета фильтр по логарифму размера
            const float z = std::log(std::max(4.f, std::max(sil.w, sil.h)));
            const float rmax = std::log(1.f + cfg_.size_rate_max);
            if (!L.lsN) {
                L.ls = z;
                L.dls = 0;
                ++L.lsN;
            } else {
                const float pred = L.ls + L.dls, res = z - pred;
                // замер сильно меньше ожидаемого — часть цели, в размер не идёт; несколько
                // таких подряд (или больше ожидаемого) — ожидание неверно, следуем замерам
                const bool outlier = std::fabs(res) > std::log(cfg_.size_tol) * 0.8f;
                if (outlier && ++L.sizeRej < 4) {
                    L.ls = pred;
                } else {
                    const float g = outlier ? 0.7f : 0.3f;
                    L.ls = pred + g * res;
                    L.dls = std::max(-rmax, std::min(rmax, L.dls + 0.05f * res));
                    if (!outlier) L.sizeRej = 0;
                    if (L.lsN < 255) ++L.lsN;
                }
            }
        }
        // треки на частях той же цели (крыло, хвост) сливаются с захваченным
        if (size > 2.f * cfg_.dup_px) {
            const int keep = L.id;
            tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                         [&](const Track& t) {
                                             return t.id != keep && t.x >= bx0 && t.x <= bx1 && t.y >= by0 && t.y <= by1;
                                         }),
                          tracks_.end());
            for (int i = 0; i < (int)tracks_.size(); ++i)
                if (tracks_[i].id == keep) lockIdx = i;
        }
    } else if (lockIdx >= 0) {
        Track& L = tracks_[lockIdx];
        // силуэта нет: ожидание размера остаётся прежним (продолжать его по скорости
        // изменения без замеров хуже — расходится с целью)
        L.bw = L.bh = 0;
    }

    // "тени": слабый отклик рядом с более сильным (светлый ореол у тёмной цели от резкости
    // камеры, цель на границе тайлов, на двух масштабах) — это та же цель, в трекинг не идёт
    std::vector<char> shadow(dets.size(), 0);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (part[i]) { shadow[i] = 1; continue; }
        if ((int)i == silhDet) continue;
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
            if (shadow[di] || di == silhDet) continue;
            const float S = t.p00 + measVar(dets[di].level);
            // крупная цель (грубый масштаб) — протяжённая, её отклик гуляет по корпусу
            const float gmin = cfg_.gate_min * (float)(1 << std::min(4, std::max(dets[di].level, t.level)));
            const float r = std::min(gateCap(t), std::max(gmin, cfg_.gate_sigma * std::sqrt(S)));
            const float ex = dets[di].x - t.x, ey = dets[di].y - t.y, d2 = ex * ex + ey * ey;
            if (d2 < r * r) pairs.push_back({d2 / S, ti, di});
        }
    }
    std::stable_sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.d < b.d; });
    std::vector<int> match(tracks_.size(), -1), owner(dets.size(), -1);
    for (const auto& p : pairs)
        if (match[p.t] < 0 && owner[p.k] < 0) { match[p.t] = p.k; owner[p.k] = p.t; }
    if (silhDet >= 0 && lockIdx >= 0) {   // захваченный трек ведёт силуэт
        if (match[lockIdx] >= 0) owner[match[lockIdx]] = -1;
        match[lockIdx] = silhDet;
        owner[silhDet] = lockIdx;
    }

    // клаттер: всё, что не продолжает движущуюся цель, греет карту; в горячих ячейках —
    // помеха (её не отдаём ни трекам, ни в новые треки)
    auto mover = [&](const Track& t) {
        // трек, переживший горячую ячейку, — помеха, пока не уполз дальше creep_px: скачок отклика
        // в первые кадры даёт пробег ~mover_px, и неподвижная точка становилась целью (wt22)
        return t.confirmed && t.travel >= (t.crept ? std::max(cfg_.mover_px, cfg_.creep_px) : cfg_.mover_px) * (t.edgeBorn ? cfg_.edge_mover_k : 1.f) &&
               t.travel >= cfg_.mover_cam_k * t.camPath && !t.overlay && !t.anchored &&
               !edgeJitter(t);
    };
    // спокойное небо: рядом с треком появился другой отклик (кромка облака, листва) или
    // камера сдвинулась — это уже не одинокая точка, дальше как обычный трек
    const bool camStill = !movingCam_ && std::fabs(camDx) < 0.5f && std::fabs(camDy) < 0.5f;
    if (!camStill) quietEpoch_ = frame_;
    if (cfg_.quiet_frames > 0)
        for (size_t ti = 0; ti < tracks_.size(); ++ti) {
            Track& t = tracks_[ti];
            if (!t.quiet) continue;
            if (!camStill || !quietHold(t)) { t.quiet = false; continue; }
            const float dup = cfg_.dup_px * (float)(1 << std::min(4, t.level)), r = std::max(cfg_.quiet_r, dup);
            for (size_t di = 0; di < dets.size() && t.quiet; ++di) {
                if (owner[di] == (int)ti || shadow[di] || dets[di].weak) continue;
                const float dx = std::fabs(dets[di].x - t.x), dy = std::fabs(dets[di].y - t.y);
                if (dx < r && dy < r && (dx >= dup || dy >= dup)) t.quiet = false;
            }
        }
    std::vector<char> isMover(dets.size(), 0);
    for (size_t di = 0; di < dets.size(); ++di) {
        isMover[di] = (owner[di] >= 0 && (mover(tracks_[owner[di]]) || quietHold(tracks_[owner[di]]))) || (int)di == silhDet;
        if (!isMover[di] && !shadow[di] && !dets[di].weak) heat_[cellOf(dets[di])] += 1.f - decay;
    }
    // долгая память учится только при неподвижной камере: координаты ячеек — кадра
    // (наложения на кадр — и при движущейся, см. overlay_hits)
    const bool learn = cfg_.static_frames > 0 && !movingCam_ && std::fabs(camDx) < 0.5f && std::fabs(camDy) < 0.5f;
    clutterDets_.clear();
    std::vector<char> clut(dets.size(), 0);
    for (size_t di = 0; di < dets.size(); ++di) {
        if (isMover[di] || part[di] || (heat_[cellOf(dets[di])] <= cfg_.clutter_level && !staticHot(dets[di].x, dets[di].y)))
            continue;
        clut[di] = 1;
        clutterDets_.push_back(dets[di]);
        // отклик захватывавшейся цели (камера ведёт дрон, и он стоит в кадре) в память не идёт
        const bool target = owner[di] >= 0 && tracks_[owner[di]].everLocked;
        if (learn && !shadow[di] && !dets[di].weak && !target) staticAdd(cellOf(dets[di]));
        if (owner[di] >= 0 && creepKeep(tracks_[owner[di]], dets[di])) {
            tracks_[owner[di]].crept = true;
        } else if (owner[di] >= 0) {
            match[owner[di]] = -1;
            owner[di] = -1;
        }
    }

    std::vector<char> quietDet(dets.size(), 0);   // отклик трека спокойного неба
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        Track& tr = tracks_[ti];
        const int best = match[ti];
        tr.det = best;
        if (best >= 0 && tr.quiet) quietDet[best] = 1;
        tr.hist.push_back(best >= 0);
        while ((int)tr.hist.size() > cfg_.window) tr.hist.pop_front();
        ++tr.age;
        if (best < 0) {
            ++tr.miss;
            tr.evidence = std::max(0.f, tr.evidence - cfg_.ev_miss);
            continue;
        }
        const Candidate& d = dets[best];
        tr.evidence = std::min(100.f, tr.evidence + evidenceOf(d));
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
        // нашлась после пропусков (reupdate, как OC-SORT): скорость — по прямой от последней отметки
        // до этой, а не та, что фильтр накопил на прогнозе (на 15 к/с 6 кадров прогноза — 0.4 с)
        if (cfg_.reupdate > 0 && tr.miss >= 2 && !tr.obs.empty() && frame_ > tr.obs.back().t) {
            const auto& o = tr.obs.back();
            const float gap = (float)(frame_ - o.t), w = std::min(1.f, cfg_.reupdate);
            const float vx = (d.x - gx_ - o.x) / gap, vy = (d.y - gy_ - o.y) / gap;
            tr.vsx += w * (vx - tr.vsx);
            tr.vsy += w * (vy - tr.vsy);
        }
        tr.polarity = d.polarity;
        tr.contrast = d.contrast;
        tr.level = d.level;
        tr.miss = 0;
        ++tr.hits;
        tr.path.push_back({tr.x, tr.y});
        while (tr.path.size() > 30) tr.path.pop_front();
        const float sx = d.x - gx_, sy = d.y - gy_;
        if (tr.hits <= 5) tr.travel = tr.camPath = tr.travelRel = 0;
        else if (!tr.obs.empty())
            tr.camPath += std::hypot((d.x - sx) - (tr.obs.back().fx - tr.obs.back().x), (d.y - sy) - (tr.obs.back().fy - tr.obs.back().y));
        tr.obs.push_back({frame_, sx, sy, d.contrast, d.level, d.energy, d.x, d.y, driftAx_, driftAy_});
        while ((int)tr.obs.size() > std::max({48, cfg_.cls_window, cfg_.flap_window, cfg_.overlay_hits})) tr.obs.pop_front();
        // прямота пути: последние до 48 попаданий, 8 отрезков (шум замера на коротких шагах
        // не делает медленный дрон "дрожащим")
        if (tr.obs.size() >= 16) {
            const int n = (int)std::min<size_t>(48, tr.obs.size()), a = (int)tr.obs.size() - n;
            float path = 0;
            for (int s = 0; s < 8; ++s) {
                const auto& p = tr.obs[a + (n - 1) * s / 8];
                const auto& q = tr.obs[a + (n - 1) * (s + 1) / 8];
                path += std::hypot(q.x - p.x, q.y - p.y);
            }
            const float net = std::hypot(tr.obs.back().x - tr.obs[a].x, tr.obs.back().y - tr.obs[a].y);
            tr.straight = path > 1e-3f ? net / path : 0.f;
        }
        // как далеко ушёл по сцене: по медиане последних 5 попаданий — одиночный скачок
        // отклика (соседнее пятно, силуэт захватил лишнее) полётом не считается
        if (tr.hits >= 5 && tr.obs.size() >= 5) {
            float xs[5], ys[5];
            for (int i = 0; i < 5; ++i) {
                xs[i] = tr.obs[tr.obs.size() - 5 + i].x;
                ys[i] = tr.obs[tr.obs.size() - 5 + i].y;
            }
            std::nth_element(xs, xs + 2, xs + 5);
            std::nth_element(ys, ys + 2, ys + 5);
            // точка рождения — медиана первых 5 попаданий: скачок второго отклика (тень,
            // соседний тайл) не делает из неподвижной помехи "летавшую" цель
            if (tr.hits == 5) { tr.bx = xs[2]; tr.by = ys[2]; }
            tr.travel = std::max(tr.travel, std::hypot(xs[2] - tr.bx, ys[2] - tr.by));
            // то же относительно фона (за вычетом дрейфа, см. drift_min); без дрейфа — как travel
            for (int i = 0; i < 5; ++i) {
                const auto& o = tr.obs[tr.obs.size() - 5 + i];
                xs[i] = o.x - o.dx;
                ys[i] = o.y - o.dy;
            }
            std::nth_element(xs, xs + 2, xs + 5);
            std::nth_element(ys, ys + 2, ys + 5);
            if (tr.hits == 5) { tr.rbx = xs[2]; tr.rby = ys[2]; }
            tr.travelRel = std::max(tr.travelRel, std::hypot(xs[2] - tr.rbx, ys[2] - tr.rby));
        }
        // наложение на кадр: стоит в одной точке кадра, пока сцена (камера) уехала
        const int no = cfg_.overlay_hits;
        if (no > 0 && (int)tr.obs.size() >= no) {
            double mx = 0, my = 0, vv = 0;
            const size_t from = tr.obs.size() - (size_t)no;
            for (size_t i = from; i < tr.obs.size(); ++i) { mx += tr.obs[i].fx; my += tr.obs[i].fy; }
            mx /= no;
            my /= no;
            for (size_t i = from; i < tr.obs.size(); ++i)
                vv += (tr.obs[i].fx - mx) * (tr.obs[i].fx - mx) + (tr.obs[i].fy - my) * (tr.obs[i].fy - my);
            const float spread = std::hypot(tr.obs.back().x - tr.obs[from].x, tr.obs.back().y - tr.obs[from].y);
            // за захваченной целью камера следит сама (setFollowing): стоять в кадре — её дело
            // в середине кадра надписей и логотипов нет, а цель, которую ведёт камера, есть
            const float sd = (float)std::sqrt(vv / no);
            const bool still = sd < cfg_.overlay_px ||
                               (cfg_.overlay_rel > 0 && sd < cfg_.overlay_rel * spread && !midFrame((float)mx, (float)my));
            if (still && spread > cfg_.overlay_cam && !(following_ && tr.id == lockId_) &&
                !(cfg_.follow_k > 0 && midFrame((float)mx, (float)my)))
                tr.overlay = true;
        }
        if (tr.overlay && cfg_.static_frames > 0) staticAdd(cellOf(d));
        // полетел — его отклики не помеха; стоит долго — уже ясно, что помеха
        if (mover(tr)) {
            tr.everMover = true;
            tr.pend.clear();
        } else if (learn && !tr.everMover && !tr.everLocked && !quietHold(tr)) {
            tr.pend.push_back((uint32_t)cellOf(d));
            if (tr.pend.size() >= 256 && tr.id != lockId_) {
                for (uint32_t c : tr.pend) staticAdd(c);
                tr.pend.clear();
            }
        }
    }
    // дрейф фона: медиана скорости устоявшихся треков этого кадра (большинство из них — неподвижные
    // предметы), сглаженная; учитывается, пока быстрее drift_min (с запасом на выход)
    if (cfg_.drift_min > 0) {
        std::vector<float> vx, vy;
        // скорость трека — по его последним drift_obs попаданиям (у фильтра Калмана она скачет от
        // квантования положения отклика)
        const size_t W = (size_t)std::max(2, cfg_.drift_obs);
        if (!camKnown)
            for (const Track& t : tracks_) {
                if (t.hits < cfg_.cls_min_hits || t.miss != 0 || t.followed || t.overlay || (hard_ && t.id == lockId_) ||
                    t.obs.size() < W)
                    continue;
                const auto& a = t.obs[t.obs.size() - W];
                const auto& b = t.obs.back();
                if (b.t <= a.t) continue;
                vx.push_back((b.x - a.x) / (float)(b.t - a.t));
                vy.push_back((b.y - a.y) / (float)(b.t - a.t));
            }
        // большинство (половина) — в пределах допуска от медианы: фон плывёт как целое, а не разброд
        bool agree = false;
        float mx = 0, my = 0;
        if ((int)vx.size() >= std::max(3, cfg_.drift_tracks)) {
            std::vector<float> sx = vx, sy = vy;
            const size_t m = sx.size() / 2;
            std::nth_element(sx.begin(), sx.begin() + (std::ptrdiff_t)m, sx.end());
            std::nth_element(sy.begin(), sy.begin() + (std::ptrdiff_t)m, sy.end());
            mx = sx[m];
            my = sy[m];
            const float tol = std::max(cfg_.drift_min, 0.5f * std::hypot(mx, my));
            size_t near = 0;
            for (size_t i = 0; i < vx.size(); ++i) near += std::hypot(vx[i] - mx, vy[i] - my) < tol;
            agree = near * 2 >= vx.size();
        }
        if (agree) {
            driftX_ = 0.7f * driftX_ + 0.3f * mx;
            driftY_ = 0.7f * driftY_ + 0.3f * my;
        } else {
            driftX_ *= 0.7f;
            driftY_ *= 0.7f;
        }
        const float dv = std::hypot(driftX_, driftY_);
        driftOn_ = dv > (driftOn_ ? 0.7f : 1.f) * cfg_.drift_min;
        if (driftOn_) {
            driftAx_ += driftX_;
            driftAy_ += driftY_;
        }
    } else {
        driftOn_ = false;
    }
    // трек умер, так и не полетев: его отклики — в долгую память
    if (cfg_.static_frames > 0)
        for (const Track& t : tracks_)
            if (t.miss > (t.confirmed ? coastLimit(t) : cfg_.tentative_miss) && !t.everMover && !t.everLocked &&
                t.id != lockId_)
                for (uint32_t c : t.pend) staticAdd(c);
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Track& t) {
                                     return t.miss > (t.confirmed ? coastLimit(t) : cfg_.tentative_miss);
                                 }),
                  tracks_.end());
    // новые треки из свободных детекций; начальная скорость неизвестна — широкие ворота
    const float v0 = cfg_.gate / 3.f;
    // родился в спокойном небе: камера стоит quiet_frames кадров, и столько же вокруг на
    // quiet_r не было откликов (не кромка крон, не облако, не запомненная помеха)
    auto quietBirth = [&](const Candidate& d) {
        if (cfg_.quiet_frames <= 0 || d.edge || frame_ - quietEpoch_ < cfg_.quiet_frames || staticHot(d.x, d.y))
            return false;
        const int r = (int)std::ceil(cfg_.quiet_r / (float)cell);
        const int cx = std::min(gw_ - 1, std::max(0, (int)d.x / cell)), cy = std::min(gh_ - 1, std::max(0, (int)d.y / cell));
        for (int y = std::max(0, cy - r); y <= std::min(gh_ - 1, cy + r); ++y)
            for (int x = std::max(0, cx - r); x <= std::min(gw_ - 1, cx + r); ++x)
                if (frame_ - lastDet_[(size_t)y * gw_ + x] < cfg_.quiet_frames) return false;
        return true;
    };
    for (int di = 0; di < (int)dets.size(); ++di) {
        // слабый отклик (пониженный порог у прогноза) только продолжает трек, новый не рождает
        if (owner[di] >= 0 || clut[di] || shadow[di] || dets[di].weak || di == silhDet) continue;
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
        t.evidence = std::max(0.f, evidenceOf(d));
        t.det = di;
        t.bx = d.x - gx_;
        t.by = d.y - gy_;
        t.hist.push_back(true);
        t.path.push_back({t.x, t.y});
        t.obs.push_back({frame_, t.bx, t.by, t.contrast, d.level, d.energy, d.x, d.y, driftAx_, driftAy_});
        t.edgeBorn = d.edge;
        t.quiet = quietBirth(d);
        if (learn && !t.quiet) t.pend.push_back((uint32_t)cellOf(d));
        if (t.quiet) quietDet[di] = 1;
        tracks_.push_back(std::move(t));
    }
    // время последнего отклика по ячейкам; отклики треков спокойного неба не в счёт — иначе
    // тусклая цель, трек которой оборвался, на том же месте уже не родится "в тишине"
    if (cfg_.quiet_frames > 0)
        for (size_t di = 0; di < dets.size(); ++di) {
            if (shadow[di] || dets[di].weak || part[di] || quietDet[di]) continue;
            lastDet_[cellOf(dets[di])] = frame_;
        }
    // стая: по взмахам соседей с прошлого кадра
    for (auto& tr : tracks_) {
        tr.flock = false;
        const float v = std::hypot(tr.vsx, tr.vsy);
        if (cfg_.flock_px <= 0 || !tr.confirmed || v < cfg_.flock_speed) continue;
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
        if (tr.miss == 0) tr.solid = nh >= cfg_.confirm && (tr.confirmed || tr.evidence >= cfg_.ev_confirm);
        if (tr.solid) tr.confirmed = true;
        tr.clutter = tr.miss == 0 && !mover(tr) && !(hard_ && tr.id == lockId_) && (hot(tr.x, tr.y) || staticHot(tr.x, tr.y));
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
        tr.lockW = tr.score;
        if (cfg_.lock_cmax > 0 && mc > cfg_.lock_cmax) tr.lockW *= (float)(cfg_.lock_cmax / mc);
        // медленную цель предпочитаем, только если она уже пролетела по сцене: дрожащий отклик
        // в кроне тоже "медленный"
        // в кроне тоже "медленный", а зависший на месте (неподвижная точка в кроне, к которой
        // перешёл трек пролетевшей цели) уступает летящим
        // при дрейфе фона — пробег и скорость относительно фона: плывущее с ним весит как зависшее
        if (tr.followed)
            tr.lockW *= cfg_.follow_k;
        else if (cfg_.lock_slow > 0) {
            const float sp = relSpeed(tr), tv = driftOn_ ? tr.travelRel : tr.travel;
            tr.lockW *= std::min(1.f, tv / 64.f) * (sp < cfg_.hover_speed ? 0.05f : 1.f) / (1.f + sp / cfg_.lock_slow);
        }
        // родился у кромки неба (кроны, крыши): цель, только если полетела
        if (!tr.solid || tr.miss > coastLimit(tr) || (tr.edgeBorn && (!tr.everMover || edgeJitter(tr)))) continue;
        out_.push_back(tr);
        if (tr.det >= 0) inTrack[tr.det] = 1;
    }
    // захват: держим текущую цель, пока она жива и не стала помехой
    const int prevLock = lockId_;
    bool same = false;   // захват перешёл на трек, подхвативший ту же цель
    const Track* cur = nullptr;
    const Track* best = nullptr;
    // прогноз, ушедший за край кадра, новой целью не становится, а захват на нём сразу уступает
    // лучшей в кадре: с рук камеру уводят от птицы, и захват полсекунды сидел бы за кадром
    auto outside = [&](const Track& t) {
        return t.miss > 0 && (t.x < 0 || t.y < 0 || t.x >= (float)frameW_ || t.y >= (float)frameH_);
    };
    // новый захват (lock_mover): только на устоявшийся трек, который уже летал по сцене (или
    // ведомый камерой, или точку спокойного неба) — не на первый подтверждённый блик
    auto lockable = [&](const Track& t) {
        return !cfg_.lock_mover || t.followed || quietHold(t) ||
               (t.hits >= cfg_.cls_min_hits && t.everMover && (!driftOn_ || t.travelRel >= cfg_.mover_px));
    };
    for (const Track& t : out_) {
        if (t.id == lockId_ && !outside(t)) cur = &t;
        if (t.score > 0 && !outside(t) && lockable(t) && (!best || t.lockW > best->lockW)) best = &t;
    }
    bool lockOut = false;
    for (const Track& t : out_) lockOut |= t.id == lockId_ && outside(t);
    // захваченное оказалось помехой (неподвижное, наложение на кадр) — не ждём его, как
    // пропавшую цель, а сразу отпускаем
    bool lockClutter = false;
    if (cur && cur->score <= 0) {
        lockClutter = cur->cls == TargetClass::Clutter;
        cur = nullptr;
    }
    // ближайший живой трек к точке, где ждём захваченную цель (её же подхватил другой трек)
    auto nearest = [&](float x, float y, float r) {
        const Track* n = nullptr;
        float bd = r * r;
        for (const Track& t : out_) {
            if (t.score <= 0 || t.miss > 0 || &t == cur) continue;
            const float d = (t.x - x) * (t.x - x) + (t.y - y) * (t.y - y);
            if (d < bd) { bd = d; n = &t; }
        }
        // мягкий захват не подхватывает слабую мелочь у прогноза, когда в кадре есть куда весомее:
        // с рук прогноз пропавшей цели уплывает, и захват переходил бы с блика на блик, не
        // давая настоящей цели набрать перевес (lock_dwell)
        if (n && !hard_ && best && n->lockW < 0.25f * best->lockW) n = nullptr;
        return n;
    };
    if (cur) {
        lostFor_ = 0;
        lastX_ = cur->x; lastY_ = cur->y; lastVx_ = cur->vsx; lastVy_ = cur->vsy;
        if (cur->miss >= 2) {   // цель на прогнозе, а рядом её уже ведёт свежий трек
            if (const Track* n = nearest(cur->x, cur->y, std::max(cfg_.gate, 4 * cur->sigma))) { cur = n; same = true; }
        } else if (!manual_) {
            // претенденты: весомее текущей (при жёстком захвате — только если задан hard_switch)
            // за lock_dwell кадров; из выдержавших — самый весомый
            auto wins = [&](const Track& t) {
                // жёсткий — только на цель, которую классификатор зовёт дроном: медленная
                // птица рядом весомее (lock_slow), и на 15 к/с захват уходил с дрона на неё (wt16, wt18)
                // с самолёта на медленный дрон, не на яркую; на ведомую камерой — и без этого
                // (её скорость по сцене — скорость камеры, hard_switch_follow)
                return hard_ ? cfg_.hard_switch > 0 && t.lockW > cfg_.hard_switch * cur->lockW &&
                                   ((cfg_.hard_switch_follow && t.followed) || relSpeed(t) < 0.5f * relSpeed(*cur)) &&
                                   (!cfg_.hard_switch_drone || t.followed ||   // камера ведёт её — это и есть цель (wt17)
                                    (t.cls == TargetClass::Drone && t.birdScore < 0.5f))
                             : t.lockW > cfg_.lock_switch * cur->lockW;
            };
            std::vector<std::pair<int, int>> nb;
            const Track* sw = nullptr;
            for (const Track& t : out_) {
                if (&t == cur || t.score <= 0) continue;
                int n = 0;
                for (const auto& b : beat_)
                    if (b.first == t.id) n = b.second;
                // счёт с утечкой: кадр перевеса +1, кадр без него -1 — претендент, весомый
                // большую часть времени, своё набирает, хоть вес и скачет
                const bool w = wins(t);
                // мягкий захват на «птице» уступает «дрону» сразу (lock_bird_yield)
                if (w && cfg_.lock_bird_yield && !hard_ && cur->cls == TargetClass::Bird && t.cls == TargetClass::Drone &&
                    t.birdScore < 0.5f)
                    n = std::max(n, cfg_.lock_dwell - 1);
                n += w ? 1 : -1;
                if (n <= 0) continue;
                nb.emplace_back(t.id, n);
                if (w && n >= cfg_.lock_dwell && (!sw || t.lockW > sw->lockW)) sw = &t;
            }
            // претендент жив, но в выдаче его нет (на прогнозе дольше выдачи) — счёт стоит
            if (cfg_.lock_dwell_keep)
                for (const auto& b : beat_) {
                    bool shown = false, alive = false;
                    for (const Track& t : out_) shown |= t.id == b.first;
                    for (const Track& t : tracks_) alive |= t.id == b.first;
                    if (!shown && alive) nb.push_back(b);
                }
            beat_.swap(nb);
            if (sw) cur = sw, beat_.clear();
        }
        lockId_ = cur->id;
    } else if (lockId_ && !lockClutter && !lockOut && lostFor_ < (hard_ ? cfg_.hard_miss : cfg_.max_miss)) {
        // захваченный трек пропал: ищем цель там, где она должна быть, и там, где её видели
        // последний раз (если скорость была оценена неверно)
        ++lostFor_;
        lastX_ += camDx;
        lastY_ += camDy;
        const float r = cfg_.gate + cfg_.lost_grow * lostFor_;
        const Track* n = nearest(lastX_ + lastVx_ * lostFor_, lastY_ + lastVy_ * lostFor_, r);
        if (!n) n = nearest(lastX_, lastY_, r);
        if (n) { lockId_ = n->id; lostFor_ = 0; same = true; }
    } else {
        lockId_ = best ? best->id : 0;
        lostFor_ = 0;
        if (best) { lastX_ = best->x; lastY_ = best->y; lastVx_ = best->vsx; lastVy_ = best->vsy; }
    }

    // ожидание размера переходит к треку, подхватившему ту же цель
    if (cfg_.size_predict && lockId_) {
        for (Track& t : tracks_) {
            if (t.id != lockId_) continue;
            if (same && !t.lsN && lastLsN_) {
                t.ls = lastLs_ + (float)std::max(1, lostFor_) * lastDls_;
                t.dls = lastDls_;
                t.lsN = lastLsN_;
            }
            lastLs_ = t.ls;
            lastDls_ = t.dls;
            lastLsN_ = t.lsN;
        }
    } else {
        lastLsN_ = 0;
    }

    // проверка захваченной цели: verify_frames кадров подряд дрон — жёсткий захват
    if (lockId_ != prevLock) beat_.clear();
    if (lockId_ != prevLock && !same) manual_ = false;   // ручной выбор — до потери цели
    if (lockId_ != prevLock && !same) {   // подхват той же цели другим треком — проверка не сбрасывается
        hard_ = false;
        verify_ = 0;
    }
    if (!lockId_) {
        hard_ = false;
        verify_ = 0;
    }
    for (Track& t : tracks_) {
        if (t.id != lockId_) continue;
        t.everLocked = true;
        if (t.miss == 0) {
            // дрон по классификатору и уже летал по сцене (неподвижное — не проверить)
            if (t.cls == TargetClass::Drone && t.birdScore < 0.3f && t.everMover) ++verify_;
            else if (t.cls == TargetClass::Bird || t.cls == TargetClass::Clutter) verify_ = 0;
            // с приоритетом летящих (lock_slow) зависшая цель жёсткий захват не держит: иначе
            // трек пролетевшей цели, перешедший на неподвижную точку в кроне, держит его до конца
            if (cfg_.lock_slow > 0 && relSpeed(t) < cfg_.hover_speed) verify_ = 0, hard_ = false;
        }
        if (cfg_.hard_lock && verify_ >= cfg_.verify_frames) hard_ = true;
        if (!cfg_.hard_lock || t.overlay || t.anchored || (hard_ && t.cls == TargetClass::Bird && t.birdScore > 0.8f)) {
            hard_ = false;
            if (t.cls == TargetClass::Bird) verify_ = 0;
        }
    }
    for (Track& t : out_)
        if (t.id == lockId_) t.everLocked = true;

    // шум: детекции, которые не помеха и не продолжают подтверждённый трек
    noiseDets_.clear();
    for (size_t di = 0; di < dets.size(); ++di)
        if (!inTrack[di] && !clut[di] && !dets[di].weak) noiseDets_.push_back(dets[di]);
    return out_;
}

}  // namespace dronolok
