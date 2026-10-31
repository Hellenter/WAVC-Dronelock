#include "dronolok/multicam.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace dronolok {

namespace {

inline Vec3 add(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 sub(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 mul(const Vec3& a, double k) { return {a.x * k, a.y * k, a.z * k}; }
inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 unit(const Vec3& a) {
    const double n = norm(a);
    return n > 0 ? mul(a, 1 / n) : Vec3{0, 1, 0};
}
inline double ang(double az1, double el1, double az2, double el2) {
    return angle_between(dir_from_azel(az1, el1), dir_from_azel(az2, el2));
}

// обращение симметричной 3x3
bool inv3(const double* m, double* o) {
    const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
    const double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const double det = a * A + b * B + c * C;
    if (std::fabs(det) < 1e-300) return false;
    const double k = 1 / det;
    o[0] = A * k, o[1] = -(b * i - c * h) * k, o[2] = (b * f - c * e) * k;
    o[3] = B * k, o[4] = (a * i - c * g) * k, o[5] = -(a * f - c * d) * k;
    o[6] = C * k, o[7] = -(a * h - b * g) * k, o[8] = (a * e - b * d) * k;
    return true;
}

inline Vec3 rotate_small(const double* w, const Vec3& d) { return unit(add(d, cross({w[0], w[1], w[2]}, d))); }
inline Vec3 unrotate_small(const double* w, const Vec3& d) { return unit(sub(d, cross({w[0], w[1], w[2]}, d))); }

// прогноз сообщения (направление и СКО) на момент t
inline void pred(const TargetReport& r, double t, double& az, double& el, double& s) { predict(r, t, az, el, s); }

}  // namespace

void Fusion::reset(const std::vector<RigCamera>& cams, const FusionConfig& c) {
    c_ = c;
    cams_ = cams;
    cal_.assign(cams.size(), CameraCalib());
    for (auto& k : cal_) k.sigmaDeg = c.calib ? c.calibInitDeg : 0;
    camT_.assign(cams.size(), -1e9);
    now_ = 0;
    tg_.clear();
    map_.clear();
    alias_.clear();
    merges_.clear();
    pt_.clear();
    nextId_ = 1;
    counters = Counters();
}

Fusion::Tgt* Fusion::get(int id) {
    for (auto& g : tg_)
        if (g.f.id == id) return &g;
    return nullptr;
}
const Fusion::Tgt* Fusion::get(int id) const {
    for (const auto& g : tg_)
        if (g.f.id == id) return &g;
    return nullptr;
}

bool Fusion::listed(const Tgt& g, int k) const {
    if (k < 0 || k >= (int)g.obs.size()) return false;
    const Obs& o = g.obs[k];
    return o.local != 0 && o.t >= camT_[k] - 1e-9 && camT_[k] >= now_ - c_.dropS;
}
bool Fusion::listedAny(const Tgt& g) const {
    for (int k = 0; k < (int)cams_.size(); ++k)
        if (listed(g, k)) return true;
    return false;
}

const TargetReport& Fusion::best(const Tgt& g, double t) const {
    (void)t;
    // точная видит цель точнее широких: её сообщение, если оно свежее
    if (g.haveRef && g.refRep.t >= g.last.t) return g.refRep;
    return g.last;
}

Vec3 Fusion::refPoint(int k, const Vec3& dcam, double R) const { return add(cams_[k].pos, mul(dcam, R)); }

double Fusion::calibSigma(int k, const Vec3& d) const {
    if (!c_.calib) return 0;
    const CameraCalib& cb = cal_[k];
    const double s0 = deg2rad(std::max(1e-4f, c_.calibInitDeg));
    double A[9], C[9];
    for (int i = 0; i < 9; ++i) A[i] = cb.A[i];
    A[0] += 1 / (s0 * s0), A[4] += 1 / (s0 * s0), A[8] += 1 / (s0 * s0);
    if (!inv3(A, C)) return c_.calibInitDeg;
    // ошибка направления от неточности поворота: след ковариации поперёк d
    const double tr = C[0] + C[4] + C[8];
    const double dCd = d.x * (C[0] * d.x + C[1] * d.y + C[2] * d.z) + d.y * (C[3] * d.x + C[4] * d.y + C[5] * d.z) +
                       d.z * (C[6] * d.x + C[7] * d.y + C[8] * d.z);
    return rad2deg(std::sqrt(std::max(0.0, 0.5 * (tr - dCd))));
}

TargetReport Fusion::toRef(int k, const TargetReport& r) const {
    if ((c_.passthrough1 && cams_.size() == 1) || k < 0 || k >= (int)cams_.size()) return r;
    TargetReport o = r;
    const double dt = 0.2;
    Vec3 d = dir_from_azel(r.az, r.el), d2 = dir_from_azel(r.az + r.vaz * dt, r.el + r.vel * dt);
    const bool calib = c_.calib && !c_.independent;
    if (calib) {
        d = rotate_small(cal_[k].w, d);
        d2 = rotate_small(cal_[k].w, d2);
    }
    const Vec3& p = cams_[k].pos;
    const double b = norm(p);
    const double R = std::max(1.f, c_.rangeGuess);
    double az, el, az2, el2;
    if (b > 1e-6) {
        azel_from_dir(refPoint(k, d, R), az, el);
        azel_from_dir(refPoint(k, d2, R), az2, el2);
    } else {
        azel_from_dir(d, az, el);
        azel_from_dir(d2, az2, el2);
    }
    o.az = (float)az;
    o.el = (float)el;
    o.vaz = (float)(az_diff(az2, az) / dt);
    o.vel = (float)((el2 - el) / dt);
    // неуверенность: калибровка позы камеры и параллакс (дальность неизвестна)
    double par = 0;
    if (b > 1e-6) {
        const double bp = norm(cross(p, d));   // база поперёк луча
        const double inv = std::max(std::fabs(1.0 / std::max(1.f, c_.rangeMin) - 1.0 / R), std::fabs(1.0 / std::max(1.f, c_.rangeMax) - 1.0 / R));
        par = rad2deg(0.5 * bp * inv);
    }
    const double cs = calib ? calibSigma(k, d) : 0.0;
    o.sigma = (float)std::sqrt((double)r.sigma * r.sigma + cs * cs + par * par);
    return o;
}

bool Fusion::projectTo(int k, double az, double el, double& u, double& v) const {
    if (k < 0 || k >= (int)cams_.size()) return false;
    Vec3 d = dir_from_azel(az, el);
    if (!(c_.passthrough1 && cams_.size() == 1)) {
        if (norm(cams_[k].pos) > 1e-6) d = unit(sub(mul(d, std::max(1.f, c_.rangeGuess)), cams_[k].pos));
        if (c_.calib && !c_.independent) d = unrotate_small(cal_[k].w, d);
    }
    return project(cams_[k].model, cams_[k].pose, d, u, v);
}

void Fusion::calibPair(int k, int id, const TargetReport& w, const TargetReport& ref) {
    CameraCalib& cb = cal_[k];
    const Vec3 d = dir_from_azel(w.az, w.el);   // как видит камера по паспорту
    Vec3 dobs = dir_from_azel(ref.az, ref.el);  // как видит точная, из положения камеры
    if (norm(cams_[k].pos) > 1e-6) dobs = unit(sub(mul(dobs, std::max(1.f, c_.rangeGuess)), cams_[k].pos));
    const double res = angle_between(rotate_small(cb.w, d), dobs);
    if (res > c_.calibMaxDeg) {
        ++cb.rejected;
        return;
    }
    // у широкой в этом кадре отклик, а не прогноз трека
    {
        double u, v;
        const RigCamera& rc = cams_[k];
        if (project(rc.model, rc.pose, d, u, v) && w.sigma > c_.calibSigmaPx * pixel_deg(rc.model, rc.pose, u, v)) return;
    }
    // скорости совпадают: трек широкой, сползающий на помеху, движется не так, как цель у точной
    const double ce = std::cos(deg2rad(0.5 * (w.el + ref.el)));
    const double dv = std::hypot((w.vaz - ref.vaz) * ce, w.vel - ref.vel);
    if (dv > c_.calibVelDeg + 3 * std::hypot(w.sigma_v, ref.sigma_v)) {
        ++cb.rejected;
        return;
    }
    // невязка — в пределах неуверенности; подряд много мимо от разных целей — камеру сдвинули, учиться заново
    const double sc = calibSigma(k, d);
    const double gate = std::max<double>(c_.calibGateMinDeg, c_.calibGateSigma * std::sqrt(sc * sc + (double)w.sigma * w.sigma +
                                                                                           (double)ref.sigma * ref.sigma));
    if (res > gate) {
        ++cb.rejected;
        if (cb.streak == 0) cb.streakId = id, cb.streakMulti = false;
        cb.streakMulti = cb.streakMulti || id != cb.streakId;
        if (++cb.streak >= c_.calibRelearn && cb.streakMulti) {   // заново, от паспорта
            for (double& a : cb.A) a = 0;
            for (double& a : cb.b) a = 0;
            for (double& a : cb.w) a = 0;
            cb.pairs = 0;
            cb.streak = 0;
            ++cb.relearned;
            cb.sigmaDeg = calibSigma(k, dir_from_azel(cams_[k].pose.az, cams_[k].pose.el));
        }
        return;
    }
    cb.streak = 0;
    const Vec3 e = sub(dobs, d);
    const double sm = deg2rad(std::max(0.01, std::sqrt((double)w.sigma * w.sigma + (double)ref.sigma * ref.sigma)));
    const double wt = 1 / (sm * sm);
    const double lam = std::pow(0.5, 1.0 / std::max(1.f, c_.calibHalfLife));
    for (double& a : cb.A) a *= lam;
    for (double& a : cb.b) a *= lam;
    cb.pairs = cb.pairs * lam + 1;
    const double dd[3] = {d.x, d.y, d.z};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) cb.A[i * 3 + j] += wt * ((i == j ? 1.0 : 0.0) - dd[i] * dd[j]);
    const Vec3 de = cross(d, e);
    cb.b[0] += wt * de.x, cb.b[1] += wt * de.y, cb.b[2] += wt * de.z;
    const double s0 = deg2rad(std::max(1e-4f, c_.calibInitDeg));
    double A[9], C[9];
    for (int i = 0; i < 9; ++i) A[i] = cb.A[i];
    A[0] += 1 / (s0 * s0), A[4] += 1 / (s0 * s0), A[8] += 1 / (s0 * s0);
    if (inv3(A, C))
        for (int i = 0; i < 3; ++i) cb.w[i] = C[i * 3] * cb.b[0] + C[i * 3 + 1] * cb.b[1] + C[i * 3 + 2] * cb.b[2];
    const double r2 = angle_between(rotate_small(cb.w, d), dobs);
    cb.rmsDeg = std::sqrt(cb.rmsDeg * cb.rmsDeg * lam + r2 * r2 * (1 - lam));
    cb.sigmaDeg = calibSigma(k, dir_from_azel(cams_[k].pose.az, cams_[k].pose.el));
    ++cb.used;
}

void Fusion::choosePrimary(Tgt& g) {
    // сообщения — от одной камеры, пока цель не подошла к краю её кадра: у точной не скачет направление
    // (у камер свои ошибки позы), а цель не теряется на стыке — соседняя берёт её раньше, чем прежняя выпустит
    const int n = (int)cams_.size();
    int bk = -1;
    double be = -1e18;
    for (int k = 0; k < n; ++k)
        if (listed(g, k) && g.obs[k].edge > be) {
            be = g.obs[k].edge;
            bk = k;
        }
    if (bk < 0) return;   // ни одна камера не видит: ждёт по прогнозу
    int pick = bk;
    if (listed(g, g.primary)) {
        const double pe = g.obs[g.primary].edge;
        if (!(pe < c_.edgeDeg && be > pe + 0.5 * c_.edgeDeg)) pick = g.primary;
    }
    if (g.lastCam >= 0 && g.lastCam != pick) {
        ++counters.handoffs;
        ++g.f.handoffs;
    }
    g.primary = pick;
    g.lastCam = pick;
}

void Fusion::mergeInto(Tgt& keep, Tgt& gone) {
    for (int k = 0; k < (int)cams_.size(); ++k)
        if (gone.obs[k].local && (!keep.obs[k].local || gone.obs[k].t > keep.obs[k].t)) keep.obs[k] = gone.obs[k];
    for (auto& m : map_)
        if (m.id == gone.f.id) m.id = keep.f.id;
    if (gone.haveRef && (!keep.haveRef || gone.refRep.t > keep.refRep.t)) {
        keep.refRep = gone.refRep;
        keep.haveRef = true;
        keep.refConfident = gone.refConfident;
        keep.f.refSeen = gone.f.refSeen;
    }
    keep.lastHardT = std::max(keep.lastHardT, gone.lastHardT);
    if (gone.lastClsT > keep.lastClsT) {
        keep.lastClsT = gone.lastClsT;
        keep.lastCls = gone.lastCls;
    }
    keep.f.born = std::min(keep.f.born, gone.f.born);
    alias_.push_back({gone.f.id, keep.f.id});
    if (alias_.size() > 400) alias_.erase(alias_.begin(), alias_.begin() + 100);
    merges_.push_back({gone.f.id, keep.f.id});
    ++counters.merged;
}

void Fusion::update(int k, double t, const std::vector<TargetReport>& reps) {
    if (k < 0 || k >= (int)cams_.size()) return;
    if (passthrough()) {   // одна камера (как раньше) или без объединения: треки камеры и есть цели
        const int base = c_.independent ? k * 100000 : 0;
        if ((int)pt_.size() != (int)cams_.size()) pt_.assign(cams_.size(), {});
        if (c_.independent && t < camT_[k] - 1e-9) return;
        auto& out = pt_[k];
        out.clear();
        for (size_t i = 0; i < reps.size(); ++i) {
            FusedTarget f;
            f.id = base + reps[i].id;
            f.rep = c_.independent ? toRef(k, reps[i]) : reps[i];
            f.rep.id = f.id;
            f.cam = k;
            f.local = reps[i].id;
            f.seenBy = 1u << k;
            f.cls = reps[i].cls;
            f.hard = reps[i].hard != 0;
            f.rank = (int)i;
            f.seen = t;
            out.push_back(f);
        }
        now_ = std::max(now_, t);
        camT_[k] = t;
        return;
    }
    if (t < camT_[k] - 1e-9) return;   // старое сообщение пришло позже нового
    camT_[k] = t;
    now_ = std::max(now_, t);
    const int n = (int)cams_.size();
    std::vector<TargetReport> ref(reps.size());
    std::vector<double> edge(reps.size(), 0);
    const CameraModel& cm = cams_[k].model;
    for (size_t i = 0; i < reps.size(); ++i) {
        ref[i] = toRef(k, reps[i]);
        double u, v;
        if (project(cm, cams_[k].pose, dir_from_azel(reps[i].az, reps[i].el), u, v)) {
            const double px = std::min(std::min(u, cm.w - u), std::min(v, cm.h - v));
            edge[i] = px * pixel_deg(cm, cams_[k].pose, u, v);
        }
    }
    auto mapped = [&](int local) -> Map* {
        for (auto& m : map_)
            if (m.cam == k && m.local == local) return &m;
        return nullptr;
    };
    // 1. треки, уже связанные с целями
    std::vector<int> unm;
    std::vector<int> touched;
    for (size_t i = 0; i < reps.size(); ++i) {
        Map* m = mapped(reps[i].id);
        Tgt* g = m ? get(m->id) : nullptr;
        if (!g) {
            if (m) m->id = 0;
            unm.push_back((int)i);
            continue;
        }
        m->lastT = t;
        Obs& o = g->obs[k];
        if (o.local != reps[i].id) o.bornT = t;
        o.local = reps[i].id;
        o.cam = reps[i];
        o.ref = ref[i];
        o.t = t;
        o.rank = (int)i;
        o.edge = edge[i];
        touched.push_back(g->f.id);
    }
    map_.erase(std::remove_if(map_.begin(), map_.end(), [](const Map& m) { return m.id == 0; }), map_.end());
    // 2. новые треки камеры: цель другой камеры (или ждущая по прогнозу) там же — это она
    struct Cand {
        double d;
        int i, id;
    };
    std::vector<Cand> cs;
    for (int i : unm)
        for (const auto& g : tg_) {
            if (listed(g, k)) continue;   // у этой камеры у цели свой трек
            double az, el, s;
            pred(best(g, t), t, az, el, s);
            const double d = ang(az, el, ref[i].az, ref[i].el);
            const double gate = std::min<double>(c_.gateMaxDeg, c_.gateDeg + c_.gateSigma * std::sqrt(s * s + (double)ref[i].sigma * ref[i].sigma)) +
                                0.5 * std::max(best(g, t).size_deg, ref[i].size_deg);
            if (d < gate) cs.push_back({d, i, g.f.id});
        }
    std::sort(cs.begin(), cs.end(), [](const Cand& a, const Cand& b) { return a.d < b.d; });
    std::vector<int> doneI, doneId;
    for (const Cand& c : cs) {
        if (std::find(doneI.begin(), doneI.end(), c.i) != doneI.end() ||
            std::find(doneId.begin(), doneId.end(), c.id) != doneId.end())
            continue;
        doneI.push_back(c.i);
        doneId.push_back(c.id);
        Tgt* g = get(c.id);
        if (!listedAny(*g)) ++counters.reacquired;
        // у этой камеры у цели был другой трек — связь переходит к новому
        map_.erase(std::remove_if(map_.begin(), map_.end(), [&](const Map& m) { return m.cam == k && m.id == c.id; }), map_.end());
        map_.push_back({k, reps[c.i].id, c.id, t});
        Obs& o = g->obs[k];
        o.local = reps[c.i].id;
        o.cam = reps[c.i];
        o.ref = ref[c.i];
        o.t = t;
        o.rank = c.i;
        o.bornT = t;
        o.edge = edge[c.i];
        touched.push_back(c.id);
    }
    for (int i : unm) {
        if (std::find(doneI.begin(), doneI.end(), i) != doneI.end()) continue;
        Tgt g;
        g.f.id = nextId_++;
        g.f.born = t;
        g.obs.assign(n, Obs());
        Obs& o = g.obs[k];
        o.local = reps[i].id;
        o.cam = reps[i];
        o.ref = ref[i];
        o.t = t;
        o.rank = i;
        o.bornT = t;
        o.edge = edge[i];
        g.last = ref[i];
        map_.push_back({k, reps[i].id, g.f.id, t});
        tg_.push_back(g);
        touched.push_back(g.f.id);
        ++counters.created;
    }
    // 3. связи с треками, которых камера давно не присылает, снимаются
    map_.erase(std::remove_if(map_.begin(), map_.end(), [&](const Map& m) { return m.cam == k && m.lastT < t - c_.dropS; }), map_.end());
    for (auto& g : tg_)
        if (g.obs[k].local && g.obs[k].t < t - c_.dropS) g.obs[k].local = 0;
    // 4. цели: от какой камеры сообщения, класс, захват; калибровка по точной
    for (auto& g : tg_) {
        const bool any = listedAny(g);
        if (any) choosePrimary(g);
        if (!any) continue;
        const Obs& p = g.obs[g.primary];
        FusedTarget& f = g.f;
        f.rep = p.ref;
        f.rep.id = f.id;
        f.cam = g.primary;
        f.local = p.local;
        f.seenBy = 0;
        f.rank = 1000;
        int cls = p.cam.cls;
        bool hard = false;
        for (int j = 0; j < n; ++j) {
            if (!listed(g, j)) continue;
            f.seenBy |= 1u << j;
            f.rank = std::min(f.rank, g.obs[j].rank);
            if (!cls) cls = g.obs[j].cam.cls;
            hard = hard || g.obs[j].cam.hard;
        }
        if (cls) {
            g.lastCls = cls;
            g.lastClsT = now_;
        } else if (now_ - g.lastClsT < c_.keepS) {
            cls = g.lastCls;   // новая камера ещё не разобралась, что это: класс от прежней
        }
        if (hard) g.lastHardT = now_;
        else hard = now_ - g.lastHardT < c_.keepS && cls != 2;
        f.cls = cls;
        f.hard = hard;
        f.rep.cls = cls;
        f.rep.hard = hard;
        f.seen = std::max(f.seen, p.t);
        if (p.t >= g.last.t) g.last = f.rep;
        // пара для калибровки: эта камера видит цель в этом кадре, точная уверенно ведёт её же
        if (c_.calib && g.haveRef && g.refConfident && listed(g, k) && g.obs[k].t >= t - 1e-9 &&
            std::fabs(g.refRep.t - t) < c_.calibPairMaxDt) {
            TargetReport r = g.refRep;
            double az, el, s;
            pred(r, t, az, el, s);
            r.az = (float)az, r.el = (float)el, r.t = t;
            calibPair(k, g.f.id, g.obs[k].cam, r);
        }
    }
    // 5. две цели от разных камер совпадают какое-то время — это одна
    for (size_t a = 0; a < tg_.size(); ++a) {
        Tgt& ga = tg_[a];
        if (!listedAny(ga)) continue;
        for (size_t b = 0; b < tg_.size(); ++b) {
            if (a == b) continue;
            Tgt& gb = tg_[b];
            if (!listedAny(gb) || gb.f.born < ga.f.born || (gb.f.born == ga.f.born && gb.f.id < ga.f.id)) continue;   // gb — младшая
            bool shared = false;
            for (int j = 0; j < n; ++j) shared = shared || (listed(ga, j) && listed(gb, j));
            if (shared) continue;
            double az1, el1, s1, az2, el2, s2;
            pred(ga.f.rep, now_, az1, el1, s1);
            pred(gb.f.rep, now_, az2, el2, s2);
            const double gate = c_.gateDeg + c_.gateSigma * std::sqrt(s1 * s1 + s2 * s2);
            if (ang(az1, el1, az2, el2) < gate) {
                if (gb.nearId != ga.f.id) {
                    gb.nearId = ga.f.id;
                    gb.nearT = now_;
                } else if (now_ - gb.nearT >= c_.mergeS) {
                    mergeInto(ga, gb);
                    gb.f.id = 0;   // удалить ниже
                }
            } else if (gb.nearId == ga.f.id) {
                gb.nearId = 0;
            }
        }
    }
    tg_.erase(std::remove_if(tg_.begin(), tg_.end(), [](const Tgt& g) { return g.f.id == 0; }), tg_.end());
    for (auto& g : tg_)
        if (listedAny(g)) choosePrimary(g);
    // 6. цели, которых давно не видит никто (ни широкие, ни точная), забываются
    std::vector<int> dead;
    for (auto& g : tg_) {
        if (listedAny(g)) continue;
        const double unseen = now_ - std::max(g.f.seen, g.f.refSeen);
        if (unseen < c_.coastS) continue;
        bool keep = false;
        if (unseen < c_.coastGapS) {   // в щели между секторами видеть её некому — ждёт дольше
            double az, el, s, u, v;
            pred(best(g, now_), now_, az, el, s);
            bool gap = s < c_.coastGapSigma;
            for (int j = 0; j < n && gap; ++j)
                if (projectTo(j, az, el, u, v) && in_frame(cams_[j].model, u, v)) gap = false;
            if (gap) g.gapT = now_;
            keep = s < c_.coastGapSigma && now_ - g.gapT < c_.coastEmergeS;   // вышла из щели — время её найти
        }
        if (!keep) dead.push_back(g.f.id);
    }
    if (!dead.empty()) {
        tg_.erase(std::remove_if(tg_.begin(), tg_.end(),
                                 [&](const Tgt& g) { return std::find(dead.begin(), dead.end(), g.f.id) != dead.end(); }),
                  tg_.end());
        map_.erase(std::remove_if(map_.begin(), map_.end(),
                                  [&](const Map& m) { return std::find(dead.begin(), dead.end(), m.id) != dead.end(); }),
                   map_.end());
    }
}

void Fusion::observeReference(int id, const TargetReport& r, bool confident) {
    if (passthrough()) return;
    Tgt* g = get(resolve(id));
    if (!g) return;
    g->refRep = r;
    g->refRep.id = g->f.id;
    g->haveRef = true;
    g->refConfident = confident;
    g->f.refSeen = std::max(g->f.refSeen, r.t);
}

std::vector<FusedTarget> Fusion::targets() const {
    if (passthrough()) {
        std::vector<FusedTarget> v;
        for (int k = 0; k < (int)pt_.size(); ++k)
            if (camT_[k] >= now_ - c_.dropS || pt_.size() == 1) v.insert(v.end(), pt_[k].begin(), pt_[k].end());
        if (pt_.size() > 1)
            std::stable_sort(v.begin(), v.end(), [](const FusedTarget& a, const FusedTarget& b) {
                return a.rank != b.rank ? a.rank < b.rank : a.cam < b.cam;
            });
        return v;
    }
    std::vector<FusedTarget> v;
    for (const auto& g : tg_)
        if (listedAny(g)) v.push_back(g.f);
    std::stable_sort(v.begin(), v.end(), [](const FusedTarget& a, const FusedTarget& b) {
        if (a.rank != b.rank) return a.rank < b.rank;
        if (a.cam != b.cam) return a.cam < b.cam;
        return a.id < b.id;
    });
    return v;
}

const FusedTarget* Fusion::find(int id) const {
    if (passthrough()) {
        for (const auto& v : pt_)
            for (const auto& f : v)
                if (f.id == id) return &f;
        return nullptr;
    }
    const Tgt* g = get(id);
    return g ? &g->f : nullptr;
}

int Fusion::resolve(int id) const {
    if (passthrough()) return id;
    for (int hop = 0; hop < 16; ++hop) {
        if (get(id)) return id;
        int next = 0;
        for (const auto& a : alias_)
            if (a.first == id) next = a.second;
        if (!next) return 0;
        id = next;
    }
    return 0;
}

int Fusion::localOf(int id, int k) const {
    if (passthrough()) return c_.independent ? (id / 100000 == k ? id % 100000 : 0) : k == 0 ? id : 0;
    const Tgt* g = get(id);
    if (!g || k < 0 || k >= (int)g->obs.size()) return 0;
    return g->obs[k].local;
}

int Fusion::globalOf(int k, int local) const {
    if (passthrough()) return c_.independent ? k * 100000 + local : k == 0 ? local : 0;
    for (const auto& m : map_)
        if (m.cam == k && m.local == local) return m.id;
    return 0;
}

std::vector<CueArea> Fusion::cues(int k, double t) const {
    std::vector<CueArea> out;
    if (passthrough() || !c_.cue || k < 0 || k >= (int)cams_.size()) return out;
    const CameraModel& m = cams_[k].model;
    for (const auto& g : tg_) {
        if (listed(g, k)) continue;
        const TargetReport& r = best(g, t);
        if (t - r.t > c_.coastS && t - g.gapT > c_.coastEmergeS) continue;
        double az, el, s, u, v;
        pred(r, t, az, el, s);
        if (!projectTo(k, az, el, u, v) || !in_frame(m, u, v)) continue;
        const double pd = pixel_deg(m, cams_[k].pose, u, v);
        const double rad = (3 * s + r.size_deg) / std::max(1e-6, pd);
        out.push_back({(float)u, (float)v, (float)std::max<double>(c_.cueMinPx, std::min<double>(c_.cueMaxPx, rad)), g.f.id});
    }
    return out;
}

std::string Fusion::saveCalib() const {
    std::ostringstream o;
    o.precision(10);
    for (size_t k = 0; k < cal_.size(); ++k) {
        const CameraCalib& c = cal_[k];
        o << "calib " << k << ' ' << c.pairs;
        for (double x : c.w) o << ' ' << x;
        for (double x : c.A) o << ' ' << x;
        for (double x : c.b) o << ' ' << x;
        o << ' ' << c.rmsDeg << '\n';
    }
    return o.str();
}

bool Fusion::loadCalib(const std::string& s) {
    std::istringstream in(s);
    std::string tag;
    bool any = false;
    while (in >> tag) {
        if (tag != "calib") {
            std::string rest;
            std::getline(in, rest);
            continue;
        }
        size_t k;
        CameraCalib c;
        if (!(in >> k >> c.pairs)) return false;
        for (double& x : c.w) in >> x;
        for (double& x : c.A) in >> x;
        for (double& x : c.b) in >> x;
        in >> c.rmsDeg;
        if (!in) return false;
        if (k < cal_.size()) {
            c.sigmaDeg = 0;
            cal_[k] = c;
            cal_[k].sigmaDeg = calibSigma((int)k, dir_from_azel(cams_[k].pose.az, cams_[k].pose.el));
            any = true;
        }
    }
    return any;
}

}  // namespace dronolok
