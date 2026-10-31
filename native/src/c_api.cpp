#include "dronolok/dronolok.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "dronolok/detector.hpp"
#include "dronolok/geometry.hpp"
#include "dronolok/gimbal.hpp"

using namespace dronolok;

struct dl_detector {
    Detector det;
    std::vector<Candidate> cands;
    std::string backend;
    explicit dl_detector(const DetectorConfig& c) : det(c), backend(det.backendName()) {}
};

struct dl_tracker {
    Tracker trk;
    std::vector<Candidate> dets;
    std::vector<FocusArea> focus;
    float dx = 0, dy = 0;
    bool staticSet = false;   // память помех выставлена dl_tracker_set_static
    dl_tracker(int w, int h) : trk(w, h) {}
};

struct dl_motion {
    GlobalMotion gm;
};

extern "C" {

dl_motion* dl_motion_create(void) { return new dl_motion; }
void dl_motion_destroy(dl_motion* m) { delete m; }
int dl_motion_estimate(dl_motion* m, const uint8_t* gray, int w, int h, int stride, float* dx, float* dy,
                       float* quality) {
    const MotionEstimate e = m->gm.estimate(gray, w, h, stride);
    if (dx) *dx = e.dx;
    if (dy) *dy = e.dy;
    if (quality) *quality = e.quality;
    return e.valid ? 1 : 0;
}

void dl_motion_set_fps(dl_motion* m, float fps) { m->gm.setFps(fps); }

void dl_config_default(dl_config* c) {
    DetectorConfig d;
    c->k = d.k; c->pool = d.pool; c->thr = d.thr; c->clutter_k = d.clutter_k; c->tile = d.tile;
    c->max_density = d.max_density; c->dark = d.dark; c->bright = d.bright; c->max_cands = d.max_cands;
    c->threads = d.threads; c->isa = (int)d.isa; c->job_rows = d.job_rows; c->job_cols = d.job_cols;
    c->ncpus = 0; c->cpus = nullptr;
    c->scales = d.scales; c->focus_k = d.focus_k;
}

dl_detector* dl_detector_create(const dl_config* c) {
    DetectorConfig d;
    // C API повторяет детектор Python (detector/): один кандидат на тайл, маска с запасом, без линии горизонта
    d.per_tile = 1;
    d.bg_map = false;
    d.skyline = false;
    if (c) {
        d.k = c->k; d.pool = c->pool; d.thr = c->thr; d.clutter_k = c->clutter_k; d.tile = c->tile;
        d.max_density = c->max_density; d.dark = c->dark != 0; d.bright = c->bright != 0;
        d.max_cands = c->max_cands; d.threads = c->threads; d.isa = (Isa)c->isa;
        d.job_rows = c->job_rows; d.job_cols = c->job_cols;
        if (c->cpus && c->ncpus > 0) d.cpus.assign(c->cpus, c->cpus + c->ncpus);
        d.scales = c->scales; d.focus_k = c->focus_k;
    }
    try {
        return new dl_detector(d);
    } catch (...) {
        return nullptr;
    }
}

void dl_detector_destroy(dl_detector* d) { delete d; }

// исключения не должны уходить через границу C (Python, другие языки): на ошибке — 0 / nullptr
int dl_detect(dl_detector* d, const uint8_t* gray, int w, int h, int stride, dl_candidate* out, int cap) {
    if (!d || !gray || w <= 0 || h <= 0 || stride < w) return 0;
    try {
        d->det.detect(gray, w, h, stride, d->cands);
    } catch (...) {
        d->cands.clear();
        return 0;
    }
    const int n = (int)d->cands.size();
    for (int i = 0; i < std::min(n, cap); ++i) {
        const Candidate& c = d->cands[i];
        out[i] = {c.x, c.y, c.contrast, c.snr, c.polarity, c.level, c.energy, c.margin};
    }
    return n;
}

void dl_detector_timing(dl_detector* d, dl_timing* t) {
    const DetectTiming& s = d->det.timing();
    *t = {s.total_ms, s.response_ms, s.select_ms, s.jobs, s.jobs_skipped};
}

const char* dl_detector_backend(dl_detector* d) { return d->backend.c_str(); }
int dl_detector_threads(dl_detector* d) { return d->det.threads(); }
void dl_mask_begin(dl_detector* d) { d->det.beginMaskLearning(); }
void dl_mask_end(dl_detector* d, float frac) { d->det.endMaskLearning(frac); }

int dl_mask_get(dl_detector* d, uint8_t* out, int cap, int* tw, int* th) {
    const auto& m = d->det.tileMask();
    if (tw) *tw = d->det.tilesX();
    if (th) *th = d->det.tilesY();
    if (m.empty()) return 0;
    std::copy(m.begin(), m.begin() + std::min<size_t>(m.size(), (size_t)std::max(0, cap)), out);
    return (int)m.size();
}

int dl_hardware_threads(void) { return hardware_threads(); }

int dl_available_isas(int* out, int cap) {
    auto v = available_isas();
    for (int i = 0; i < std::min((int)v.size(), cap); ++i) out[i] = (int)v[i];
    return (int)v.size();
}

dl_tracker* dl_tracker_create(int w, int h) {
    if (w <= 0 || h <= 0) return nullptr;
    try {
        return new dl_tracker(w, h);
    } catch (...) {
        return nullptr;
    }
}
void dl_tracker_destroy(dl_tracker* t) { delete t; }
void dl_tracker_reset(dl_tracker* t) { t->trk.resetTracks(); }

int dl_tracker_queue(dl_tracker* t, int* ids, int cap) {
    std::vector<int> q;
    t->trk.lockQueue(q);
    for (int i = 0; ids && i < std::min((int)q.size(), cap); ++i) ids[i] = q[i];
    return (int)q.size();
}
int dl_tracker_locked(dl_tracker* t) { return t->trk.lockedId(); }
int dl_tracker_select(dl_tracker* t, int id) { return t->trk.selectLock(id) ? 1 : 0; }
int dl_tracker_next(dl_tracker* t) { return t->trk.nextLock(); }

int dl_tracker_update(dl_tracker* t, const dl_candidate* dets, int n, dl_track* out, int cap) {
    if (!dets || n < 0) n = 0;   // resize(-1) — это 4 миллиарда элементов
    t->dets.clear();
    for (int i = 0; i < n; ++i) {
        const dl_candidate& c = dets[i];
        // nan/inf в координатах от вызывающего: приведение к int в сетке помех — неопределённое поведение
        if (!std::isfinite(c.x) || !std::isfinite(c.y)) continue;
        t->dets.push_back({c.x, c.y, c.contrast, c.snr, c.polarity, c.level});
        t->dets.back().energy = c.energy;
        t->dets.back().margin = c.margin;
    }
    const std::vector<Track>* trp;
    try {
        trp = &t->trk.update(t->dets, t->dx, t->dy);
    } catch (...) {
        return 0;
    }
    const auto& tr = *trp;
    t->dx = t->dy = 0;
    for (int i = 0; i < std::min((int)tr.size(), cap); ++i) {
        const Track& s = tr[i];
        out[i] = {s.id, s.x, s.y, s.vx, s.vy, s.polarity, s.contrast, s.clutter ? 1 : 0, (int)s.cls,
                  s.birdScore, s.speed, s.resid, s.contrastCv, s.hits, s.age, s.miss, s.sigma, s.score,
                  s.id == t->trk.lockedId() ? 1 : 0, s.flapR2, s.flapPeriod};
    }
    return (int)tr.size();
}

int dl_tracker_clutter(dl_tracker* t, dl_candidate* out, int cap) {
    const auto& c = t->trk.clutterDets();
    for (int i = 0; i < std::min((int)c.size(), cap); ++i)
        out[i] = {c[i].x, c[i].y, c[i].contrast, c[i].snr, c[i].polarity, c[i].level, c[i].energy, c[i].margin};
    return (int)c.size();
}

void dl_tracker_set_fps(dl_tracker* t, float fps) {
    // от умолчаний (они для 25 к/с): повторный вызов не пересчитывает уже пересчитанное
    TrackerConfig c = tracker_for_fps(TrackerConfig(), fps);
    if (t->staticSet) {   // выставлено dl_tracker_set_static — не трогаем
        c.static_frames = t->trk.config().static_frames;
        c.static_hits = t->trk.config().static_hits;
    }
    t->trk.config() = c;
}

void dl_tracker_set_static(dl_tracker* t, int frames, float hits) {
    t->trk.config().static_frames = frames;
    t->trk.config().static_hits = hits;
    t->staticSet = true;
}
int dl_tracker_static_save(dl_tracker* t, const char* path) { return t->trk.saveStatic(path) ? 1 : 0; }
int dl_tracker_static_load(dl_tracker* t, const char* path) { return t->trk.loadStatic(path) ? 1 : 0; }
void dl_tracker_static_clear(dl_tracker* t) { t->trk.clearStatic(); }

void dl_tracker_set_motion(dl_tracker* t, float dx, float dy) {
    t->dx = dx;
    t->dy = dy;
}

int dl_tracker_focus(dl_tracker* t, float* xyr, int cap) {
    t->trk.focus(t->dx, t->dy, t->focus);
    const int n = (int)t->focus.size();
    for (int i = 0; i < std::min(n, cap); ++i) {
        xyr[3 * i] = t->focus[i].x;
        xyr[3 * i + 1] = t->focus[i].y;
        xyr[3 * i + 2] = t->focus[i].r;
    }
    return n;
}

void dl_detector_set_focus(dl_detector* d, const float* xyr, int n) {
    if (!xyr) n = 0;
    std::vector<FocusArea> a(std::max(0, n));
    for (int i = 0; i < n; ++i) a[i] = {xyr[3 * i], xyr[3 * i + 1], xyr[3 * i + 2]};
    d->det.setFocus(a);
}

static CameraModel cam(const dl_camera* c) {
    CameraModel m;
    if (!c) return m;
    m.w = c->w;
    m.h = c->h;
    m.hfov_deg = c->hfov_deg;
    m.lens = c->lens == 1 ? Lens::Fisheye : Lens::Pinhole;
    m.cx = c->cx;
    m.cy = c->cy;
    m.k1 = c->k1;
    return m;
}
static Pose pose(const dl_pose* p) { return p ? Pose{p->az, p->el, p->roll} : Pose{}; }
static TargetReport report(const dl_target* t) {
    TargetReport r;
    r.id = t->id; r.t = t->t; r.az = t->az; r.el = t->el; r.vaz = t->vaz; r.vel = t->vel; r.sigma = t->sigma;
    r.sigma_v = t->sigma_v; r.cls = t->cls; r.confidence = t->confidence; r.hard = t->hard; r.size_deg = t->size_deg;
    return r;
}

int dl_project(const dl_camera* c, const dl_pose* p, float az, float el, float* u, float* v) {
    double uu = 0, vv = 0;
    const bool ok = project(cam(c), pose(p), dir_from_azel(az, el), uu, vv);
    if (u) *u = (float)uu;
    if (v) *v = (float)vv;
    return ok && in_frame(cam(c), uu, vv) ? 1 : 0;
}
void dl_unproject(const dl_camera* c, const dl_pose* p, float u, float v, float* az, float* el) {
    double a, e;
    azel_from_dir(unproject(cam(c), pose(p), u, v), a, e);
    if (az) *az = (float)a;
    if (el) *el = (float)e;
}
float dl_pixel_deg(const dl_camera* c, const dl_pose* p, float u, float v) {
    return (float)pixel_deg(cam(c), pose(p), u, v);
}
void dl_target_from_track(const dl_camera* c, const dl_pose* p, double t, float fps, const dl_track* tr, int hard,
                          dl_target* out) {
    if (!tr || !out) return;
    const TargetReport r = make_report(cam(c), pose(p), t, fps > 0 ? fps : 25.f, tr->x, tr->y, tr->vx, tr->vy, tr->sigma);
    *out = {tr->id, r.t, r.az, r.el, r.vaz, r.vel, r.sigma, r.sigma_v, tr->cls,
            // уверенность по классу: подтверждённый дрон, дрон, неизвестно, птица или помеха
            hard == 2 ? 0.9f : tr->cls == 1 ? 0.6f : tr->cls == 0 ? 0.4f : 0.15f, hard == 2 ? 1 : 0, 0.f};
}
void dl_target_predict(const dl_target* tg, double t, float* az, float* el, float* sigma) {
    double a = 0, e = 0, s = 0;
    if (tg) predict(report(tg), t, a, e, s);
    if (az) *az = (float)a;
    if (el) *el = (float)e;
    if (sigma) *sigma = (float)s;
}
dl_window dl_target_window(const dl_target* tg, double t, const dl_camera* c, const dl_pose* p, float k_sigma) {
    dl_window w{0, 0, 0, 0};
    if (!tg) return w;
    const SearchWindow s = search_window(report(tg), t, cam(c), pose(p), k_sigma > 0 ? k_sigma : 3.f);
    return {s.u, s.v, s.radius, s.visible};
}
dl_pose dl_target_aim(const dl_target* tg, double t, double lead) {
    if (!tg) return {0, 0, 0};
    const Pose p = aim(report(tg), t, lead);
    return {p.az, p.el, p.roll};
}
void dl_pose_shift(const dl_camera* c, const dl_pose* from, const dl_pose* to, float u, float v, float* dx, float* dy) {
    double a = 0, b = 0;
    pose_shift(cam(c), pose(from), pose(to), u, v, a, b);
    if (dx) *dx = (float)a;
    if (dy) *dy = (float)b;
}
int dl_tracker_hard(dl_tracker* t) { return (int)t->trk.lockState(); }

}  // extern "C"

// ---- наведение поворотки
struct dl_gimbal {
    GimbalConfig cfg;
    GimbalFilter f;
    WideBias bias;
    PoseHistory hist;
    AxisProfile ax[2];
    double anchor[2] = {0, 0};
    bool anchored[2] = {false, false};
    double spT = -1;
};

static GimbalConfig gimbal_cfg(const dl_gimbal_config* c) {
    GimbalConfig g;
    if (!c) return g;
    g.max_rate = c->max_rate, g.max_accel = c->max_accel, g.max_jerk = c->max_jerk;
    g.kp = c->kp, g.kv = c->kv, g.brake_share = c->brake_share, g.lead_cmd = c->lead_cmd;
    g.jerk_noise = c->jerk_noise, g.acc_tau = c->acc_tau, g.sigma_pose = c->sigma_pose, g.gate = c->gate;
    g.gate_reset = c->gate_reset, g.bias_tau = c->bias_tau, g.bias_max = c->bias_max;
    return g;
}
void dl_gimbal_defaults(dl_gimbal_config* c) {
    if (!c) return;
    const GimbalConfig g;
    c->max_rate = g.max_rate, c->max_accel = g.max_accel, c->max_jerk = g.max_jerk;
    c->kp = g.kp, c->kv = g.kv, c->brake_share = g.brake_share, c->lead_cmd = g.lead_cmd;
    c->jerk_noise = g.jerk_noise, c->acc_tau = g.acc_tau, c->sigma_pose = g.sigma_pose, c->gate = g.gate;
    c->gate_reset = g.gate_reset, c->bias_tau = g.bias_tau, c->bias_max = g.bias_max;
}
dl_gimbal* dl_gimbal_create(const dl_gimbal_config* c) {
    dl_gimbal* g = new dl_gimbal();
    g->cfg = gimbal_cfg(c);
    g->f.set_config(g->cfg);
    return g;
}
void dl_gimbal_destroy(dl_gimbal* g) { delete g; }
void dl_gimbal_target(dl_gimbal* g, const dl_target* tg) {
    if (!g) return;
    if (tg) g->f.reset(report(tg));
    else g->f.clear();
}
int dl_gimbal_measure(dl_gimbal* g, double t_frame, float az, float el, float sigma) {
    return g && g->f.measure(t_frame, az, el, sigma) ? 1 : 0;
}
int dl_gimbal_estimate(const dl_gimbal* g, double t, dl_target* out) {
    if (!g || !out || !g->f.active()) return 0;
    const TargetReport r = g->f.report(t);
    *out = dl_target();
    out->t = r.t, out->az = r.az, out->el = r.el, out->vaz = r.vaz, out->vel = r.vel, out->sigma = r.sigma;
    out->sigma_v = r.sigma_v;
    return 1;
}
dl_setpoint dl_gimbal_setpoint(const dl_gimbal* g, double now) {
    dl_setpoint o{};
    o.hold = 1;
    if (!g) return o;
    const Setpoint s = gimbal_setpoint(g->f, now, g->bias.az, g->bias.el);
    o.t = s.t, o.az = (float)s.az, o.el = (float)s.el, o.vaz = (float)s.vaz, o.vel = (float)s.vel;
    o.aaz = (float)s.aaz, o.ael = (float)s.ael, o.hold = s.hold ? 1 : 0;
    return o;
}
void dl_gimbal_report_pose(dl_gimbal* g, double t, float az, float el) {
    if (!g) return;
    // азимут в историю — непрерывным (через 0° без скачка на 360)
    double a = az;
    double pa, pe;
    if (g->hist.at(t, pa, pe)) a = pa + az_diff(az, pa);
    g->hist.push(t, a, el);
}
dl_pose dl_gimbal_pose_at(const dl_gimbal* g, double t) {
    dl_pose p{0, 0, 0};
    if (!g) return p;
    double az = 0, el = 0;
    g->hist.at(t, az, el);
    p.az = (float)az_norm(az + g->bias.az);
    p.el = (float)(el + g->bias.el);
    return p;
}
void dl_gimbal_wide_check(dl_gimbal* g, const dl_target* wide, double t_frame, float az, float el, double dt) {
    if (!g || !wide) return;
    double waz, wel, wsg;
    predict(report(wide), t_frame, waz, wel, wsg);
    g->bias.update(az_diff(waz, az), wel - el, std::max(wsg, 0.005), dt, g->cfg);
}
void dl_gimbal_bias(const dl_gimbal* g, float* az, float* el) {
    if (az) *az = g ? (float)g->bias.az : 0.f;
    if (el) *el = g ? (float)g->bias.el : 0.f;
}
void dl_gimbal_drive_reset(dl_gimbal* g, float az, float el) {
    if (!g) return;
    g->ax[0] = AxisProfile();
    g->ax[1] = AxisProfile();
    g->ax[0].p = az, g->ax[1].p = el;
    g->anchored[0] = g->anchored[1] = false;
}
void dl_gimbal_drive(dl_gimbal* g, const dl_setpoint* sp, double t_recv, double t, double h, float* pos, float* vel) {
    if (!g) return;
    if (t_recv != g->spT) g->anchored[0] = g->anchored[1] = false, g->spT = t_recv;
    for (int i = 0; i < 2; ++i) {
        AxisProfile& a = g->ax[i];
        if (!sp || sp->hold) {
            a.hold(h, g->cfg, g->anchor[i], g->anchored[i], i == 0);
        } else {
            const double s = std::min(0.25, std::max(0.0, t - t_recv));
            const double p0 = i ? sp->el : sp->az, v0 = i ? sp->vel : sp->vaz, a0 = i ? sp->ael : sp->aaz;
            a.track(p0 + v0 * s + 0.5 * a0 * s * s, v0 + a0 * s, a0, h, g->cfg, i == 0);
        }
        if (pos) pos[i] = (float)(i ? a.p : az_norm(a.p));
        if (vel) vel[i] = (float)a.v;
    }
}
