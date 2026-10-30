#include "twocam.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>

namespace twocam {

constexpr double PI = 3.14159265358979323846;

using namespace dronolok;

namespace {

const int CLOUD_STEP = 8;    // сетка облаков на широкой, пикс.
const int NARROW_STEP = 4;   // сетка фона на точной, пикс.

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---- шум значений: решётка с хешем и плавной интерполяцией
inline uint32_t hash2(int x, int y, uint32_t s) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + s * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}
inline float hval(int x, int y, uint32_t s) { return (hash2(x, y, s) & 0xffffff) / 16777215.f; }
inline float vnoise(double x, double y, uint32_t s) {
    const double fx = std::floor(x), fy = std::floor(y);
    const int ix = (int)fx, iy = (int)fy;
    double tx = x - fx, ty = y - fy;
    tx = tx * tx * (3 - 2 * tx);
    ty = ty * ty * (3 - 2 * ty);
    const float a = hval(ix, iy, s), b = hval(ix + 1, iy, s), c = hval(ix, iy + 1, s), d = hval(ix + 1, iy + 1, s);
    return (float)((a + (b - a) * tx) * (1 - ty) + (c + (d - c) * tx) * ty);
}
inline float fbm(double x, double y, uint32_t s) {
    return 0.5f * vnoise(x, y, s) + 0.27f * vnoise(2.03 * x, 2.03 * y, s + 1) + 0.15f * vnoise(4.1 * x, 4.1 * y, s + 2) +
           0.08f * vnoise(8.3 * x, 8.3 * y, s + 3);
}
inline float smooth01(float e0, float e1, float x) {
    const float t = std::min(1.f, std::max(0.f, (x - e0) / (e1 - e0)));
    return t * t * (3 - 2 * t);
}
// азимут со знаком относительно севера — без шва на 0° в поле зрения
inline double saz(double az) { return az_diff(az, 0); }

inline double treeline(double az) {
    const double a = saz(az);
    return 1.6 + 1.1 * std::sin(a * 0.071 + 1.0) + 0.55 * std::sin(a * 0.29 + 0.3) + 0.6 * (vnoise(a * 0.8, 0.5, 77) - 0.5) +
           0.35 * (vnoise(a * 3.1, 0.5, 78) - 0.5);
}
inline float sky_base(double el) { return (float)(208 - 58 * std::min(1.0, std::max(0.0, el) / 60.0)); }

template <class F>
void parallel_rows(int h, F&& f) {
    const int n = std::max(1, std::min(8, (int)std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int k = 0; k < n; ++k) th.emplace_back([&, k] { f(h * k / n, h * (k + 1) / n); });
    for (auto& t : th) t.join();
}

inline double ang(double az1, double el1, double az2, double el2) {
    return angle_between(dir_from_azel(az1, el1), dir_from_azel(az2, el2));
}

}  // namespace

const char* state_name(NState s) {
    switch (s) {
        case NState::Idle: return "ждёт цель";
        case NState::Slew: return "поворот к цели";
        case NState::Acquire: return "поиск в окне";
        case NState::Track: return "ведёт цель";
        case NState::Lost: return "потеряла, ищет";
    }
    return "";
}

Vec3 Body::at(double t) const {
    const double dt = t - t0;
    Vec3 p{p0.x + v.x * dt, p0.y + v.y * dt, p0.z + v.z * dt};
    if (bird) {   // птица летит неровно: покачивание по высоте и в сторону
        p.z += 1.2 * std::sin(0.9 * dt + phase);
        p.x += 0.8 * std::sin(0.6 * dt + 2 * phase);
    }
    return p;
}

Emulator::Emulator() {}
Emulator::~Emulator() { stop(); }

void Emulator::event(double t, const std::string& s) {
    events_.push_back({t, s});
    while (events_.size() > 60) events_.pop_front();
}

float Emulator::background(double az, double el, double t, bool fine) const {
    const double a = saz(az);
    const double tl = treeline(az);
    if (el < tl) {   // лес и поле: крупные пятна, кроны, мелкая текстура
        const double depth = tl - el;
        float g = 62 + 26 * vnoise(a * 0.6, el * 0.6, 11) + 22 * vnoise(a * 4.5, el * 4.5, 12);
        if (fine) g += 14 * (hval((int)std::floor(a * 40), (int)std::floor(el * 40), 13) - 0.5f);
        g += (float)(40 * std::exp(-depth * 1.5));   // дымка у кромки
        return g;
    }
    float s = sky_base(el);
    if (c_.clouds > 0) {
        const double wind = 0.04 * t;   // облака плывут по азимуту, °/с
        const float f = fbm((a + wind) * 0.055, el * 0.085, 21);
        const float th = 1.f - c_.clouds * 0.75f;
        const float cover = smooth01(th - 0.08f, th + 0.12f, f);
        if (cover > 0) {
            const float tex = vnoise((a + wind) * 0.9, el * 0.9, 22);
            const float cloud = 228 + 14 * tex - 34 * smooth01(0.4f, 1.f, f) * vnoise((a + wind) * 0.35, el * 0.35, 23);
            s = s * (1 - cover) + cloud * cover;
        }
    }
    return s;
}

void Emulator::spawn(Body& b, double t, bool first) {
    std::uniform_real_distribution<double> U(0, 1);
    b.id = nextBodyId_++;
    b.t0 = t;
    const double half = c_.wideFov * 0.42;
    for (int k = 0; k < 50; ++k) {
        const double az = az_norm(c_.wideAz + (U(rng_) * 2 - 1) * half);
        const double el = treeline(az) + 1.5 + U(rng_) * 26;
        const double R = c_.rangeMin + U(rng_) * (c_.rangeMax - c_.rangeMin);
        // новая цель влетает в поле зрения сбоку или снизу, первые — где угодно
        double u, v;
        const Vec3 d = dir_from_azel(az, el);
        if (!project(wideCam_, widePose_, d, u, v) || !in_frame(wideCam_, u, v, -40)) continue;
        b.p0 = {d.x * R, d.y * R, d.z * R};
        break;
    }
    const double sp = b.bird ? 8 + U(rng_) * 6 : c_.speedMin + U(rng_) * (c_.speedMax - c_.speedMin);
    const double hd = U(rng_) * 2 * PI;
    b.v = {sp * std::cos(hd), sp * std::sin(hd), (U(rng_) * 2 - 1) * (b.bird ? 0.5 : 1.5)};
    b.size = b.bird ? c_.birdSize : c_.droneSize * (float)(0.8 + 0.4 * U(rng_));
    b.flapHz = b.bird ? (float)(4 + U(rng_) * 4) : 0;
    b.phase = (float)(U(rng_) * 6.28);
    (void)first;
}

void Emulator::drawBodies(const CameraModel& m, const Pose& p, double t, uint8_t* px, int w, int h) {
    for (const Body& b : bodies_) {
        const Vec3 q = b.at(t);
        const double R = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
        const Vec3 d{q.x / R, q.y / R, q.z / R};
        double u, v;
        if (!project(m, p, d, u, v) || !in_frame(m, u, v, 20)) continue;
        const double pd = pixel_deg(m, p, u, v);
        double sizePx = rad2deg(b.size / R) / pd;
        double c = (b.bird ? -55.0 : -70.0) * std::exp(-R / c_.visibility);
        if (b.bird) {   // взмахи: силуэт и контраст пульсируют
            const double fl = std::sin(2 * PI * b.flapHz * t + b.phase);
            sizePx *= 0.75 + 0.25 * fl;
            c *= 0.8 + 0.2 * fl;
        }
        const double sx = std::max(0.5, 0.42 * sizePx), sy = std::max(0.5, (b.bird ? 0.3 : 0.55) * sizePx);
        // точка меньше пикселя: та же сумма контраста, размазанная по пятну датчика
        const double area = 0.785 * sizePx * sizePx * (b.bird ? 0.6 : 1.0);
        const double A = c * std::min(1.0, area / (2 * PI * sx * sy));
        const int rx = (int)std::ceil(3 * sx), ry = (int)std::ceil(3 * sy);
        const int x0 = std::max(0, (int)u - rx), x1 = std::min(w - 1, (int)u + rx);
        const int y0 = std::max(0, (int)v - ry), y1 = std::min(h - 1, (int)v + ry);
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const double dx = (x + 0.5 - u) / sx, dy = (y + 0.5 - v) / sy;
                const double g = std::exp(-0.5 * (dx * dx + dy * dy));
                uint8_t& o = px[(size_t)y * w + x];
                o = (uint8_t)std::max(0.0, std::min(255.0, o + A * g));
            }
    }
}

void Emulator::renderWide(double t, std::vector<uint8_t>& out) {
    const int W = wideCam_.w, H = wideCam_.h;
    const size_t N = (size_t)W * H;
    out.resize(N);
    // облака плывут медленно (0.04 °/с — меньше пикселя за 0.5 с): фон с облаками считаем раз в
    // полсекунды, на сетке CLOUD_STEP, между узлами — линейно
    if (wideSky_.size() != N || t - wideSkyT_ >= 0.5) {
        wideSkyT_ = t;
        wideSky_.resize(N);
        std::vector<float> cg((size_t)cgW_ * cgH_);
        parallel_rows(cgH_, [&](int ja, int jb) {
            for (int j = ja; j < jb; ++j)
                for (int i = 0; i < cgW_; ++i) {
                    const size_t k = (size_t)j * cgW_ + i;
                    const double az = wideAz_[k], el = wideEl_[k];
                    cg[k] = el >= treeline(az) ? background(az, el, t, false) - sky_base(el) : 0.f;
                }
        });
        parallel_rows(H, [&](int ya, int yb) {
            for (int y = ya; y < yb; ++y) {
                const float gy = (float)y / CLOUD_STEP;
                const int j0 = std::min(cgH_ - 2, (int)gy);
                const float ty = gy - j0;
                const float* c0 = &cg[(size_t)j0 * cgW_];
                const float* c1 = c0 + cgW_;
                const float* base = &wideBase_[(size_t)y * W];
                const uint8_t* gr = &wideGround_[(size_t)y * W];
                int16_t* o = &wideSky_[(size_t)y * W];
                for (int x = 0; x < W; ++x) {
                    float v = base[x];
                    if (!gr[x]) {
                        const float gx = (float)x / CLOUD_STEP;
                        const int i0 = std::min(cgW_ - 2, (int)gx);
                        const float tx = gx - i0;
                        v += (c0[i0] + (c0[i0 + 1] - c0[i0]) * tx) * (1 - ty) + (c1[i0] + (c1[i0 + 1] - c1[i0]) * tx) * ty;
                    }
                    o[x] = (int16_t)std::lround(v * 8);
                }
            }
        });
    }
    // шум датчика: заранее набранный, со случайным сдвигом на каждый кадр
    if (wideNoise_.size() != N) {
        wideNoise_.resize(N);
        uint32_t r = noiseSeed_ ^ 0xabcdefu;
        for (auto& v : wideNoise_) {
            r ^= r << 13;
            r ^= r >> 17;
            r ^= r << 5;
            v = noiseTab_[r >> 16];
        }
    }
    noiseSeed_ = noiseSeed_ * 1664525u + 1013904223u;
    const size_t off = (size_t)(noiseSeed_ % (uint32_t)N);
    parallel_rows(H, [&](int ya, int yb) {
        for (size_t k = (size_t)ya * W, e = (size_t)yb * W; k < e; ++k) {
            const int v = (wideSky_[k] + wideNoise_[(k + off) % N] + 4) >> 3;
            out[k] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    });
    drawBodies(wideCam_, widePose_, t, out.data(), W, H);
}

void Emulator::renderNarrow(double t, const Pose& p, std::vector<uint8_t>& out) {
    const int W = narrowCam_.w, H = narrowCam_.h;
    out.resize((size_t)W * H);
    const int gw = W / NARROW_STEP + 2, gh = H / NARROW_STEP + 2;
    std::vector<float> bg((size_t)gw * gh), gaz(bg.size()), gel(bg.size());
    std::vector<uint8_t> ground(bg.size());
    parallel_rows(gh, [&](int ja, int jb) {
        for (int j = ja; j < jb; ++j)
            for (int i = 0; i < gw; ++i) {
                double az, el;
                azel_from_dir(unproject(narrowCam_, p, i * NARROW_STEP, j * NARROW_STEP), az, el);
                const size_t k = (size_t)j * gw + i;
                gaz[k] = (float)saz(az);
                gel[k] = (float)el;
                ground[k] = el < treeline(az);
                bg[k] = background(az, el, t, false);
            }
    });
    const uint32_t seed = noiseSeed_ += 0x7f4a7c15u;
    parallel_rows(H, [&](int ya, int yb) {
    for (int y = ya; y < yb; ++y) {
        // свой генератор на строку: шум не зависит от числа потоков (ядер ПК)
        uint32_t r = (seed ^ (uint32_t)(y * 2654435761u) ^ 0x51ed27u) | 1u;
        const float gy = (float)y / NARROW_STEP;
        const int j0 = std::min(gh - 2, (int)gy);
        const float ty = gy - j0;
        for (int x = 0; x < W; ++x) {
            const float gx = (float)x / NARROW_STEP;
            const int i0 = std::min(gw - 2, (int)gx);
            const float tx = gx - i0;
            const size_t k = (size_t)j0 * gw + i0;
            auto lerp = [&](const std::vector<float>& a) {
                return (a[k] + (a[k + 1] - a[k]) * tx) * (1 - ty) + (a[k + gw] + (a[k + gw + 1] - a[k + gw]) * tx) * ty;
            };
            float v = lerp(bg);
            if (ground[k] | ground[k + 1] | ground[k + gw] | ground[k + gw + 1]) {
                // земля: точная граница кромки и мелкая текстура крон — по направлению пикселя
                const double az = lerp(gaz), el = lerp(gel);
                if (el < treeline(az)) v = background(az, el, t, true);
                else v = sky_base(el);
            }
            r ^= r << 13;
            r ^= r >> 17;
            r ^= r << 5;
            v += noiseTab_[r >> 16] * 0.125f;
            out[(size_t)y * W + x] = (uint8_t)std::max(0.f, std::min(255.f, v + 0.5f));
        }
    }
    });
    drawBodies(narrowCam_, p, t, out.data(), W, H);
}

int Emulator::nearestBody(double t, double az, double el, double maxDeg) const {
    int best = 0;
    double bd = maxDeg;
    const Vec3 d = dir_from_azel(az, el);
    for (const Body& b : bodies_) {
        const double a = angle_between(d, b.at(t));
        if (a < bd) {
            bd = a;
            best = b.id;
        }
    }
    return best;
}

void Emulator::start(const Config& c) {
    stop();
    c_ = c;
    switchMode_ = (int)c.switchMode;
    speed_ = c.simSpeed;
    rng_.seed(c.seed);
    noiseSeed_ = c.seed * 7919u + 1;
    nextBodyId_ = 1;
    wideCam_ = CameraModel();
    wideCam_.w = c.wideW;
    wideCam_.h = c.wideH;
    wideCam_.hfov_deg = c.wideFov;
    wideCam_.lens = c.wideFisheye ? Lens::Fisheye : Lens::Pinhole;
    widePose_ = {c.wideAz, c.wideEl, 0};
    narrowCam_ = CameraModel();
    narrowCam_.w = c.narrowW;
    narrowCam_.h = c.narrowH;
    narrowCam_.hfov_deg = c.narrowFov;
    // фон широкой без облаков — один раз
    const int W = c.wideW, H = c.wideH;
    wideBase_.assign((size_t)W * H, 0);
    wideGround_.assign((size_t)W * H, 0);
    parallel_rows(H, [&](int ya, int yb) {
        for (int y = ya; y < yb; ++y)
            for (int x = 0; x < W; ++x) {
                double az, el;
                azel_from_dir(unproject(wideCam_, widePose_, x + 0.5, y + 0.5), az, el);
                const bool g = el < treeline(az);
                const size_t k = (size_t)y * W + x;
                wideGround_[k] = g;
                wideBase_[k] = g ? background(az, el, 0, true) : sky_base(el);
            }
    });
    cgW_ = W / CLOUD_STEP + 2;
    cgH_ = H / CLOUD_STEP + 2;
    wideAz_.assign((size_t)cgW_ * cgH_, 0);
    wideEl_.assign(wideAz_.size(), 0);
    for (int j = 0; j < cgH_; ++j)
        for (int i = 0; i < cgW_; ++i) {
            double az, el;
            azel_from_dir(unproject(wideCam_, widePose_, i * CLOUD_STEP, j * CLOUD_STEP), az, el);
            wideAz_[(size_t)j * cgW_ + i] = (float)az;
            wideEl_[(size_t)j * cgW_ + i] = (float)el;
        }
    ngm_.reset();
    wideSky_.clear();
    wideNoise_.clear();
    wideSkyT_ = -1;
    noiseTab_.resize(65536);
    {
        std::mt19937 g(c.seed + 99);
        std::normal_distribution<float> N(0, c.noise * 8);
        for (auto& v : noiseTab_) v = (int16_t)std::max(-32000.f, std::min(32000.f, N(g)));
    }
    bodies_.clear();
    for (int i = 0; i < c.drones + c.birds; ++i) {
        Body b;
        b.bird = i >= c.drones;
        spawn(b, 0, true);
        bodies_.push_back(b);
    }
    // точная
    DetectorConfig dc;
    dc.scales = c.narrowScales;   // цель в узком поле крупнее: несколько масштабов
    dc.threads = 2;
    ndet_.reset(new Detector(dc));
    ntrk_.reset(new Tracker(c.narrowW, c.narrowH, tracker_for_fps(TrackerConfig(), c.narrowFps)));
    ntrk_->setMovingCamera(true);
    mount_ = Mount();
    mount_.az = c.wideAz;
    mount_.el = c.wideEl;
    cmdPose_ = {(float)mount_.az, (float)mount_.el, 0};
    ffAz_ = ffEl_ = 0;
    havePrev_ = false;
    state_ = NState::Idle;
    assigned_ = assignedBody_ = nlock_ = 0;
    haveNrep_ = false;
    winOn_ = false;
    st_ = Stats();
    events_.clear();
    targets_.clear();
    inbox_.clear();
    visited_.clear();
    // широкая: настоящий конвейер, кадры — от нас
    Settings s;
    s.pace = Pace::MaxSpeed;
    s.loop = false;
    s.fresh = true;
    s.staticFile = false;
    s.maskFile = false;
    pipe_.setSettings(s);
    wideReady_ = false;
    wideIdx_ = -1;
    wideDone_ = -1;
    wideRes_.reset();
    pipe_.setObserver([this](const Result& r) {
        auto p = std::make_shared<Result>(r);
        std::lock_guard<std::mutex> lk(wm_);
        wideRes_ = p;
        wideDone_ = r.frame ? r.frame->index : wideDone_;
        wcv_.notify_all();
    });
    VideoInfo vi;
    vi.w = W;
    vi.h = H;
    vi.fps = c.wideFps;
    vi.codec = "эмулятор";
    stop_ = false;
    pipe_.openGenerator(vi,
                        [this](Frame& f) {
                            std::unique_lock<std::mutex> lk(wm_);
                            wcv_.wait(lk, [&] { return wideReady_ || stop_; });
                            if (stop_) return false;
                            memcpy(f.px.data(), wideFrame_.data(), std::min(f.px.size(), wideFrame_.size()));
                            f.index = wideIdx_;
                            wideReady_ = false;
                            return true;
                        },
                        "эмулятор");
    event(0, "старт: целей " + std::to_string(c.drones) + " дронов и " + std::to_string(c.birds) + " птиц");
    running_ = true;
    th_ = std::thread([this] { loop(); });
}

void Emulator::stop() {
    stop_ = true;
    wcv_.notify_all();
    if (th_.joinable()) th_.join();
    pipe_.close();
    running_ = false;
}

void Emulator::onWide(double t, const Result& r) {
    std::vector<WideTarget> v;
    for (const Track& tr : r.tracks) {
        if (tr.clutter || tr.miss > 2) continue;
        WideTarget w;
        w.id = tr.id;
        w.rep = make_report(wideCam_, widePose_, t, c_.wideFps, tr.x, tr.y, tr.vsx, tr.vsy, tr.sigma);
        w.rep.id = tr.id;
        w.cls = (int)tr.cls;
        w.hard = r.lockId == tr.id && r.lockState == LockState::Hard;
        w.rep.cls = w.cls;
        w.rep.hard = w.hard;
        w.truth = nearestBody(t, w.rep.az, w.rep.el, 0.25);
        if (w.truth)
            for (const Body& b : bodies_)
                if (b.id == w.truth) {
                    double az, el;
                    azel_from_dir(b.at(t), az, el);
                    st_.errWide.push_back((float)ang(az, el, w.rep.az, w.rep.el));
                }
        v.push_back(w);
    }
    // очередь в порядке весомости широкой: захваченная первой
    std::vector<int> order = r.lockQueue;
    std::stable_sort(v.begin(), v.end(), [&](const WideTarget& a, const WideTarget& b) {
        auto pos = [&](int id) {
            auto it = std::find(order.begin(), order.end(), id);
            return it == order.end() ? 1000 : (int)(it - order.begin());
        };
        return pos(a.id) < pos(b.id);
    });
    inbox_.push_back({t + c_.handoffLatency, std::move(v)});
    st_.wideMs += r.procMs;
}

void Emulator::assign(double t, int wideId, const char* why) {
    if (wideId == assigned_) return;
    const WideTarget* w = nullptr;
    for (const auto& x : targets_)
        if (x.id == wideId) w = &x;
    assigned_ = wideId;
    assignedBody_ = w ? w->truth : 0;
    state_ = wideId ? NState::Slew : NState::Idle;
    nlock_ = 0;
    haveNrep_ = false;
    tAssign_ = tState_ = t;
    tInView_ = -1;
    tSearch_ = -1;
    gotThis_ = false;
    ntrk_->setFollowing(false);
    if (!wideId) return;
    ++st_.assignments;
    double slew = 0;
    if (w) slew = ang(mount_.az, mount_.el, w->rep.az, w->rep.el);
    char b[160];
    snprintf(b, sizeof(b), "цель %d (%s): %s, поворот %.1f°", wideId, w && w->cls == 1 ? "дрон" : w && w->cls == 2 ? "птица" : "?",
             why, slew);
    event(t, b);
}

void Emulator::choose(double t) {
    cool_.erase(std::remove_if(cool_.begin(), cool_.end(), [&](const std::pair<int, double>& c) { return c.second <= t; }),
                cool_.end());
    auto ok = [&](const WideTarget& w) {
        if (c_.dronesOnly && w.cls == 2) return false;
        for (const auto& c : cool_)
            if (c.first == w.id) return false;
        return true;
    };
    auto find = [&](int id) -> const WideTarget* {
        for (const auto& x : targets_)
            if (x.id == id) return &x;
        return nullptr;
    };
    const SwitchMode mode = (SwitchMode)switchMode_.load();
    auto pick = [&](bool skipVisited) -> int {
        int best = 0;
        double bestCost = 1e18;
        int rank = 0;
        for (const auto& w : targets_) {
            ++rank;
            if (!ok(w) || w.id == assigned_) continue;
            if (skipVisited && std::find(visited_.begin(), visited_.end(), w.id) != visited_.end()) continue;
            // по очереди — в порядке весомости широкой; ближайшая — по времени поворота
            const double cost = mode == SwitchMode::Nearest ? ang(mount_.az, mount_.el, w.rep.az, w.rep.el) : rank;
            if (cost < bestCost) {
                bestCost = cost;
                best = w.id;
            }
        }
        return best;
    };
    const int cmd = cmd_.exchange(0);
    if (cmd > 0) {
        if (find(cmd)) assign(t, cmd, "выбрана вручную");
    } else if (cmd < 0) {
        if (assigned_) visited_.push_back(assigned_);
        int n = pick(true);
        if (!n) {
            visited_.clear();
            n = pick(false);
        }
        if (n) assign(t, n, "следующая");
    }
    // назначенную широкая потеряла: если точная её ведёт — ведём дальше, иначе отпускаем
    if (assigned_ && !find(assigned_) && state_ != NState::Track && t - tState_ > 1.5) {
        event(t, "цель " + std::to_string(assigned_) + " пропала у широкой");
        assign(t, 0, "");
    }
    if (mode == SwitchMode::Manual) {
        if (!assigned_ && !targets_.empty())
            for (const auto& w : targets_)
                if (ok(w) && w.hard) {   // вручную: сами берём только подтверждённый дрон
                    assign(t, w.id, "жёсткий захват широкой");
                    break;
                }
        return;
    }
    if (!assigned_) {
        int n = pick(true);
        if (!n) {
            visited_.clear();
            n = pick(false);
        }
        if (n) assign(t, n, mode == SwitchMode::Nearest ? "ближайшая" : "первая в очереди");
    } else if (state_ == NState::Track && t - tTrackStart_ > c_.dwell) {
        visited_.push_back(assigned_);
        int n = pick(true);
        if (!n) {
            visited_.clear();
            visited_.push_back(assigned_);
            n = pick(true);
        }
        if (n) assign(t, n, "следующая по очереди");
    }
}

void Emulator::stepNarrow(double t, double dt) {
    const WideTarget* wt = nullptr;
    for (const auto& x : targets_)
        if (x.id == assigned_) wt = &x;
    // 1. куда смотреть: по точной, пока она ведёт цель (или недавно видела), иначе по широкой
    const bool useN = haveNrep_ && (state_ == NState::Track || t - tLastSeen_ < 0.6);
    const TargetReport* rep = useN ? &nrep_ : wt ? &wt->rep : nullptr;
    const double lead = 1.5 / c_.narrowFps;   // команда исполняется к следующему кадру
    if (rep) {
        cmdPose_ = aim(*rep, t, lead);
        ffAz_ = c_.feedForward ? rep->vaz : 0;
        ffEl_ = c_.feedForward ? rep->vel : 0;
        if (!c_.feedForward) cmdPose_ = aim(*rep, t, 0);
    } else {
        ffAz_ = ffEl_ = 0;
    }
    // 2. поворотка: упреждение по скорости цели + доля ошибки, торможение до цели, пределы
    const int sub = 4;
    const double h = dt / sub, K = 6.0;
    for (int i = 0; i < sub; ++i) {
        auto axis = [&](double& pos, double& vel, double err, double ff) {
            double want = ff + K * err;
            const double brake = std::sqrt(2 * c_.maxAccel * std::fabs(err)) + std::fabs(ff);
            want = std::max(-brake, std::min(brake, want));
            want = std::max<double>(-c_.maxRate, std::min<double>(c_.maxRate, want));
            const double dv = std::max(-c_.maxAccel * h, std::min(c_.maxAccel * h, want - vel));
            vel += dv;
            pos += vel * h;
        };
        axis(mount_.az, mount_.vaz, az_diff(cmdPose_.az, mount_.az), ffAz_);
        axis(mount_.el, mount_.vel, cmdPose_.el - mount_.el, ffEl_);
        mount_.az = az_norm(mount_.az);
        mount_.el = std::max(-5.0, std::min(89.0, mount_.el));
    }
    const Pose truth{(float)mount_.az, (float)mount_.el, 0};
    const float q = std::max(1e-4f, c_.encoderDeg);
    const Pose enc{(float)(std::round(mount_.az / q) * q), (float)(std::round(mount_.el / q) * q), 0};
    // 3. кадр точной и наш алгоритм на нём; сдвиг картинки — по датчикам угла, не по кадру
    const double t0 = now_s();
    renderNarrow(t, truth, nframe_);
    st_.renderNarrowMs += (now_s() - t0) * 1000;
    const double t1 = now_s();
    double dx = 0, dy = 0;
    if (c_.encoderMotion) {
        if (havePrev_) pose_shift(narrowCam_, prevEnc_, enc, narrowCam_.w * 0.5, narrowCam_.h * 0.5, dx, dy);
    } else {   // как с рук: сдвиг по картинке (GlobalMotion)
        const MotionEstimate e = ngm_.estimate(nframe_.data(), narrowCam_.w, narrowCam_.h, narrowCam_.w);
        if (e.valid) {
            dx = e.dx;
            dy = e.dy;
        }
    }
    prevEnc_ = enc;
    havePrev_ = true;
    // окно поиска: где цель должна быть в этом кадре
    const TargetReport* prior = useN ? &nrep_ : wt ? &wt->rep : nullptr;
    winOn_ = prior != nullptr;
    if (prior) win_ = search_window(*prior, t, narrowCam_, enc, 3.f, 10.f);
    // в окне детектор ищет с пониженным порогом (Detector::setFocus)
    std::vector<FocusArea> fa;
    if (state_ == NState::Track) ntrk_->focus((float)dx, (float)dy, fa);
    if (c_.windowFocus && winOn_ && win_.visible && state_ != NState::Track)
        fa.push_back({win_.u, win_.v, std::min(win_.radius, 120.f)});
    // ведём: вокруг прогноза своей цели тоже ниже порог. Трекер сам даёт это только устоявшемуся
    // треку, а слабая дальняя цель без этого то пропадает, то находится заново другим треком
    if (c_.lockFocus && state_ == NState::Track && winOn_ && win_.visible)
        fa.push_back({win_.u, win_.v, std::min(win_.radius, 40.f)});
    ndet_->setFocus(fa);
    ndet_->detect(nframe_.data(), narrowCam_.w, narrowCam_.h, narrowCam_.w, ncands_);
    const std::vector<Track>& tr = ntrk_->update(ncands_, (float)dx, (float)dy);
    if (dumpAt_ > 0 && t >= dumpAt_ && t < dumpAt_ + dumpLen_) {   // отладка: кадры точной в PGM и её треки
        char fn[64];
        snprintf(fn, sizeof(fn), "narrow_%07.3f.pgm", t);
        if (FILE* f = fopen(fn, "wb")) {
            fprintf(f, "P5 %d %d 255\n", narrowCam_.w, narrowCam_.h);
            fwrite(nframe_.data(), 1, nframe_.size(), f);
            fclose(f);
        }
        printf("t=%.3f state=%s lock=%d win=(%.0f,%.0f r%.0f vis%d) cands=%zu\n", t, state_name(state_), nlock_, win_.u,
               win_.v, win_.radius, win_.visible, ncands_.size());
        for (const Track& x : tr)
            printf("  трек %d (%.1f,%.1f) hits %d miss %d clutter %d cls %d\n", x.id, x.x, x.y, x.hits, x.miss, (int)x.clutter,
                   (int)x.cls);
        for (const Body& b : bodies_) {
            double u, v;
            const Vec3 q = b.at(t);
            const double R = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
            if (project(narrowCam_, truth, {q.x / R, q.y / R, q.z / R}, u, v) && in_frame(narrowCam_, u, v))
                printf("  тело %d%s в (%.1f,%.1f), %.0f м\n", b.id, b.bird ? " птица" : "", u, v, R);
        }
    }
    st_.narrowMs += (now_s() - t1) * 1000;
    ++st_.narrowFrames;
    auto truthOf = [&](double az, double el) { return nearestBody(t, az, el, 0.2); };
    auto bodyDir = [&](int id, double& az, double& el) {
        for (const Body& b : bodies_)
            if (b.id == id) {
                azel_from_dir(b.at(t), az, el);
                return true;
            }
        return false;
    };
    // 4. что делает точная
    if (!assigned_) st_.idleTime += dt;
    else if (!assignedBody_) st_.falseTarget += dt;
    if (!assigned_) {
        state_ = NState::Idle;
    } else if (state_ == NState::Slew) {
        if (winOn_ && win_.visible) {
            state_ = NState::Acquire;
            tState_ = t;
            if (tInView_ < 0) {
                tInView_ = t;
                st_.slewSum += t - tAssign_;
                ++st_.slewN;
            }
        }
    } else if (state_ == NState::Acquire || state_ == NState::Lost) {
        // в окне — трек, ближайший к ожидаемому месту (летящий, не помеха, держится minHold)
        const int minHits = std::max(3, (int)std::lround(c_.minHold * c_.narrowFps));
        if (tSearch_ < 0) tSearch_ = t;
        // где цель по широкой сейчас: сосед рядом (другой дрон в окне) не годится
        double waz = 0, wel = 0, wsg = 0;
        if (wt) predict(wt->rep, t, waz, wel, wsg);
        int best = 0;
        double bd = 1e18;
        for (const Track& x : tr) {
            if (!winOn_) break;   // ни широкая, ни своя память не говорят, где цель: любой трек в кадре — не она
            if (x.clutter || x.miss > 0 || x.hits < minHits) continue;
            const double d = std::hypot(x.x - win_.u, x.y - win_.v);
            if (c_.windowFocus && winOn_ && d > win_.radius) continue;
            if (wt && c_.wideCheck) {
                double az, el;
                azel_from_dir(unproject(narrowCam_, enc, x.x, x.y), az, el);
                if (ang(az, el, waz, wel) > std::max(0.4, 5 * wsg)) continue;
            }
            if (d < bd) {
                bd = d;
                best = x.id;
            }
        }
        if (best) {
            nlock_ = best;
            ntrk_->selectLock(best);
            ntrk_->setFollowing(true);
            const bool again = state_ == NState::Lost;
            state_ = NState::Track;
            tState_ = tTrackStart_ = tLastSeen_ = t;
            for (const Track& x : tr)
                if (x.id == best) {
                    double az, el;
                    azel_from_dir(unproject(narrowCam_, enc, x.x, x.y), az, el);
                    const int who = truthOf(az, el);
                    if (who != assignedBody_ || !who) ++st_.wrongLocks;
                    char b[160];
                    snprintf(b, sizeof(b), "точная %s цель %d за %.2f с%s", again ? "снова взяла" : "взяла", assigned_,
                             t - tAssign_, who == assignedBody_ && who ? "" : " — НЕ та цель");
                    event(t, b);
                }
            if (!gotThis_) {
                gotThis_ = true;
                ++st_.acquired;
                st_.acqSum += t - tAssign_;
                ++st_.acqN;
            }
        } else if (winOn_ && !win_.visible && t - tState_ > 0.5) {
            state_ = NState::Slew;
            tState_ = t;
        } else if (state_ == NState::Lost && t - tLastSeen_ > 3 && !wt) {
            event(t, "точная потеряла цель " + std::to_string(assigned_) + ", широкая её не видит");
            assign(t, 0, "");
        }
    } else if (state_ == NState::Track) {
        const Track* me = nullptr;
        for (const Track& x : tr)
            if (x.id == nlock_) me = &x;
        if (me && me->miss <= std::max(2, (int)(0.15 * c_.narrowFps))) {
            nrep_ = make_report(narrowCam_, enc, t, c_.narrowFps, me->x, me->y, me->vsx, me->vsy, me->sigma);
            nrep_.id = assigned_;
            haveNrep_ = true;
            if (!me->miss) tLastSeen_ = t;
            if (t - tTrackStart_ > 0.4) tSearch_ = -1;   // держит устойчиво
            st_.trackTime += dt;
            const int who = truthOf(nrep_.az, nrep_.el);
            double baz, bel;
            if (assignedBody_ && bodyDir(assignedBody_, baz, bel)) {
                if (who == assignedBody_) st_.trackCorrect += dt;
                st_.errLock.push_back((float)ang(baz, bel, nrep_.az, nrep_.el));
                st_.errAxis.push_back((float)ang(baz, bel, mount_.az, mount_.el));
            }
            // широкая видит назначенную цель далеко от того, что ведёт точная: не та цель.
            // Сравниваем с прогнозом широкой на этот момент: её сообщение старше на задержку и кадр
            bool off = false;
            if (wt) {
                double waz = wt->rep.az, wel = wt->rep.el, wsg = wt->rep.sigma;
                if (c_.wideCheck) predict(wt->rep, t, waz, wel, wsg);
                const double tol = c_.wideCheck ? std::max(0.5, 6.0 * wsg) : std::max(1.5, 6.0 * wsg);
                off = ang(waz, wel, nrep_.az, nrep_.el) > tol && t - tTrackStart_ > (c_.wideCheck ? 0.3 : 1.0);
            }
            if (off) {
                event(t, "точная ведёт не то, что широкая: ищет заново");
                state_ = NState::Lost;
                haveNrep_ = false;
                tState_ = t;
                ntrk_->setFollowing(false);
            }
        } else {
            ++st_.losses;
            state_ = NState::Lost;
            tState_ = t;
            ntrk_->setFollowing(false);
            event(t, "точная потеряла цель " + std::to_string(assigned_) + ", ищет по прогнозу");
        }
    }
    // не может удержать цель giveUp секунд (ищет, берёт и теряет): точная её не подтверждает —
    // это шум или облако у широкой, или цель для точной слишком слабая. В авто-режимах — к следующей
    if (assigned_ && tSearch_ >= 0 && t - tSearch_ > c_.giveUp && (SwitchMode)switchMode_.load() != SwitchMode::Manual) {
        ++st_.unconfirmed;
        cool_.push_back({assigned_, t + c_.coolDown});
        event(t, "точная не подтверждает цель " + std::to_string(assigned_) + ": за " +
                     std::to_string((int)c_.giveUp) + " с не удержала, к ней — не раньше чем через " +
                     std::to_string((int)c_.coolDown) + " с");
        assign(t, 0, "");
    }
    // 5. для интерфейса
    std::lock_guard<std::mutex> lk(vm_);
    view_.narrow = nframe_;
    view_.nw = narrowCam_.w;
    view_.nh = narrowCam_.h;
    view_.ntracks = tr;
    view_.nlock = state_ == NState::Track ? nlock_ : 0;
    view_.state = state_;
    view_.assigned = assigned_;
    view_.targets = targets_;
    view_.axis = truth;
    view_.cmd = cmdPose_;
    view_.rateAz = (float)mount_.vaz;
    view_.rateEl = (float)mount_.vel;
    view_.win = win_;
    view_.winOn = winOn_;
    view_.truth.clear();
    for (const Body& b : bodies_) {
        View::TruthPt p{};
        const Vec3 d = b.at(t);
        double u, v;
        p.id = b.id;
        p.bird = b.bird;
        p.wu = p.wv = -1;
        if (project(wideCam_, widePose_, d, u, v)) {
            p.wu = (float)u;
            p.wv = (float)v;
        }
        p.inNarrow = project(narrowCam_, truth, d, u, v) && in_frame(narrowCam_, u, v);
        p.nu = (float)u;
        p.nv = (float)v;
        view_.truth.push_back(p);
    }
    view_.footprint.clear();
    const double W = narrowCam_.w, H = narrowCam_.h;
    const double pts[][2] = {{0, 0}, {W / 2, 0}, {W, 0}, {W, H / 2}, {W, H}, {W / 2, H}, {0, H}, {0, H / 2}};
    for (const auto& p : pts) {
        double u, v;
        if (project(wideCam_, widePose_, unproject(narrowCam_, truth, p[0], p[1]), u, v))
            view_.footprint.push_back({(float)u, (float)v});
    }
    view_.events = events_;
    view_.st = st_;
    view_.st.simT = t + dt;   // шаг [t, t+dt] уже учтён в долях времени
}

void Emulator::loop() {
    double t = 0, nextWide = 0;
    const double dtN = 1.0 / std::max(1.f, c_.narrowFps);
    const double wall0 = now_s();
    double simAtWall = 0, wallBase = wall0;
    float lastSpeed = speed_;
    int64_t idx = 0;
    while (!stop_) {
        if (c_.stopAt > 0 && t >= c_.stopAt - 1e-9) break;   // замер: ровно столько модельного времени
        if (paused_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            wallBase = now_s();
            simAtWall = t;
            continue;
        }
        if (t >= nextWide - 1e-9) {
            // тела, вылетевшие из поля зрения широкой или ушедшие далеко, — заново
            for (Body& b : bodies_) {
                const Vec3 d = b.at(t);
                const double R = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
                double u, v, az, el;
                azel_from_dir(d, az, el);
                if (!project(wideCam_, widePose_, d, u, v) || !in_frame(wideCam_, u, v, 30) || R > c_.rangeMax * 1.7 ||
                    R < 60 || el < treeline(az) + 0.2) {
                    if (b.id == assignedBody_) event(t, "цель " + std::to_string(assigned_) + " вышла из зоны");
                    spawn(b, t, false);
                }
            }
            const double r0 = now_s();
            std::vector<uint8_t> buf;
            renderWide(t, buf);
            st_.renderWideMs += (now_s() - r0) * 1000;
            std::shared_ptr<const Result> res;
            {
                std::unique_lock<std::mutex> lk(wm_);
                wideFrame_.swap(buf);
                wideIdx_ = idx;
                wideReady_ = true;
                wcv_.notify_all();
                // ждём, пока конвейер обработает этот кадр
                wcv_.wait_for(lk, std::chrono::seconds(10), [&] { return wideDone_ >= idx || stop_; });
                if (stop_) break;
                res = wideRes_;
            }
            if (res && wideDone_ >= idx) {
                onWide(t, *res);
                std::lock_guard<std::mutex> lk(vm_);
                view_.wide = res;
                view_.status.clear();
            } else {
                std::lock_guard<std::mutex> lk(vm_);
                view_.status = "широкая камера не ответила";
            }
            ++st_.wideFrames;
            ++idx;
            nextWide += 1.0 / std::max(1.f, c_.wideFps);
        }
        while (!inbox_.empty() && inbox_.front().first <= t + 1e-9) {
            targets_ = std::move(inbox_.front().second);
            inbox_.pop_front();
            // трек широкой, назначенный точной, перешёл на другое тело (перескок трека у широкой):
            // точная честно ведёт то, что ей указали, — это ошибка широкой, считаем отдельно
            for (const auto& w : targets_)
                if (w.id == assigned_ && w.truth && assignedBody_ && w.truth != assignedBody_) {
                    ++st_.wideSwaps;
                    event(t, "широкая: трек " + std::to_string(w.id) + " перескочил на другую цель");
                    assignedBody_ = w.truth;
                } else if (w.id == assigned_ && w.truth && !assignedBody_) {
                    assignedBody_ = w.truth;   // назначили по шуму рядом с целью, теперь трек на ней
                } else if (w.id == assigned_ && !w.truth && assignedBody_) {
                    // трек широкой ушёл с цели на пустое место (помеха, облако): цель уже не та
                    for (const Body& b : bodies_)
                        if (b.id == assignedBody_) {
                            double az, el;
                            azel_from_dir(b.at(t), az, el);
                            if (ang(az, el, w.rep.az, w.rep.el) > 1.0) {
                                ++st_.wideSwaps;
                                event(t, "широкая: трек " + std::to_string(w.id) + " ушёл с цели на помеху");
                                assignedBody_ = 0;
                            }
                        }
                }
        }
        choose(t);
        stepNarrow(t, dtN);
        t += dtN;
        // темп: модельное время идёт в speed раз быстрее настенного (0 — как успевает ПК)
        const float sp = speed_;
        if (sp != lastSpeed) {
            wallBase = now_s();
            simAtWall = t;
            lastSpeed = sp;
        }
        if (sp > 0) {
            const double wallWant = wallBase + (t - simAtWall) / sp;
            const double wait = wallWant - now_s();
            if (wait > 0) std::this_thread::sleep_for(std::chrono::microseconds((int64_t)(wait * 1e6)));
            else if (wait < -0.5) {   // ПК не успевает: не догоняем рывком
                wallBase = now_s();
                simAtWall = t;
            }
        }
        std::lock_guard<std::mutex> lk(vm_);
        view_.running = true;
        view_.paused = paused_;
    }
    std::lock_guard<std::mutex> lk(vm_);
    view_.running = false;
}

View Emulator::view() const {
    std::lock_guard<std::mutex> lk(vm_);
    return view_;
}

static void pct(std::vector<float> v, float& med, float& p90) {
    med = p90 = 0;
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    med = v[v.size() / 2];
    p90 = v[std::min(v.size() - 1, v.size() * 9 / 10)];
}

Stats Emulator::runBlocking(const Config& c, double seconds, bool verbose) {
    Config cc = c;
    cc.simSpeed = 0;
    cc.stopAt = seconds;
    start(cc);
    double last = -1;
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const View v = view();
        if (verbose && v.st.simT - last >= 5) {
            last = v.st.simT;
            printf("  %.0f с: назначений %d, захватов %d, ведёт %.0f%% времени\n", v.st.simT, v.st.assignments,
                   v.st.acquired, v.st.simT > 0 ? 100 * v.st.trackTime / v.st.simT : 0);
            fflush(stdout);
        }
        if (v.st.simT >= seconds - 1e-6 || (!v.running && v.st.simT > 0)) break;
    }
    if (th_.joinable()) th_.join();   // цикл остановился сам на stopAt: итог — ровно на этот момент
    const View v = view();
    if (verbose)
        for (const auto& e : v.events) printf("  [%6.2f] %s\n", e.t, e.text.c_str());
    stop();
    return v.st;
}

int run_cli(int argc, char** argv) {
    Config c;
    Emulator e;
    double seconds = 60;
    std::string json;
    bool verbose = true;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--seconds") seconds = atof(val().c_str());
        else if (a == "--drones") c.drones = atoi(val().c_str());
        else if (a == "--birds") c.birds = atoi(val().c_str());
        else if (a == "--seed") c.seed = (uint32_t)atoi(val().c_str());
        else if (a == "--range") sscanf(val().c_str(), "%f,%f", &c.rangeMin, &c.rangeMax);
        else if (a == "--speed") sscanf(val().c_str(), "%f,%f", &c.speedMin, &c.speedMax);
        else if (a == "--clouds") c.clouds = (float)atof(val().c_str());
        else if (a == "--noise") c.noise = (float)atof(val().c_str());
        else if (a == "--visibility") c.visibility = (float)atof(val().c_str());
        else if (a == "--size") c.droneSize = (float)atof(val().c_str());
        else if (a == "--wide") sscanf(val().c_str(), "%dx%d", &c.wideW, &c.wideH);
        else if (a == "--wide-fps") c.wideFps = (float)atof(val().c_str());
        else if (a == "--fisheye") c.wideFisheye = true;
        else if (a == "--fov") c.narrowFov = (float)atof(val().c_str());
        else if (a == "--fps") c.narrowFps = (float)atof(val().c_str());
        else if (a == "--rate") c.maxRate = (float)atof(val().c_str());
        else if (a == "--accel") c.maxAccel = (float)atof(val().c_str());
        else if (a == "--latency") c.handoffLatency = (float)atof(val().c_str());
        else if (a == "--dwell") c.dwell = (float)atof(val().c_str());
        else if (a == "--switch") {
            const std::string m = val();
            c.switchMode = m == "manual" ? SwitchMode::Manual : m == "nearest" ? SwitchMode::Nearest : SwitchMode::Queue;
        } else if (a == "--json") json = val();
        else if (a == "--quiet") verbose = false;
        else if (a == "--image-motion") c.encoderMotion = false;
        else if (a == "--no-ff") c.feedForward = false;
        else if (a == "--no-window") c.windowFocus = false;
        else if (a == "--no-lock-focus") c.lockFocus = false;
        else if (a == "--nscales") c.narrowScales = atoi(val().c_str());
        else if (a == "--dump") sscanf(val().c_str(), "%lf,%lf", &e.dumpAt_, &e.dumpLen_);
        else if (a == "--no-wide-check") c.wideCheck = false;
        else if (a == "--min-hold") c.minHold = (float)atof(val().c_str());
        else if (a == "--give-up") c.giveUp = (float)atof(val().c_str());
        else if (a == "--cool-down") c.coolDown = (float)atof(val().c_str());
        else {
            printf("dronolok_app --twocam [--seconds 60] [--drones 3] [--birds 1] [--seed 1] [--range 250,1200]\n"
                   "  [--speed 6,22] [--clouds 0.35] [--noise 2] [--visibility 6000] [--size 0.5] [--wide 4608x3456] [--wide-fps 14] [--fisheye] [--fov 20]\n"
                   "  [--fps 60] [--rate 30] [--accel 150] [--latency 0.06] [--dwell 6]\n"
                   "  [--switch queue|nearest|manual] [--json out.json] [--quiet]\n"
                   "  сравнение доработок точной: [--image-motion] [--no-ff] [--no-window] [--no-lock-focus]\n"
                   "  [--min-hold 0.08] [--give-up 3] [--cool-down 10]\n");
            return a == "--help" ? 0 : 2;
        }
    }
    const Stats s = e.runBlocking(c, seconds, verbose);
    float lm, l9, am, a9, wm, w9;
    pct(s.errLock, lm, l9);
    pct(s.errAxis, am, a9);
    pct(s.errWide, wm, w9);
    const double T = std::max(1e-9, s.simT);
    const double nPd = c.narrowFov / c.narrowW;
    printf("модельное время %.1f с: кадров широкой %d, точной %d\n", s.simT, s.wideFrames, s.narrowFrames);
    printf("назначений %d, точная взяла %d, не ту цель %d, потерь %d\n", s.assignments, s.acquired, s.wrongLocks, s.losses);
    printf("до цели в кадре точной %.2f с, до захвата %.2f с (среднее)\n", s.slewN ? s.slewSum / s.slewN : 0,
           s.acqN ? s.acqSum / s.acqN : 0);
    printf("ведёт цель %.0f%% времени, из них на назначенной %.0f%%\n", 100 * s.trackTime / T,
           s.trackTime > 0 ? 100 * s.trackCorrect / s.trackTime : 0);
    printf("нечего вести %.0f%% времени, назначена ложная цель широкой %.0f%%; точная не подтвердила целей: %d; перескоков трека у широкой: %d\n",
           100 * s.idleTime / T, 100 * s.falseTarget / T, s.unconfirmed, s.wideSwaps);
    printf("ошибка: захват точной %.3f° (90%% %.3f°, %.1f пикс. точной), ось %.3f° (90%% %.3f°), широкая %.3f° (90%% %.3f°)\n",
           lm, l9, lm / nPd, am, a9, wm, w9);
    printf("обработка: широкая %.1f мс/кадр, точная %.2f мс/кадр; отрисовка широкой %.1f, точной %.1f мс/кадр\n",
           s.wideFrames ? s.wideMs / s.wideFrames : 0, s.narrowFrames ? s.narrowMs / s.narrowFrames : 0,
           s.wideFrames ? s.renderWideMs / s.wideFrames : 0, s.narrowFrames ? s.renderNarrowMs / s.narrowFrames : 0);
    if (!json.empty()) {
        if (FILE* f = fopen(json.c_str(), "w")) {
            fprintf(f,
                    "{\"sim_s\": %.2f, \"assignments\": %d, \"acquired\": %d, \"wrong_locks\": %d, \"losses\": %d,\n"
                    " \"slew_s\": %.3f, \"acquire_s\": %.3f, \"track_pct\": %.2f, \"track_correct_pct\": %.2f,\n"
                    " \"err_lock_deg\": [%.4f, %.4f], \"err_axis_deg\": [%.4f, %.4f], \"err_wide_deg\": [%.4f, %.4f],\n"
                    " \"unconfirmed\": %d, \"wide_swaps\": %d, \"idle_pct\": %.2f, \"false_target_pct\": %.2f, \"wide_ms\": %.2f, \"narrow_ms\": %.3f}\n",
                    s.simT, s.assignments, s.acquired, s.wrongLocks, s.losses, s.slewN ? s.slewSum / s.slewN : 0,
                    s.acqN ? s.acqSum / s.acqN : 0, 100 * s.trackTime / T,
                    s.trackTime > 0 ? 100 * s.trackCorrect / s.trackTime : 0, lm, l9, am, a9, wm, w9,
                    s.unconfirmed, s.wideSwaps, 100 * s.idleTime / T, 100 * s.falseTarget / T, s.wideFrames ? s.wideMs / s.wideFrames : 0, s.narrowFrames ? s.narrowMs / s.narrowFrames : 0);
            fclose(f);
        }
    }
    return 0;
}

}  // namespace twocam
