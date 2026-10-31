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

// Взмахи по ряду x (последние n): размах (СКО к среднему после снятия наклона) и наибольшая
// автокорреляция на лагах lmin..lmax (период взмаха, кадров), у которой на половине лага — обратная
// (колебание, а не медленный дрейф размера или шум). Возвращает автокорреляцию
float flap_shape(const std::deque<float>& x, int n, int lmin, int lmax, float& amp, int& lag) {
    amp = 0;
    lag = 0;
    n = std::min(n, (int)x.size());
    if (n < 6 || lmax < lmin) return 0;
    std::vector<double> v(x.end() - n, x.end());
    double mx = 0, mi = (n - 1) * 0.5, sxy = 0, sxx = 0;
    for (double a : v) mx += a;
    mx /= n;
    if (mx <= 0) return 0;
    for (int i = 0; i < n; ++i) {
        sxy += (i - mi) * (v[i] - mx);
        sxx += (i - mi) * (i - mi);
    }
    const double k = sxy / sxx;
    double var = 0;
    for (int i = 0; i < n; ++i) {
        v[i] -= mx + k * (i - mi);
        var += v[i] * v[i];
    }
    var /= n;
    amp = (float)(std::sqrt(var) / mx);
    if (var <= 1e-12) return 0;
    auto acf = [&](int L) {
        double c = 0;
        for (int i = L; i < n; ++i) c += v[i] * v[i - L];
        return c / ((n - L) * var);
    };
    double best = 0;
    // лаг не длиннее половины ряда: на длинных лагах слагаемых мало, и шум даёт ложный пик (3 Гц у дронов)
    for (int L = std::max(2, lmin); L <= std::min(lmax, n / 2); ++L) {
        const double c = acf(L);
        if (c > best && acf((L + 1) / 2) <= -0.15) {
            best = c;
            lag = L;
        }
    }
    return (float)best;
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
    if (bird && wobble) {   // птица летит неровно: покачивание по высоте и в сторону
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
    // тип дрона по долям из настроек; у каждого свой размах и скорость
    if (b.bird) {
        b.shape = Shape::Bird;
    } else {
        const double tot = std::max(1e-6f, c_.quadShare + c_.deltaShare + c_.planeShare), r = U(rng_) * tot;
        b.shape = r < c_.quadShare ? Shape::Quad : r < c_.quadShare + c_.deltaShare ? Shape::Delta : Shape::Plane;
    }
    double sp = c_.speedMin + U(rng_) * (c_.speedMax - c_.speedMin);
    switch (b.shape) {
    case Shape::Bird: sp = 8 + U(rng_) * 6; b.size = (float)(0.3 + U(rng_) * 0.9); break;   // от воробья до вороны
    case Shape::Quad: b.size = c_.droneSize * (float)(0.6 + U(rng_)); break;
    case Shape::Delta: sp = 25 + U(rng_) * 20; b.size = (float)(1.8 + U(rng_) * 1.7); break;
    case Shape::Plane: sp = 15 + U(rng_) * 15; b.size = (float)(1.5 + U(rng_) * 2.5); break;
    }
    const double hd = U(rng_) * 2 * PI;
    b.v = {sp * std::cos(hd), sp * std::sin(hd), (U(rng_) * 2 - 1) * (b.bird ? 0.5 : 1.5)};
    b.flapHz = b.bird ? (float)(4 + U(rng_) * 4) : 0;
    b.phase = (float)(U(rng_) * 6.28);
    (void)first;
}

void Emulator::drawBodies(const CameraModel& m, const Pose& p, double t, uint8_t* px, int w, int h) {
    std::vector<Tri> tris;
    for (const Body& b : bodies_) {
        const Vec3 q = b.at(t);
        const double R = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
        double u, v;
        if (!project(m, p, q, u, v) || !in_frame(m, u, v, 20)) continue;
        double c = (b.bird ? -55.0 : -70.0) * std::exp(-R / c_.visibility);
        double flap = 0, pitch = 0, bank = 0;
        const double sp = std::sqrt(b.v.x * b.v.x + b.v.y * b.v.y);
        if (b.bird) {   // взмахи: крылья ходят вверх-вниз, силуэт меняется
            flap = 0.7 * std::sin(2 * PI * b.flapHz * (t - b.t0) + b.phase);
            bank = 0.15 * std::sin(0.7 * t + b.phase);
        } else if (b.shape == Shape::Quad) {
            pitch = -std::min(0.45, 0.02 * sp);   // квадрокоптер летит с наклоном носом вниз
            bank = 0.05 * std::sin(1.3 * t + b.phase);
        } else {
            bank = 0.12 * std::sin(0.4 * t + b.phase);
        }
        build_silhouette(b.shape, b.size, flap, tris);
        draw_silhouette(tris, q, body_frame(b.v, pitch, bank), c, c_.psf,
                        [&](const Vec3& d, double& uu, double& vv) { return project(m, p, d, uu, vv); }, px, w, h);
    }
}

// Кадр из 3D-сцены: Godot рисует камеру при повороте p на момент t; шум датчика добавляем здесь,
// цели (правда) — из ответа сцены.
bool Emulator::sceneRender(double t, int cam, const Pose& p, int W, int H, std::vector<uint8_t>& out) {
    int w = 0, h = 0;
    std::string err;
    if (envReq_ != envSeen_) {   // погоду поменяли: со следующим кадром, и при перезапуске Godot — она же
        std::lock_guard<std::mutex> lk(envM_);
        envSeen_ = envReq_;
        link_->setEnv(envJson_);
        linkHello_ = linkHelloBase_ + ", \"env\": " + envJson_ + "}";
        event(t, "3D-сцена: погода " + envJson_);
    }
    bool ok = link_->render(t, cam, p.az, p.el, out, w, h, sbodies_, err);
    // Godot закрыли или он упал: поднимаем заново (до 3 раз), мир тот же по seed, кадр просим повторно
    while (!ok && !stop_ && linkRestarts_ < 3) {
        ++linkRestarts_;
        fprintf(stderr, "сцена: %s, перезапуск %d\n", err.c_str(), linkRestarts_);
        event(t, "3D-сцена: " + err + ", перезапуск");
        {
            std::lock_guard<std::mutex> lk(vm_);
            view_.status = "перезапуск 3D-сцены...";
        }
        err.clear();
        ok = link_->start(linkOpt_, linkHello_, err) && link_->render(t, cam, p.az, p.el, out, w, h, sbodies_, err);
        std::lock_guard<std::mutex> lk(vm_);
        view_.status.clear();
    }
    if (!ok || w != W || h != H) {
        if (err.empty()) err = "сцена прислала кадр не того размера";
        std::lock_guard<std::mutex> lk(vm_);
        view_.status = "3D-сцена: " + err;
        stop_ = true;
        out.assign((size_t)W * H, 0);
        return false;
    }
    if (c_.noise > 0) {
        const uint32_t seed = noiseSeed_ += 0x7f4a7c15u;
        parallel_rows(H, [&](int ya, int yb) {
            for (int y = ya; y < yb; ++y) {
                uint32_t r = (seed ^ (uint32_t)(y * 2654435761u) ^ 0x51ed27u) | 1u;
                uint8_t* o = &out[(size_t)y * W];
                for (int x = 0; x < W; ++x) {
                    r ^= r << 13;
                    r ^= r >> 17;
                    r ^= r << 5;
                    const int v = o[x] + ((noiseTab_[r >> 16] + 4) >> 3);
                    o[x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
                }
            }
        });
    }
    bodies_.clear();
    for (const scene::Body& sb : sbodies_) {
        Body b;
        b.id = sb.id;
        b.bird = sb.bird;
        b.shape = (Shape)std::max(0, std::min(3, sb.shape));
        b.size = sb.size;
        b.p0 = {sb.x, sb.y, sb.z};
        b.v = {sb.vx, sb.vy, sb.vz};
        b.t0 = t;
        b.wobble = false;
        b.hidden = sb.hidden;
        bodies_.push_back(b);
    }
    return true;
}

void Emulator::renderWide(double t, std::vector<uint8_t>& out) {
    const int W = wideCam_.w, H = wideCam_.h;
    const size_t N = (size_t)W * H;
    out.resize(N);
    if (sceneMode_) {
        sceneRender(t, 0, widePose_, W, H, out);
        return;
    }
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
    if (sceneMode_) {
        sceneRender(t, 1, p, W, H, out);
        return;
    }
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
        if (b.hidden) continue;
        // крупная цель — не точка: попадание в её силуэт (крыло, хвост) — тоже она
        const Vec3 q = b.at(t);
        const double R = std::max(1.0, std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z));
        const double a = angle_between(d, q) - 0.5 * b.size / R * 180 / PI;
        if (a < bd) {
            bd = a;
            best = b.id;
        }
    }
    return best;
}

void Emulator::start(const Config& cfg) {
    stop();
    // кадр меньше шага облаков, частота 0 или угол 0 дают деление на ноль и NaN в повороте
    Config c = cfg;
    c.wideW = std::max(c.wideW, 64), c.wideH = std::max(c.wideH, 64);
    c.narrowW = std::max(c.narrowW, 64), c.narrowH = std::max(c.narrowH, 64);
    c.wideFps = std::max(c.wideFps, 1.f), c.narrowFps = std::max(c.narrowFps, 1.f);
    c.wideFov = std::min(std::max(c.wideFov, 0.5f), 179.f), c.narrowFov = std::min(std::max(c.narrowFov, 0.5f), 179.f);
    {
        std::lock_guard<std::mutex> lk(vm_);
        view_ = View();
    }
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
    sceneMode_ = !c.scene.empty();
    poseSeen_ = poseReq_;
    scanReq_ = false;
    scan_ = false;
    pano_.clear();   // панорама — своя у каждой сцены
    panoTw_ = panoTh_ = panoTile_ = 0;
    sincePose_ = 0;
    lastLearning_ = true;
    if (!sceneMode_) prepareWide();
    ngm_.reset();
    noiseTab_.resize(65536);
    {
        std::mt19937 g(c.seed + 99);
        std::normal_distribution<float> N(0, c.noise * 8);
        for (auto& v : noiseTab_) v = (int16_t)std::max(-32000.f, std::min(32000.f, N(g)));
    }
    bodies_.clear();
    if (!sceneMode_)
        for (int i = 0; i < c.drones + c.birds; ++i) {
            Body b;
            b.bird = i >= c.drones;
            spawn(b, 0, true);
            bodies_.push_back(b);
        }
    wideNoise_.clear();
    // точная
    DetectorConfig dc;
    dc.scales = c.narrowScales;   // цель в узком поле крупнее: несколько масштабов
    dc.auto_thr = c.narrowAutoThr;
    dc.thr_min = c.narrowThrMin;
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
    cool_.clear();
    veto_.clear();
    wideBirds_.clear();
    nConf_ = nOnly_ = manualPick_ = afterBird_ = false;
    detachId_ = 0;
    tDetach_ = -10;
    tBird_ = -1;
    lastU_ = lastV_ = lastSz_ = 0;
    pipe_.setCue({});
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
    vi.w = c.wideW;
    vi.h = c.wideH;
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
    event(0, "старт: целей " + std::to_string(c.drones) + " дронов и " + std::to_string(c.birds) + " птиц" +
                 (sceneMode_ ? ", 3D-сцена " + c.scene : std::string()));
    running_ = true;
    th_ = std::thread([this] { loop(); });
}

// фон широкой без облаков — при запуске и после поворота широкой
void Emulator::prepareWide() {
    const int W = wideCam_.w, H = wideCam_.h;
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
    wideSky_.clear();
    wideSkyT_ = -1;
}

// маска широкой (по тайлам) -> ячейки панорамы, видимые сейчас
void Emulator::panoLearn(const Result& r) {
    if ((int)pano_.size() != PANO_W * PANO_H) pano_.assign((size_t)PANO_W * PANO_H, 0);
    const int tw = r.tilesX, th = r.tilesY, tp = r.tilePx;
    if (tw <= 0 || th <= 0 || tp <= 0 || r.mask.size() != (size_t)tw * th) return;
    const double half = c_.wideFov * 0.5 + 5;
    const int n = (int)std::ceil(half / PANO_STEP);
    const int i0 = (int)std::floor(widePose_.az / PANO_STEP);
    for (int j = 0; j < PANO_H; ++j) {
        const double el = PANO_EL0 + (j + 0.5) * PANO_STEP;
        for (int di = -n; di <= n; ++di) {
            const int i = ((i0 + di) % PANO_W + PANO_W) % PANO_W;
            double u, v;
            if (!project(wideCam_, widePose_, dir_from_azel((i + 0.5) * PANO_STEP, el), u, v)) continue;
            const int tx = (int)(u / tp), ty = (int)(v / tp);
            if (u < 0 || v < 0 || tx >= tw || ty >= th) continue;
            pano_[(size_t)j * PANO_W + i] = r.mask[(size_t)ty * tw + tx] ? 1 : 2;
        }
    }
}

// маска широкой для текущего поворота из панорамы; возвращает долю тайлов, которые панорама покрывает
float Emulator::panoMaskFor(std::vector<uint8_t>& m) const {
    m.assign((size_t)panoTw_ * panoTh_, 1);
    if ((int)pano_.size() != PANO_W * PANO_H) return 0.f;
    int known = 0;
    for (int ty = 0; ty < panoTh_; ++ty)
        for (int tx = 0; tx < panoTw_; ++tx) {
            double az, el;
            azel_from_dir(unproject(wideCam_, widePose_, (tx + 0.5) * panoTile_, (ty + 0.5) * panoTile_), az, el);
            const int j = (int)std::floor((el - PANO_EL0) / PANO_STEP);
            uint8_t v = 0;
            if (j >= PANO_H) v = 1;        // выше панорамы — небо
            else if (j < 0) v = 2;         // ниже — земля
            else v = pano_[(size_t)j * PANO_W + ((int)std::floor(az / PANO_STEP) % PANO_W + PANO_W) % PANO_W];
            if (v) ++known;
            m[(size_t)ty * panoTw_ + tx] = v != 2;
        }
    return (float)known / std::max(1, panoTw_ * panoTh_);
}

void Emulator::stop() {
    stop_ = true;
    wcv_.notify_all();
    if (th_.joinable()) th_.join();
    pipe_.close();
    if (link_) link_->stop();
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
        // новый трек широкой там, где точная отсеяла птицу: сказать широкой, что это птица
        if (c_.vetoToWide && w.cls != 2 && std::find(wideBirds_.begin(), wideBirds_.end(), w.id) == wideBirds_.end() &&
            vetoed(t, w.rep)) {
            pipe_.markBird(w.id);
            wideBirds_.push_back(w.id);
        }
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
    tAssign_ = tState_ = tWideSeen_ = t;
    tInView_ = -1;
    tSearch_ = -1;
    gotThis_ = false;
    manualPick_ = !strcmp(why, "выбрана вручную");
    nConf_ = nOnly_ = false;
    tBird_ = -1;
    shapeE_.clear();
    shapeId_ = shapeBirdN_ = 0;
    shapeRigid_ = false;
    ntrk_->setFollowing(false);
    nextPick_ = false;
    if (!wideId) return;
    afterBird_ = false;
    ++st_.assignments;
    double slew = 0;
    if (w) slew = ang(mount_.az, mount_.el, w->rep.az, w->rep.el);
    char b[160];
    snprintf(b, sizeof(b), "цель %d (%s): %s, поворот %.1f°", wideId, w && w->cls == 1 ? "дрон" : w && w->cls == 2 ? "птица" : "?",
             why, slew);
    event(t, b);
}

// цель широкой там, где точная видела птицу (по прогнозу отсеянной): новый трек широкой на той же
// птице не назначается заново
bool Emulator::vetoed(double t, const TargetReport& r) const {
    for (const auto& v : veto_) {
        double a1, e1, s1, a2, e2, s2;
        predict(v.first, t, a1, e1, s1);
        predict(r, t, a2, e2, s2);
        if (ang(a1, e1, a2, e2) < std::max(0.5, 0.05 * (t - v.first.t)) + 3 * s2) return true;
    }
    return false;
}

// Цель есть только у точной: широкая снова видит её — свой трек широкой у неё теперь этот
void Emulator::rebind(double t) {
    if (!nOnly_ || !haveNrep_ || (state_ != NState::Track && t - tLastSeen_ > c_.narrowLostHold)) return;
    double naz, nel, nsg;
    predict(nrep_, t, naz, nel, nsg);
    const double half = 0.5 * nrep_.size_deg;
    int best = 0;
    double bd = 1e9;
    for (const auto& w : targets_) {
        if (c_.dronesOnly && w.cls == 2) continue;
        if (w.id == detachId_ && t - tDetach_ < 1.5) continue;   // только что ушёл с цели: не мечемся
        double az, el, sg;
        predict(w.rep, t, az, el, sg);
        const double a = ang(az, el, naz, nel);
        // строже, чем расхождение, по которому точная от трека отказывается (0.5° / 6 СКО): без запаса
        // трек на границе то отдаётся, то берётся каждый кадр
        if (a < std::max(0.25, 3 * sg) + half && a < bd) {
            bd = a;
            best = w.id;
        }
    }
    if (!best) return;
    const int was = assigned_;
    assigned_ = best;
    nOnly_ = false;
    ++st_.rebinds;
    tWideSeen_ = t;
    for (const auto& w : targets_)
        if (w.id == best && w.truth) assignedBody_ = w.truth;
    char b[160];
    snprintf(b, sizeof(b), "широкая снова видит цель %d: теперь это её трек %d", was, best);
    event(t, b);
}

void Emulator::choose(double t) {
    cool_.erase(std::remove_if(cool_.begin(), cool_.end(), [&](const std::pair<int, double>& c) { return c.second <= t; }),
                cool_.end());
    veto_.erase(std::remove_if(veto_.begin(), veto_.end(),
                               [&](const std::pair<TargetReport, double>& c) { return c.second <= t; }),
                veto_.end());
    auto ok = [&](const WideTarget& w) {
        if (c_.dronesOnly && w.cls == 2) return false;
        for (const auto& c : cool_)
            if (c.first == w.id) return false;
        return !vetoed(t, w.rep);
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
    // назначенную широкая потеряла: точная, уверенная в дроне, держит её сама (и ищет по своему
    // прогнозу); иначе — если точная её ведёт, ведём дальше, а нет — отпускаем
    // (трек широкой выпадает из её сообщения на кадр-другой, когда цель слабая: это ещё не потеря)
    if (assigned_ && find(assigned_)) tWideSeen_ = t;
    if (assigned_ && !nOnly_ && !find(assigned_) && t - tWideSeen_ > 0.5 && c_.narrowHold && nConf_ &&
        (state_ == NState::Track || t - tLastSeen_ < c_.narrowLostHold)) {
        nOnly_ = true;
        ++st_.kept;
        event(t, "широкая потеряла цель " + std::to_string(assigned_) + ", точная видит дрон и держит его сама");
    }
    if (assigned_ && !nOnly_ && !find(assigned_) && state_ != NState::Track && t - tState_ > 1.5) {
        event(t, "цель " + std::to_string(assigned_) + " пропала у широкой");
        const bool next = nextPick_;
        assign(t, 0, "");
        afterBird_ = next;   // взяли её сами вместо отсеянной птицы — ищем дальше
    }
    if (mode == SwitchMode::Manual) {
        // цель взяли сами после птицы, точная её ещё не подтвердила, а широкая жёстко захватила другую — к ней
        if (assigned_ && nextPick_ && !nConf_) {
            const WideTarget* cur = find(assigned_);
            if (!(cur && cur->hard))
                for (const auto& w : targets_)
                    if (ok(w) && w.hard && w.id != assigned_) {
                        assign(t, w.id, "жёсткий захват широкой");
                        break;
                    }
        }
        if (!assigned_ && !targets_.empty())
            for (const auto& w : targets_)
                if (ok(w) && w.hard) {   // вручную: сами берём только подтверждённый дрон
                    assign(t, w.id, "жёсткий захват широкой");
                    break;
                }
        // ...а отсеяв птицу — следующую цель широкой: жёсткий захват, потом «дрон», потом прочие (птиц широкой —
        // нет, если «только дроны»), среди равных — ближайшую по повороту; если целей нет — ждём, пока появятся
        if (!assigned_ && afterBird_ && c_.nextOnBird) {
            int best = 0;
            double bestP = 1e18;
            for (const auto& w : targets_) {
                if (!ok(w)) continue;
                const double p = (w.hard ? 0 : w.cls == 1 ? 1 : w.cls == 2 ? 3 : 2) * 1000 + ang(mount_.az, mount_.el, w.rep.az, w.rep.el);
                if (p < bestP) {
                    bestP = p;
                    best = w.id;
                }
            }
            if (best) {
                assign(t, best, "следующая после птицы");
                nextPick_ = true;
                ++st_.nextAfterBird;
            }
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
    // цель только у точной: трек широкой с этим номером (если ещё жив) — уже не она
    const WideTarget* wt = nullptr;
    if (!nOnly_)
        for (const auto& x : targets_)
            if (x.id == assigned_) wt = &x;
    // 1. куда смотреть: по точной, пока она ведёт цель (или недавно видела), иначе по широкой
    const bool useN = haveNrep_ && (state_ == NState::Track || t - tLastSeen_ < (nOnly_ ? c_.narrowLostHold : 0.6));
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
    // силуэт захваченной цели: крупный дрон (крыло, самолёт) на точной — десятки пикселей, детектор
    // находит на нём несколько точек (концы крыльев, нос, хвост). Без силуэта захват перескакивал
    // между ними, а прицел вместе с поворотом — с одного края цели на другой
    if (c_.narrowSilhouette) {
        float lx, ly, lr;
        int lp;
        if (ntrk_->lockSearch((float)dx, (float)dy, lx, ly, lr, lp)) {
            // только крупной цели: у дальней в 2–4 пикс. силуэт по шуму дёргает трек на пиксель-другой, его
            // скорость сбивается, и поворотка по ней уводит цель из центра (сцена 5: крыло за 1 км потеряно за 0.4 с)
            const Silhouette sh = measure_silhouette(nframe_.data(), narrowCam_.w, narrowCam_.h, narrowCam_.w, lx, ly, lr, lp,
                                                     std::max(6.f, 0.8f * ndet_->threshold()), ntrk_->config().silh_max);
            if (sh.valid && std::max(sh.w, sh.h) >= c_.silhMinPx) ntrk_->setLockSilhouette(sh);
        }
    }
    const std::vector<Track>& tr = ntrk_->update(ncands_, (float)dx, (float)dy);
    const double npd = pixel_deg(narrowCam_, enc, narrowCam_.w * 0.5, narrowCam_.h * 0.5);   // град на пиксель точной
    if (dumpAt_ > 0 && t >= dumpAt_ && t < dumpAt_ + dumpLen_) {   // отладка: кадры точной в PGM и её треки
        char fn[64];
        snprintf(fn, sizeof(fn), "narrow_%07.3f.pgm", t);
        if (FILE* f = fopen(fn, "wb")) {
            fprintf(f, "P5 %d %d 255\n", narrowCam_.w, narrowCam_.h);
            fwrite(nframe_.data(), 1, nframe_.size(), f);
            fclose(f);
        }
        printf("t=%.3f state=%s lock=%d win=(%.0f,%.0f r%.0f vis%d) cands=%zu thr=%.1f noise=%.2f\n", t, state_name(state_), nlock_, win_.u,
               win_.v, win_.radius, win_.visible, ncands_.size(), ndet_->threshold(), ndet_->noiseLevel());
        for (const Track& x : tr)
            printf("  трек %d (%.1f,%.1f) hits %d miss %d clutter %d cls %d\n", x.id, x.x, x.y, x.hits, x.miss, (int)x.clutter,
                   (int)x.cls);
        for (const Body& b : bodies_) {
            double u, v;
            const Vec3 q = b.at(t);
            const double R = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
            if (project(narrowCam_, truth, {q.x / R, q.y / R, q.z / R}, u, v) && in_frame(narrowCam_, u, v))
                printf("  тело %d %s %.1f м в (%.1f,%.1f), %.0f м\n", b.id, shape_name(b.shape), b.size, u, v, R);
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
            // цель только у точной, и та её потеряла: сама заново не берёт — широкая не подтвердит, что
            // найденное — она, а это обычно кромка леса или облако, куда цель ушла (на сценах: 1–2 с на
            // чужом треке). Поворотка идёт по прогнозу, пока широкая не увидит цель снова (rebind)
            if (nOnly_) break;
            if (x.clutter || x.miss > 0 || x.hits < minHits) continue;
            const double d = std::hypot(x.x - win_.u, x.y - win_.v);
            if (c_.windowFocus && winOn_ && d > win_.radius) continue;
            if (wt && c_.wideCheck) {
                double az, el;
                azel_from_dir(unproject(narrowCam_, enc, x.x, x.y), az, el);
                const double a = ang(az, el, waz, wel), tol = std::max(0.4, 5 * wsg);
                if (a > tol) {
                    // крупная цель: широкая указывает её середину, а трек точной — на её части (крыло, хвост)
                    if (!c_.narrowSilhouette || a > tol + 3) continue;
                    const Silhouette sh = measure_silhouette(nframe_.data(), narrowCam_.w, narrowCam_.h, narrowCam_.w, x.x, x.y, 6.f,
                                                             x.polarity, std::max(6.f, 0.8f * ndet_->threshold()), 200.f);
                    if (!sh.valid || std::max(sh.w, sh.h) < c_.silhMinPx) continue;
                    double saz, sel;
                    azel_from_dir(unproject(narrowCam_, enc, sh.x, sh.y), saz, sel);
                    if (ang(saz, sel, waz, wel) > tol + 0.5 * npd * std::max(sh.w, sh.h)) continue;
                }
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
                    lastU_ = x.x;
                    lastV_ = x.y;
                    lastSz_ = 0;
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
                tGot_ = t;
                ++st_.acquired;
                st_.acqSum += t - tAssign_;
                ++st_.acqN;
            }
        } else if (winOn_ && !win_.visible && t - tState_ > 0.5) {
            state_ = NState::Slew;
            tState_ = t;
        } else if (state_ == NState::Lost && t - tLastSeen_ > (nOnly_ ? c_.narrowLostHold : 3) && !wt) {
            event(t, "точная потеряла цель " + std::to_string(assigned_) + ", широкая её не видит");
            assign(t, 0, "");
        }
    } else if (state_ == NState::Track) {
        const Track* me = nullptr;
        for (const Track& x : tr)
            if (x.id == nlock_) me = &x;
        // захват перешёл на другой трек той же цели (трекер передал его наследнику, или трек части крупной
        // цели пропал, а рядом, внутри её силуэта, — другой): ведём его, а не теряем цель
        const int keepMiss = std::max(2, (int)(0.15 * c_.narrowFps));
        if (c_.narrowSilhouette && (!me || me->miss > 0)) {
            const float px = me ? me->x : lastU_, py = me ? me->y : lastV_;
            const float reach = std::max(10.f, 0.6f * lastSz_ + 6.f);
            const Track* alt = nullptr;
            float bd = reach;
            for (const Track& x : tr) {
                if (x.id == nlock_ || x.clutter || x.miss > 0 || x.cls == TargetClass::Bird) continue;
                const float d = std::hypot(x.x - px, x.y - py);
                if (d < bd) {
                    bd = d;
                    alt = &x;
                }
            }
            if (alt && (!me || me->miss > keepMiss / 2)) {
                ++st_.partJumps;
                nlock_ = alt->id;
                ntrk_->selectLock(alt->id);
                me = alt;
            }
        }
        if (me && me->miss <= keepMiss) {
            const float szPx = std::max(me->bw, me->bh);
            lastU_ = me->x + me->vx;
            lastV_ = me->y + me->vy;
            lastSz_ = szPx;
            nrep_ = make_report(narrowCam_, enc, t, c_.narrowFps, me->x, me->y, me->vsx, me->vsy, me->sigma, szPx);
            nrep_.size_deg = (float)(szPx * npd);
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
            bool whoBird = false;
            for (const Body& b : bodies_)
                if (b.id == who) {
                    whoBird = b.bird;
                    (b.bird ? st_.trackBird : st_.trackDrone) += dt;
                }
            if (nOnly_) {
                st_.narrowOnly += dt;
                if (who && who == assignedBody_) st_.narrowOnlyCorrect += dt;
            }
            // птица по силуэту: «масса» силуэта (контраст x площадь) по кадрам, взмахи — её колебания
            // с периодом 1/12–1/3 с. Классификатор трекера смотрит то же по отклику детектора и копит
            // дольше; по силуэту крупной на точной цели решение — за полсекунды после захвата
            bool shapeBird = false;
            if (c_.narrowShape) {
                if (shapeId_ != me->id) {   // другой трек — та же цель, но ряд начинаем заново
                    shapeE_.clear();
                    shapeId_ = me->id;
                    shapeBirdN_ = 0;
                    shapeRigid_ = false;
                }
                if (!me->miss) {
                    const float r = std::max(4.f, 0.6f * szPx + 3.f);
                    const Silhouette sh = measure_silhouette(nframe_.data(), narrowCam_.w, narrowCam_.h, narrowCam_.w, me->x, me->y, r,
                                                             me->polarity, std::max(6.f, 0.8f * ndet_->threshold()), 120.f);
                    // мельче 6 пикс. (дальняя цель без зума) форма — это шум и размытие, взмахов там не видно
                    if (sh.valid && sh.area >= 6) shapeE_.push_back(sh.contrast * (float)sh.area);
                    if (dumpAt_ > 0 && t >= dumpAt_ && t < dumpAt_ + dumpLen_)
                        printf("  силуэт %d: %.1fx%.1f площадь %d контраст %.1f\n", (int)sh.valid, sh.w, sh.h, sh.area, sh.contrast);
                    const int win = std::max(8, (int)std::lround(c_.shapeWin * c_.narrowFps));
                    while ((int)shapeE_.size() > win) shapeE_.pop_front();
                    if ((int)shapeE_.size() >= win) {
                        float amp;
                        int lag;
                        const int lmin = std::max(2, (int)(c_.narrowFps / 12)), lmax = (int)std::ceil(c_.narrowFps / 3);
                        const float R = flap_shape(shapeE_, win, lmin, lmax, amp, lag);
                        const bool flap = amp >= c_.shapeAmp && R >= c_.shapeR;
                        shapeBirdN_ = flap ? shapeBirdN_ + 1 : 0;
                        if (flap) shapeHz_ = c_.narrowFps / std::max(1, lag);
                        shapeRigid_ = !flap && amp < 0.5f * c_.shapeAmp;
                        shapeBird = shapeBirdN_ >= (nConf_ ? 2 : 1) * c_.shapeHits;
                        if (dumpAt_ > 0 && t >= dumpAt_ && t < dumpAt_ + dumpLen_)
                            printf("  силуэт: размах %.3f, автокорр. %.2f на %d кадрах, подряд %d\n", amp, R, lag, shapeBirdN_);
                    }
                }
            }
            // точная сама решает, дрон ли это: у неё цель крупнее и кадров вчетверо больше (взмахи видны).
            // Пока силуэт машет крыльями — не дрон, что бы ни говорил классификатор
            // Силуэт крупный (ряд копится) — сначала досмотреть окно взмахов: птицу в дроны не записывать
            const bool shapeWait = c_.narrowShape && !shapeE_.empty() &&
                                   (int)shapeE_.size() < std::max(8, (int)std::lround(c_.shapeWin * c_.narrowFps));
            const bool droneNow = (ntrk_->lockState() == LockState::Hard || me->cls == TargetClass::Drone) && !shapeBird &&
                                  shapeBirdN_ == 0 && !shapeWait;
            if (!nConf_ && droneNow && t - tTrackStart_ >= c_.confirmT) {
                nConf_ = true;
                ++st_.confirmed;
                if (whoBird) ++st_.confirmedBird;
                event(t, "точная: цель " + std::to_string(assigned_) + " — дрон");
            }
            if (me->cls == TargetClass::Bird) {
                if (tBird_ < 0) tBird_ = t;
            } else {
                tBird_ = -1;
            }
            // цель выбрал оператор: отсеиваем, только если точная уверена вдвое (вдвое больше оценок силуэта или
            // вдвое дольше птица у классификатора)
            const bool birdSure = manualPick_ ? c_.vetoManual && (shapeBirdN_ >= 2 * c_.shapeHits || (tBird_ >= 0 && t - tBird_ >= 2 * c_.vetoT))
                                              : shapeBird || (tBird_ >= 0 && t - tBird_ >= c_.vetoT);
            if (c_.narrowVeto && birdSure) {
                bool bird = false;
                for (const Body& b : bodies_)
                    if (b.id == who) bird = b.bird;
                ++(bird ? st_.vetoBird : who ? st_.vetoDrone : st_.vetoNone);
                if (bird) st_.vetoDelay += t - tGot_;
                if (shapeBird) ++st_.vetoShape;
                veto_.push_back({nrep_, t + c_.vetoCool});
                if (c_.vetoToWide && !nOnly_ && assigned_) {
                    pipe_.markBird(assigned_);
                    wideBirds_.push_back(assigned_);
                }
                if (!nOnly_) cool_.push_back({assigned_, t + c_.vetoCool});
                char why[64];
                if (shapeBird) snprintf(why, sizeof(why), "силуэт машет крыльями, %.0f Гц", shapeHz_);
                else snprintf(why, sizeof(why), "машет крыльями");
                event(t, "точная: цель " + std::to_string(assigned_) + " — птица (" + why + "), отсеяна" +
                             (bird || !who ? "" : " — НА САМОМ ДЕЛЕ ДРОН"));
                assign(t, 0, "");
                afterBird_ = true;
                goto ui;
            }
            // широкая видит назначенную цель далеко от того, что ведёт точная: не та цель.
            // Сравниваем с прогнозом широкой на этот момент: её сообщение старше на задержку и кадр
            bool off = false;
            if (wt) {
                double waz = wt->rep.az, wel = wt->rep.el, wsg = wt->rep.sigma;
                if (c_.wideCheck) predict(wt->rep, t, waz, wel, wsg);
                // крупная цель: широкая указывает её середину, а захват точной может быть у края силуэта
                const double tol = (c_.wideCheck ? std::max(0.5, 6.0 * wsg) : std::max(1.5, 6.0 * wsg)) +
                                   (c_.narrowSilhouette ? 0.5 * nrep_.size_deg : 0.0);
                off = ang(waz, wel, nrep_.az, nrep_.el) > tol && t - tTrackStart_ > (c_.wideCheck ? 0.3 : 1.0);
            }
            if (off && c_.narrowHold && nConf_) {
                // широкая ушла с цели (перескок трека на соседа, помеху), а точная видит дрон: держит его
                // сама, а трек широкой отпускает
                nOnly_ = true;
                ++st_.kept;
                detachId_ = assigned_;
                tDetach_ = t;
                event(t, "широкая ушла с цели " + std::to_string(assigned_) + ", точная видит дрон и держит его сама");
            } else if (off) {
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
    // (и в ручном, если цель точная взяла сама после птицы)
    if (assigned_ && tSearch_ >= 0 && t - tSearch_ > c_.giveUp && ((SwitchMode)switchMode_.load() != SwitchMode::Manual || nextPick_)) {
        ++st_.unconfirmed;
        cool_.push_back({assigned_, t + c_.coolDown});
        event(t, "точная не подтверждает цель " + std::to_string(assigned_) + ": за " +
                     std::to_string((int)c_.giveUp) + " с не удержала, к ней — не раньше чем через " +
                     std::to_string((int)c_.coolDown) + " с");
        const bool next = nextPick_;
        assign(t, 0, "");
        afterBird_ = next;
    }
ui:
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
    for (auto& w : view_.targets) {
        if (w.id == assigned_ && !nOnly_ && nConf_) w.verdict = 1;
        if (vetoed(t, w.rep)) w.verdict = 2;
    }
    view_.nConf = nConf_;
    view_.nOnly = nOnly_;
    view_.nWu = view_.nWv = -1;
    if (haveNrep_ && assigned_) {
        double u, v;
        if (project(wideCam_, widePose_, dir_from_azel(nrep_.az, nrep_.el), u, v)) {
            view_.nWu = (float)u;
            view_.nWv = (float)v;
        }
    }
    view_.vetoW.clear();
    for (const auto& vt : veto_) {
        double az, el, sg, u, v;
        predict(vt.first, t, az, el, sg);
        if (project(wideCam_, widePose_, dir_from_azel(az, el), u, v)) view_.vetoW.push_back({(float)u, (float)v});
    }
    view_.axis = truth;
    view_.cmd = cmdPose_;
    view_.rateAz = (float)mount_.vaz;
    view_.rateEl = (float)mount_.vel;
    view_.win = win_;
    view_.winOn = winOn_;
    view_.widePose = widePose_;
    view_.truth.clear();
    for (const Body& b : bodies_) {
        if (b.hidden) continue;
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
    {
        // окну хватает хвоста ошибок (медиана по 3000): полные векторы за час — мегабайты на каждый кадр точной
        auto tail = [](const std::vector<float>& v, std::vector<float>& o) {
            o.assign(v.end() - std::min<size_t>(v.size(), 3000), v.end());
        };
        std::vector<float> l = std::move(st_.errLock), a = std::move(st_.errAxis), w = std::move(st_.errWide);
        view_.st = st_;   // без векторов ошибок
        st_.errLock = std::move(l), st_.errAxis = std::move(a), st_.errWide = std::move(w);
        tail(st_.errLock, view_.st.errLock), tail(st_.errAxis, view_.st.errAxis), tail(st_.errWide, view_.st.errWide);
    }
    view_.st.simT = t + dt;   // шаг [t, t+dt] уже учтён в долях времени
}

void Emulator::loop() {
    if (sceneMode_) {   // Godot — отдельный процесс на своих ядрах; сцена строится несколько секунд
        {
            std::lock_guard<std::mutex> lk(vm_);
            view_.status = "запуск 3D-сцены...";
        }
        if (!link_) link_.reset(new scene::Link());
        link_->setCancel(&stop_);
        scene::Options o;
        o.godot = c_.godot;
        o.project = c_.simProject;
        o.scene = c_.scene;
        o.sector = c_.sector;
        o.affinity = c_.renderAffinity;
        char hello[700];
        snprintf(hello, sizeof(hello),
                 "{\"wide\": [%d, %d, %.3f, %.3f], \"fine\": [%d, %d, %.3f, %.3f], \"seed\": %u, \"drones\": %d, "
                 "\"birds\": %d, \"range\": [%.1f, %.1f], \"types\": {\"quad\": %.3f, \"delta\": %.3f, \"plane\": %.3f}, "
                 "\"az\": %.3f, \"el\": %.3f, \"sector\": %s",
                 c_.wideW, c_.wideH, c_.wideFov, c_.wideFps, c_.narrowW, c_.narrowH, c_.narrowFov, c_.narrowFps, c_.seed,
                 c_.drones, c_.birds, c_.rangeMin, c_.rangeMax, c_.quadShare, c_.deltaShare, c_.planeShare, widePose_.az,
                 widePose_.el, c_.sector.empty() ? "false" : "true");
        std::string err;
        linkOpt_ = o;
        linkHelloBase_ = hello;
        {
            std::lock_guard<std::mutex> lk(envM_);
            envJson_ = scene_env_json(c_);
            envSeen_ = envReq_;
            linkHello_ = linkHelloBase_ + ", \"env\": " + envJson_ + "}";
        }
        linkRestarts_ = 0;
        if (!link_->start(o, linkHello_, err)) {
            std::lock_guard<std::mutex> lk(vm_);
            view_.status = "3D-сцена: " + err;
            view_.running = false;
            running_ = false;
            return;
        }
        std::lock_guard<std::mutex> lk(vm_);
        view_.status.clear();
    }
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
            if (scanReq_.exchange(false) && !scan_) {
                scan_ = true;
                scanStep_ = 0;
                scanAz0_ = widePose_.az;
                scanEl0_ = widePose_.el;
                event(t, "обзор 360°: маска неба по кругу");
            }
            if (scan_ && (!lastLearning_ && sincePose_ >= 3)) {   // этот вид выучен — дальше
                const float step = std::max(20.f, c_.wideFov * 0.5f);
                const int n = (int)std::ceil(360.f / step - 1e-3f);
                if (scanStep_ + 1 >= n) {
                    scan_ = false;
                    wantAz_ = scanAz0_;
                    wantEl_ = scanEl0_;
                    event(t, "обзор 360° закончен");
                } else {
                    ++scanStep_;
                    wantAz_ = std::fmod(scanAz0_ + scanStep_ * step, 360.f);
                    wantEl_ = scanEl0_;
                }
                ++poseReq_;
            }
            if (poseReq_ != poseSeen_) {   // широкую повернули кнопками
                poseSeen_ = poseReq_;
                widePose_ = {wantAz_, std::max(-10.f, std::min(80.f, (float)wantEl_)), 0};
                if (!sceneMode_) prepareWide();
                sincePose_ = 0;
                lastLearning_ = true;
                char b[160];
                // маска неба для нового вида: из панорамы, если она его покрывает, иначе — учить заново
                std::vector<uint8_t> m;
                const float known = c_.panoMask && !scan_ && panoTw_ > 0 ? panoMaskFor(m) : 0.f;
                if (known >= 0.85f) {
                    pipe_.setExternalMask(std::move(m), panoTw_, panoTh_, (int)std::lround(c_.wideFps));
                    snprintf(b, sizeof(b), "широкая повёрнута: азимут %.0f°, угол места %.0f°; маска из панорамы (%.0f%% вида)",
                             widePose_.az, widePose_.el, known * 100);
                } else {
                    pipe_.relearnMask();
                    snprintf(b, sizeof(b), "широкая повёрнута: азимут %.0f°, угол места %.0f°; маска учится%s", widePose_.az,
                             widePose_.el, c_.panoMask && !scan_ ? " (панорама покрывает мало)" : "");
                }
                {
                    std::lock_guard<std::mutex> lk(vm_);
                    view_.panoKnown = known;
                }
                event(t, b);
            }
            // тела, вылетевшие из поля зрения широкой или ушедшие далеко, — заново (в 3D-сцене их ведёт сцена)
            for (Body& b : bodies_) {
                if (sceneMode_) break;
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
            // точная уверена в дроне и видит его: широкой — где он будет на этом кадре (там её порог ниже,
            // и потерянную цель она находит снова, а свою слабую не теряет)
            {
                std::vector<FocusArea> cue;
                if (c_.wideCue && nConf_ && haveNrep_ && assigned_ && t - tLastSeen_ < 0.5) {
                    double az, el, sg, u, v;
                    predict(nrep_, t, az, el, sg);
                    if (project(wideCam_, widePose_, dir_from_azel(az, el), u, v) && in_frame(wideCam_, u, v)) {
                        const double wpd = pixel_deg(wideCam_, widePose_, u, v);
                        cue.push_back({(float)u, (float)v, (float)std::max(12.0, std::min(60.0, (3 * sg + nrep_.size_deg) / wpd))});
                    }
                }
                pipe_.setCue(cue);
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
                // панорамная маска: маска этого вида — в панораму (после обучения и раз в полсекунды)
                const bool learned = lastLearning_ && !res->learning;
                lastLearning_ = res->learning;
                ++sincePose_;
                if (res->tilesX > 0) {
                    panoTw_ = res->tilesX;
                    panoTh_ = res->tilesY;
                    panoTile_ = res->tilePx;
                }
                if (c_.panoMask && !res->learning && !res->mask.empty() &&
                    (learned || st_.wideFrames % std::max(1, (int)(c_.wideFps / 2)) == 0)) {
                    panoLearn(*res);
                    std::lock_guard<std::mutex> lk(vm_);
                    view_.pano = pano_;
                }
                {
                    std::lock_guard<std::mutex> lk(vm_);
                    view_.scanning = scan_;
                }
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
                if (nOnly_) {
                    break;   // трек широкой с этим номером уже не наша цель: её держит точная
                } else if (w.id == assigned_ && w.truth && assignedBody_ && w.truth != assignedBody_ && c_.narrowHold && nConf_ &&
                           state_ == NState::Track) {
                    // точная ведёт и уверена: назначенная цель — та, что у неё; уйдёт ли широкая с неё,
                    // точная решит сама (см. stepNarrow), а оценка — по прежнему телу
                    ++st_.wideSwaps;
                    event(t, "широкая: трек " + std::to_string(w.id) + " перескочил на другую цель");
                } else if (w.id == assigned_ && w.truth && assignedBody_ && w.truth != assignedBody_) {
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
                                if (!(c_.narrowHold && nConf_ && state_ == NState::Track)) assignedBody_ = 0;
                            }
                        }
                }
        }
        rebind(t);
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

std::string scene_env_json(const Config& c) {
    std::string j = "{";
    auto num = [&](const char* k, double v, const char* fmt) {
        char b[64];
        snprintf(b, sizeof(b), fmt, v);
        j += (j.size() > 1 ? ", \"" : "\"") + std::string(k) + "\": " + b;
    };
    if (!c.sceneWeather.empty()) j += "\"weather\": \"" + c.sceneWeather + "\"";
    if (c.sceneTime >= 0) {
        num("time", c.sceneTime, "%.4f");
        num("time_rate", c.sceneTimeRate, "%.2f");
        num("day", c.sceneDay, "%.0f");
        num("lat", c.sceneLat, "%.3f");
    }
    if (c.sceneClouds >= 0) num("clouds", c.sceneClouds, "%.3f");
    if (c.sceneVis >= 0) num("visibility", c.sceneVis, "%.0f");
    if (c.sceneRain >= 0) num("rain", c.sceneRain, "%.3f");
    if (c.sceneSnow >= 0) num("snow", c.sceneSnow, "%.3f");
    return j + "}";
}

bool scene_weather_preset(const std::string& name, float& clouds, float& vis, float& rain, float& snow) {
    // как WEATHER в sim/godot/sim.gd
    static const struct {
        const char* n;
        float c, v, r, s;
    } P[] = {{"clear", 0.06f, 30000, 0, 0},  {"cumulus", 0.38f, 20000, 0, 0}, {"cloudy", 0.65f, 14000, 0, 0},
             {"overcast", 0.97f, 9000, 0, 0}, {"haze", 0.12f, 3500, 0, 0},    {"mist", 0.85f, 1200, 0, 0},
             {"fog", 1.0f, 300, 0, 0},        {"rain", 1.0f, 4000, 0.5f, 0},  {"storm", 1.0f, 1800, 1.0f, 0},
             {"snow", 1.0f, 2500, 0, 0.6f}};
    for (const auto& p : P)
        if (name == p.n) {
            clouds = p.c, vis = p.v, rain = p.r, snow = p.s;
            return true;
        }
    return false;
}

float sun_elevation(float hours, int day, float lat) {
    const double D = 3.14159265358979 / 180;
    const double decl = 23.44 * D * std::sin(2 * 3.14159265358979 * (day - 81) / 365.0);
    const double ha = (hours - 12) * 15 * D, la = lat * D;
    return (float)(std::asin(std::sin(la) * std::sin(decl) + std::cos(la) * std::cos(decl) * std::cos(ha)) / D);
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
    Stats s = v.st;   // ошибки — полностью (в окне только хвост), поток уже остановлен
    s.errLock = st_.errLock, s.errAxis = st_.errAxis, s.errWide = st_.errWide;
    return s;
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
        else if (a == "--types") sscanf(val().c_str(), "%f,%f,%f", &c.quadShare, &c.deltaShare, &c.planeShare);
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
        else if (a == "--nauto") c.narrowAutoThr = atoi(val().c_str()) != 0;
        else if (a == "--nthr-min") c.narrowThrMin = (float)atof(val().c_str());
        else if (a == "--dump") sscanf(val().c_str(), "%lf,%lf", &e.dumpAt_, &e.dumpLen_);
        else if (a == "--no-wide-check") c.wideCheck = false;
        else if (a == "--min-hold") c.minHold = (float)atof(val().c_str());
        else if (a == "--give-up") c.giveUp = (float)atof(val().c_str());
        else if (a == "--cool-down") c.coolDown = (float)atof(val().c_str());
        else if (a == "--no-silhouette") c.narrowSilhouette = false;
        else if (a == "--silh-min") c.silhMinPx = (float)atof(val().c_str());
        else if (a == "--no-hold") c.narrowHold = false;
        else if (a == "--no-veto") c.narrowVeto = false;
        else if (a == "--no-shape") c.narrowShape = false;
        else if (a == "--no-veto-wide") c.vetoToWide = false;
        else if (a == "--with-birds") c.dronesOnly = false;   // широкая указывает и тех, кого считает птицами
        else if (a == "--shape") sscanf(val().c_str(), "%f,%f,%d,%f", &c.shapeAmp, &c.shapeR, &c.shapeHits, &c.shapeWin);
        else if (a == "--no-cue") c.wideCue = false;
        else if (a == "--no-next") c.nextOnBird = false;
        else if (a == "--no-veto-manual") c.vetoManual = false;
        else if (a == "--legacy")
            c.narrowSilhouette = c.narrowHold = c.narrowVeto = c.wideCue = c.narrowShape = c.vetoToWide = c.nextOnBird = c.vetoManual = false;
        else if (a == "--scene") c.scene = val();
        else if (a == "--sector") c.sector = val();
        else if (a == "--godot") c.godot = val();
        else if (a == "--weather") c.sceneWeather = val();
        else if (a == "--time") {   // 19.5 или 19:30
            const std::string v = val();
            int hh = 0, mm = 0;
            c.sceneTime = v.find(':') != std::string::npos && sscanf(v.c_str(), "%d:%d", &hh, &mm) == 2 ? hh + mm / 60.f
                                                                                                    : (float)atof(v.c_str());
        } else if (a == "--time-rate") c.sceneTimeRate = (float)atof(val().c_str());
        else if (a == "--date") {   // ММ-ДД
            int m = 9, d = 15;
            static const int cum[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
            if (sscanf(val().c_str(), "%d-%d", &m, &d) == 2) c.sceneDay = cum[std::max(1, std::min(12, m)) - 1] + d;
        } else if (a == "--lat") c.sceneLat = (float)atof(val().c_str());
        else if (a == "--rain") c.sceneRain = (float)atof(val().c_str());
        else if (a == "--snow") c.sceneSnow = (float)atof(val().c_str());
        else if (a == "--scene-clouds") c.sceneClouds = (float)atof(val().c_str());
        else if (a == "--scene-visibility") c.sceneVis = (float)atof(val().c_str());
        else if (a == "--az") c.wideAz = (float)atof(val().c_str());
        else if (a == "--el") c.wideEl = (float)atof(val().c_str());
        else {
            printf("dronolok_app --twocam [--seconds 60] [--drones 3] [--birds 1] [--seed 1] [--range 100,750]\n"
                   "  [--speed 6,22] [--clouds 0.35] [--noise 2] [--visibility 6000] [--size 0.5] [--types 0.5,0.25,0.25] [--wide 4608x3456] [--wide-fps 14] [--fisheye] [--fov 5 (x4; 20 — без зума)]\n"
                   "  [--fps 60] [--rate 30] [--accel 150] [--latency 0.06] [--dwell 6]\n"
                   "  [--switch manual|queue|nearest] [--json out.json] [--quiet]\n"
                   "  сравнение доработок точной: [--image-motion] [--no-ff] [--no-window] [--no-lock-focus]\n"
                   "  [--min-hold 0.08] [--give-up 3] [--cool-down 10]\n"
                   "  обмен между камерами: [--no-silhouette] [--no-hold] [--no-veto] [--no-cue] [--no-shape] [--shape амп,автокорр,оценок,окно] [--no-veto-wide] [--no-next] [--no-veto-manual] [--with-birds] [--legacy — всё выключено]\n"
                   "  3D-сцена Godot вместо синтетического неба: [--scene sim/scenes/city_sectors.json] [--sector highrise]\n"
                   "  [--godot путь] [--az 0] [--el 32]\n"
                   "  погода и время суток сцены: [--weather clear|cumulus|cloudy|overcast|haze|mist|fog|rain|storm|snow]\n"
                   "  [--time 19:30] [--time-rate 60] [--date 09-15] [--lat 55.75] [--scene-clouds 0.8] [--scene-visibility 3000]\n"
                   "  [--rain 0.5] [--snow 0.5]\n");
            return a == "--help" ? 0 : 2;
        }
    }
    if (!c.scene.empty()) {
        std::string sim, godot = c.godot;
        if (c.simProject.empty() && scene::find_sim(".", sim, godot)) c.simProject = sim + "/godot";
        if (c.simProject.empty()) {   // путь к сцене sim/scenes/x.json — проект рядом
            const std::string d = c.scene.substr(0, c.scene.find_last_of("/\\"));
            c.simProject = d + "/../godot";
        }
        if (c.godot.empty()) c.godot = godot;
        if (c.godot.empty()) {
            printf("не найден Godot 4.3: --godot путь или переменная GODOT\n");
            return 2;
        }
        if (!c.sector.empty())   // ось широкой — на середину сектора
            for (const auto& sc : scene::scene_sectors(c.scene))
                if (sc.name == c.sector) c.wideAz = sc.mid;
        printf("погода сцены (поверх своей): %s\n", scene_env_json(c).c_str());
    }
    const Stats s = e.runBlocking(c, seconds, verbose);
    if (!c.scene.empty() && s.wideFrames == 0) {
        printf("%s\n", e.view().status.c_str());
        return 1;
    }
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
    printf("ведёт настоящий дрон %.0f%% времени; цель только у точной %.1f с (на назначенной %.0f%%), не бросила цель %d раз,\n"
           "  вернула широкой %d; подтвердила дронов %d; отсеяла как птицу: птиц %d, дронов %d, пустых мест %d; переходов по частям цели %d\n",
           100 * s.trackDrone / T, s.narrowOnly, s.narrowOnly > 0 ? 100 * s.narrowOnlyCorrect / s.narrowOnly : 0, s.kept,
           s.rebinds, s.confirmed, s.vetoBird, s.vetoDrone, s.vetoNone, s.partJumps);
    printf("птицы у точной: вела птицу %.1f с, назвала дроном птиц %d, отсеяла птицу в среднем через %.2f с после захвата, по силуэту %d, "
           "сразу на следующую %d\n",
           s.trackBird, s.confirmedBird, s.vetoBird ? s.vetoDelay / s.vetoBird : 0, s.vetoShape, s.nextAfterBird);
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
                    " \"unconfirmed\": %d, \"wide_swaps\": %d, \"idle_pct\": %.2f, \"false_target_pct\": %.2f, \"wide_ms\": %.2f, \"narrow_ms\": %.3f,\n"
                    " \"track_drone_pct\": %.2f, \"narrow_only_s\": %.2f, \"narrow_only_correct_s\": %.2f, \"kept\": %d, \"rebinds\": %d,\n"
                    " \"confirmed\": %d, \"veto_bird\": %d, \"veto_drone\": %d, \"veto_none\": %d, \"part_jumps\": %d,\n"
                    " \"track_bird_s\": %.2f, \"confirmed_bird\": %d, \"veto_delay_s\": %.3f, \"veto_shape\": %d, \"next_after_bird\": %d}\n",
                    s.simT, s.assignments, s.acquired, s.wrongLocks, s.losses, s.slewN ? s.slewSum / s.slewN : 0,
                    s.acqN ? s.acqSum / s.acqN : 0, 100 * s.trackTime / T,
                    s.trackTime > 0 ? 100 * s.trackCorrect / s.trackTime : 0, lm, l9, am, a9, wm, w9,
                    s.unconfirmed, s.wideSwaps, 100 * s.idleTime / T, 100 * s.falseTarget / T, s.wideFrames ? s.wideMs / s.wideFrames : 0, s.narrowFrames ? s.narrowMs / s.narrowFrames : 0,
                    100 * s.trackDrone / T, s.narrowOnly, s.narrowOnlyCorrect, s.kept, s.rebinds, s.confirmed, s.vetoBird,
                    s.vetoDrone, s.vetoNone, s.partJumps, s.trackBird, s.confirmedBird, s.vetoBird ? s.vetoDelay / s.vetoBird : 0, s.vetoShape,
                    s.nextAfterBird);
            fclose(f);
        }
    }
    return 0;
}

}  // namespace twocam
