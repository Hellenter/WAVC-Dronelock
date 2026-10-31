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

inline Vec3 vsub(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

}  // namespace

// Широкая камера установки в эмуляторе: модель, поза по паспорту и на самом деле, свой конвейер
// (Pipeline — настоящий, кадры от нас), фон синтетического неба, маска.
struct Emulator::WideUnit {
    int k = 0;
    WideCam cfg;
    CameraModel cam;
    Pose pose;       // по паспорту: по ней алгоритм считает направления
    Pose truePose;   // на самом деле (паспорт + ошибка установки): так рисуется кадр
    Vec3 pos;        // м от поворотки
    // синтетическое небо: фон без облаков, земля, направления узлов сетки облаков
    std::vector<float> base;
    std::vector<uint8_t> ground;
    std::vector<float> gaz, gel;
    int cgW = 0, cgH = 0;
    std::vector<int16_t> sky, noise;   // x8
    double skyT = -1;
    // конвейер и ожидание его результата
    Pipeline pipe;
    std::mutex m;
    std::condition_variable cv;
    std::vector<uint8_t> frame;
    bool ready = false;   // кадр отрисован, генератор может забрать
    int64_t idx = -1, done = -1, n = 0;
    std::shared_ptr<const Result> res;
    double nextT = 0;
    // маска неба: тайлы её результата, кадров после поворота
    int panoTw = 0, panoTh = 0, panoTile = 0, sincePose = 0;
    bool lastLearning = true;
    std::vector<int> birds;                     // её треки, которым точная сказала «птица»
    std::vector<std::pair<int, int>> truth;     // её трек -> тело по правде (последний кадр)
    std::vector<FocusArea> cueNet;              // подсказка о соседнем секторе, пришедшая по сети
    double cueNetT = -1;
    uint32_t seq = 0;
    dronolok::net::ClockSync sync;              // устройство поворотки: часы этой камеры
    double nextPing = 0;
    double ms = 0;
    int frames = 0;
};

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
    const int nw = (int)wu_.size();
    // несколько широких: в поле случайной камеры, а доля seamShare — у стыка соседних секторов, курсом через него
    bool seam = false;
    double seamAz = 0, seamHalf = 0;   // стык и полуширина щели между секторами (0 — перекрываются)
    if (nw > 1 && c_.seamShare > 0 && U(rng_) < c_.seamShare) {
        std::vector<double> az;
        for (const auto& w : wu_) az.push_back(w->pose.az);
        std::sort(az.begin(), az.end());
        std::vector<double> seams;
        for (int i = 0; i < nw; ++i) {
            const double a = az[i], b2 = az[(i + 1) % nw];
            const double gap = az_norm(b2 - a);
            if (gap > 1e-3 && gap < 2 * c_.wideFov) seams.push_back(az_norm(a + gap / 2));   // стык — посередине между осями
        }
        if (!seams.empty()) {
            seam = true;
            const size_t i = std::min((size_t)(U(rng_) * seams.size()), seams.size() - 1);
            seamAz = seams[i];
            seamHalf = std::max(0.0, (360.0 / nw - c_.wideFov) / 2);
            for (int j = 0; j < nw; ++j)   // щель — по соседним осям этого стыка
                if (std::fabs(az_diff(az[j], seamAz)) < 360.0 / nw)
                    seamHalf = std::max(0.0, std::fabs(az_diff(az[j], seamAz)) - c_.wideFov / 2);
        }
    }
    const WideUnit* home = wu_[0].get();
    if (nw > 1 && !seam) home = wu_[std::min(nw - 1, (int)(U(rng_) * nw))].get();
    auto visible = [&](const Vec3& d) {
        for (const auto& w : wu_) {
            double u, v;
            if (project(w->cam, w->truePose, d, u, v) && in_frame(w->cam, u, v, -40)) return true;
        }
        return false;
    };
    for (int k = 0; k < 50; ++k) {
        // у стыка: в 3–18° от края щели (или от стыка, если секторы перекрываются), курсом к нему — точная успевает
        // взять цель до перелёта
        const double o = U(rng_) * 2 - 1;
        const double az = seam ? az_norm(seamAz + (o < 0 ? -1 : 1) * (seamHalf + 3 + std::fabs(o) * 15))
                               : az_norm(home->pose.az + o * half);
        const double el = treeline(az) + 1.5 + U(rng_) * 26;
        const double R = c_.rangeMin + U(rng_) * (c_.rangeMax - c_.rangeMin);
        // новая цель влетает в поле зрения сбоку или снизу, первые — где угодно
        double u, v;
        const Vec3 d = dir_from_azel(az, el);
        if (nw == 1) {
            if (!project(wu_[0]->cam, wu_[0]->truePose, d, u, v) || !in_frame(wu_[0]->cam, u, v, -40)) continue;
        } else if (!seam && !visible(d)) {
            continue;
        }
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
    double hd = U(rng_) * 2 * PI;
    if (seam) {   // поперёк луча зрения, к стыку и через него (±25°)
        double az, el;
        azel_from_dir(b.p0, az, el);
        hd = -deg2rad(az) + (az_diff(az, seamAz) < 0 ? 0 : PI) + (U(rng_) * 2 - 1) * deg2rad(25);
    }
    b.v = {sp * std::cos(hd), sp * std::sin(hd), (U(rng_) * 2 - 1) * (b.bird ? 0.5 : 1.5)};
    b.flapHz = b.bird ? (float)(4 + U(rng_) * 4) : 0;
    b.phase = (float)(U(rng_) * 6.28);
    if (!b.bird && c_.droneWobble) {   // порывы ветра: дрон иногда качается
        const double f1 = std::fmod(b.phase * 5.13, 1.0), f2 = std::fmod(b.phase * 2.71, 1.0), f3 = std::fmod(b.phase * 9.07, 1.0);
        b.wobPer = (float)(3 + 5 * f1);
        b.wobOn = (float)(0.2 + 0.4 * f2);
        b.wobHz = (float)(3 + 3 * f3);
    }
    if (b.bird && c_.birdGlide) {   // как в 3D-сцене (sim.gd): пачки взмахов, между ними крылья раскрыты
        // из фазы, а не новыми числами генератора: остальная сцена (очерёдность, места целей) та же, что без планирования
        const double f1 = std::fmod(b.phase * 7.31, 1.0), f2 = std::fmod(b.phase * 3.17, 1.0);
        b.flapPer = (float)(2 + 3 * f1);
        b.flapOn = (float)(0.8 + 1.2 * f2);
    }
    (void)first;
}

void Emulator::drawBodies(const CameraModel& m, const Pose& p, const Vec3& pos, double t, uint8_t* px, int w, int h,
                          const Pose* pa, const Pose* pb, double ex) {
    std::vector<Tri> tris;
    for (const Body& b : bodies_) {
        const Vec3 q = vsub(b.at(t), pos);   // от камеры
        const double R = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
        double u, v;
        if (!project(m, p, q, u, v) || !in_frame(m, u, v, 20)) continue;
        // смаз за выдержку: где цель на кадре в начале и в конце выдержки (камера и цель движутся)
        double su = 0, sv = 0;
        if (pa && pb && ex > 0) {
            double ua, va, ub, vb;
            if (project(m, *pa, vsub(b.at(t - 0.5 * ex), pos), ua, va) && project(m, *pb, vsub(b.at(t + 0.5 * ex), pos), ub, vb))
                su = ub - ua, sv = vb - va;
        }
        double c = (b.bird ? -55.0 : -70.0) * std::exp(-R / c_.visibility);
        double flap = 0, pitch = 0, bank = 0;
        const double sp = std::sqrt(b.v.x * b.v.x + b.v.y * b.v.y);
        if (b.bird) {   // взмахи: крылья ходят вверх-вниз, силуэт меняется
            flap = 0.7 * std::sin(2 * PI * b.flapHz * (t - b.t0) + b.phase);
            if (b.flapPer > 0 && std::fmod(t - b.t0 + b.id * 0.37, (double)b.flapPer) > b.flapOn) flap = 0.12;   // планирует
            bank = 0.15 * std::sin(0.7 * t + b.phase);
        } else if (b.shape == Shape::Quad) {
            pitch = -std::min(0.45, 0.02 * sp);   // квадрокоптер летит с наклоном носом вниз
            bank = 0.05 * std::sin(1.3 * t + b.phase);
        } else {
            bank = 0.12 * std::sin(0.4 * t + b.phase);
        }
        if (b.wobPer > 0 && std::fmod(t - b.t0 + b.id * 0.53, (double)b.wobPer) < b.wobOn) {   // качнуло порывом
            const double w = 2 * PI * b.wobHz * (t - b.t0);
            bank += 0.35 * std::sin(w);
            pitch += 0.2 * std::sin(w + 1.1);
        }
        build_silhouette(b.shape, b.size, flap, tris);
        draw_silhouette(tris, q, body_frame(b.v, pitch, bank), c, c_.psf,
                        [&](const Vec3& d, double& uu, double& vv) { return project(m, p, d, uu, vv); }, px, w, h, su, sv);
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
    bool ok = link_->render(t, cam, p.az, p.el, out, w, h, sbodies_, err, p.roll);
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
        ok = link_->start(linkOpt_, linkHello_, err) && link_->render(t, cam, p.az, p.el, out, w, h, sbodies_, err, p.roll);
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

void Emulator::renderWide(WideUnit& w, double t, std::vector<uint8_t>& out) {
    const int W = w.cam.w, H = w.cam.h;
    const size_t N = (size_t)W * H;
    out.resize(N);
    if (sceneMode_) {   // камеры сцены: 0 — первая широкая, 1 — точная, 2... — вторая и следующие широкие
        sceneRender(t, w.k == 0 ? 0 : w.k + 1, w.truePose, W, H, out);
        return;
    }
    // облака плывут медленно (0.04 °/с — меньше пикселя за 0.5 с): фон с облаками считаем раз в
    // полсекунды, на сетке CLOUD_STEP, между узлами — линейно
    if (w.sky.size() != N || t - w.skyT >= 0.5) {
        w.skyT = t;
        w.sky.resize(N);
        const int cgW = w.cgW, cgH = w.cgH;
        std::vector<float> cg((size_t)cgW * cgH);
        parallel_rows(cgH, [&](int ja, int jb) {
            for (int j = ja; j < jb; ++j)
                for (int i = 0; i < cgW; ++i) {
                    const size_t k = (size_t)j * cgW + i;
                    const double az = w.gaz[k], el = w.gel[k];
                    cg[k] = el >= treeline(az) ? background(az, el, t, false) - sky_base(el) : 0.f;
                }
        });
        parallel_rows(H, [&](int ya, int yb) {
            for (int y = ya; y < yb; ++y) {
                const float gy = (float)y / CLOUD_STEP;
                const int j0 = std::min(cgH - 2, (int)gy);
                const float ty = gy - j0;
                const float* c0 = &cg[(size_t)j0 * cgW];
                const float* c1 = c0 + cgW;
                const float* base = &w.base[(size_t)y * W];
                const uint8_t* gr = &w.ground[(size_t)y * W];
                int16_t* o = &w.sky[(size_t)y * W];
                for (int x = 0; x < W; ++x) {
                    float v = base[x];
                    if (!gr[x]) {
                        const float gx = (float)x / CLOUD_STEP;
                        const int i0 = std::min(cgW - 2, (int)gx);
                        const float tx = gx - i0;
                        v += (c0[i0] + (c0[i0 + 1] - c0[i0]) * tx) * (1 - ty) + (c1[i0] + (c1[i0 + 1] - c1[i0]) * tx) * ty;
                    }
                    o[x] = (int16_t)std::lround(v * 8);
                }
            }
        });
    }
    // шум датчика: заранее набранный, со случайным сдвигом на каждый кадр
    if (w.noise.size() != N) {
        w.noise.resize(N);
        uint32_t r = (noiseSeed_ ^ 0xabcdefu) + (uint32_t)w.k * 0x9e3779b9u;
        for (auto& v : w.noise) {
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
            const int v = (w.sky[k] + w.noise[(k + off) % N] + 4) >> 3;
            out[k] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    });
    drawBodies(w.cam, w.truePose, w.pos, t, out.data(), W, H);
}

void Emulator::renderNarrow(double t, const Pose& p, std::vector<uint8_t>& out, const Pose* pa, const Pose* pb, double ex) {
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
    drawBodies(narrowCam_, p, Vec3{}, t, out.data(), W, H, pa, pb, ex);
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

// тело по правде у направления d из точки pos (камера не в начале отсчёта)
int Emulator::nearestBodyFrom(double t, const Vec3& pos, const Vec3& d, double maxDeg) const {
    int best = 0;
    double bd = maxDeg;
    for (const Body& b : bodies_) {
        if (b.hidden) continue;
        const Vec3 q = vsub(b.at(t), pos);
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
    netRng_.seed(c.seed * 31u + 5);
    nextBodyId_ = 1;
    // широкие камеры: первая — wideAz/wideEl и wide0, следующие — moreWide; размер, поле, частота общие
    wu_.clear();
    std::vector<RigCamera> rig;
    for (int k = 0; k < c.wideCount(); ++k) {
        std::unique_ptr<WideUnit> w(new WideUnit());
        w->k = k;
        w->cfg = k == 0 ? c.wide0 : c.moreWide[k - 1];
        if (k == 0) w->cfg.az = c.wideAz, w->cfg.el = c.wideEl;
        w->cam.w = c.wideW;
        w->cam.h = c.wideH;
        w->cam.hfov_deg = c.wideFov;
        w->cam.lens = c.wideFisheye ? Lens::Fisheye : Lens::Pinhole;
        w->pose = {w->cfg.az, w->cfg.el, 0};
        w->truePose = {(float)az_norm(w->cfg.az + w->cfg.errAz), w->cfg.el + w->cfg.errEl, w->cfg.errRoll};
        w->pos = {w->cfg.x, w->cfg.y, w->cfg.z};
        // устройства не в такт: кадры камеры позже на phase (на одном устройстве — в такт)
        w->nextT = c.deploy == Deploy::PerCamera ? std::max(0.f, w->cfg.phase) : 0;
        RigCamera rc;
        rc.model = w->cam;
        rc.pose = w->pose;
        rc.pos = w->pos;
        rig.push_back(rc);
        wu_.push_back(std::move(w));
    }
    FusionConfig fc;
    fc.calib = c.fusionCalib;
    fc.calibInitDeg = c.calibInit;
    fc.cue = c.crossCue;
    fc.independent = !c.fusion && c.wideCount() > 1;
    fc.rangeMin = std::max(30.f, 0.6f * c.rangeMin);
    fc.rangeMax = std::max(c.rangeMax * 1.7f, fc.rangeMin + 1);
    fc.rangeGuess = std::sqrt(std::max(1.f, c.rangeMin) * std::max(1.f, c.rangeMax));
    fusion_.reset(rig, fc);
    net_.clear();
    udpTx_.close();
    udpRx_.close();
    if (c.netUdp && c.deploy == Deploy::PerCamera) {
        std::string err;
        if (!udpRx_.bind("127.0.0.1", 0, err) || !udpTx_.bind("127.0.0.1", 0, err)) {
            fprintf(stderr, "UDP: %s — сообщения без сокета\n", err.c_str());
            udpTx_.close();
            udpRx_.close();
        }
    }
    primaryCam_ = -1;
    bodySeen_.clear();
    seamCheck_.clear();
    lastSnapT_ = -1;
    offRefT_ = offTrack_ = -1;
    offN_ = 0;
    offCam_ = -1;
    narrowCam_ = CameraModel();
    narrowCam_.w = c.narrowW;
    narrowCam_.h = c.narrowH;
    narrowCam_.hfov_deg = c.narrowFov;
    sceneMode_ = !c.scene.empty();
    poseSeen_ = poseReq_;
    {
        std::lock_guard<std::mutex> lk(poseM_);
        poseWant_.clear();
    }
    scanReq_ = false;
    scan_ = false;
    pano_.clear();   // панорама — своя у каждой сцены
    if (!sceneMode_)
        for (auto& w : wu_) prepareWide(*w);
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
    // поворотка как на железе
    gcfg_ = GimbalConfig();
    gcfg_.max_rate = c.maxRate;
    gcfg_.max_accel = c.maxAccel;
    gcfg_.max_jerk = c.maxJerk;
    gcfg_.kp = c.mountKp;
    gcfg_.kv = c.mountKv;
    gcfg_.lead_cmd = c.cmdLatency;   // задержку команды контроллер знает (номинальную)
    gcfg_.jerk_noise = c.targetJerk;
    // ошибка положения на момент кадра: полшага, люфт (без датчика и без поправки) или датчик на оси
    gcfg_.sigma_pose = (float)std::sqrt(std::pow(0.5 * (c.encoder == 1 ? c.encoderDeg : c.stepDeg), 2) +
                                        std::pow(c.encoder == 1 || c.backlashComp ? 0.0 : 0.5 * c.backlash, 2) + 0.003 * 0.003);
    gf_ = GimbalFilter(gcfg_);
    wbias_ = WideBias();
    gfWideT_ = biasT_ = -1;
    gfAssignN_ = -1;
    nq_.clear();
    jitBody_ = jitK_ = 0;
    jitT_ = etaT_ = -1;
    {
        MountParams mp;
        mp.ctl = c.mountCtl;
        mp.cmdLatency = c.cmdLatency;
        mp.cmdJitter = c.cmdJitter;
        mp.stepDeg = c.stepDeg;
        mp.backlash = c.backlash;
        mp.backlashComp = c.backlashComp;
        mp.resonanceHz = c.resonanceHz;
        mp.damping = c.damping;
        mp.slipEvery = c.slipEvery;
        mp.slipSteps = c.slipSteps;
        mp.encoder = c.encoder;
        mp.encoderDeg = c.encoderDeg;
        mp.poseModel = c.poseModel;
        mp.modelHz = c.modelHz;
        mp.modelDamping = c.modelDamping;
        mp.g = gcfg_;
        msim_.reset(mp, c.wideAz, c.wideEl, c.seed);
    }
    state_ = NState::Idle;
    assigned_ = assignedBody_ = nlock_ = 0;
    haveNrep_ = false;
    winOn_ = false;
    st_ = Stats();
    events_.clear();
    targets_.clear();
    visited_.clear();
    cool_.clear();
    veto_.clear();
    nConf_ = nOnly_ = manualPick_ = afterBird_ = false;
    detachId_ = 0;
    tDetach_ = -10;
    tBird_ = -1;
    birdT_ = 0;
    lastU_ = lastV_ = lastSz_ = 0;
    stop_ = false;
    // широкие: настоящий конвейер у каждой, кадры — от нас
    for (auto& wp : wu_) {
        WideUnit& w = *wp;
        w.pipe.setCue({});
        Settings s;
        s.pace = Pace::MaxSpeed;
        s.loop = false;
        s.fresh = true;
        s.staticFile = false;
        s.maskFile = false;
        w.pipe.setSettings(s);
        w.ready = false;
        w.idx = -1;
        w.done = -1;
        w.res.reset();
        WideUnit* wq = &w;
        w.pipe.setObserver([wq](const Result& r) {
            auto p = std::make_shared<Result>(r);
            std::lock_guard<std::mutex> lk(wq->m);
            wq->res = p;
            wq->done = r.frame ? r.frame->index : wq->done;
            wq->cv.notify_all();
        });
        VideoInfo vi;
        vi.w = c.wideW;
        vi.h = c.wideH;
        vi.fps = c.wideFps;
        vi.codec = "эмулятор";
        w.pipe.openGenerator(vi,
                             [this, wq](Frame& f) {
                                 std::unique_lock<std::mutex> lk(wq->m);
                                 wq->cv.wait(lk, [&] { return wq->ready || stop_; });
                                 if (stop_) return false;
                                 memcpy(f.px.data(), wq->frame.data(), std::min(f.px.size(), wq->frame.size()));
                                 f.index = wq->idx;
                                 wq->ready = false;
                                 return true;
                             },
                             "эмулятор");
    }
    {
        std::string ev = "старт: целей " + std::to_string(c.drones) + " дронов и " + std::to_string(c.birds) + " птиц" +
                         (sceneMode_ ? ", 3D-сцена " + c.scene : std::string());
        if (c.wideCount() > 1) {
            char b[160];
            snprintf(b, sizeof(b), "; широких %d (%s)%s", c.wideCount(),
                     c.deploy == Deploy::OneDevice ? "все на одном устройстве" : "у каждой своё устройство, обмен по сети",
                     !c.fusion ? ", без объединения целей" : "");
            ev += b;
        }
        event(0, ev);
    }
    running_ = true;
    th_ = std::thread([this] { loop(); });
}

// фон широкой без облаков — при запуске и после поворота широкой (рисуется в настоящей позе камеры)
void Emulator::prepareWide(WideUnit& w) {
    const int W = w.cam.w, H = w.cam.h;
    w.base.assign((size_t)W * H, 0);
    w.ground.assign((size_t)W * H, 0);
    parallel_rows(H, [&](int ya, int yb) {
        for (int y = ya; y < yb; ++y)
            for (int x = 0; x < W; ++x) {
                double az, el;
                azel_from_dir(unproject(w.cam, w.truePose, x + 0.5, y + 0.5), az, el);
                const bool g = el < treeline(az);
                const size_t k = (size_t)y * W + x;
                w.ground[k] = g;
                w.base[k] = g ? background(az, el, 0, true) : sky_base(el);
            }
    });
    w.cgW = W / CLOUD_STEP + 2;
    w.cgH = H / CLOUD_STEP + 2;
    w.gaz.assign((size_t)w.cgW * w.cgH, 0);
    w.gel.assign(w.gaz.size(), 0);
    for (int j = 0; j < w.cgH; ++j)
        for (int i = 0; i < w.cgW; ++i) {
            double az, el;
            azel_from_dir(unproject(w.cam, w.truePose, i * CLOUD_STEP, j * CLOUD_STEP), az, el);
            w.gaz[(size_t)j * w.cgW + i] = (float)az;
            w.gel[(size_t)j * w.cgW + i] = (float)el;
        }
    w.sky.clear();
    w.skyT = -1;
}

// маска широкой (по тайлам) -> ячейки панорамы, видимые сейчас (панорама общая: камеры стоят рядом)
void Emulator::panoLearn(const WideUnit& w, const Result& r) {
    if ((int)pano_.size() != PANO_W * PANO_H) pano_.assign((size_t)PANO_W * PANO_H, 0);
    const int tw = r.tilesX, th = r.tilesY, tp = r.tilePx;
    if (tw <= 0 || th <= 0 || tp <= 0 || r.mask.size() != (size_t)tw * th) return;
    const double half = c_.wideFov * 0.5 + 5;
    const int n = (int)std::ceil(half / PANO_STEP);
    const int i0 = (int)std::floor(w.pose.az / PANO_STEP);
    for (int j = 0; j < PANO_H; ++j) {
        const double el = PANO_EL0 + (j + 0.5) * PANO_STEP;
        for (int di = -n; di <= n; ++di) {
            const int i = ((i0 + di) % PANO_W + PANO_W) % PANO_W;
            double u, v;
            if (!project(w.cam, w.pose, dir_from_azel((i + 0.5) * PANO_STEP, el), u, v)) continue;
            const int tx = (int)(u / tp), ty = (int)(v / tp);
            if (u < 0 || v < 0 || tx >= tw || ty >= th) continue;
            pano_[(size_t)j * PANO_W + i] = r.mask[(size_t)ty * tw + tx] ? 1 : 2;
        }
    }
}

// маска широкой для текущего поворота из панорамы; возвращает долю тайлов, которые панорама покрывает
float Emulator::panoMaskFor(const WideUnit& w, std::vector<uint8_t>& m) const {
    m.assign((size_t)w.panoTw * w.panoTh, 1);
    if ((int)pano_.size() != PANO_W * PANO_H) return 0.f;
    int known = 0;
    for (int ty = 0; ty < w.panoTh; ++ty)
        for (int tx = 0; tx < w.panoTw; ++tx) {
            double az, el;
            azel_from_dir(unproject(w.cam, w.pose, (tx + 0.5) * w.panoTile, (ty + 0.5) * w.panoTile), az, el);
            const int j = (int)std::floor((el - PANO_EL0) / PANO_STEP);
            uint8_t v = 0;
            if (j >= PANO_H) v = 1;        // выше панорамы — небо
            else if (j < 0) v = 2;         // ниже — земля
            else v = pano_[(size_t)j * PANO_W + ((int)std::floor(az / PANO_STEP) % PANO_W + PANO_W) % PANO_W];
            if (v) ++known;
            m[(size_t)ty * w.panoTw + tx] = v != 2;
        }
    return (float)known / std::max(1, w.panoTw * w.panoTh);
}

void Emulator::stop() {
    stop_ = true;
    for (auto& w : wu_) w->cv.notify_all();
    if (th_.joinable()) th_.join();
    for (auto& w : wu_) w->pipe.close();
    if (link_) link_->stop();
    udpTx_.close();
    udpRx_.close();
    running_ = false;
}

void Emulator::onWide(WideUnit& wu, double t, const Result& r) {
    const bool single = wu_.size() == 1;
    std::vector<WideTarget> v;
    wu.truth.clear();
    for (const Track& tr : r.tracks) {
        if (tr.clutter || tr.miss > 2) continue;
        WideTarget w;
        w.id = tr.id;
        w.cam = wu.k;
        w.local = tr.id;
        w.rep = make_report(wu.cam, wu.pose, t, c_.wideFps, tr.x, tr.y, tr.vsx, tr.vsy, tr.sigma);
        w.rep.id = tr.id;
        w.cls = (int)tr.cls;
        w.hard = r.lockId == tr.id && r.lockState == LockState::Hard;
        w.rep.cls = w.cls;
        w.rep.hard = w.hard;
        // правда: тело на луче трека — в настоящей позе камеры и из её положения
        w.truth = single ? nearestBody(t, w.rep.az, w.rep.el, 0.25)
                         : nearestBodyFrom(t, wu.pos, unproject(wu.cam, wu.truePose, tr.x, tr.y), 0.25);
        const TargetReport ref = fusion_.toRef(wu.k, w.rep);   // в системе поворотки (одна камера — как есть)
        // новый трек широкой там, где точная отсеяла птицу: сказать широкой, что это птица
        if (c_.vetoToWide && w.cls != 2 && std::find(wu.birds.begin(), wu.birds.end(), w.id) == wu.birds.end() && vetoed(t, ref)) {
            wu.pipe.markBird(w.id);
            wu.birds.push_back(w.id);
        }
        if (w.truth)
            for (const Body& b : bodies_)
                if (b.id == w.truth) {
                    double az, el;
                    azel_from_dir(b.at(t), az, el);
                    st_.errWide.push_back((float)ang(az, el, ref.az, ref.el));
                }
        wu.truth.push_back({w.id, w.truth});
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
    st_.wideMs += r.procMs;
    wu.ms += r.procMs;
    ++wu.frames;
    std::vector<TargetReport> reps;
    for (const auto& w : v) reps.push_back(w.rep);
    if (c_.deploy == Deploy::OneDevice) {
        fusion_.update(wu.k, t, reps);   // объединение — на этом же устройстве; поворотке — после кадров всех камер
        return;
    }
    // устройство камеры: цели — устройству поворотки по сети, время кадра — по своим часам
    net::WideMsg m;
    m.cam = wu.k;
    m.seq = ++wu.seq;
    m.t = t + wu.cfg.clock;
    m.sent = m.t + r.procMs * 1e-3;
    m.reps = reps;
    Msg msg;
    msg.kind = 1;
    msg.cam = wu.k;
    msg.data = net::encode(m);
    std::exponential_distribution<double> J(1.0 / std::max(1e-6f, c_.netJitter));
    msg.at = t + c_.handoffLatency + (c_.netJitter > 0 ? J(netRng_) : 0);
    if (udpRx_.open()) {   // по настоящему сокету: что ушло, то и пришло
        std::string got;
        if (udpTx_.sendTo("127.0.0.1", udpRx_.port(), msg.data) && udpRx_.recv(got, 500)) msg.data = got;
        else msg.data.clear();
    }
    ++st_.netMsgs;
    st_.netBytes += (int)msg.data.size();
    post(std::move(msg));
}

void Emulator::post(Msg m) {
    // по времени доставки; равные — в порядке отправки
    auto it = std::upper_bound(net_.begin(), net_.end(), m.at, [](double a, const Msg& b) { return a < b.at; });
    net_.insert(it, std::move(m));
}

void Emulator::fusedSnapshot(double t, std::vector<WideTarget>& out) {
    (void)t;
    out.clear();
    for (const FusedTarget& f : fusion_.targets()) {
        WideTarget w;
        w.id = f.id;
        w.rep = f.rep;
        w.cls = f.cls;
        w.hard = f.hard;
        w.cam = f.cam;
        w.local = f.local;
        w.seenBy = f.seenBy;
        if (!fusion_.passthrough() && fusion_.config().calib && f.cam >= 0 && f.cam < fusion_.cameras())
            w.bias = (float)fusion_.calib(f.cam).sigmaDeg;
        if (f.cam >= 0 && f.cam < (int)wu_.size())
            for (const auto& p : wu_[f.cam]->truth)
                if (p.first == f.local) w.truth = p.second;
        out.push_back(w);
    }
}

void Emulator::markWideBird(double t, int id) {
    for (auto& wp : wu_) {
        WideUnit& w = *wp;
        const int local = fusion_.localOf(id, w.k);
        if (!local) continue;
        if (wu_.size() == 1 && c_.deploy == Deploy::OneDevice) {   // одна камера на том же устройстве — сразу, как было
            w.pipe.markBird(local);
            w.birds.push_back(local);
            continue;
        }
        Msg m;
        m.kind = 3;
        m.cam = w.k;
        m.id = local;
        m.at = t + c_.netLatency;
        if (c_.deploy == Deploy::PerCamera) m.data = net::encode(net::BirdMsg{w.k, local});
        post(std::move(m));
    }
}

void Emulator::deliver(double t) {
    while (!net_.empty() && net_.front().at <= t + 1e-9) {
        Msg m = std::move(net_.front());
        net_.pop_front();
        switch (m.kind) {
        case 0:   // цели установки дошли до поворотки (объединение на устройстве широких)
            for (const auto& mg : m.merges) {
                if (assigned_ == mg.first) assigned_ = mg.second;
                if (detachId_ == mg.first) detachId_ = mg.second;
            }
            setTargets(t, std::move(m.snap));
            break;
        case 1: {   // цели камеры дошли до устройства поворотки (по сети): объединить здесь
            net::WideMsg w;
            if (m.data.empty() || !net::decode(m.data, w) || w.cam < 0 || w.cam >= (int)wu_.size()) break;
            WideUnit& u = *wu_[w.cam];
            const double off = c_.clockSync && u.sync.valid() ? u.sync.offset() : 0.0;
            const double tc = w.t - off;   // время кадра по часам поворотки
            for (auto& r : w.reps) r.t = tc;
            fusion_.update(w.cam, tc, w.reps);
            for (const auto& mg : fusion_.takeMerges()) {
                if (assigned_ == mg.first) assigned_ = mg.second;
                if (detachId_ == mg.first) detachId_ = mg.second;
            }
            std::vector<WideTarget> snap;
            fusedSnapshot(t, snap);
            setTargets(t, std::move(snap));
            // подсказки соседним камерам: где ждать цели, которых они пока не видят (их система, градусы)
            if (c_.crossCue && wu_.size() > 1)
                for (auto& op : wu_) {
                    if (op->k == w.cam) continue;
                    const double tq = t + c_.netLatency + 0.5 / std::max(1.f, c_.wideFps);
                    net::CueMsg cm;
                    cm.cam = op->k;
                    cm.t = tq;
                    for (const auto& a : fusion_.cues(op->k, tq)) {
                        double az, el;
                        azel_from_dir(unproject(op->cam, op->pose, a.u, a.v), az, el);
                        cm.areas.push_back({(float)az, (float)el, (float)(a.r * pixel_deg(op->cam, op->pose, a.u, a.v))});
                    }
                    Msg q;
                    q.kind = 2;
                    q.cam = op->k;
                    q.data = net::encode(cm);
                    q.at = t + c_.netLatency;
                    ++st_.netMsgs;
                    st_.netBytes += (int)q.data.size();
                    post(std::move(q));
                }
            break;
        }
        case 2: {   // подсказка дошла до устройства камеры
            net::CueMsg cm;
            if (!net::decode(m.data, cm) || cm.cam < 0 || cm.cam >= (int)wu_.size()) break;
            WideUnit& u = *wu_[cm.cam];
            u.cueNet.clear();
            for (const auto& a : cm.areas) {
                double uu, vv;
                if (!project(u.cam, u.pose, dir_from_azel(a.az, a.el), uu, vv) || !in_frame(u.cam, uu, vv)) continue;
                const double pd = pixel_deg(u.cam, u.pose, uu, vv);
                u.cueNet.push_back({(float)uu, (float)vv, (float)(a.r / std::max(1e-6, pd))});
            }
            u.cueNetT = t;
            break;
        }
        case 3: {   // «птица» дошла до устройства камеры
            int cam = m.cam, local = m.id;
            if (!m.data.empty()) {
                net::BirdMsg b;
                if (!net::decode(m.data, b)) break;
                cam = b.cam, local = b.local;
            }
            if (cam < 0 || cam >= (int)wu_.size()) break;
            wu_[cam]->pipe.markBird(local);
            wu_[cam]->birds.push_back(local);
            break;
        }
        case 4:   // наблюдение точной дошло до объединения (оно на устройстве широких)
            fusion_.observeReference(m.id, m.rep, m.flag);
            break;
        case 5: {   // сверка часов: запрос дошёл до камеры (ответить) или ответ — до поворотки
            if (m.cam < 0 || m.cam >= (int)wu_.size()) break;
            WideUnit& u = *wu_[m.cam];
            std::exponential_distribution<double> J(1.0 / std::max(1e-6f, c_.netJitter));
            // время событий — когда сообщение на самом деле дошло (m.at), а не шаг, на котором его разобрали
            if (!m.flag) {
                net::PingMsg p;
                if (!net::decode(m.data, p)) break;
                const double t2 = m.at + u.cfg.clock, t3 = t2 + 0.0002;
                Msg q;
                q.kind = 5;
                q.cam = m.cam;
                q.flag = true;
                q.data = net::encode(net::PongMsg{p.seq, p.t1, t2, t3});
                q.at = m.at + 0.0002 + c_.netLatency + (c_.netJitter > 0 ? J(netRng_) : 0);
                post(std::move(q));
            } else {
                net::PongMsg p;
                if (!net::decode(m.data, p)) break;
                u.sync.add(p.t1, p.t2, p.t3, m.at);
                if (u.sync.samples() >= 8)   // после первых обменов: до них смещение по одному-двум, грубое
                    st_.clockErrMs = std::max(st_.clockErrMs, 1000 * std::fabs(u.sync.offset() - u.cfg.clock));
            }
            break;
        }
        }
    }
}

// поворотка получила цели установки
void Emulator::setTargets(double t, std::vector<WideTarget> v) {
    targets_ = std::move(v);
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
    if (wu_.size() > 1) wideStats(t);
}

// Несколько широких: цель (тело по правде) перешла от камеры к камере — с тем же номером цели или нет;
// у одного тела два номера одновременно (не объединились); точная на стыке
void Emulator::wideStats(double t) {
    const double dt = lastSnapT_ >= 0 ? std::min(0.5, t - lastSnapT_) : 0;
    lastSnapT_ = t;
    std::vector<std::pair<int, const WideTarget*>> byBody;
    for (const auto& w : targets_)
        if (w.truth) byBody.push_back({w.truth, &w});
    std::sort(byBody.begin(), byBody.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second->id < b.second->id;
    });
    for (size_t i = 0; i < byBody.size();) {
        size_t j = i;
        while (j < byBody.size() && byBody[j].first == byBody[i].first) ++j;
        const int body = byBody[i].first;
        if (j - i > 1) st_.dupTime += dt;
        // номер — тот, что был у тела (если он ещё есть), иначе старший
        const WideTarget* w = byBody[i].second;
        BodySeen* bs = nullptr;
        for (auto& b : bodySeen_)
            if (b.body == body) bs = &b;
        if (bs)
            for (size_t q = i; q < j; ++q)
                if (byBody[q].second->id == bs->gid) w = byBody[q].second;
        if (bs && t - bs->t < fusion_.config().coastS + 1) {
            if (w->cam != bs->cam) {
                if (w->id == bs->gid) ++st_.handoffOk;
                else ++st_.handoffBroken;
                // точная вела это тело — продолжит ли через 1 с
                if (body == assignedBody_ && state_ == NState::Track) {
                    ++st_.seamTrack;
                    seamCheck_.push_back({t + 1.0, body});
                }
            } else if (w->id != bs->gid) {
                ++st_.renumbered;
            }
        }
        if (bs) *bs = {body, w->id, w->cam, t};
        else bodySeen_.push_back({body, w->id, w->cam, t});
        i = j;
    }
    bodySeen_.erase(std::remove_if(bodySeen_.begin(), bodySeen_.end(), [&](const BodySeen& b) { return t - b.t > 30; }),
                    bodySeen_.end());
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
    birdT_ = 0;
    nBird_ = 0;
    shapeE_.clear();
    shapeId_ = shapeBirdN_ = shapeLag_ = 0;
    shapeEv_ = 0;
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
    if (!c_.mountIdeal) {   // поворотка как на железе: задержки, рывок, люфт, упругость
        stepNarrowReal(t, dt);
        return;
    }
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
    processNarrow(t, dt, truth, enc, t);
}

// Кадр точной (снят в t при положении камеры truth; enc — где она по знанию поворотки) — наш алгоритм
// на нём. tNow — модельное время сейчас (кадр мог дойти с задержкой)
void Emulator::processNarrow(double t, double dt, const Pose& truth, const Pose& enc, double tNow) {
    const WideTarget* wt = nullptr;
    if (!nOnly_)
        for (const auto& x : targets_)
            if (x.id == assigned_) wt = &x;
    const bool useN = haveNrep_ && (state_ == NState::Track || t - tLastSeen_ < (nOnly_ ? c_.narrowLostHold : 0.6));
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
    // поворотка: насколько вычтенный трекером сдвиг кадра расходится с настоящим, и её положение — с правдой
    // (при ведении цели — тогда это и важно)
    if (havePrev_ && state_ == NState::Track) {
        double tx, ty;
        pose_shift(narrowCam_, prevTruth_, truth, narrowCam_.w * 0.5, narrowCam_.h * 0.5, tx, ty);
        st_.motionSq += (tx - dx) * (tx - dx) + (ty - dy) * (ty - dy);
        ++st_.motionN;
        const double pe = ang(enc.az, enc.el, truth.az, truth.el);
        st_.poseSq += pe * pe;
        ++st_.poseN;
    }
    prevTruth_ = truth;
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
                st_.slewSum += tNow - tAssign_;
                ++st_.slewN;
            }
        } else if (nOnly_ && !winOn_ && t - tLastSeen_ > c_.narrowLostHold) {
            // цель была только у точной, та её потеряла, окно ушло из кадра, а память кончилась: искать
            // больше негде (без этого точная навсегда оставалась в повороте к окну, которого нет)
            event(t, "точная потеряла цель " + std::to_string(assigned_) + ", широкая её не видит");
            assign(t, 0, "");
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
                             tNow - tAssign_, who == assignedBody_ && who ? "" : " — НЕ та цель");
                    event(t, b);
                }
            if (!gotThis_) {
                gotThis_ = true;
                tGot_ = t;
                ++st_.acquired;
                st_.acqSum += tNow - tAssign_;
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
            if (c_.birdMemory) {   // накопленное сгорает вдвое за birdHalf секунд
                const double burn = std::exp2(-dt / std::max(0.05f, c_.birdHalf));
                shapeEv_ *= burn;
                birdT_ *= burn;
            }
            bool shapeBird = false;
            if (c_.narrowShape) {
                if (shapeId_ != me->id) {   // другой трек — та же цель, но ряд начинаем заново
                    shapeE_.clear();
                    shapeId_ = me->id;
                    shapeBirdN_ = shapeLag_ = 0;
                    shapeEv_ = 0;
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
                        if (c_.birdMemory) {
                            // пачка взмахов, пауза (планирует), снова пачка: накопленное не обнуляется, а сгорает
                            const double fr = 1.0 / c_.narrowFps;
                            if (flap) shapeEv_ += fr;
                            else if (amp >= c_.shapeAmp && R >= c_.shapeRWeak && lag && std::abs(lag - shapeLag_) <= 1) shapeEv_ += c_.shapeWeakW * fr;
                            shapeLag_ = lag;
                            shapeBird = shapeEv_ >= memLevel(confK(t) * c_.shapeHits * fr);
                        } else {
                            shapeBird = shapeBirdN_ >= (nConf_ ? 2 : 1) * c_.shapeHits;
                        }
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
                                  (c_.birdMemory ? shapeEv_ < 0.25 * memLevel(c_.shapeHits / c_.narrowFps) && birdT_ < 0.25 * memLevel(c_.vetoT)
                                                 : shapeBirdN_ == 0) && !shapeWait;
            if (!nConf_ && droneNow && t - tTrackStart_ >= c_.confirmT) {
                nConf_ = true;
                tConf_ = t;
                ++st_.confirmed;
                if (whoBird) ++st_.confirmedBird;
                event(t, "точная: цель " + std::to_string(assigned_) + " — дрон");
            }
            if (me->cls == TargetClass::Bird) {
                if (tBird_ < 0) tBird_ = t;
                birdT_ += dt;
            } else {
                tBird_ = -1;
            }
            const double birdFor = tBird_ >= 0 ? t - tBird_ : 0;
            // цель выбрал оператор: отсеиваем, только если точная уверена вдвое (вдвое больше оценок силуэта или
            // вдвое дольше птица у классификатора)
            // доли до отсева: силуэт и классификатор (1 — отсеять). Выбранную вручную — при вдвое большей
            double fShape, fCls;
            const double k = manualPick_ ? 2 : 1;
            if (c_.birdMemory) {
                fShape = shapeEv_ / memLevel(k * confK(t) * c_.shapeHits / c_.narrowFps);
                fCls = birdT_ / memLevel(k * c_.vetoT);
            } else {
                fShape = (double)shapeBirdN_ / (k * (nConf_ ? 2 : 1) * c_.shapeHits);
                fCls = birdFor / (k * c_.vetoT);
            }
            const bool birdSure = manualPick_ ? c_.vetoManual && (fShape >= 1 || fCls >= 1) : shapeBird || fCls >= 1;
            nBird_ = (float)std::min(1.0, std::max(fShape, fCls));
            if (c_.narrowVeto && birdSure) {
                bool bird = false;
                for (const Body& b : bodies_)
                    if (b.id == who) bird = b.bird;
                ++(bird ? st_.vetoBird : who ? st_.vetoDrone : st_.vetoNone);
                if (bird) st_.vetoDelay += t - tGot_;
                if (shapeBird) ++st_.vetoShape;
                veto_.push_back({nrep_, t + c_.vetoCool});
                if (c_.vetoToWide && !nOnly_ && assigned_) markWideBird(t, assigned_);
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
                // калибровка камеры ещё не сошлась: сдвиг её направлений — общий для всех целей, а не шум
                // сообщения; допуск на него checkBias его СКО (иначе 6 СКО калибровки 0.5° — допуск 3°, и точная
                // сидит на помехе рядом с целью). Одна широкая — без калибровки, как было
                double bias = 0;
                if (wt->bias > 0) {
                    bias = c_.checkBias * wt->bias;
                    wsg = std::sqrt(std::max(0.0, wsg * wsg - (double)wt->bias * wt->bias));
                }
                // крупная цель: широкая указывает её середину, а захват точной может быть у края силуэта
                const double tol = (c_.wideCheck ? std::max(0.5, 6.0 * wsg) : std::max(1.5, 6.0 * wsg)) + bias +
                                   (c_.narrowSilhouette ? 0.5 * nrep_.size_deg : 0.0);
                off = ang(waz, wel, nrep_.az, nrep_.el) > tol && t - tTrackStart_ > (c_.wideCheck ? 0.3 : 1.0);
                // сдвиг калибровки у всех сообщений камеры один и тот же: точная, взявшая цель, видит её на том же
                // смещении от прогноза широкой, что и в начале. Смещение поползло (трек точной сползает с цели на
                // помеху) — заметно с обычным допуском, без запаса на калибровку. Пока точная не подтвердила дрон,
                // пар для калибровки нет, и сдвиг не меняется; потом калибровка его убирает — сверка по началу
                // захвата уже не годится. Сообщения перешли к другой широкой — у неё свой сдвиг, смещение заново
                if (wt->bias > 0 && !nConf_) {
                    const double cosE = std::cos(deg2rad(nrep_.el));
                    const double dx = az_diff(nrep_.az, waz) * cosE, dy = nrep_.el - wel;
                    if (offTrack_ != tTrackStart_ || offCam_ != wt->cam) {
                        offRefT_ = offTrack_ != tTrackStart_ ? tTrackStart_ : t;   // откуда считать смещение
                        offTrack_ = tTrackStart_;
                        offCam_ = wt->cam;
                        offN_ = 0;
                        offX_ = offY_ = 0;
                    }
                    if (t - offRefT_ <= 0.15) {   // смещение — по первым кадрам
                        offX_ = (offX_ * offN_ + dx) / (offN_ + 1);
                        offY_ = (offY_ * offN_ + dy) / (offN_ + 1);
                        ++offN_;
                    } else if (offN_ > 0 && std::hypot(dx - offX_, dy - offY_) > tol - bias && t - tTrackStart_ > 0.3) {
                        off = true;
                    }
                }
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
    mountFeed(t);
ui:
    // точная ведёт цель: объединению широких — где она (прогноз между секторами, калибровка широких)
    if (wu_.size() > 1 && state_ == NState::Track && haveNrep_ && assigned_ && t - tLastSeen_ < 1e-6) {
        bool listed = false;
        for (const auto& w : targets_) listed = listed || w.id == assigned_;
        // точная держит цель сама, а номер у широких — уже на другом: не путать объединение
        if (!nOnly_ || !listed) {
            const bool sure = nConf_ && !nOnly_ && t - tTrackStart_ > 0.5;
            if (c_.deploy == Deploy::OneDevice) {   // объединение — на устройстве широких: по связи
                Msg m;
                m.kind = 4;
                m.id = assigned_;
                m.rep = nrep_;
                m.flag = sure;
                m.at = t + c_.netLatency;
                post(std::move(m));
            } else {
                fusion_.observeReference(assigned_, nrep_, sure);
            }
        }
    }
    // от какой широкой сообщения о назначенной цели
    primaryCam_ = -1;
    for (const auto& w : targets_)
        if (w.id == assigned_ && !nOnly_) primaryCam_ = w.cam;
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
    view_.nBird = state_ == NState::Track ? nBird_ : 0.f;
    view_.nOnly = nOnly_;
    view_.axis = truth;
    view_.cmd = cmdPose_;
    view_.rateAz = (float)mount_.vaz;
    view_.rateEl = (float)mount_.vel;
    view_.believed = enc;
    if (!c_.mountIdeal) {
        view_.rateAz = msim_.rate(0);
        view_.rateEl = msim_.rate(1);
        view_.accAz = msim_.accel(0);
        view_.accEl = msim_.accel(1);
        view_.narrowLagFrames = std::max(0, (int)nq_.size() - 1);   // кроме этого, уже дошедшего
    }
    view_.win = win_;
    view_.winOn = winOn_;
    view_.widePose = wu_[0]->pose;
    // по широким: проекции на кадр каждой (по паспорту и калибровке — как видит алгоритм; правда — как на самом деле)
    if (view_.wides.size() != wu_.size()) view_.wides.resize(wu_.size());
    for (size_t k = 0; k < wu_.size(); ++k) {
        const WideUnit& w = *wu_[k];
        View::WideView& wv = view_.wides[k];
        wv.pose = w.pose;
        wv.primary = (int)k == primaryCam_;
        wv.gid.clear();
        if (wv.res)
            for (const auto& x : wv.res->tracks) {
                const int g = fusion_.globalOf((int)k, x.id);
                if (g) wv.gid.push_back({x.id, g});
            }
        wv.nWu = wv.nWv = -1;
        double u, v;
        if (haveNrep_ && assigned_ && fusion_.projectTo((int)k, nrep_.az, nrep_.el, u, v)) {
            wv.nWu = (float)u;
            wv.nWv = (float)v;
        }
        wv.vetoW.clear();
        for (const auto& vt : veto_) {
            double az, el, sg;
            predict(vt.first, t, az, el, sg);
            if (fusion_.projectTo((int)k, az, el, u, v)) wv.vetoW.push_back({(float)u, (float)v});
        }
        wv.truth.clear();
        for (const Body& b : bodies_) {
            if (b.hidden) continue;
            if (project(w.cam, w.truePose, vsub(b.at(t), w.pos), u, v)) wv.truth.push_back({(float)u, (float)v, b.id, b.bird});
        }
        wv.footprint.clear();
        const double W = narrowCam_.w, H = narrowCam_.h;
        const double pts[][2] = {{0, 0}, {W / 2, 0}, {W, 0}, {W, H / 2}, {W, H}, {W / 2, H}, {0, H}, {0, H / 2}};
        for (const auto& p : pts) {
            double az, el;
            azel_from_dir(unproject(narrowCam_, truth, p[0], p[1]), az, el);
            if (fusion_.projectTo((int)k, az, el, u, v)) wv.footprint.push_back({(float)u, (float)v});
        }
        wv.calibSigma = (float)fusion_.calib((int)k).sigmaDeg;
        wv.calibPairs = fusion_.calib((int)k).used;
        if (wu_.size() > 1 && (++uiCalibN_ % 30 == 0 || wv.calibErr == 0)) wv.calibErr = (float)calibTruthDeg(w);
    }
    {
        const auto& fc = fusion_.counters;
        st_.fusionCreated = fc.created;
        st_.fusionHandoffs = fc.handoffs;
        st_.fusionReacquired = fc.reacquired;
        st_.fusionMerged = fc.merged;
    }
    // первая широкая — в прежних полях (интерфейс одной широкой)
    {
        const View::WideView& w0 = view_.wides[0];
        view_.nWu = w0.nWu;
        view_.nWv = w0.nWv;
        view_.vetoW = w0.vetoW;
        view_.footprint = w0.footprint;
    }
    view_.truth.clear();
    for (const Body& b : bodies_) {
        if (b.hidden) continue;
        View::TruthPt p{};
        const Vec3 d = b.at(t);
        double u, v;
        p.id = b.id;
        p.bird = b.bird;
        p.wu = p.wv = -1;
        if (project(wu_[0]->cam, wu_[0]->truePose, vsub(d, wu_[0]->pos), u, v)) {
            p.wu = (float)u;
            p.wv = (float)v;
        }
        p.inNarrow = project(narrowCam_, truth, d, u, v) && in_frame(narrowCam_, u, v);
        p.nu = (float)u;
        p.nv = (float)v;
        view_.truth.push_back(p);
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
    view_.st.mount = msim_.stats();
    view_.st.biasAz = (float)wbias_.az;
    view_.st.biasEl = (float)wbias_.el;
    view_.st.slipAz = (float)msim_.slip(0);
    view_.st.slipEl = (float)msim_.slip(1);
    view_.st.simT = tNow + dt;   // шаг [t, t+dt] уже учтён в долях времени
}

// Поворотка как на железе: кадр снят сейчас (середина выдержки), до алгоритма дойдёт через camLatency;
// алгоритм обрабатывает те кадры, что уже дошли; команда уходит контроллеру двигателя и дойдёт через
// cmdLatency; физика поворотки — на шаг вперёд
void Emulator::stepNarrowReal(double t, double dt) {
    const double ex = std::max(0.f, c_.exposure);
    NFrame f;
    f.t = t - 0.5 * ex;
    f.ready = t + std::max(0.f, c_.camLatency);
    f.truth = msim_.camera(f.t);
    f.bel = msim_.believed(f.t);
    const Pose pa = msim_.camera(t - ex), pb = msim_.camera(t);
    const double r0 = now_s();
    renderNarrow(f.t, f.truth, f.px, ex > 0 ? &pa : nullptr, ex > 0 ? &pb : nullptr, ex);
    st_.renderNarrowMs += (now_s() - r0) * 1000;
    // смаз назначенной цели за выдержку
    if (state_ == NState::Track && assignedBody_ && ex > 0)
        for (const Body& b : bodies_)
            if (b.id == assignedBody_) {
                double ua, va, ub, vb;
                if (project(narrowCam_, pa, b.at(t - ex), ua, va) && project(narrowCam_, pb, b.at(t), ub, vb)) {
                    st_.blurSum += std::hypot(ub - ua, vb - va);
                    ++st_.blurN;
                }
            }
    nq_.push_back(std::move(f));
    while (!nq_.empty() && nq_.front().ready <= t + 1e-9) {
        NFrame& g = nq_.front();
        nframe_.swap(g.px);
        Pose enc = g.bel;
        if (c_.mountCtl == 1 && c_.encoder == 0 && c_.wideBias) {   // поправка ухода нуля по широкой
            enc.az = (float)az_norm(enc.az + wbias_.az);
            enc.el = (float)(enc.el + wbias_.el);
        }
        mount_.az = g.truth.az, mount_.el = g.truth.el;   // ось на момент кадра (ошибка оси — по ней)
        processNarrow(g.t, dt, g.truth, enc, t);
        nq_.pop_front();
    }
    controlReal(t);
    msim_.advance(t, dt, state_ == NState::Track);
    const Pose now = msim_.camera(t + dt);   // ось сейчас: выбор ближайшей цели, поворот в событиях
    mount_.az = now.az, mount_.el = now.el;
    mount_.vaz = msim_.rate(0), mount_.vel = msim_.rate(1);
}

// Команда поворотке на момент t: прежний закон — куда смотреть через полтора кадра и скорость цели;
// плавный — уставка по фильтру цели на момент, когда команда дойдёт до двигателя
void Emulator::controlReal(double t) {
    const WideTarget* wt = nullptr;
    if (!nOnly_)
        for (const auto& x : targets_)
            if (x.id == assigned_) wt = &x;
    const bool useN = haveNrep_ && (state_ == NState::Track || t - tLastSeen_ < (nOnly_ ? c_.narrowLostHold : 0.6));
    const TargetReport* rep = useN ? &nrep_ : wt ? &wt->rep : nullptr;
    Setpoint sp;
    sp.t = t;
    if (c_.mountCtl == 0) {
        const double lead = 1.5 / c_.narrowFps;
        if (rep) {
            cmdPose_ = aim(*rep, t, c_.feedForward ? lead : 0);
            ffAz_ = c_.feedForward ? rep->vaz : 0;
            ffEl_ = c_.feedForward ? rep->vel : 0;
        } else {
            ffAz_ = ffEl_ = 0;
        }
        sp.az = cmdPose_.az, sp.el = cmdPose_.el, sp.vaz = ffAz_, sp.vel = ffEl_;
        sp.hold = false;
    } else {
        if (!assigned_) {
            gf_.clear();
            gfAssignN_ = -1;
        } else if ((st_.assignments != gfAssignN_ || !gf_.active()) && rep) {
            // новая цель (не та же, что широкая снова увидела под другим номером): фильтр — с сообщения о ней;
            // сколько до неё поворачиваться — прогноз
            gf_.reset(*rep);
            gfAssignN_ = st_.assignments;
            gfWideT_ = rep == &nrep_ ? -1 : rep->t;
            const double tol = 0.45 * c_.narrowFov * c_.narrowH / std::max(1, c_.narrowW);
            etaPred_ = slew_eta(msim_.profile(0), msim_.profile(1), gf_, t, tol);
            etaT_ = tAssign_;
        }
        // пока точная цель не видит — по широкой (каждое её новое сообщение)
        if (gf_.active() && !useN && wt && wt->rep.t > gfWideT_ + 1e-9) {
            gf_.measure(wt->rep.t, wt->rep.az, wt->rep.el, wt->rep.sigma);
            gfWideT_ = wt->rep.t;
        }
        const bool bias = c_.encoder == 0 && c_.wideBias;
        const double baz = bias ? wbias_.az : 0, bel = bias ? wbias_.el : 0;
        if (useN && state_ != NState::Track) {
            // точная потеряла цель и ищет по последнему своему сообщению: туда же и поворотка
            sp = report_setpoint(nrep_, t, gcfg_.lead_cmd, baz, bel);
        } else if (!rep) {
            sp.hold = true;   // ни точная, ни широкая не знают, где цель: остановиться (как прежний закон)
        } else {
            sp = gimbal_setpoint(gf_, t, baz, bel);
        }
        if (!assigned_) sp.hold = true;
        if (!sp.hold) cmdPose_ = {(float)sp.az, (float)sp.el, 0};
    }
    // прогноз времени поворота против того, когда цель на самом деле оказалась в кадре
    if (etaT_ >= 0 && etaT_ == tAssign_ && tInView_ >= 0) {
        st_.etaSum += etaPred_;
        st_.etaErr += std::fabs((tInView_ - tAssign_) - etaPred_);
        ++st_.etaN;
        etaT_ = -1;
    }
    msim_.command(t, sp);
}

// Отметка точной — фильтру цели поворотки, невязка с широкой — в уход нуля (плавный регулятор); и как ровно
// поворотка ведёт цель — по правде, для замеров
void Emulator::mountFeed(double t) {
    bool seen = false;
    if (state_ == NState::Track && assignedBody_)
        for (const Body& b : bodies_)
            if (b.id == assignedBody_) {
                double u, v;
                if (!project(narrowCam_, prevTruth_, b.at(t), u, v)) break;
                seen = true;
                const double du = u - narrowCam_.w * 0.5, dv = v - narrowCam_.h * 0.5;
                st_.centerSq += du * du + dv * dv;
                ++st_.centerN;
                const bool next = jitBody_ == assignedBody_ && jitK_ > 0 && std::fabs(t - jitT_ - 1.0 / c_.narrowFps) < 0.25 / c_.narrowFps;
                if (!next) {
                    jitBody_ = assignedBody_;
                    jitK_ = 0;
                } else if (jitK_ >= 2) {
                    const double ju = u - 2 * jitU_[0] + jitU_[1], jv = v - 2 * jitV_[0] + jitV_[1];
                    st_.jitterSq += ju * ju + jv * jv;
                    ++st_.jitterN;
                }
                jitU_[1] = jitU_[0], jitV_[1] = jitV_[0];
                jitU_[0] = u, jitV_[0] = v;
                jitK_ = std::min(jitK_ + 1, 2);
                jitT_ = t;
            }
    if (!seen) jitK_ = 0;
    if (c_.mountIdeal || c_.mountCtl != 1) return;
    const bool fresh = state_ == NState::Track && haveNrep_ && nrep_.t == t && tLastSeen_ == t;
    if (!fresh) return;
    gf_.measure(t, nrep_.az, nrep_.el, nrep_.sigma);
    // уход нуля без датчиков: широкая видит ту же цель (точечную — у крупной широкая и точная могут
    // указывать разные её места)
    if (c_.encoder != 0 || !c_.wideBias || nOnly_ || t - tTrackStart_ < 0.3 || nrep_.size_deg > 0.1f) return;
    for (const auto& w : targets_)
        if (w.id == assigned_ && w.rep.t > biasT_ + 1e-9) {
            double waz, wel, wsg;
            predict(w.rep, t, waz, wel, wsg);
            const double dtb = biasT_ < 0 ? 1.0 / c_.wideFps : std::min(0.5, w.rep.t - biasT_);
            wbias_.update(az_diff(waz, nrep_.az), wel - nrep_.el, std::max(wsg, 0.005), dtb, gcfg_);
            biasT_ = w.rep.t;
        }
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
        const Pose& p0 = wu_[0]->truePose;
        char hello[600];
        snprintf(hello, sizeof(hello),
                 "{\"wide\": [%d, %d, %.3f, %.3f], \"fine\": [%d, %d, %.3f, %.3f], \"seed\": %u, \"drones\": %d, "
                 "\"birds\": %d, \"range\": [%.1f, %.1f], \"types\": {\"quad\": %.3f, \"delta\": %.3f, \"plane\": %.3f}, "
                 "\"az\": %.3f, \"el\": %.3f, \"sector\": %s",
                 c_.wideW, c_.wideH, c_.wideFov, c_.wideFps, c_.narrowW, c_.narrowH, c_.narrowFov, c_.narrowFps, c_.seed,
                 c_.drones, c_.birds, c_.rangeMin, c_.rangeMax, c_.quadShare, c_.deltaShare, c_.planeShare, p0.az, p0.el,
                 c_.sector.empty() ? "false" : "true");
        std::string h = hello;
        // вторая и следующие широкие: камеры сцены 2, 3...; цели сцена держит в поле любой из широких
        if (wu_.size() > 1 || wu_[0]->pos.x != 0 || wu_[0]->pos.y != 0 || wu_[0]->pos.z != 0) {
            h += ", \"wides\": [";
            for (size_t k = 0; k < wu_.size(); ++k) {
                const WideUnit& w = *wu_[k];
                char b[160];
                snprintf(b, sizeof(b), "%s[%.3f, %.3f, %.3f, %.3f, %.3f]", k ? ", " : "", w.truePose.az, w.truePose.el, w.pos.x, w.pos.y,
                         w.pos.z);
                h += b;
            }
            h += "]";
        }
        std::string err;
        linkOpt_ = o;
        linkHelloBase_ = h;   // без закрывающей скобки: следом погода
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
    double t = 0;
    const double dtN = 1.0 / std::max(1.f, c_.narrowFps);
    const double wall0 = now_s();
    double simAtWall = 0, wallBase = wall0;
    float lastSpeed = speed_;
    const bool multi = wu_.size() > 1;
    {
        std::lock_guard<std::mutex> lk(vm_);
        view_.wides.assign(wu_.size(), View::WideView());
    }
    while (!stop_) {
        if (c_.stopAt > 0 && t >= c_.stopAt - 1e-9) break;   // замер: ровно столько модельного времени
        if (paused_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            wallBase = now_s();
            simAtWall = t;
            continue;
        }
        // широкие, у которых сейчас кадр (на одном устройстве — в такт, на своих — каждая в свой момент)
        std::vector<WideUnit*> due;
        for (auto& w : wu_)
            if (t >= w->nextT - 1e-9) due.push_back(w.get());
        bool waitFail = false;
        if (!due.empty()) {
            WideUnit& w0 = *wu_[0];
            const bool due0 = due[0] == &w0;
            if (due0 && scanReq_.exchange(false) && !scan_) {
                scan_ = true;
                scanStep_ = 0;
                scanAz0_ = w0.pose.az;
                scanEl0_ = w0.pose.el;
                event(t, "обзор 360°: маска неба по кругу");
            }
            if (due0 && scan_ && (!w0.lastLearning && w0.sincePose >= 3)) {   // этот вид выучен — дальше
                const float step = std::max(20.f, c_.wideFov * 0.5f);
                const int n = (int)std::ceil(360.f / step - 1e-3f);
                std::lock_guard<std::mutex> lk(poseM_);
                if (scanStep_ + 1 >= n) {
                    scan_ = false;
                    poseWant_.push_back({0, scanAz0_, scanEl0_});
                    event(t, "обзор 360° закончен");
                } else {
                    ++scanStep_;
                    poseWant_.push_back({0, std::fmod(scanAz0_ + scanStep_ * step, 360.f), scanEl0_});
                }
                ++poseReq_;
            }
            if (poseReq_ != poseSeen_) {   // широкую повернули кнопками (вступает с её кадра)
                std::vector<PoseCmd> cmds, rest;
                {
                    std::lock_guard<std::mutex> lk(poseM_);
                    cmds.swap(poseWant_);
                    poseSeen_ = poseReq_;
                }
                for (const PoseCmd& pc : cmds) {
                    if (pc.cam < 0 || pc.cam >= (int)wu_.size()) continue;
                    WideUnit& w = *wu_[pc.cam];
                    if (std::find(due.begin(), due.end(), &w) == due.end()) {
                        rest.push_back(pc);
                        continue;
                    }
                    w.pose = {pc.az, std::max(-10.f, std::min(80.f, pc.el)), 0};
                    w.truePose = {(float)az_norm(w.pose.az + w.cfg.errAz), w.pose.el + w.cfg.errEl, w.cfg.errRoll};
                    fusion_.setPose(w.k, w.pose);
                    if (!sceneMode_) prepareWide(w);
                    w.sincePose = 0;
                    w.lastLearning = true;
                    char b[200];
                    const std::string who = multi ? "широкая " + std::to_string(w.k + 1) : "широкая";
                    // маска неба для нового вида: из панорамы, если она его покрывает, иначе — учить заново
                    std::vector<uint8_t> m;
                    const float known = c_.panoMask && !scan_ && w.panoTw > 0 ? panoMaskFor(w, m) : 0.f;
                    if (known >= 0.85f) {
                        w.pipe.setExternalMask(std::move(m), w.panoTw, w.panoTh, (int)std::lround(c_.wideFps));
                        snprintf(b, sizeof(b), "%s повёрнута: азимут %.0f°, угол места %.0f°; маска из панорамы (%.0f%% вида)", who.c_str(),
                                 w.pose.az, w.pose.el, known * 100);
                    } else {
                        w.pipe.relearnMask();
                        snprintf(b, sizeof(b), "%s повёрнута: азимут %.0f°, угол места %.0f°; маска учится%s", who.c_str(), w.pose.az,
                                 w.pose.el, c_.panoMask && !scan_ ? " (панорама покрывает мало)" : "");
                    }
                    if (w.k == 0) {
                        std::lock_guard<std::mutex> lk(vm_);
                        view_.panoKnown = known;
                    }
                    event(t, b);
                }
                if (!rest.empty()) {   // остальным камерам — с их кадра
                    std::lock_guard<std::mutex> lk(poseM_);
                    poseWant_.insert(poseWant_.begin(), rest.begin(), rest.end());
                    ++poseReq_;
                }
            }
            // тела, вылетевшие из поля зрения всех широких или ушедшие далеко, — заново (в 3D-сцене их ведёт сцена)
            for (Body& b : bodies_) {
                if (sceneMode_) break;
                const Vec3 d = b.at(t);
                const double R = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
                double u, v, az, el;
                azel_from_dir(d, az, el);
                bool inView = false;
                for (const auto& w : wu_)
                    inView = inView || (project(w->cam, w->truePose, vsub(d, w->pos), u, v) && in_frame(w->cam, u, v, 30));
                if (!inView || R > c_.rangeMax * 1.7 || R < 60 || el < treeline(az) + 0.2) {
                    if (b.id == assignedBody_) event(t, "цель " + std::to_string(assigned_) + " вышла из зоны");
                    spawn(b, t, false);
                }
            }
            // подсказки широким: точная уверена в дроне и видит его — где он будет на этом кадре (там порог ниже,
            // и потерянную цель камера находит снова, а свою слабую не теряет); и цели соседних секторов, которые
            // по прогнозу уже в кадре этой камеры
            for (WideUnit* wp : due) {
                WideUnit& w = *wp;
                std::vector<FocusArea> cue;
                if (c_.wideCue && nConf_ && haveNrep_ && assigned_ && t - tLastSeen_ < 0.5) {
                    double az, el, sg, u, v;
                    predict(nrep_, t, az, el, sg);
                    if (fusion_.projectTo(w.k, az, el, u, v) && in_frame(w.cam, u, v)) {
                        const double wpd = pixel_deg(w.cam, w.pose, u, v);
                        cue.push_back({(float)u, (float)v, (float)std::max(12.0, std::min(60.0, (3 * sg + nrep_.size_deg) / wpd))});
                    }
                }
                if (multi && c_.crossCue) {
                    if (c_.deploy == Deploy::OneDevice) {
                        for (const auto& a : fusion_.cues(w.k, t)) cue.push_back({a.u, a.v, a.r});
                    } else if (w.cueNetT >= 0 && t - w.cueNetT < 0.5) {
                        cue.insert(cue.end(), w.cueNet.begin(), w.cueNet.end());
                    }
                }
                w.pipe.setCue(cue);
            }
            // кадры широких — их конвейерам; ждём, пока все обработают
            for (WideUnit* wp : due) {
                WideUnit& w = *wp;
                const double r0 = now_s();
                std::vector<uint8_t> buf;
                renderWide(w, t, buf);
                st_.renderWideMs += (now_s() - r0) * 1000;
                std::lock_guard<std::mutex> lk(w.m);
                w.frame.swap(buf);
                w.idx = w.n;
                w.ready = true;
                w.cv.notify_all();
            }
            for (WideUnit* wp : due) {
                WideUnit& w = *wp;
                std::shared_ptr<const Result> res;
                {
                    std::unique_lock<std::mutex> lk(w.m);
                    w.cv.wait_for(lk, std::chrono::seconds(10), [&] { return w.done >= w.idx || stop_; });
                    if (stop_) break;
                    if (w.done >= w.idx) res = w.res;
                }
                if (res) {
                    onWide(w, t, *res);
                    // панорамная маска: маска этого вида — в панораму (после обучения и раз в полсекунды)
                    const bool learned = w.lastLearning && !res->learning;
                    w.lastLearning = res->learning;
                    ++w.sincePose;
                    if (res->tilesX > 0) {
                        w.panoTw = res->tilesX;
                        w.panoTh = res->tilesY;
                        w.panoTile = res->tilePx;
                    }
                    if (c_.panoMask && !res->learning && !res->mask.empty() &&
                        (learned || w.frames % std::max(1, (int)(c_.wideFps / 2)) == 0)) {
                        panoLearn(w, *res);
                        std::lock_guard<std::mutex> lk(vm_);
                        view_.pano = pano_;
                    }
                    std::lock_guard<std::mutex> lk(vm_);
                    view_.scanning = scan_;
                    if (w.k == 0) view_.wide = res;
                    if (w.k < (int)view_.wides.size()) {
                        view_.wides[w.k].res = res;
                        view_.wides[w.k].pose = w.pose;
                    }
                    view_.status.clear();
                } else {
                    waitFail = true;
                }
                ++st_.wideFrames;
                ++w.n;
                w.nextT += 1.0 / std::max(1.f, c_.wideFps);
            }
            if (stop_) break;
            if (waitFail) {
                std::lock_guard<std::mutex> lk(vm_);
                view_.status = "широкая камера не ответила";
            }
            // одно устройство: цели установки после кадров всех его камер — поворотке через handoffLatency
            if (c_.deploy == Deploy::OneDevice) {
                Msg m;
                m.kind = 0;
                m.at = t + c_.handoffLatency;
                fusedSnapshot(t, m.snap);
                m.merges = fusion_.takeMerges();
                post(std::move(m));
            }
        }
        // устройство на камеру: сверка часов с каждой раз в секунду
        if (c_.deploy == Deploy::PerCamera)
            for (auto& wp : wu_) {
                WideUnit& w = *wp;
                if (t < w.nextPing) continue;
                w.nextPing = t + 1.0;
                std::exponential_distribution<double> J(1.0 / std::max(1e-6f, c_.netJitter));
                Msg m;
                m.kind = 5;
                m.cam = w.k;
                m.data = net::encode(net::PingMsg{(uint32_t)(t * 10), t});
                m.at = t + c_.netLatency + (c_.netJitter > 0 ? J(netRng_) : 0);
                post(std::move(m));
            }
        deliver(t);
        rebind(t);
        choose(t);
        stepNarrow(t, dtN);
        // точная вела цель на стыке камер: через секунду — всё ещё на ней?
        for (auto it = seamCheck_.begin(); it != seamCheck_.end();)
            if (t >= it->first) {
                if (state_ == NState::Track && assignedBody_ == it->second) ++st_.seamKept;
                it = seamCheck_.erase(it);
            } else {
                ++it;
            }
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
    // итог по камерам: калибровка (неуверенность и правда), объединение
    {
        const auto& fc = fusion_.counters;
        st_.fusionCreated = fc.created;
        st_.fusionHandoffs = fc.handoffs;
        st_.fusionReacquired = fc.reacquired;
        st_.fusionMerged = fc.merged;
        st_.calibDeg.clear();
        st_.calibSigma.clear();
        st_.calibPairs.clear();
        st_.camFrames.clear();
        for (const auto& w : wu_) {
            st_.camFrames.push_back(w->frames);
            st_.calibSigma.push_back((float)fusion_.calib(w->k).sigmaDeg);
            st_.calibPairs.push_back(fusion_.calib(w->k).used);
            st_.calibDeg.push_back((float)calibTruthDeg(*w));
        }
    }
    std::lock_guard<std::mutex> lk(vm_);
    view_.st.fusionCreated = st_.fusionCreated;
    view_.st.fusionHandoffs = st_.fusionHandoffs;
    view_.st.fusionReacquired = st_.fusionReacquired;
    view_.st.fusionMerged = st_.fusionMerged;
    view_.st.calibDeg = st_.calibDeg;
    view_.st.calibSigma = st_.calibSigma;
    view_.st.calibPairs = st_.calibPairs;
    view_.st.camFrames = st_.camFrames;
    view_.running = false;
}

// правда: насколько направление по камере (с калибровкой) отличается от настоящего — худшее по сетке кадра
double Emulator::calibTruthDeg(const WideUnit& w) const {
    double worst = 0;
    const double R = fusion_.config().rangeGuess;
    for (int j = 1; j < 4; ++j)
        for (int i = 1; i < 4; ++i) {
            const double u = w.cam.w * i / 4.0, v = w.cam.h * j / 4.0;
            const TargetReport r = make_report(w.cam, w.pose, 0, c_.wideFps, (float)u, (float)v, 0, 0, 0);
            const TargetReport ref = fusion_.toRef(w.k, r);
            const Vec3 d = unproject(w.cam, w.truePose, u, v);
            const Vec3 q{w.pos.x + d.x * R, w.pos.y + d.y * R, w.pos.z + d.z * R};
            worst = std::max(worst, angle_between(q, dir_from_azel(ref.az, ref.el)));
        }
    return worst;
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

void layout_wides(Config& c, int count, float step, float installErr, float clockErr, float phase) {
    std::mt19937 g(c.seed * 977u + 13);
    std::normal_distribution<float> N(0, 1);
    auto err = [&](WideCam& w) {
        w.errAz = w.errEl = w.errRoll = 0;
        if (installErr > 0) {
            w.errAz = installErr * N(g);
            w.errEl = installErr * N(g);
            w.errRoll = installErr * N(g);
        }
    };
    err(c.wide0);
    std::vector<WideCam> old = std::move(c.moreWide);
    c.moreWide.clear();
    for (int k = 1; k < count; ++k) {
        WideCam w;
        if (k - 1 < (int)old.size()) w.x = old[k - 1].x, w.y = old[k - 1].y, w.z = old[k - 1].z;
        w.az = (float)az_norm(c.wideAz + k * step);
        w.el = c.wideEl;
        err(w);
        // устройство на камеру: кадры не в такт, часы до сверки разные
        w.phase = phase >= 0 ? phase * k : std::fmod(0.031f * k, 1.f / std::max(1.f, c.wideFps));
        w.clock = clockErr * (k % 2 ? 1.f : -0.6f);
        c.moreWide.push_back(w);
    }
}

int run_cli(int argc, char** argv) {
    Config c;
    Emulator e;
    double seconds = 60;
    std::string json;
    bool verbose = true;
    // несколько широких
    int wides = 1;
    float wideStep = 110, installErr = 0, clockErr = 0.05f, phase = -1;
    std::vector<float> wideAz;
    std::vector<std::pair<int, Vec3>> camPos;
    std::vector<std::pair<int, Vec3>> camErr;
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
        else if (a == "--no-glide") c.birdGlide = false;
        else if (a == "--no-veto") c.narrowVeto = false;
        else if (a == "--no-shape") c.narrowShape = false;
        else if (a == "--no-veto-wide") c.vetoToWide = false;
        else if (a == "--with-birds") c.dronesOnly = false;   // широкая указывает и тех, кого считает птицами
        else if (a == "--shape") sscanf(val().c_str(), "%f,%f,%d,%f", &c.shapeAmp, &c.shapeR, &c.shapeHits, &c.shapeWin);
        else if (a == "--no-cue") c.wideCue = false;
        else if (a == "--no-next") c.nextOnBird = false;
        else if (a == "--no-veto-manual") c.vetoManual = false;
        else if (a == "--no-bird-memory") c.birdMemory = false;
        else if (a == "--bird-memory") sscanf(val().c_str(), "%f,%f,%f,%f,%f,%f", &c.shapeRWeak, &c.shapeWeakW, &c.birdHalf, &c.shapeConfK0, &c.shapeConfK, &c.shapeProveT);
        else if (a == "--drone-wobble") c.droneWobble = true;
        else if (a == "--legacy")
            c.narrowSilhouette = c.narrowHold = c.narrowVeto = c.wideCue = c.narrowShape = c.vetoToWide = c.nextOnBird = c.vetoManual = c.birdMemory = false;
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
        else if (a == "--ideal-mount") c.mountIdeal = true;   // поворотка как до 31.10 (бит в бит)
        else if (a == "--old-ctl") c.mountCtl = 0;
        else if (a == "--jerk") c.maxJerk = (float)atof(val().c_str());
        else if (a == "--gains") sscanf(val().c_str(), "%f,%f", &c.mountKp, &c.mountKv);
        else if (a == "--target-jerk") c.targetJerk = (float)atof(val().c_str());
        else if (a == "--cam-latency") c.camLatency = (float)atof(val().c_str());
        else if (a == "--cmd-latency") sscanf(val().c_str(), "%f,%f", &c.cmdLatency, &c.cmdJitter);
        else if (a == "--exposure") c.exposure = (float)atof(val().c_str());
        else if (a == "--step") c.stepDeg = (float)atof(val().c_str());
        else if (a == "--backlash") c.backlash = (float)atof(val().c_str());
        else if (a == "--no-backlash-comp") c.backlashComp = false;
        else if (a == "--resonance") sscanf(val().c_str(), "%f,%f", &c.resonanceHz, &c.damping);
        else if (a == "--slip") sscanf(val().c_str(), "%f,%d", &c.slipEvery, &c.slipSteps);
        else if (a == "--encoder") c.encoder = 1;
        else if (a == "--encoder-deg") c.encoderDeg = (float)atof(val().c_str());
        else if (a == "--no-wide-bias") c.wideBias = false;
        else if (a == "--pose-model") c.poseModel = atoi(val().c_str());
        else if (a == "--mount-model") sscanf(val().c_str(), "%f,%f", &c.modelHz, &c.modelDamping);
        else if (a == "--perfect-mount") {   // настоящий путь без несовершенств: только регулятор
            c.camLatency = c.cmdLatency = c.cmdJitter = c.exposure = c.stepDeg = c.backlash = c.resonanceHz = c.slipEvery = 0;
            c.modelHz = 0;
        }
        else if (a == "--az") c.wideAz = (float)atof(val().c_str());
        else if (a == "--el") c.wideEl = (float)atof(val().c_str());
        else if (a == "--wides") wides = std::max(1, std::min(8, atoi(val().c_str())));
        else if (a == "--wide-step") wideStep = (float)atof(val().c_str());
        else if (a == "--wide-az") {
            std::string v = val();
            for (size_t p = 0; p < v.size();) {
                const size_t q = v.find(',', p);
                wideAz.push_back((float)atof(v.substr(p, q - p).c_str()));
                if (q == std::string::npos) break;
                p = q + 1;
            }
        } else if (a == "--cam-pos" || a == "--cam-err") {
            int k = 0;
            double x = 0, y = 0, z = 0;
            if (sscanf(val().c_str(), "%d:%lf,%lf,%lf", &k, &x, &y, &z) == 4) (a == "--cam-pos" ? camPos : camErr).push_back({k - 1, Vec3{x, y, z}});
        } else if (a == "--install-err") installErr = (float)atof(val().c_str());
        else if (a == "--deploy") c.deploy = val() == "net" ? Deploy::PerCamera : Deploy::OneDevice;
        else if (a == "--net-latency") c.netLatency = (float)atof(val().c_str());
        else if (a == "--net-jitter") c.netJitter = (float)atof(val().c_str());
        else if (a == "--clock-err") clockErr = (float)atof(val().c_str());
        else if (a == "--phase") phase = (float)atof(val().c_str());
        else if (a == "--no-clock-sync") c.clockSync = false;
        else if (a == "--udp") c.netUdp = true;
        else if (a == "--no-fusion") c.fusion = false;
        else if (a == "--no-calib") c.fusionCalib = false;
        else if (a == "--calib-init") c.calibInit = (float)atof(val().c_str());
        else if (a == "--check-bias") c.checkBias = (float)atof(val().c_str());
        else if (a == "--no-cross-cue") c.crossCue = false;
        else if (a == "--seam") c.seamShare = (float)atof(val().c_str());
        else {
            printf("dronolok_app --twocam [--seconds 60] [--drones 3] [--birds 1] [--no-glide] [--seed 1] [--range 100,750]\n"
                   "  [--speed 6,22] [--clouds 0.35] [--noise 2] [--visibility 6000] [--size 0.5] [--types 0.5,0.25,0.25] [--wide 4608x3456] [--wide-fps 14] [--fisheye] [--fov 5 (x4; 20 — без зума)]\n"
                   "  [--fps 60] [--rate 30] [--accel 150] [--latency 0.06] [--dwell 6]\n"
                   "  [--switch manual|queue|nearest] [--json out.json] [--quiet]\n"
                   "  сравнение доработок точной: [--image-motion] [--no-ff] [--no-window] [--no-lock-focus]\n"
                   "  [--min-hold 0.08] [--give-up 3] [--cool-down 10]\n"
                   "  обмен между камерами: [--no-silhouette] [--no-hold] [--no-veto] [--no-cue] [--no-shape] [--shape амп,автокорр,оценок,окно] [--no-veto-wide] [--no-next] [--no-veto-manual] [--no-bird-memory] [--bird-memory rслаб,вес,полураспад_с,K_подтв,K_проверен,проверка_с] [--drone-wobble] [--with-birds] [--legacy — всё выключено]\n"
                   "  3D-сцена Godot вместо синтетического неба: [--scene sim/scenes/city_sectors.json] [--sector highrise]\n"
                   "  [--godot путь] [--az 0] [--el 32]\n"
                   "  несколько широких: [--wides 2] [--wide-step 110 (между осями, °) | --wide-az 0,110] [--cam-pos N:x,y,z (м)]\n"
                   "  [--cam-err N:аз,у.м.,поворот (ошибка установки, °) | --install-err СКО°] [--seam 0.6 (доля целей у стыка)]\n"
                   "  [--deploy one|net (одно устройство | устройство на камеру)] [--net-latency 0.004] [--net-jitter 0.003]\n"
                   "  [--clock-err 0.05 (часы камер, с)] [--phase с] [--no-clock-sync] [--udp] [--no-fusion] [--no-calib]\n"
                   "  [--calib-init 0.5] [--check-bias 3] [--no-cross-cue]\n"
                   "  погода и время суток сцены: [--weather clear|cumulus|cloudy|overcast|haze|mist|fog|rain|storm|snow]\n"
                   "  [--time 19:30] [--time-rate 60] [--date 09-15] [--lat 55.75] [--scene-clouds 0.8] [--scene-visibility 3000]\n"
                   "  [--rain 0.5] [--snow 0.5]\n"
                   "  поворотка (по умолчанию — как на железе, плавный регулятор): [--ideal-mount — как до 31.10] [--old-ctl — прежний закон]\n"
                   "  [--jerk 3000] [--gains 8,32] [--target-jerk 2] [--cam-latency 0.033] [--cmd-latency 0.01,0.002] [--exposure 0.001] [--step 0.01125]\n"
                   "  [--backlash 0.05] [--no-backlash-comp] [--resonance 12,0.15] [--slip раз_в_с,шагов] [--encoder] [--encoder-deg 0.01]\n"
                   "  [--no-wide-bias] [--perfect-mount — без задержек, люфта, шагов, упругости]\n");
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
    // широкие: вторая и следующие — через wideStep по азимуту (или по списку), ошибки установки по seed
    {
        if (!wideAz.empty()) {
            c.wideAz = wideAz[0];
            wides = std::max(wides, (int)wideAz.size());
        }
        layout_wides(c, wides, wideStep, installErr, clockErr, phase);
        for (int k = 1; k < wides && k < (int)wideAz.size(); ++k) c.moreWide[k - 1].az = (float)az_norm(wideAz[k]);
        for (const auto& p : camPos) {
            WideCam* w = p.first == 0 ? &c.wide0 : p.first - 1 < (int)c.moreWide.size() ? &c.moreWide[p.first - 1] : nullptr;
            if (w) w->x = (float)p.second.x, w->y = (float)p.second.y, w->z = (float)p.second.z;
        }
        for (const auto& p : camErr) {
            WideCam* w = p.first == 0 ? &c.wide0 : p.first - 1 < (int)c.moreWide.size() ? &c.moreWide[p.first - 1] : nullptr;
            if (w) w->errAz = (float)p.second.x, w->errEl = (float)p.second.y, w->errRoll = (float)p.second.z;
        }
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
    const double jit = s.jitterN ? std::sqrt(s.jitterSq / s.jitterN) : 0, mot = s.motionN ? std::sqrt(s.motionSq / s.motionN) : 0;
    const double poseE = s.poseN ? std::sqrt(s.poseSq / s.poseN) : 0, blur = s.blurN ? s.blurSum / s.blurN : 0;
    const double ctr = s.centerN ? std::sqrt(s.centerSq / s.centerN) : 0;
    const double accR = s.mount.n ? std::sqrt(s.mount.accSq / (2.0 * s.mount.n)) : 0, jrkR = s.mount.n ? std::sqrt(s.mount.profJerkSq / (2.0 * s.mount.n)) : 0;
    printf("поворотка (%s, %s): цель от центра %.1f пикс., дрожание %.2f пикс./кадр, ошибка сдвига кадра %.2f пикс., положения %.4f°,\n"
           "  смаз %.2f пикс.; камера при ведении — ускорение %.1f °/с² (до %.0f), рывок профиля %.0f °/с³ (до %.0f), скорость до %.1f °/с;\n"
           "  пропусков шагов %d (%.3f°/%.3f°), поправка по широкой %.3f°/%.3f°; поворот к цели — прогноз %.2f с, ошибка %.2f с\n",
           c.mountIdeal ? "идеальная" : "как на железе", !c.mountIdeal && c.mountCtl ? "плавный регулятор" : "прежний закон", ctr, jit, mot, poseE, blur, accR,
           s.mount.peakAcc, jrkR, s.mount.peakProfJerk, s.mount.peakRate, s.mount.slips, s.slipAz, s.slipEl, s.biasAz, s.biasEl,
           s.etaN ? s.etaSum / s.etaN : 0, s.etaN ? s.etaErr / s.etaN : 0);
    printf("обработка: широкая %.1f мс/кадр, точная %.2f мс/кадр; отрисовка широкой %.1f, точной %.1f мс/кадр\n",
           s.wideFrames ? s.wideMs / s.wideFrames : 0, s.narrowFrames ? s.narrowMs / s.narrowFrames : 0,
           s.wideFrames ? s.renderWideMs / s.wideFrames : 0, s.narrowFrames ? s.renderNarrowMs / s.narrowFrames : 0);
    const int nw = c.wideCount();
    if (nw > 1) {
        printf("широких %d (%s): цель перешла к другой камере %d раз — с тем же номером %d, с новым %d; у одной цели два номера %.1f с;\n"
               "  та же камера нашла цель под новым номером %d; точная вела цель на стыке %d раз, через 1 с на ней %d\n",
               nw, c.deploy == Deploy::OneDevice ? "одно устройство" : "устройство на камеру", s.handoffOk + s.handoffBroken, s.handoffOk,
               s.handoffBroken, s.dupTime, s.renumbered, s.seamTrack, s.seamKept);
        printf("объединение: целей %d, переходов %d, найдено снова после пропажи %d, слияний %d; калибровка:", s.fusionCreated,
               s.fusionHandoffs, s.fusionReacquired, s.fusionMerged);
        for (size_t k = 0; k < s.calibDeg.size(); ++k)
            printf(" камера %zu — ошибка позы %.3f°, неуверенность %.3f°, пар %d%s", k + 1, s.calibDeg[k],
                   k < s.calibSigma.size() ? s.calibSigma[k] : 0.f, k < s.calibPairs.size() ? s.calibPairs[k] : 0,
                   k + 1 < s.calibDeg.size() ? ";" : "\n");
        if (c.deploy == Deploy::PerCamera)
            printf("сеть: %d сообщений, %.1f КБ/с; сверка часов: ошибка до %.2f мс\n", s.netMsgs, s.netBytes / 1024.0 / T, s.clockErrMs);
        const double per = s.wideFrames ? s.wideMs / s.wideFrames : 0;
        printf("нагрузка широких: %.1f мс/кадр на камеру; %s %.1f мс за кадр (%.0f мс при %.0f к/с)\n", per,
               c.deploy == Deploy::OneDevice ? "на устройстве со всеми камерами" : "на устройстве камеры", c.deploy == Deploy::OneDevice ? per * nw : per,
               1000 / std::max(1.f, c.wideFps), c.wideFps);
    }
    if (!json.empty()) {
        if (FILE* f = fopen(json.c_str(), "w")) {
            fprintf(f,
                    "{\"sim_s\": %.2f, \"assignments\": %d, \"acquired\": %d, \"wrong_locks\": %d, \"losses\": %d,\n"
                    " \"slew_s\": %.3f, \"acquire_s\": %.3f, \"track_pct\": %.2f, \"track_correct_pct\": %.2f,\n"
                    " \"err_lock_deg\": [%.4f, %.4f], \"err_axis_deg\": [%.4f, %.4f], \"err_wide_deg\": [%.4f, %.4f],\n"
                    " \"unconfirmed\": %d, \"wide_swaps\": %d, \"idle_pct\": %.2f, \"false_target_pct\": %.2f, \"wide_ms\": %.2f, \"narrow_ms\": %.3f,\n"
                    " \"track_drone_pct\": %.2f, \"narrow_only_s\": %.2f, \"narrow_only_correct_s\": %.2f, \"kept\": %d, \"rebinds\": %d,\n"
                    " \"confirmed\": %d, \"veto_bird\": %d, \"veto_drone\": %d, \"veto_none\": %d, \"part_jumps\": %d,\n"
                    " \"track_bird_s\": %.2f, \"confirmed_bird\": %d, \"veto_delay_s\": %.3f, \"veto_shape\": %d, \"next_after_bird\": %d,\n"
                    " \"mount_ideal\": %d, \"mount_ctl\": %d, \"center_px\": %.3f, \"jitter_px\": %.4f, \"motion_err_px\": %.4f, \"pose_err_deg\": %.5f,\n"
                    " \"blur_px\": %.3f, \"cam_acc_rms\": %.2f, \"cam_acc_peak\": %.1f, \"prof_jerk_rms\": %.1f, \"prof_jerk_peak\": %.0f, \"peak_rate\": %.2f,\n"
                    " \"slips\": %d, \"slip_deg\": [%.4f, %.4f], \"bias_deg\": [%.4f, %.4f], \"eta_s\": %.3f, \"eta_err_s\": %.3f",
                    s.simT, s.assignments, s.acquired, s.wrongLocks, s.losses, s.slewN ? s.slewSum / s.slewN : 0,
                    s.acqN ? s.acqSum / s.acqN : 0, 100 * s.trackTime / T,
                    s.trackTime > 0 ? 100 * s.trackCorrect / s.trackTime : 0, lm, l9, am, a9, wm, w9,
                    s.unconfirmed, s.wideSwaps, 100 * s.idleTime / T, 100 * s.falseTarget / T, s.wideFrames ? s.wideMs / s.wideFrames : 0, s.narrowFrames ? s.narrowMs / s.narrowFrames : 0,
                    100 * s.trackDrone / T, s.narrowOnly, s.narrowOnlyCorrect, s.kept, s.rebinds, s.confirmed, s.vetoBird,
                    s.vetoDrone, s.vetoNone, s.partJumps, s.trackBird, s.confirmedBird, s.vetoBird ? s.vetoDelay / s.vetoBird : 0, s.vetoShape,
                    s.nextAfterBird, (int)c.mountIdeal, c.mountIdeal ? 0 : c.mountCtl, ctr, jit, mot, poseE, blur, accR, s.mount.peakAcc, jrkR,
                    s.mount.peakProfJerk, s.mount.peakRate, s.mount.slips, s.slipAz, s.slipEl, s.biasAz, s.biasEl,
                    s.etaN ? s.etaSum / s.etaN : 0, s.etaN ? s.etaErr / s.etaN : 0);
            if (nw > 1) {   // несколько широких
                fprintf(f, ",\n \"wides\": %d, \"deploy\": \"%s\", \"handoff_ok\": %d, \"handoff_broken\": %d, \"dup_s\": %.2f, "
                           "\"renumbered\": %d, \"seam_track\": %d, \"seam_kept\": %d,\n \"fusion_created\": %d, \"fusion_handoffs\": %d, "
                           "\"fusion_reacquired\": %d, \"fusion_merged\": %d, \"net_msgs\": %d, \"net_bytes\": %d, \"clock_err_ms\": %.3f, \"calib_deg\": [",
                        nw, c.deploy == Deploy::OneDevice ? "one" : "net", s.handoffOk, s.handoffBroken, s.dupTime, s.renumbered, s.seamTrack,
                        s.seamKept, s.fusionCreated, s.fusionHandoffs, s.fusionReacquired, s.fusionMerged, s.netMsgs, s.netBytes, s.clockErrMs);
                for (size_t k = 0; k < s.calibDeg.size(); ++k) fprintf(f, "%s%.4f", k ? ", " : "", s.calibDeg[k]);
                fprintf(f, "], \"cam_frames\": [");
                for (size_t k = 0; k < s.camFrames.size(); ++k) fprintf(f, "%s%d", k ? ", " : "", s.camFrames[k]);
                fprintf(f, "]");
            }
            fprintf(f, "}\n");
            fclose(f);
        }
    }
    return 0;
}

}  // namespace twocam
