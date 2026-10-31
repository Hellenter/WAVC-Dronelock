// Наведение поворотки точной камеры — то, что работает на Pi и в контроллере двигателей, а не только в эмуляторе.
//
// 1. Фильтр цели (GimbalFilter): Калман по осям поворотки (азимут, угол места) — положение, скорость и
//    ускорение цели. Отметка — со временем кадра (середина выдержки), а не со временем, когда кадр обработан:
//    задержка камеры и обработки учитывается сама, прогноз — на любой момент.
// 2. Уставка (setpoint): где будет цель, когда команда дойдёт до двигателя (lead_cmd), с её скоростью и
//    ускорением: поворотка идёт вместе с целью, а не догоняет её по ошибке.
// 3. Профиль (AxisProfile — контроллер двигателя, ~1 кГц): плавное ведение уставки с пределами скорости,
//    ускорения и рывка; к цели — с торможением, рассчитанным с учётом рывка; между командами уставка
//    продолжается по своей скорости. Без рывков и бросков ускорения не раскачивается крепление камеры.
// 4. Без датчиков угла: где была поворотка в момент кадра — по шагам, которые выдал контроллер
//    (PoseHistory) с поправкой на люфт (BacklashComp); уход нуля от пропущенных шагов — по широкой камере
//    (WideBias): обе камеры видят одну цель.
//
// Углы — градусы, время — секунды по общим часам (как в geometry.hpp). Азимут профиля и истории —
// без свёртки в 0..360 (непрерывный), наружу — az_norm.
#pragma once
#include <algorithm>
#include <cmath>
#include <deque>

#include "geometry.hpp"

namespace dronolok {

struct GimbalConfig {
    // пределы движения (что двигатель выдерживает с запасом, без пропуска шагов)
    float max_rate = 30, max_accel = 150, max_jerk = 3000;   // град/с, град/с², град/с³
    // ведение: скорость к ошибке положения kp (1/с), ускорение к ошибке скорости kv (1/с); kv = 4 kp — без
    // перерегулирования, полоса ~sqrt(kp·kv)/2π Гц — заметно ниже резонанса крепления камеры
    float kp = 8, kv = 32;
    float brake_share = 0.85f;   // доля предела ускорения на торможение к цели (остальное — на манёвр цели)
    // задержка команды: от решения на Pi до движения двигателя (связь, цикл контроллера), с
    float lead_cmd = 0.01f;
    // фильтр цели
    float jerk_noise = 2;        // манёвр цели: СКО рывка, град/с³ (подбор на эмуляторе: 1–2 — вдвое меньше дрожание, чем 4)
    float acc_tau = 1.0f;        // ускорение цели затухает за столько секунд (манёвр не длится вечно)
    float sigma_pose = 0.006f;   // ошибка положения поворотки на момент кадра (микрошаг, упругость, люфт), град
    float gate = 5;              // отметка дальше стольких СКО от прогноза — выброс
    int gate_reset = 3;          // ...столько выбросов подряд — цель сманеврировала: берём отметку
    // без датчиков угла: уход нуля по широкой
    float bias_tau = 3;          // с
    float bias_max = 1;          // град
};

// Одна ось фильтра: положение, скорость, ускорение (затухает за tau — модель Зингера)
struct AxisKF {
    double t = 0;
    double x[3] = {0, 0, 0};
    double P[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

    void reset(double t0, double p, double v, double a, double sp, double sv, double sa) {
        t = t0;
        x[0] = p, x[1] = v, x[2] = a;
        for (auto& r : P)
            for (double& e : r) e = 0;
        P[0][0] = sp * sp, P[1][1] = sv * sv, P[2][2] = sa * sa;
    }
    static void trans(double dt, double tau, double F[3][3]) {
        const double e = std::exp(-dt / tau);
        F[0][0] = 1, F[0][1] = dt, F[0][2] = tau * tau * (dt / tau - 1 + e);
        F[1][0] = 0, F[1][1] = 1, F[1][2] = tau * (1 - e);
        F[2][0] = 0, F[2][1] = 0, F[2][2] = e;
    }
    // прогноз на t1 (только вперёд); q — спектральная плотность рывка, (град/с³)²·с
    void predict(double t1, double q, double tau) {
        const double dt = t1 - t;
        if (dt <= 0) return;
        double F[3][3];
        trans(dt, tau, F);
        double y[3];
        for (int i = 0; i < 3; ++i) y[i] = F[i][0] * x[0] + F[i][1] * x[1] + F[i][2] * x[2];
        for (int i = 0; i < 3; ++i) x[i] = y[i];
        double FP[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) FP[i][j] = F[i][0] * P[0][j] + F[i][1] * P[1][j] + F[i][2] * P[2][j];
        const double d2 = dt * dt, d3 = d2 * dt, d4 = d3 * dt, d5 = d4 * dt;
        const double Q[3][3] = {{d5 / 20, d4 / 8, d3 / 6}, {d4 / 8, d3 / 3, d2 / 2}, {d3 / 6, d2 / 2, dt}};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                P[i][j] = FP[i][0] * F[j][0] + FP[i][1] * F[j][1] + FP[i][2] * F[j][2] + q * Q[i][j];
        t = t1;
    }
    // отметка положения: y — невязка (отметка минус прогноз), r — дисперсия отметки
    void update(double y, double r) {
        const double S = P[0][0] + r;
        if (S <= 0) return;
        const double K[3] = {P[0][0] / S, P[1][0] / S, P[2][0] / S};
        for (int i = 0; i < 3; ++i) x[i] += K[i] * y;
        const double P0[3] = {P[0][0], P[0][1], P[0][2]};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) P[i][j] -= K[i] * P0[j];
    }
    // состояние на t1 без изменения фильтра (t1 может быть и раньше t — тогда назад по той же модели)
    void at(double t1, double tau, double& p, double& v, double& a) const {
        const double dt = t1 - t;
        double F[3][3];
        trans(dt, tau, F);
        p = F[0][0] * x[0] + F[0][1] * x[1] + F[0][2] * x[2];
        v = F[1][1] * x[1] + F[1][2] * x[2];
        a = F[2][2] * x[2];
    }
    double sigma_at(double t1, double q, double tau) const {
        AxisKF k = *this;
        k.predict(t1, q, tau);
        return std::sqrt(std::max(0.0, k.P[0][0]));
    }
};

// Цель для поворотки: направление, угловая скорость и ускорение по осям поворотки
class GimbalFilter {
public:
    explicit GimbalFilter(const GimbalConfig& c = GimbalConfig()) : c_(c) {}
    void set_config(const GimbalConfig& c) { c_ = c; }
    const GimbalConfig& config() const { return c_; }
    bool active() const { return on_; }
    void clear() {
        on_ = false;
        rej_ = 0;
    }
    double time() const { return el_.t; }

    // начать по сообщению о цели (обычно — широкой при назначении): положение и скорость с их СКО
    void reset(const TargetReport& r) {
        const double ce = cos_el(r.el);
        const double sv = std::max(0.05f, r.sigma_v);
        az_.reset(r.t, r.az, r.vaz, 0, std::max(1e-4f, r.sigma) / ce, sv / ce, 3 / ce);
        el_.reset(r.t, r.el, r.vel, 0, std::max(1e-4f, r.sigma), sv, 3);
        on_ = true;
        rej_ = 0;
    }

    // отметка: направление на момент t (время кадра), СКО в градусах (без ошибки положения поворотки — она
    // добавляется здесь). false — выброс, не принята
    bool measure(double t, double az, double el, double sigma) {
        const double r = (double)sigma * sigma + (double)c_.sigma_pose * c_.sigma_pose;
        if (!on_) {
            TargetReport s;
            s.t = t, s.az = (float)az, s.el = (float)el, s.sigma = (float)std::sqrt(r), s.sigma_v = 5;
            reset(s);
            return true;
        }
        const double q = (double)c_.jerk_noise * c_.jerk_noise, tau = std::max(0.05f, c_.acc_tau);
        double zaz = az, zel = el, rr = r;
        if (t < el_.t) {   // отметка старше состояния (другой источник с большей задержкой): переносим её вперёд
            double p0, v0, a0, p1, v1, a1;
            az_.at(t, tau, p0, v0, a0);
            az_.at(az_.t, tau, p1, v1, a1);
            zaz += p1 - p0;
            el_.at(t, tau, p0, v0, a0);
            el_.at(el_.t, tau, p1, v1, a1);
            zel += p1 - p0;
            const double d = el_.t - t;
            rr += std::max(az_.P[1][1], el_.P[1][1]) * d * d;
        } else {
            az_.predict(t, q, tau);
            el_.predict(t, q, tau);
        }
        const double ce = cos_el(el_.x[0]);
        const double raz = rr / (ce * ce);
        const double yaz = az_diff(zaz, az_.x[0]), yel = zel - el_.x[0];
        const double nis = yaz * yaz / (az_.P[0][0] + raz) + yel * yel / (el_.P[0][0] + rr);
        if (nis > (double)c_.gate * c_.gate) {
            if (++rej_ < c_.gate_reset) return false;
            // цель сманеврировала (или захват перешёл на другую часть крупной цели): положение — по отметке
            rej_ = 0;
            az_.x[0] += yaz;
            el_.x[0] += yel;
            az_.P[0][0] = raz, el_.P[0][0] = rr;
            az_.P[0][1] = az_.P[1][0] = az_.P[0][2] = az_.P[2][0] = 0;
            el_.P[0][1] = el_.P[1][0] = el_.P[0][2] = el_.P[2][0] = 0;
            return true;
        }
        rej_ = 0;
        az_.update(yaz, raz);
        el_.update(yel, rr);
        return true;
    }

    // оценка на момент t: направление (азимут 0..360), скорость, ускорение
    void state(double t, double& az, double& el, double& vaz, double& vel, double& aaz, double& ael) const {
        const double tau = std::max(0.05f, c_.acc_tau);
        az_.at(t, tau, az, vaz, aaz);
        el_.at(t, tau, el, vel, ael);
        az = az_norm(az);
    }
    // СКО положения на момент t, градусы по небу
    double sigma(double t) const {
        const double q = (double)c_.jerk_noise * c_.jerk_noise, tau = std::max(0.05f, c_.acc_tau);
        const double sa = az_.sigma_at(t, q, tau) * cos_el(el_.x[0]), se = el_.sigma_at(t, q, tau);
        return std::sqrt(0.5 * (sa * sa + se * se));
    }
    // как сообщение о цели (для окна поиска, сверки с широкой): на момент t
    TargetReport report(double t) const {
        TargetReport r;
        double az, el, vaz, vel, aaz, ael;
        state(t, az, el, vaz, vel, aaz, ael);
        r.t = t, r.az = (float)az, r.el = (float)el, r.vaz = (float)vaz, r.vel = (float)vel;
        r.sigma = (float)sigma(t);
        r.sigma_v = (float)std::sqrt(std::max(0.0, 0.5 * (az_.P[1][1] * sqr(cos_el(el)) + el_.P[1][1])));
        return r;
    }
    int rejected() const { return rej_; }

private:
    static double cos_el(double el) { return std::max(0.05, std::cos(deg2rad(el))); }
    static double sqr(double x) { return x * x; }
    GimbalConfig c_;
    AxisKF az_, el_;
    bool on_ = false;
    int rej_ = 0;
};

// Уставка двигателю: положение, скорость, ускорение на момент t (в координатах привода — с поправкой на уход нуля)
struct Setpoint {
    double t = 0;
    double az = 0, el = 0, vaz = 0, vel = 0, aaz = 0, ael = 0;
    bool hold = true;   // цели нет: остановиться плавно там, где есть
};

// уставка по фильтру: цель на момент now + lead_cmd, минус уход нуля привода (bias_az, bias_el)
inline Setpoint gimbal_setpoint(const GimbalFilter& f, double now, double bias_az = 0, double bias_el = 0) {
    Setpoint s;
    s.t = now + f.config().lead_cmd;
    if (!f.active()) return s;
    f.state(s.t, s.az, s.el, s.vaz, s.vel, s.aaz, s.ael);
    s.az = az_norm(s.az - bias_az);
    s.el -= bias_el;
    s.hold = false;
    return s;
}

// уставка по одному сообщению о цели (по прямой, без ускорения): пока цель потеряна и её ищут по последнему
// сообщению, поворотка идёт туда же, где окно поиска. Ускорение фильтра за секунду без отметок уводит ось
// на градусы в сторону, и окно выходит из кадра
inline Setpoint report_setpoint(const TargetReport& r, double now, double lead, double bias_az = 0, double bias_el = 0) {
    Setpoint s;
    s.t = now + lead;
    const double dt = s.t - r.t;
    s.az = az_norm(r.az + r.vaz * dt - bias_az);
    s.el = std::min(90.0, r.el + r.vel * dt) - bias_el;
    s.vaz = r.vaz, s.vel = r.vel;
    s.hold = false;
    return s;
}

// Профиль одной оси (контроллер двигателя): плавное ведение уставки
struct AxisProfile {
    double p = 0, v = 0, a = 0;

    // шаг h к уставке sp (скорость sv, ускорение sa — на этот же момент); wrap — ось азимута
    void track(double sp, double sv, double sa, double h, const GimbalConfig& c, bool wrap) {
        const double A = std::max(1.f, c.max_accel), J = std::max(1.f, c.max_jerk), V = std::max(0.1f, c.max_rate);
        const double e = wrap ? az_diff(sp, p) : sp - p;
        // скорость к цели, с которой ещё можно затормозить: путь торможения v²/2A + v·A/2J (рывок на сброс
        // ускорения) равен ошибке
        const double Ab = std::max(0.05, (double)c.brake_share) * A, k = Ab * Ab / (2 * J);
        const double vb = std::sqrt(k * k + 2 * Ab * std::fabs(e)) - k;
        const double vr = std::min((double)c.kp * std::fabs(e), vb);
        const double vg = std::max(-V, std::min(V, sv + (e >= 0 ? vr : -vr)));
        double ag = sa + c.kv * (vg - v);
        // к пределу скорости подходим плавно: ускорение успевает сойти на нет по рывку
        ag = std::min(ag, std::sqrt(2 * J * std::max(0.0, V - v)));
        ag = std::max(ag, -std::sqrt(2 * J * std::max(0.0, V + v)));
        ag = std::max(-A, std::min(A, ag));
        a += std::max(-J * h, std::min(J * h, ag - a));
        double v1 = v + a * h;
        if (v1 > V || v1 < -V) {
            v1 = std::max(-V, std::min(V, v1));
            a = (v1 - v) / h;
        }
        p += 0.5 * (v + v1) * h;
        v = v1;
    }
    // стоять: уставка — где профиль сейчас, скорость ноль
    void hold(double h, const GimbalConfig& c, double& anchor, bool& anchored, bool wrap) {
        if (!anchored) {
            // точка остановки: где профиль встанет при торможении с нынешней скорости
            const double A = std::max(1.f, c.max_accel) * 0.5;
            anchor = p + v * std::fabs(v) / (2 * A);
            anchored = true;
        }
        track(anchor, 0, 0, h, c, wrap);
    }
};

// Поправка на люфт передачи: двигатель доворачивает на половину люфта в сторону движения, и выход
// передачи (камера) оказывается там, где профиль. Направление меняется, когда скорость профиля перешла
// через ноль дальше hyst.
struct BacklashComp {
    int dir = 0;
    double motor(double p, double v, double backlash, double hyst = 0.02) {
        if (v > hyst) dir = 1;
        else if (v < -hyst) dir = -1;
        return p + 0.5 * dir * backlash;
    }
    // где выход передачи по положению двигателя (что знает контроллер без датчиков на оси)
    double output(double motor_pos, double backlash) const { return motor_pos - 0.5 * dir * backlash; }
};

// История положения поворотки (что сообщил контроллер двигателя, со временем): где была ось в момент кадра
class PoseHistory {
public:
    void clear() { h_.clear(); }
    void push(double t, double az, double el, double keep = 1.0) {
        if (!h_.empty() && t < h_.back().t) h_.clear();
        h_.push_back({t, az, el});
        while (h_.size() > 2 && h_.front().t < t - keep) h_.pop_front();
    }
    bool empty() const { return h_.empty(); }
    // на момент t: между записями — линейно, за краями — крайняя
    bool at(double t, double& az, double& el) const {
        if (h_.empty()) return false;
        if (t <= h_.front().t) {
            az = h_.front().az, el = h_.front().el;
        } else if (t >= h_.back().t) {
            az = h_.back().az, el = h_.back().el;
        } else {
            size_t lo = 0, hi = h_.size() - 1;
            while (hi - lo > 1) {
                const size_t m = (lo + hi) / 2;
                (h_[m].t <= t ? lo : hi) = m;
            }
            const S& a = h_[lo];
            const S& b = h_[hi];
            const double k = b.t > a.t ? (t - a.t) / (b.t - a.t) : 0;
            az = a.az + (b.az - a.az) * k;
            el = a.el + (b.el - a.el) * k;
        }
        return true;
    }

private:
    struct S {
        double t, az, el;
    };
    std::deque<S> h_;
};

// Уход нуля поворотки без датчиков угла (пропущенные шаги): широкая и точная видят одну цель — разница
// между прогнозом широкой и направлением по точной (с нынешней поправкой) медленно уходит в поправку.
struct WideBias {
    double az = 0, el = 0;
    // daz, del — прогноз широкой на время кадра точной минус направление по точной; sigma — СКО широкой;
    // dt — сколько прошло с прошлого раза
    void update(double daz, double del, double sigma, double dt, const GimbalConfig& c) {
        const double lim = std::max(0.05, 3.0 * sigma);
        daz = std::max(-lim, std::min(lim, daz));
        del = std::max(-lim, std::min(lim, del));
        const double w = std::min(1.0, std::pow(0.03 / std::max(1e-4, sigma), 2));
        const double k = std::min(1.0, dt / std::max(0.1f, c.bias_tau)) * w;
        az = std::max(-(double)c.bias_max, std::min((double)c.bias_max, az + k * daz));
        el = std::max(-(double)c.bias_max, std::min((double)c.bias_max, el + k * del));
    }
};

// Сколько секунд поворотке до цели (обе оси ближе tol): профиль идёт за прогнозом фильтра без новых отметок
inline double slew_eta(AxisProfile az, AxisProfile el, const GimbalFilter& f, double t, double tol, double max_t = 5,
                       double h = 0.002) {
    if (!f.active()) return 0;
    const GimbalConfig& c = f.config();
    for (double s = 0; s < max_t; s += h) {
        double taz, tel, vaz, vel, aaz, ael;
        f.state(t + s, taz, tel, vaz, vel, aaz, ael);
        if (std::fabs(az_diff(taz, az.p)) * std::cos(deg2rad(tel)) < tol && std::fabs(tel - el.p) < tol) return s;
        az.track(taz, vaz, aaz, h, c, true);
        el.track(tel, vel, ael, h, c, false);
    }
    return max_t;
}

}  // namespace dronolok
