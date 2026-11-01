// Настоящая поворотка точной камеры для эмулятора — вместо идеальной (до 31.10 поворотка исполняла команду
// мгновенно, знала своё положение точно, и кадр снимался без задержки).
//
// Цепочка, как на железе: уставка с Pi доходит до контроллера двигателя через задержку команды (связь, цикл
// контроллера) -> профиль движения (плавный из dronolok/gimbal.hpp или прежний закон) -> поправка на люфт ->
// микрошаг шагового двигателя (ротор доходит до шага не мгновенно; бывают пропуски шагов) -> люфт передачи ->
// упругое крепление камеры (резонанс, раскачка после рывков) -> камера. Датчик угла на оси (если он есть)
// видит выход передачи, но не раскачку камеры на креплении; без датчика контроллер знает только выданные шаги.
#pragma once
#include <cmath>
#include <deque>
#include <random>

#include "dronolok/gimbal.hpp"

namespace twocam {

constexpr double kPi = 3.14159265358979323846;

struct MountParams {
    int ctl = 1;                       // 0 — прежний закон (P + скорость цели, без рывка), 1 — плавный профиль
    float cmdLatency = 0.010f;         // от команды Pi до движения двигателя, с
    float cmdJitter = 0.002f;          // разброс этой задержки, с
    float stepDeg = 0.01125f;          // микрошаг на выходе передачи, град (0 — плавно)
    float backlash = 0.05f;            // люфт передачи, град
    bool backlashComp = true;          // контроллер доворачивает люфт (только плавный профиль)
    float resonanceHz = 12;            // крепление камеры: собственная частота, Гц (0 — жёсткое)
    float damping = 0.15f;             // ...и затухание (доля от критического)
    float rotorHz = 0;                 // ротор доходит до нового микрошага с такой полосой, Гц (0 — сразу)
    float slipEvery = 0;               // пропуск шагов (толчок, ветер, кабель): в среднем раз в столько секунд, 0 — нет
    int slipSteps = 8;                 // ...на столько микрошагов
    int encoder = 0;                   // 0 — датчиков угла нет (положение по шагам), 1 — на оси
    float encoderDeg = 0.01f;          // дискретность датчика на оси, град
    // без датчика — где камера по знанию контроллера: 0 — выданные шаги, 1 — профиль (без лестницы микрошагов),
    // 2 — профиль через модель крепления (резонанс modelHz, затухание modelDamping — по замеру на железе)
    int poseModel = 1;
    float modelHz = 12, modelDamping = 0.15f;
    dronolok::GimbalConfig g;          // пределы и усиления профиля
};

// прежний закон поворотки (до 31.10): скорость цели + 6·ошибка, торможение по корню, предел ускорения
inline void legacy_axis(double& pos, double& vel, double err, double ff, double h, double maxRate, double maxAccel) {
    const double K = 6.0;
    double want = ff + K * err;
    const double brake = std::sqrt(2 * maxAccel * std::fabs(err)) + std::fabs(ff);
    want = std::max(-brake, std::min(brake, want));
    want = std::max(-maxRate, std::min(maxRate, want));
    const double dv = std::max(-maxAccel * h, std::min(maxAccel * h, want - vel));
    vel += dv;
    pos += vel * h;
}

class MountSim {
public:
    struct Stats {
        double accSq = 0, profJerkSq = 0;   // при ведении цели: ускорение камеры, рывок профиля (в квадрате, сумма)
        long n = 0;
        double peakRate = 0, peakAcc = 0, peakProfJerk = 0;
        int slips = 0;
    };

    void reset(const MountParams& p, double az, double el, uint32_t seed) {
        p_ = p;
        rng_.seed(seed * 2654435761u + 17);
        const double pos[2] = {az, el};
        for (int i = 0; i < 2; ++i) {
            Axis& a = ax_[i];
            a = Axis();
            a.prof.p = pos[i];
            a.motor = a.rotor = a.shaft = a.cam = quant(pos[i]);
            a.cmdQ = a.motor;
            a.mdl = a.prof.p;
        }
        queue_.clear();
        haveCur_ = false;
        cur_ = dronolok::Setpoint();
        curT_ = 0;
        cam_.clear();
        bel_.clear();
        cam_.push(0, ax_[0].cam, ax_[1].cam);
        bel_.push(0, believedAxis(0), believedAxis(1));
        st_ = Stats();
        prevProfA_[0] = prevProfA_[1] = 0;
    }

    // уставка с Pi, отправлена в момент t; до контроллера дойдёт через задержку команды
    void command(double t, const dronolok::Setpoint& s) {
        double d = p_.cmdLatency;
        if (p_.cmdJitter > 0) d += std::uniform_real_distribution<double>(-p_.cmdJitter, p_.cmdJitter)(rng_);
        d = std::max(0.0, d);
        if (!queue_.empty()) d = std::max(d, queue_.back().first - t);   // по порядку
        queue_.push_back({t + d, s});
    }

    // физика на [t, t+dt]
    void advance(double t, double dt, bool tracking) {
        const int sub = std::max(4, (int)std::ceil(dt / 0.001 - 1e-9));
        const double h = dt / sub;
        for (int k = 0; k < sub; ++k) {
            const double ts = t + k * h;
            while (!queue_.empty() && queue_.front().first <= ts + 1e-12) {
                cur_ = queue_.front().second;
                curT_ = queue_.front().first;
                haveCur_ = true;
                queue_.pop_front();
                ax_[0].anchored = ax_[1].anchored = false;
            }
            for (int i = 0; i < 2; ++i) stepAxis(i, ts, h);
            // пропуск шагов
            if (p_.slipEvery > 0 && std::uniform_real_distribution<double>(0, 1)(rng_) < h / p_.slipEvery) {
                Axis& a = ax_[std::uniform_int_distribution<int>(0, 1)(rng_)];
                a.slip += (std::uniform_int_distribution<int>(0, 1)(rng_) ? 1 : -1) * p_.slipSteps * std::max(1e-4f, p_.stepDeg);
                ++st_.slips;
            }
            const double te = ts + h;
            cam_.push(te, ax_[0].cam, ax_[1].cam);
            bel_.push(te, believedAxis(0), believedAxis(1));
            for (int i = 0; i < 2; ++i) {
                const Axis& a = ax_[i];
                const double pj = (a.prof.a - prevProfA_[i]) / h;
                prevProfA_[i] = a.prof.a;
                st_.peakRate = std::max(st_.peakRate, std::fabs(a.prof.v));
                if (tracking) {
                    st_.accSq += a.camA * a.camA;
                    st_.profJerkSq += pj * pj;
                    st_.peakAcc = std::max(st_.peakAcc, std::fabs(a.camA));
                    st_.peakProfJerk = std::max(st_.peakProfJerk, std::fabs(pj));
                }
            }
            if (tracking) ++st_.n;
        }
    }

    // где камера на самом деле в момент t
    dronolok::Pose camera(double t) const { return pose(cam_, t); }
    // что о своём положении в момент t знает алгоритм: выданные шаги (с поправкой на люфт) или датчик на оси
    dronolok::Pose believed(double t) const { return pose(bel_, t); }
    // уставка профиля (что ведёт контроллер) и скорость
    dronolok::Pose profilePose() const { return {(float)dronolok::az_norm(ax_[0].prof.p), (float)ax_[1].prof.p, 0}; }
    float rate(int i) const { return (float)ax_[i].prof.v; }
    float accel(int i) const { return (float)ax_[i].prof.a; }
    const dronolok::AxisProfile& profile(int i) const { return ax_[i].prof; }
    double slip(int i) const { return ax_[i].slip; }
    const Stats& stats() const { return st_; }

private:
    struct Axis {
        dronolok::AxisProfile prof;
        dronolok::BacklashComp comp;
        double motor = 0;      // команда двигателю с поправкой на люфт, град на выходе передачи
        double cmdQ = 0;       // она же по микрошагам
        double slip = 0;       // накопленный пропуск шагов
        double rotor = 0;      // где ротор (доходит до микрошага с полосой rotorHz) + пропуски
        double shaft = 0;      // выход передачи (после люфта)
        double cam = 0, camV = 0, camA = 0;   // камера на упругом креплении
        double anchor = 0;
        bool anchored = false;
        double mdl = 0, mdlV = 0;   // модель крепления у контроллера: где по ней камера
    };

    double quant(double x) const { return p_.stepDeg > 0 ? std::round(x / p_.stepDeg) * p_.stepDeg : x; }
    bool comp() const { return p_.ctl == 1 && p_.backlashComp; }

    void stepAxis(int i, double ts, double h) {
        Axis& a = ax_[i];
        const bool wrap = i == 0;
        // 1. профиль: уставка с Pi (без общих часов — продолжается по своей скорости от того, как дошла)
        if (p_.ctl == 0) {
            if (haveCur_) {
                const double cmd = wrap ? cur_.az : cur_.el, ff = wrap ? cur_.vaz : cur_.vel;
                const double err = wrap ? dronolok::az_diff(cmd, a.prof.p) : cmd - a.prof.p;
                const double v0 = a.prof.v;
                legacy_axis(a.prof.p, a.prof.v, err, ff, h, p_.g.max_rate, p_.g.max_accel);
                a.prof.a = (a.prof.v - v0) / h;
            }
        } else if (!haveCur_ || cur_.hold) {
            a.prof.hold(h, p_.g, a.anchor, a.anchored, wrap);
        } else {
            const double s = std::min(0.25, std::max(0.0, ts - curT_));   // дальше 0.25 с без команд не продолжаем
            const double p0 = wrap ? cur_.az : cur_.el, v0 = wrap ? cur_.vaz : cur_.vel, a0 = wrap ? cur_.aaz : cur_.ael;
            a.prof.track(p0 + v0 * s + 0.5 * a0 * s * s, v0 + a0 * s, a0, h, p_.g, wrap);
        }
        if (!wrap) a.prof.p = std::max(-5.0, std::min(89.0, a.prof.p));
        // 2. двигатель: поправка на люфт, микрошаг, ротор доходит до шага с полосой rotorHz
        a.motor = comp() ? a.comp.motor(a.prof.p, a.prof.v, p_.backlash) : a.prof.p;
        a.cmdQ = quant(a.motor);
        const double target = a.cmdQ + a.slip;
        if (p_.rotorHz > 0) a.rotor += (target - a.rotor) * (1 - std::exp(-2 * kPi * p_.rotorHz * h));
        else a.rotor = target;
        // 3. люфт передачи: выход стоит, пока ротор не выберет зазор
        const double hb = 0.5 * std::max(0.f, p_.backlash);
        if (a.rotor - a.shaft > hb) a.shaft = a.rotor - hb;
        else if (a.rotor - a.shaft < -hb) a.shaft = a.rotor + hb;
        // модель крепления у контроллера (по профилю: где выход передачи, если люфт выбран)
        if (p_.modelHz > 0) {
            const double w = 2 * kPi * p_.modelHz;
            const double out = comp() ? a.prof.p : a.cmdQ;
            const double ma = w * w * (out - a.mdl) - 2 * p_.modelDamping * w * a.mdlV;
            a.mdlV += ma * h;
            a.mdl += a.mdlV * h;
        } else {
            a.mdl = comp() ? a.prof.p : a.cmdQ;
        }
        // 4. упругое крепление камеры
        if (p_.resonanceHz > 0) {
            const double w = 2 * kPi * p_.resonanceHz;
            a.camA = w * w * (a.shaft - a.cam) - 2 * p_.damping * w * a.camV;
            a.camV += a.camA * h;
            a.cam += a.camV * h;
        } else {
            const double v1 = (a.shaft - a.cam) / h;
            a.camA = (v1 - a.camV) / h;
            a.camV = v1;
            a.cam = a.shaft;
        }
    }
    double believedAxis(int i) const {
        const Axis& a = ax_[i];
        if (p_.encoder == 1) {
            const double q = std::max(1e-4f, p_.encoderDeg);
            return std::round(a.shaft / q) * q;
        }
        if (p_.ctl == 1 && p_.poseModel == 2) return a.mdl;
        if (p_.ctl == 1 && p_.poseModel == 1) return comp() ? a.prof.p : a.motor;
        return comp() ? a.comp.output(a.cmdQ, p_.backlash) : a.cmdQ;   // выданные шаги, без пропусков
    }
    static dronolok::Pose pose(const dronolok::PoseHistory& h, double t) {
        double az = 0, el = 0;
        h.at(t, az, el);
        return {(float)dronolok::az_norm(az), (float)el, 0};
    }

    MountParams p_;
    std::mt19937 rng_;
    Axis ax_[2];
    std::deque<std::pair<double, dronolok::Setpoint>> queue_;
    dronolok::Setpoint cur_;
    bool haveCur_ = false;
    double curT_ = 0;
    dronolok::PoseHistory cam_, bel_;
    Stats st_;
    double prevProfA_[2] = {0, 0};
};

}  // namespace twocam
