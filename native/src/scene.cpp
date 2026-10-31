// Оценка сцены и погоды (см. dronolok/scene.hpp)
#include "dronolok/scene.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace dronolok {

const char* light_name(Light l) {
    switch (l) {
        case Light::Day: return "день";
        case Light::Dusk: return "сумерки";
        case Light::Night: return "ночь";
    }
    return "?";
}

const char* terrain_name(Terrain t) {
    switch (t) {
        case Terrain::Open: return "поле";
        case Terrain::Forest: return "лес";
        case Terrain::Urban: return "город";
        default: return "неясно";
    }
}

// медиана (портит порядок v)
static float median(std::vector<float>& v) {
    if (v.empty()) return 0.f;
    auto m = v.begin() + v.size() / 2;
    std::nth_element(v.begin(), m, v.end());
    return *m;
}

static float quantile(std::vector<float>& v, float q) {
    if (v.empty()) return 0.f;
    auto m = v.begin() + std::min(v.size() - 1, (size_t)(q * (float)v.size()));
    std::nth_element(v.begin(), m, v.end());
    return *m;
}

void SceneAnalyzer::reset() {
    st_ = SceneState();
    w_ = h_ = 0;
    havePrev_ = false;
    last_ = -1000000;
    ref_.clear();
    refLine_.clear();
    lastLine_.clear();
    movedRun_ = 0;
}

bool SceneAnalyzer::update(const uint8_t* gray, int w, int h, int stride, float fps, bool movingCamera) {
    ++frame_;
    const int every = std::max(1, (int)std::lround(cfg_.interval_s * (fps > 0 ? fps : 25.f)));
    if (frame_ - last_ < every) return false;
    const float dt = (float)(frame_ - std::max(last_, frame_ - 10 * every)) / (fps > 0 ? fps : 25.f);
    last_ = frame_;
    const int bs = 4;   // выборок на сторону блока
    if (w != w_ || h != h_) {
        w_ = w;
        h_ = h;
        step_ = cfg_.step > 0 ? cfg_.step : 4;
        while (cfg_.step <= 0 && (double)w * h / ((double)step_ * step_) > 90000.0) step_ *= 2;
        bw_ = w / step_ / bs;
        bh_ = h / step_ / bs;
        havePrev_ = false;
        ref_.clear();
        refLine_.clear();
        lastLine_.clear();
    }
    if (bw_ < 4 || bh_ < 4) return false;
    const int step = step_;
    const int sw = bw_ * bs, sh = bh_ * bs;
    cur_.resize((size_t)sw * sh);
    for (int y = 0; y < sh; ++y) {
        const uint8_t* row = gray + (size_t)y * step * stride;
        uint8_t* o = &cur_[(size_t)y * sw];
        for (int x = 0; x < sw; ++x) o[x] = row[x * step];
    }
    const size_t nb = (size_t)bw_ * bh_;
    mean_.assign(nb, 0.f);
    act_.assign(nb, 0.f);
    rough_.assign(nb, 0.f);
    out_.assign(nb, 0);
    std::vector<uint16_t> glare(nb, 0);
    const bool tm = havePrev_ && prev_.size() == cur_.size();
    const float sig0 = st_.noise > 0 ? st_.noise : 2.f;
    const int T = (int)std::max(10.f, 6.f * sig0);   // точечная вспышка во времени (снежинка, капля)
    for (int by = 0; by < bh_; ++by)
        for (int bx = 0; bx < bw_; ++bx) {
            int sum = 0, rough = 0, nr = 0, act = 0, outs = 0, gl = 0;
            for (int y = by * bs; y < (by + 1) * bs; ++y) {
                const uint8_t* c = &cur_[(size_t)y * sw];
                const uint8_t* p = tm ? &prev_[(size_t)y * sw] : nullptr;
                for (int x = bx * bs; x < (bx + 1) * bs; ++x) {
                    const int v = c[x];
                    sum += v;
                    gl += v >= 250;
                    if (x + 1 < (bx + 1) * bs) {
                        rough += std::abs(v - c[x + 1]);
                        ++nr;
                    }
                    if (y + 1 < (by + 1) * bs) {
                        rough += std::abs(v - c[x + sw]);
                        ++nr;
                    }
                    if (p) {
                        const int d = std::abs(v - p[x]);
                        act += d;
                        // вспышка — точка: соседние выборки (4 пикс.) почти не изменились
                        if (d > T && x > 0 && x + 1 < sw && y > 0 && y + 1 < sh &&
                            std::abs(c[x - 1] - p[x - 1]) * 2 < T && std::abs(c[x + 1] - p[x + 1]) * 2 < T &&
                            std::abs(c[x - sw] - p[x - sw]) * 2 < T && std::abs(c[x + sw] - p[x + sw]) * 2 < T)
                            ++outs;
                    }
                }
            }
            const size_t b = (size_t)by * bw_ + bx;
            const float n = (float)(bs * bs);
            mean_[b] = (float)sum / n;
            rough_[b] = (float)rough / (float)std::max(1, nr);
            act_[b] = (float)act / n;
            out_[b] = (uint16_t)outs;
            glare[b] = (uint16_t)gl;
        }
    prev_.swap(cur_);
    havePrev_ = true;

    // шум матрицы, грубо: нижний дециль шероховатости блоков (самые гладкие — чистое небо;
    // неба в кадре бывает и 30%); E|n1 - n2| = 1.128 СКО. Точнее — ниже, по найденному небу
    tmp_.assign(rough_.begin(), rough_.end());
    float sigma = std::max(0.3f, quantile(tmp_, 0.1f) / 1.128f);

    // небо и земля: блок земли, под которым больше половины блоков земли, — кромка;
    // всё ниже — земля (как skyline в детекторе)
    const float rt = std::max(5.f, 2.5f * 1.128f * sigma + 1.f);
    // опора яркости неба — верхний ряд блоков: светлое облако не должно делать небо под ним "тёмным"
    // (первый сверху ряд, который не чёрная рамка записи экрана)
    float cap = 0;
    for (int by = 0; by < bh_ && cap < 30.f; ++by) {
        tmp_.assign(mean_.begin() + (size_t)by * bw_, mean_.begin() + (size_t)(by + 1) * bw_);
        cap = 1.15f * median(tmp_);
    }
    sky_.assign(nb, 1);
    std::vector<int16_t> line(bw_, (int16_t)bh_);
    std::vector<int> tex(bh_);
    for (int bx = 0; bx < bw_; ++bx) {
        int below = 0;
        // земля — текстура или заметно темнее неба над ней (в сумерках, дымке и ночью
        // текстура земли тонет в шуме, а по яркости кромка видна)
        float top = 0;
        for (int by = 0; by < bh_; ++by) {
            const float m = mean_[(size_t)by * bw_ + bx];
            tex[by] = rough_[(size_t)by * bw_ + bx] > rt || (top > 0 && m < 0.85f * top - 2.f);
            if (!tex[by]) top = std::min(cap, std::max(top, m));
        }
        // и сразу под ним земля сплошь (4 блока из 5): кромка облака так не выглядит, даже
        // когда земли в кадре больше половины (записи с низко опущенной камерой)
        int run = 0;
        for (int by = bh_ - 1; by >= 0; --by) {
            below += tex[by];
            run = 0;
            for (int k = by; k < std::min(bh_, by + 5); ++k) run += tex[k];
            if (tex[by] && 2 * below > bh_ - by && run >= std::min(4, bh_ - by)) line[bx] = (int16_t)by;
        }
        for (int by = line[bx]; by < bh_; ++by) sky_[(size_t)by * bw_ + bx] = 0;
    }
    tmp_.clear();
    for (size_t b = 0; b < nb; ++b)
        if (sky_[b]) tmp_.push_back(rough_[b]);
    if (tmp_.size() >= 8) sigma = std::max(0.3f, quantile(tmp_, 0.25f) / 1.128f);
    std::vector<float> skyM, gndM, skyAct;
    long skyN = 0, skyOut = 0, skyGl = 0;
    for (size_t b = 0; b < nb; ++b) {
        if (sky_[b]) {
            skyM.push_back(mean_[b]);
            skyOut += out_[b];
            skyGl += glare[b];
            ++skyN;
        } else {
            gndM.push_back(mean_[b]);
        }
    }
    SceneState& s = st_;
    const bool first = !s.ready;
    const float a = first ? 1.f : std::min(1.f, dt / std::max(0.05f, cfg_.tau_s));
    auto ema = [&](float& v, float x, float al) { v = first ? x : v + al * (x - v); };
    s.skyFrac = (float)skyN / (float)nb;
    s.lightJump = s.moved = false;
    s.skyline = line;
    if (skyN < 8) {   // неба нет (камера смотрит в землю) — погоду не оцениваем
        s.ready = true;
        ++s.analyses;
        return true;
    }
    const float skyLum = median(skyM);
    ema(s.skyLum, skyLum, std::min(1.f, dt / 1.f));
    ema(s.groundLum, gndM.empty() ? skyLum : median(gndM), a);
    ema(s.noise, sigma, std::min(1.f, dt / 1.f));
    ema(s.glare, (float)skyGl / (float)(skyN * bs * bs), a);

    // облачность: блоки неба, заметно отличающиеся от медианы своего ряда
    {
        long cl = 0, nn = 0;
        std::vector<float> rowv;
        for (int by = 0; by < bh_; ++by) {
            rowv.clear();
            for (int bx = 0; bx < bw_; ++bx)
                if (sky_[(size_t)by * bw_ + bx]) rowv.push_back(mean_[(size_t)by * bw_ + bx]);
            if (rowv.size() < 3) continue;
            const float med = median(rowv);
            const float lim = 2.f + 0.04f * skyLum;
            for (int bx = 0; bx < bw_; ++bx) {
                const size_t b = (size_t)by * bw_ + bx;
                if (!sky_[b]) continue;
                ++nn;
                cl += std::fabs(mean_[b] - med) > lim;
            }
        }
        ema(s.clouds, nn ? (float)cl / (float)nn : 0.f, a);
    }

    // кромка: ветер (качание крон) и дымка (контраст кромки)
    const float actN = 1.128f * sigma;
    std::vector<float> wind, haze;
    for (int bx = 0; bx < bw_; ++bx) {
        const int ys = line[bx];
        if (ys >= bh_ || ys < 1) continue;
        haze.push_back(std::fabs(mean_[(size_t)(ys - 1) * bw_ + bx] - mean_[(size_t)std::min(bh_ - 1, ys + 1) * bw_ + bx]) /
                       std::max(1.f, skyLum));
        if (!tm) continue;
        for (int by = ys; by < std::min(bh_, ys + 2); ++by) {
            const size_t b = (size_t)by * bw_ + bx;
            // сдвиг ~ шаг * (изменение во времени) / (перепад между соседними выборками)
            wind.push_back(std::min(4.f, std::max(0.f, (float)step * (act_[b] - actN) / std::max(1.f, rough_[b]))));
        }
    }
    // дымка: кромка мало отличается от неба над ней и земля в целом светлая, как небо (по
    // отдельности оба признака бывают и в ясный день: дальние холмы за лесом, светлая трава)
    if (haze.size() * 3 >= (size_t)bw_ && gndM.size() * 10 >= nb) {
        const float edge = std::min(1.f, std::max(0.f, 1.f - median(haze) / 0.8f));
        const float ratio = gndM.empty() ? 0.f : std::min(1.f, std::max(0.f, (s.groundLum / std::max(1.f, s.skyLum) - 0.3f) / 0.5f));
        ema(s.haze, 0.5f * (edge + ratio), a);
    } else {
        ema(s.haze, 0.f, a);   // кромки в кадре нет — о дымке судить не по чему
    }
    if (tm && !movingCamera && wind.size() >= 4) ema(s.wind, median(wind), a);
    if (tm && !movingCamera) ema(s.precip, 1000.f * (float)skyOut / (float)(skyN * bs * bs), a);

    // окружение: раз в 5 анализов (~1 с), по градиентам выборки на земле
    if (!gndM.empty() && (s.analyses % 5 == 0)) {
        const uint8_t* c = prev_.data();   // текущая выборка (после swap)
        double hvE = 0, allE = 0;
        for (int by = 0; by < bh_; ++by)
            for (int bx = 0; bx < bw_; ++bx) {
                const size_t b = (size_t)by * bw_ + bx;
                if (sky_[b] || by < line[bx] + 2) continue;   // не кромка: там небо в градиентах
                for (int y = std::max(1, by * bs); y < std::min(sh - 1, (by + 1) * bs); ++y)
                    for (int x = std::max(1, bx * bs); x < std::min(sw - 1, (bx + 1) * bs); ++x) {
                        const int gx = c[(size_t)y * sw + x + 1] - c[(size_t)y * sw + x - 1];
                        const int gy = c[(size_t)(y + 1) * sw + x] - c[(size_t)(y - 1) * sw + x];
                        const int ax = std::abs(gx), ay = std::abs(gy);
                        const double e = (double)(gx * gx + gy * gy);
                        if (ax + ay < 8) continue;
                        allE += e;
                        if (std::min(ax, ay) * 6 < std::max(ax, ay)) hvE += e;   // в пределах ~10 градусов
                    }
            }
        int flat = 0, pairs = 0;
        double ls = 0, ls2 = 0;
        int nl = 0;
        for (int bx = 0; bx < bw_; ++bx) {
            if (line[bx] >= bh_) continue;
            ls += line[bx];
            ls2 += (double)line[bx] * line[bx];
            ++nl;
            if (bx + 1 < bw_ && line[bx + 1] < bh_) {
                ++pairs;
                flat += line[bx] == line[bx + 1];
            }
        }
        const float hv = allE > 0 ? (float)(hvE / allE) : 0.f;
        const float fl = pairs ? (float)flat / (float)pairs : 0.f;
        const float lsd = nl ? (float)std::sqrt(std::max(0.0, ls2 / nl - (ls / nl) * (ls / nl))) : 0.f;
        const float a5 = first ? 1.f : std::min(1.f, 5 * a);
        ema(s.hv, hv, a5);
        ema(s.skylineFlat, fl, a5);
        ema(s.skylineSd, lsd * (float)(bs * step), a5);
        // город: кромка из ровных отрезков (крыши) на разной высоте; поле: ровная и низкая;
        // лес: рваная (кроны)
        Terrain t = Terrain::Forest;
        if ((float)nl < 0.3f * (float)bw_) t = Terrain::Unknown;
        else if (s.skylineFlat > 0.8f && s.skylineSd >= 0.02f * (float)h) t = Terrain::Urban;
        else if (s.skylineFlat > 0.6f && s.skylineSd < 0.015f * (float)h) t = Terrain::Open;
        s.terrain = t;
    }

    // свет: по яркости неба и шуму (автоэкспозиция держит небо светлым, но в сумерках
    // поднимает усиление — шум растёт)
    {
        const float hy = s.light == Light::Day ? 1.f : 1.1f;   // гистерезис
        Light l = Light::Day;
        if (s.skyLum < 45.f * hy) l = Light::Night;
        else if (s.skyLum < 100.f * hy || s.noise > 3.f / hy) l = Light::Dusk;
        s.light = l;
    }

    // перемены: резкий скачок яркости неба и сдвиг сцены
    auto event = [&](const char* what) {
        s.lastEvent = what;
        s.lastEventFrame = frame_;
        ++s.events;
    };
    {
        const float af = std::min(1.f, dt / 0.4f), as = std::min(1.f, dt / 6.f);
        if (first) lumFast_ = lumSlow_ = skyLum;
        lumFast_ += af * (skyLum - lumFast_);
        lumSlow_ += as * (skyLum - lumSlow_);
        if (s.analyses > 5 && std::fabs(lumFast_ / std::max(1.f, lumSlow_) - 1.f) > cfg_.light_jump) {
            s.lightJump = true;
            char b[96];
            std::snprintf(b, sizeof(b), "освещение: небо %.0f -> %.0f", lumSlow_, lumFast_);
            event(b);
            lumSlow_ = lumFast_;
            s.skyLum = skyLum;
        }
    }
    if (!movingCamera && !gndM.empty()) {
        if (ref_.size() != nb || refLine_.size() != (size_t)bw_) {
            ref_ = mean_;
            refLine_ = line;
            movedRun_ = 0;
        } else {
            // кромка ушла по высоте или узор земли не совпал с опорным (нормированная
            // корреляция не зависит от экспозиции)
            double sl = 0;
            int nl = 0;
            for (int bx = 0; bx < bw_; ++bx)
                if (line[bx] < bh_ || refLine_[bx] < bh_) {
                    sl += std::abs(line[bx] - refLine_[bx]);
                    ++nl;
                }
            double ma = 0, mb = 0, n = 0;
            for (size_t b = 0; b < nb; ++b)
                if (!sky_[b]) {
                    ma += mean_[b];
                    mb += ref_[b];
                    ++n;
                }
            double ncc = 1;
            if (n > 16) {   // узор земли есть, только если он заметно сильнее шума блока (СКО/4)
                ma /= n;
                mb /= n;
                double sab = 0, saa = 0, sbb = 0;
                for (size_t b = 0; b < nb; ++b)
                    if (!sky_[b]) {
                        const double x = mean_[b] - ma, y = ref_[b] - mb;
                        sab += x * y;
                        saa += x * x;
                        sbb += y * y;
                    }
                const double floor = n * std::max(4.0, (double)sigma * sigma);
                ncc = saa > floor && sbb > floor ? sab / std::sqrt(saa * sbb) : 1;
            }
            // кромка, которая дрожит от анализа к анализу (густой туман: земля по яркости на пороге
            // "темнее неба", столбцы то находят кромку, то нет), сдвигом не считается: камеру
            // повернули — кромка встаёт на новое место и там стоит
            double jl = 0;
            int nj = 0;
            if (lastLine_.size() == line.size())
                for (int bx = 0; bx < bw_; ++bx)
                    if (line[bx] < bh_ || lastLine_[bx] < bh_) {
                        jl += std::abs(line[bx] - lastLine_[bx]);
                        ++nj;
                    }
            const bool steady = !nj || jl / nj <= 1.0;
            const bool off = (nl && sl / nl > 2.0 && steady) || ncc < 0.5;
            movedRun_ = off ? movedRun_ + 1 : 0;
            if (movedRun_ >= 3) {
                s.moved = true;
                event("сцена сдвинулась: маска неба учится заново");
                ref_ = mean_;
                refLine_ = line;
                movedRun_ = 0;
            } else if (!off) {   // медленные перемены (тени, листва) — опорный кадр следует за ними
                for (size_t b = 0; b < nb; ++b) ref_[b] += 0.02f * (mean_[b] - ref_[b]);
            }
        }
    }
    lastLine_ = line;
    s.ready = true;
    ++s.analyses;
    return true;
}

void scene_profile(const SceneState& s, const DetectorConfig& base, const TrackerConfig& baseT, DetectorConfig& d,
                   TrackerConfig& t, std::string* why) {
    d = base;
    t = baseT;
    if (why) why->clear();
    if (!s.ready) return;
    auto note = [&](const char* m) {
        if (!why) return;
        if (!why->empty()) *why += "; ";
        *why += m;
    };
    // Ночь: тёмная цель на тёмном небе не видна, дрон виден по огням — только светлые
    // (и на ~40% быстрее)
    if (s.light == Light::Night && base.dark && base.bright) {
        d.dark = false;
        note("ночь: ищем только огни");
    }
    // Шумная матрица (сумерки, ночь) или дымка: порог по шуму, а не постоянный
    if (!base.auto_thr && (s.noise > 2.6f || s.light != Light::Day || s.haze > 0.58f)) {
        d.auto_thr = true;
        note(s.light != Light::Day ? "мало света: порог по шуму" : s.haze > 0.58f ? "дымка: порог по шуму" : "шум: порог по шуму");
    }
    // Осадки: много коротких точечных вспышек — подтверждать трек строже. Снегопад (вспышек
    // вдвое больше, чем от дождя) — светлые хлопья: днём ищем только тёмные цели. Синтетика:
    // ложных 45 тыс./мин -> 0, но светлый (подсвеченный) дрон в снег не виден
    if (s.precip > 0.3f) {
        const float k = std::min(1.f, (s.precip - 0.3f) / 0.7f);
        t.ev_confirm = baseT.ev_confirm + 3.f * k;
        note("осадки: строже подтверждение");
    }
    if (s.precip > 1.0f && s.light != Light::Night && base.dark && d.bright) {
        d.bright = false;
        note("снег: только тёмные цели");
    }
    // Ветер: листва у кромки "ходит" тем дальше, чем сильнее ветер; в штиль цель, вылетевшая
    // из-за крон, выдаётся раньше
    {
        const float k = std::min(4.f, std::max(1.5f, 1.5f + 1.25f * s.wind));
        t.edge_mover_k = baseT.edge_mover_k > 0 ? k : 0.f;
        if (s.wind > 1.2f) note("ветер у кромки: цель у крон должна пролететь дальше");
        else if (s.wind < 0.4f) note("штиль: у кромки цель выдаётся раньше");
    }
    // Город: антенны, фонари, изоляторы — неподвижные точки, память помех учит их быстрее
    if (s.terrain == Terrain::Urban && baseT.static_frames > 0) {
        t.static_hits = std::min(baseT.static_hits, 5.f);
        note("город: быстрее запоминать неподвижные помехи");
    }
}

bool scene_mask_stale(const DetectorConfig& was, const DetectorConfig& now) {
    return was.auto_thr != now.auto_thr || was.dark != now.dark || was.bright != now.bright;
}

}  // namespace dronolok
