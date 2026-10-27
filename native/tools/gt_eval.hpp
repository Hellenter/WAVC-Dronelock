// Разметка синтетических роликов (*.gt.json от detector/synth_sky.py) и оценка
// качества так же, как в detector/run_demo.py: цель найдена, если рядом (< 10 px)
// есть не-клаттерный подтверждённый трек; остальные треки — ложные.
// Дополнительно — матрица "что на самом деле" x "каким классом назвал трекер".
#pragma once
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "dronolok/detector.hpp"
#include "mini_json.hpp"

namespace dronolok {

struct GtObj {
    bool bird;
    bool bright;
    float x, y;
};

struct GroundTruth {
    int w = 0, h = 0;
    std::vector<std::vector<GtObj>> frames;

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        std::stringstream ss;
        ss << f.rdbuf();
        minijson::Value v;
        if (!minijson::parse(ss.str(), v)) return false;
        w = (int)v["w"].num();
        h = (int)v["h"].num();
        frames.clear();
        for (const auto& fr : v["frames"].a) {
            std::vector<GtObj> objs;
            for (const auto& o : fr.a)
                objs.push_back({o["kind"].s == "bird", o["bright"].num() != 0, (float)o["x"].num(), (float)o["y"].num()});
            frames.push_back(std::move(objs));
        }
        return !frames.empty();
    }
};

struct EvalStats {
    long hit[2] = {0, 0}, total[2] = {0, 0};   // [0] дроны, [1] птицы
    long fp = 0, frames = 0;
    long conf[2][4] = {{0}};                    // [дрон/птица][Unknown, Drone, Bird, Clutter]

    // sx, sy: масштаб координат разметки к кадру (если видео растянуто)
    void add(const std::vector<GtObj>& gt, const std::vector<Track>& tracks, float sx = 1, float sy = 1) {
        std::vector<const Track*> live;
        for (const auto& t : tracks)
            if (!t.clutter) live.push_back(&t);
        std::vector<char> matched(live.size(), 0);
        for (const auto& o : gt) {
            const int k = o.bird ? 1 : 0;
            ++total[k];
            for (size_t j = 0; j < live.size(); ++j)
                if (std::fabs(live[j]->x - o.x * sx) < 10 && std::fabs(live[j]->y - o.y * sy) < 10) {
                    ++hit[k];
                    matched[j] = 1;
                    ++conf[k][(int)live[j]->cls];
                    break;
                }
        }
        long m = 0;
        for (char c : matched) m += c;
        fp += (long)live.size() - m;
        ++frames;
    }
    double recall(int k) const { return total[k] ? (double)hit[k] / total[k] : 0; }
    double falsePerFrame() const { return frames ? (double)fp / frames : 0; }
    // доля правильно названных среди найденных и классифицированных (Drone/Bird)
    double clsAccuracy(int k) const {
        const long ok = conf[k][k == 0 ? 1 : 2], n = conf[k][1] + conf[k][2];
        return n ? (double)ok / n : 0;
    }
};

}  // namespace dronolok
