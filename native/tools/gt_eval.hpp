// Разметка синтетических роликов (*.gt.json от detector/synth_sky.py) и оценка
// качества так же, как в detector/run_demo.py: цель найдена, если рядом (< 10 px)
// есть не-клаттерный подтверждённый трек; остальные треки — ложные.
// Дополнительно — матрица "что на самом деле" x "каким классом назвал трекер".
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "dronolok/detector.hpp"
#include "dronolok/fileutil.hpp"
#include "mini_json.hpp"

namespace dronolok {

struct GtObj {
    bool bird;
    bool bright;
    float x, y;
    int id = -1;          // номер объекта (synth_horizon.py), -1 — нет
    std::string group;    // группа целей (far, high, pair, bird), "" — нет
};

struct GroundTruth {
    int w = 0, h = 0;
    double fps = 0;
    std::vector<std::vector<GtObj>> frames;

    bool load(const std::string& path) {
        FILE* f = open_file(path, "rb");
        if (!f) return false;
        std::string text;
        char buf[65536];
        for (size_t n; (n = std::fread(buf, 1, sizeof(buf), f)) > 0;) text.append(buf, n);
        std::fclose(f);
        minijson::Value v;
        if (!minijson::parse(text, v)) return false;
        // числа из файла — в разумных пределах до приведения к int/float (иначе неопределённое поведение)
        auto clamp = [](double x, double lim) { return std::min(std::max(x, -lim), lim); };
        w = (int)clamp(v["w"].num(), 1 << 20);
        h = (int)clamp(v["h"].num(), 1 << 20);
        fps = v["fps"].num();
        frames.clear();
        for (const auto& fr : v["frames"].a) {
            std::vector<GtObj> objs;
            for (const auto& o : fr.a) {
                GtObj g{o["kind"].s == "bird", o["bright"].num() != 0, (float)clamp(o["x"].num(), 1e7),
                        (float)clamp(o["y"].num(), 1e7)};
                if (o["id"].type == minijson::Value::Num) g.id = (int)clamp(o["id"].num(), 1e9);
                g.group = o["group"].s;
                objs.push_back(g);
            }
            frames.push_back(std::move(objs));
        }
        return !frames.empty();
    }
};

struct EvalStats {
    long hit[2] = {0, 0}, total[2] = {0, 0};   // [0] дроны, [1] птицы
    long fp = 0, frames = 0;
    // ложные тревоги: треки, которые за всю жизнь ни разу не были на настоящей цели
    // (в отличие от fp сюда не попадает трек птицы, отставший от неё на рывке)
    std::map<int, std::pair<bool, long>> life;   // id трека -> (был на цели, кадров выдан)
    long conf[2][4] = {{0}};                    // [дрон/птица][Unknown, Drone, Bird, Clutter]
    // по группам целей: найдено / всего кадров с целью
    std::map<std::string, std::pair<long, long>> groups;
    // задержка обнаружения: кадр появления объекта и кадр, когда его впервые нашли
    std::map<int, std::pair<long, long>> firstSeen;   // id -> (появился, найден или -1)
    std::map<int, std::string> idGroup;

    // sx, sy: масштаб координат разметки к кадру (если видео растянуто)
    void add(const std::vector<GtObj>& gt, const std::vector<Track>& tracks, float sx = 1, float sy = 1) {
        std::vector<const Track*> live;
        for (const auto& t : tracks)
            if (!t.clutter && t.cls != TargetClass::Clutter) live.push_back(&t);
        std::vector<char> matched(live.size(), 0);
        for (const auto& o : gt) {
            const int k = o.bird ? 1 : 0;
            ++total[k];
            bool found = false;
            for (size_t j = 0; j < live.size(); ++j)
                if (std::fabs(live[j]->x - o.x * sx) < 10 && std::fabs(live[j]->y - o.y * sy) < 10) {
                    ++hit[k];
                    matched[j] = 1;
                    ++conf[k][(int)live[j]->cls];
                    found = true;
                    break;
                }
            if (!o.group.empty()) {
                auto& g = groups[o.group];
                g.first += found;
                ++g.second;
            }
            if (o.id >= 0) {
                auto it = firstSeen.find(o.id);
                if (it == firstSeen.end()) it = firstSeen.emplace(o.id, std::make_pair(frames, -1L)).first;
                if (found && it->second.second < 0) it->second.second = frames;
                idGroup[o.id] = o.group;
            }
        }
        long m = 0;
        for (char c : matched) m += c;
        fp += (long)live.size() - m;
        for (size_t j = 0; j < live.size(); ++j) {
            auto& l = life[live[j]->id];
            l.first |= matched[j] != 0;
            ++l.second;
        }
        ++frames;
    }
    double recall(int k) const { return total[k] ? (double)hit[k] / total[k] : 0; }
    double falsePerFrame() const { return frames ? (double)fp / frames : 0; }
    // ложные тревоги: сколько их было и сколько кадров на кадр они в среднем висели
    long falseAlarms() const {
        long n = 0;
        for (const auto& kv : life) n += !kv.second.first;
        return n;
    }
    double falseAlarmPerFrame() const {
        long n = 0;
        for (const auto& kv : life) n += kv.second.first ? 0 : kv.second.second;
        return frames ? (double)n / frames : 0;
    }
    // средняя задержка обнаружения группы, кадров; не найденные ни разу — в missed
    double meanDelay(const std::string& group, int& found, int& missed) const {
        double s = 0;
        found = missed = 0;
        for (const auto& kv : firstSeen) {
            if (idGroup.at(kv.first) != group) continue;
            if (kv.second.second < 0) { ++missed; continue; }
            s += (double)(kv.second.second - kv.second.first);
            ++found;
        }
        return found ? s / found : 0;
    }
    // доля правильно названных среди найденных и классифицированных (Drone/Bird)
    double clsAccuracy(int k) const {
        const long ok = conf[k][k == 0 ? 1 : 2], n = conf[k][1] + conf[k][2];
        return n ? (double)ok / n : 0;
    }
};

}  // namespace dronolok
