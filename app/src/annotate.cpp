#include "annotate.hpp"

#include <algorithm>
#include <cstdio>

#include "dronolok/fileutil.hpp"
#include "mini_json.hpp"

using namespace dronolok;

void Annotation::load(const std::string& path, int vw, int vh, double vfps) {
    w = vw;
    h = vh;
    fps = vfps;
    objs.clear();
    editable = true;
    FILE* f = open_file(path, "rb");
    if (!f) return;
    std::string text;
    char buf[65536];
    for (size_t n; (n = fread(buf, 1, sizeof(buf), f)) > 0;) text.append(buf, n);
    fclose(f);
    minijson::Value v;
    if (!minijson::parse(text, v)) return;
    const auto& ob = v["objects"];
    if (ob.type != minijson::Value::Arr) {
        editable = v["frames"].a.empty();   // разметка из synth_sky.py: только по кадрам
        return;
    }
    // опорные точки — в координатах того кадра, на котором размечали
    const double sx = v["w"].num() > 0 ? vw / v["w"].num() : 1, sy = v["h"].num() > 0 ? vh / v["h"].num() : 1;
    for (const auto& o : ob.a) {
        AnnObject a;
        a.id = (int)o["id"].num();
        a.bird = o["kind"].s == "bird";
        for (const auto& k : o["keys"].a)
            if (k.a.size() >= 3) a.keys[(int64_t)k.a[0].num()] = {(float)(k.a[1].num() * sx), (float)(k.a[2].num() * sy)};
        if (!a.keys.empty()) objs.push_back(a);
    }
}

bool Annotation::pos(const AnnObject& o, int64_t f, float& x, float& y) {
    if (o.keys.empty()) return false;
    auto hi = o.keys.lower_bound(f);
    if (hi != o.keys.end() && hi->first == f) {
        x = hi->second.first;
        y = hi->second.second;
        return true;
    }
    if (hi == o.keys.begin() || hi == o.keys.end()) return false;   // до первой или после последней точки
    auto lo = std::prev(hi);
    const float k = (float)(f - lo->first) / (float)(hi->first - lo->first);
    x = lo->second.first + k * (hi->second.first - lo->second.first);
    y = lo->second.second + k * (hi->second.second - lo->second.second);
    return true;
}

void Annotation::frameObjects(int64_t f, std::vector<GtObj>& out) const {
    out.clear();
    for (const auto& o : objs) {
        float x, y;
        if (pos(o, f, x, y)) out.push_back({o.bird, false, x, y});
    }
}

GroundTruth Annotation::toGroundTruth(int64_t frames) const {
    GroundTruth g;
    g.w = w;
    g.h = h;
    int64_t last = 0;
    for (const auto& o : objs)
        if (!o.keys.empty()) last = std::max(last, o.keys.rbegin()->first);
    frames = std::max(frames, last + 1);
    if (objs.empty()) return g;
    g.frames.resize((size_t)frames);
    for (int64_t f = 0; f < frames; ++f) frameObjects(f, g.frames[(size_t)f]);
    return g;
}

bool Annotation::save(const std::string& path, int64_t frames) const {
    if (objs.empty()) {   // всё стёрли — файла разметки больше нет
        remove_file(path);
        return true;
    }
    const GroundTruth g = toGroundTruth(frames);
    FILE* f = open_file(path, "w");
    if (!f) return false;
    fprintf(f, "{\"w\": %d, \"h\": %d, \"fps\": %g, \"source\": \"dronolok_app\",\n \"objects\": [", w, h, fps);
    for (size_t i = 0; i < objs.size(); ++i) {
        fprintf(f, "%s\n  {\"id\": %d, \"kind\": \"%s\", \"keys\": [", i ? "," : "", objs[i].id,
                objs[i].bird ? "bird" : "drone");
        bool first = true;
        for (const auto& k : objs[i].keys) {
            fprintf(f, "%s[%lld, %.1f, %.1f]", first ? "" : ", ", (long long)k.first, k.second.first, k.second.second);
            first = false;
        }
        fprintf(f, "]}");
    }
    fprintf(f, "],\n \"frames\": [");
    for (size_t i = 0; i < g.frames.size(); ++i) {
        fprintf(f, "%s\n  [", i ? "," : "");
        for (size_t j = 0; j < g.frames[i].size(); ++j) {
            const GtObj& o = g.frames[i][j];
            fprintf(f, "%s{\"kind\": \"%s\", \"bright\": false, \"x\": %.1f, \"y\": %.1f}", j ? ", " : "",
                    o.bird ? "bird" : "drone", o.x, o.y);
        }
        fprintf(f, "]");
    }
    fprintf(f, "]}\n");
    return fclose(f) == 0;
}

AnnObject* Annotation::find(int id) {
    for (auto& o : objs)
        if (o.id == id) return &o;
    return nullptr;
}

int Annotation::add(bool bird) {
    int id = 1;
    for (const auto& o : objs) id = std::max(id, o.id + 1);
    AnnObject o;
    o.id = id;
    o.bird = bird;
    objs.push_back(o);
    return id;
}
