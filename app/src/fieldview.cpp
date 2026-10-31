#include "fieldview.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include "dronolok/fileutil.hpp"
#include "mini_json.hpp"
#include "video_source.hpp"

namespace fieldview {
namespace fs = std::filesystem;

namespace {

bool read_text(const std::string& path, std::string& out) {
    FILE* f = dronolok::open_file(path, "rb");
    if (!f) return false;
    out.clear();
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    fclose(f);
    return true;
}

template <class F>
void each_line(const std::string& text, F&& f) {
    size_t i = 0;
    while (i < text.size()) {
        size_t e = text.find('\n', i);
        if (e == std::string::npos) e = text.size();
        if (e > i) f(text.data() + i, text.data() + e);
        i = e + 1;
    }
}

// Строка frames.jsonl (формат известен — свой разбор, без дерева JSON: час на 14 к/с — 50 тыс. строк)
const char* find_key(const char* b, const char* e, const char* key) {
    const size_t n = strlen(key);
    for (const char* p = b; p + n <= e; ++p)
        if (*p == '"' && !memcmp(p, key, n)) return p + n;
    return nullptr;
}

bool parse_frame(const char* b, const char* e, FrameRow& fr, std::vector<TrackRow>& tracks) {
    std::string line(b, e);   // strtod нужен нуль в конце
    const char* s = line.c_str();
    const char* end = s + line.size();
    const char* p = find_key(s, end, "\"f\":");
    if (!p) return false;
    fr.f = strtoll(p, nullptr, 10);
    if ((p = find_key(s, end, "\"t\":"))) fr.t = strtod(p, nullptr);
    else return false;
    if ((p = find_key(s, end, "\"L\":"))) fr.lock = atoi(p);
    if ((p = find_key(s, end, "\"S\":"))) fr.state = atoi(p);
    if ((p = find_key(s, end, "\"v\":"))) fr.verify = atoi(p);
    if ((p = find_key(s, end, "\"m\":"))) fr.manual = atoi(p) != 0;
    if ((p = find_key(s, end, "\"c\":["))) {
        char* q;
        fr.cdx = strtof(p, &q);
        if (*q == ',') fr.cdy = strtof(q + 1, nullptr);
    }
    fr.trackOff = (uint32_t)tracks.size();
    fr.trackN = 0;
    if ((p = find_key(s, end, "\"T\":["))) {
        while (*p == '[') {
            ++p;
            double v[13] = {};
            int n = 0;
            char* q = (char*)p;
            while (n < 13) {
                v[n++] = strtod(q, &q);
                if (*q == ',') ++q;
                else break;
            }
            while (*q && *q != ']') ++q;   // лишние поля (новая версия журнала) пропускаем
            if (*q != ']') return false;
            ++q;
            if (n >= 13) {
                TrackRow t;
                t.id = (int)v[0];
                t.x = (float)v[1];
                t.y = (float)v[2];
                t.vx = (float)v[3];
                t.vy = (float)v[4];
                t.cls = (int)v[5];
                t.flags = (int)v[6];
                t.bird = (float)v[7];
                t.weight = (float)v[8];
                t.hits = (int)v[9];
                t.miss = (int)v[10];
                t.contrast = (float)v[11];
                t.size = (float)v[12];
                tracks.push_back(t);
                ++fr.trackN;
            }
            if (*q == ',') ++q;
            p = q;
        }
    }
    return true;
}

std::string str(const minijson::Value& v) { return v.type == minijson::Value::Str ? v.s : ""; }

int64_t file_bytes(const fs::path& p) {
    std::error_code ec;
    const auto n = fs::file_size(p, ec);
    return ec ? 0 : (int64_t)n;
}

}  // namespace

int Session::frameAt(int64_t f) const {
    if (frames.empty()) return -1;
    auto it = std::upper_bound(frames.begin(), frames.end(), f, [](int64_t v, const FrameRow& r) { return v < r.f; });
    if (it == frames.begin()) return 0;
    return (int)(it - frames.begin()) - 1;
}

int Session::frameAtTime(double t) const {
    if (frames.empty()) return -1;
    auto it = std::lower_bound(frames.begin(), frames.end(), t, [](const FrameRow& r, double v) { return r.t < v; });
    if (it == frames.end()) return (int)frames.size() - 1;
    if (it != frames.begin() && t - (it - 1)->t < it->t - t) --it;
    return (int)(it - frames.begin());
}

const TrackRow* Session::track(const FrameRow& fr, int id) const {
    for (uint32_t i = 0; i < fr.trackN; ++i)
        if (tracks[fr.trackOff + i].id == id) return &tracks[fr.trackOff + i];
    return nullptr;
}

int Session::snapNear(int64_t f, int64_t maxBack) const {
    int best = -1;
    for (int i = 0; i < (int)snaps.size(); ++i)
        if (snaps[(size_t)i].f <= f && (maxBack < 0 || f - snaps[(size_t)i].f <= maxBack) &&
            (best < 0 || snaps[(size_t)i].f > snaps[(size_t)best].f))
            best = i;
    return best;
}

std::vector<std::string> list_sessions(const std::string& root) {
    std::vector<std::string> out;
    const fs::path r = fs::u8path(root);
    std::error_code ec;
    if (fs::exists(r / "session.json", ec)) return {root};
    std::vector<fs::path> dirs;
    for (fs::directory_iterator it(r, ec), end; !ec && it != end; it.increment(ec))
        if (fs::exists(it->path() / "session.json", ec)) dirs.push_back(it->path());
    std::sort(dirs.begin(), dirs.end(), [](const fs::path& a, const fs::path& b) { return a.filename() > b.filename(); });
    for (const auto& d : dirs) out.push_back(d.u8string());
    return out;
}

bool load(const std::string& dir, Session& s, std::string& err) {
    s = Session();
    s.dir = dir;
    const fs::path d = fs::u8path(dir);
    s.name = d.filename().u8string();
    std::string text;
    if (!read_text((d / "session.json").u8string(), text)) {
        err = "нет session.json в " + dir;
        return false;
    }
    minijson::Value sj;
    if (!minijson::parse(text, sj) || str(sj["format"]) != "dronolok-fieldlog") {
        err = "session.json — не журнал испытаний Дронолока";
        return false;
    }
    s.source = str(sj["source"]);
    s.started = str(sj["started"]);
    s.ended = str(sj["ended"]);
    s.w = (int)sj["w"].num();
    s.h = (int)sj["h"].num();
    s.fps = sj["fps"].num(14);
    s.crop = (int)sj["crop"].num(160);
    s.finished = sj["summary"].type == minijson::Value::Obj;
    {   // настройки — как в файле, одной строкой
        const size_t k = text.find("\"settings\":");
        if (k != std::string::npos) {
            int depth = 0;
            size_t i = k + 11, b = i;
            for (; i < text.size(); ++i) {
                if (text[i] == '{') ++depth;
                else if (text[i] == '}' && --depth == 0) break;
            }
            if (i < text.size()) s.settings = text.substr(b, i - b + 1);
        }
    }
    if (read_text((d / "frames.jsonl").u8string(), text)) {
        s.bytes += (int64_t)text.size();
        s.frames.reserve(text.size() / 60);
        each_line(text, [&](const char* b, const char* e) {
            FrameRow fr;
            if (parse_frame(b, e, fr, s.tracks)) s.frames.push_back(fr);
            else ++s.badLines;   // оборванная строка (пропало питание)
        });
        // перемотка в видео — строки не по порядку; для поиска по кадру нужен порядок
        if (!std::is_sorted(s.frames.begin(), s.frames.end(), [](const FrameRow& a, const FrameRow& b) { return a.t < b.t; }))
            std::stable_sort(s.frames.begin(), s.frames.end(), [](const FrameRow& a, const FrameRow& b) { return a.t < b.t; });
    }
    if (read_text((d / "events.jsonl").u8string(), text)) {
        s.bytes += (int64_t)text.size();
        each_line(text, [&](const char* b, const char* e) {
            minijson::Value v;
            if (!minijson::parse(std::string(b, e), v) || v.type != minijson::Value::Obj) {
                ++s.badLines;
                return;
            }
            Event ev;
            ev.n = (int)v["n"].num();
            ev.type = str(v["type"]);
            ev.f = (int64_t)v["f"].num();
            ev.t = v["t"].num();
            ev.id = (int)v["id"].num();
            ev.from = (int)v["from"].num();
            ev.was = (int)v["was"].num();
            ev.manual = v["manual"].num() != 0;
            ev.locked = v["locked"].num() != 0;
            ev.x = (float)v["x"].num(-1);
            ev.y = (float)v["y"].num(-1);
            ev.bird = (float)v["bird"].num(-1);
            ev.miss = (int)v["miss"].num(-1);
            ev.why = str(v["why"]);
            ev.cls = str(v["cls"]);
            ev.fromCls = str(v["from_cls"]);
            ev.text = str(v["text"]);
            ev.what = str(v["what"]);
            ev.snap = str(v["snap"]);
            ev.source = str(v["source"]);
            for (const auto& c : v["clips"].a)
                if (c.type == minijson::Value::Str) ev.clips.push_back(c.s);
            s.events.push_back(ev);
        });
    }
    if (read_text((d / "snap" / "index.jsonl").u8string(), text)) {
        each_line(text, [&](const char* b, const char* e) {
            minijson::Value v;
            if (!minijson::parse(std::string(b, e), v)) return;
            Snap sn;
            sn.f = (int64_t)v["f"].num();
            sn.k = std::max(1, (int)v["k"].num(1));
            sn.ox = (int)v["ox"].num();
            sn.oy = (int)v["oy"].num();
            sn.why = str(v["why"]);
            char nm[48];
            snprintf(nm, sizeof(nm), "%08lld.jpg", (long long)sn.f);
            sn.file = (d / "snap" / nm).u8string();
            std::error_code ec;
            if (fs::exists(fs::u8path(sn.file), ec)) {
                s.snapBytes += file_bytes(fs::u8path(sn.file));
                s.snaps.push_back(sn);
            }
        });
    }
    std::error_code ec;
    for (fs::directory_iterator it(d / "clips", ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().extension() != ".json") continue;
        if (!read_text(it->path().u8string(), text)) continue;
        minijson::Value v;
        if (!minijson::parse(text, v)) continue;
        Clip c;
        c.name = it->path().stem().u8string();
        c.track = (int)v["track"].num();
        c.size = (int)v["size"].num(s.crop);
        for (const auto& e : v["events"].a) c.events.push_back((int)e.num());
        for (const auto& r : v["frames"].a) {
            if (r.a.size() < 7) continue;
            ClipFrame cf;
            cf.f = (int64_t)r.a[0].num();
            cf.t = r.a[1].num();
            cf.x0 = (int)r.a[2].num();
            cf.y0 = (int)r.a[3].num();
            cf.k = std::max(1, (int)r.a[4].num(1));
            cf.tx = (float)r.a[5].num();
            cf.ty = (float)r.a[6].num();
            c.frames.push_back(cf);
        }
        const fs::path mj = fs::path(it->path()).replace_extension(".mjpeg");
        c.missing = !fs::exists(mj, ec);
        s.clipBytes += file_bytes(mj) + (int64_t)text.size();
        s.clips[c.name] = std::move(c);
    }
    // вырезки, упомянутые в событиях, но удалённые (нехватка места) — пустые с пометкой
    for (const auto& e : s.events)
        for (const auto& c : e.clips)
            if (!s.clips.count(c)) {
                Clip m;
                m.name = c;
                m.track = e.id;
                m.missing = true;
                s.clips[c] = m;
            }
    // итоги по кадрам и событиям (журнал мог оборваться без итогов)
    for (size_t i = 1; i < s.frames.size(); ++i) {
        const double dt = std::min(1.0, std::max(0.0, s.frames[i].t - s.frames[i - 1].t));
        if (s.frames[i - 1].lock) s.lockS += dt;
        if (s.frames[i - 1].state == 2) s.hardS += dt;
    }
    if (!s.frames.empty()) s.duration = s.frames.back().t - s.frames.front().t;
    for (const auto& e : s.events) {
        s.locks += e.type == "lock";
        s.lost += e.type == "lost";
        s.switches += e.type == "switch";
        s.hard += e.type == "hard";
        s.coast += e.type == "coast";
        s.resets += e.type == "reset";
        s.gaps += e.type == "gap";
        if (e.type == "cls") {
            s.droneDecisions += e.cls == "drone";
            s.birdDecisions += e.cls == "bird";
            s.flips += !e.fromCls.empty() && e.fromCls != "unknown";
        }
    }
    s.bytes += s.clipBytes + s.snapBytes;
    return true;
}

bool load_clip(const Session& s, Clip& c) {
    if (c.loaded || c.missing) return !c.missing && !c.px.empty();
    c.loaded = true;
    const std::string path = (fs::u8path(s.dir) / "clips" / fs::u8path(c.name + ".mjpeg")).u8string();
    VideoSource vs;
    if (!vs.open(path, 1, c.err)) {
        c.missing = true;
        return false;
    }
    const int n = c.size > 0 ? c.size : vs.info().w;
    c.size = n;
    std::vector<uint8_t> buf((size_t)n * n);
    while (c.px.size() < std::max<size_t>(c.frames.size(), 1) + 64 && vs.read(buf.data(), n, n, n)) c.px.push_back(buf);
    if (c.px.empty()) {
        c.err = "вырезка не читается";
        c.missing = true;
        return false;
    }
    return true;
}

bool load_image(const std::string& path, std::vector<uint8_t>& px, int& w, int& h, std::string& err) {
    VideoSource vs;
    if (!vs.open(path, 1, err)) return false;
    w = vs.info().w;
    h = vs.info().h;
    if (w <= 0 || h <= 0) return false;
    px.resize((size_t)w * h);
    return vs.read(px.data(), w, h, w);
}

const char* type_ru(const std::string& t) {
    if (t == "lock") return "захват";
    if (t == "lost") return "срыв";
    if (t == "switch") return "смена цели";
    if (t == "hard") return "жёсткий";
    if (t == "hard_off") return "жёсткий снят";
    if (t == "cls") return "дрон/птица";
    if (t == "coast") return "на прогнозе";
    if (t == "bird_ext") return "птица (точная)";
    if (t == "note") return "отметка";
    if (t == "reset") return "заново";
    if (t == "gap") return "пропуск";
    if (t == "start") return "начало";
    if (t == "stop") return "конец";
    return t.c_str();
}

const char* class_ru(int cls) {
    switch (cls) {
        case 1: return "дрон";
        case 2: return "птица";
        case 3: return "помеха";
        default: return "цель?";
    }
}

static const char* cls_ru(const std::string& c) {
    if (c == "drone") return "дрон";
    if (c == "bird") return "птица";
    if (c == "clutter") return "помеха";
    if (c == "leaf") return "лист";
    if (c == "unknown") return "не решено";
    return c.empty() ? "трека нет" : c.c_str();
}

const char* why_ru(const std::string& w) {
    if (w == "track_gone") return "трек пропал (цель не нашлась дольше, чем держится прогноз)";
    if (w == "clutter") return "цель признана помехой";
    if (w == "leaf") return "цель признана листом у камеры";
    if (w == "bird") return "цель признана птицей";
    if (w == "released") return "захват снят, трек ещё есть";
    if (w == "weight") return "другая цель весомее";
    if (w == "manual") return "выбрано вручную";
    if (w == "seek") return "перемотка видео";
    if (w == "back") return "кадры пошли назад (повтор, перезапуск камеры)";
    return w.c_str();
}

std::string describe(const Event& e) {
    char b[512];
    if (e.type == "lock")
        snprintf(b, sizeof(b), "Захват цели %d (%s)%s", e.id, cls_ru(e.cls), e.manual ? ", выбрана вручную" : "");
    else if (e.type == "lost")
        snprintf(b, sizeof(b), "Срыв захвата цели %d%s: %s", e.id, e.was == 2 ? " (был жёсткий)" : "", why_ru(e.why));
    else if (e.type == "switch")
        snprintf(b, sizeof(b), "Захват %d (%s) -> %d (%s)%s: %s", e.from, cls_ru(e.fromCls), e.id, cls_ru(e.cls),
                 e.was == 2 ? ", с жёсткого" : "", why_ru(e.why));
    else if (e.type == "hard") snprintf(b, sizeof(b), "Жёсткий захват цели %d: подтверждено, что дрон", e.id);
    else if (e.type == "hard_off") snprintf(b, sizeof(b), "Жёсткий захват цели %d снят (%s)", e.id, cls_ru(e.cls));
    else if (e.type == "cls")
        snprintf(b, sizeof(b), "Цель %d%s: %s -> %s (птичность %.2f)", e.id, e.locked ? " (захвачена)" : "", cls_ru(e.fromCls),
                 cls_ru(e.cls), e.bird);
    else if (e.type == "coast")
        snprintf(b, sizeof(b), "Захваченная цель %d не видна %d кадров, захват держится на прогнозе", e.id, e.miss);
    else if (e.type == "bird_ext") snprintf(b, sizeof(b), "Точная камера рассмотрела цель %d: птица", e.id);
    else if (e.type == "note")
        snprintf(b, sizeof(b), "%s%s%s%s", e.what.c_str(), e.id ? (" (цель " + std::to_string(e.id) + ")").c_str() : "",
                 e.text.empty() ? "" : ": ", e.text.c_str());
    else if (e.type == "reset") snprintf(b, sizeof(b), "Треки заново: %s", why_ru(e.why));
    else if (e.type == "gap") snprintf(b, sizeof(b), "Пропуск кадров (с кадра %d)", e.from);
    else if (e.type == "start") snprintf(b, sizeof(b), "Начало записи: %s", e.source.c_str());
    else if (e.type == "stop") snprintf(b, sizeof(b), "Конец записи");
    else snprintf(b, sizeof(b), "%s", e.type.c_str());
    return b;
}

int group(const std::string& t) {
    if (t == "lock" || t == "lost" || t == "switch") return 0;
    if (t == "hard" || t == "hard_off") return 1;
    if (t == "cls" || t == "bird_ext") return 2;
    if (t == "coast") return 3;
    return 4;
}

}  // namespace fieldview
