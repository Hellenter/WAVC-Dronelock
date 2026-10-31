// Журнал полевых испытаний (см. dronolok/fieldlog.hpp)
#include "dronolok/fieldlog.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <thread>

#include "dronolok/fileutil.hpp"

namespace dronolok {
namespace fs = std::filesystem;

namespace {

fs::path upath(const std::string& s) { return fs::u8path(s); }
std::string ustr(const fs::path& p) { return p.u8string(); }

// Строка для JSON: кавычки, обратная косая, управляющие символы
std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o += '\\', o += (char)c;
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) {
            char b[8];
            snprintf(b, sizeof(b), "\\u%04x", c);
            o += b;
        } else o += (char)c;
    }
    return o + "\"";
}

std::string iso_time(std::time_t t, bool utc) {
    std::tm tm{};
#ifdef _WIN32
    if (utc) gmtime_s(&tm, &t);
    else localtime_s(&tm, &t);
#else
    if (utc) gmtime_r(&t, &tm);
    else localtime_r(&t, &tm);
#endif
    char b[40];
    std::strftime(b, sizeof(b), utc ? "%Y-%m-%dT%H:%M:%SZ" : "%Y-%m-%dT%H:%M:%S", &tm);
    return b;
}

std::string stamp_name(std::time_t t) {
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char b[32];
    std::strftime(b, sizeof(b), "%Y%m%d-%H%M%S", &tm);
    return b;
}

const char* FORMAT_TAG = "dronolok-fieldlog";

// Папка — журнал Дронолока (её можно удалять при нехватке места): в ней session.json с нашей меткой.
// Чужие папки в root не трогаются никогда
bool is_session_dir(const fs::path& d) {
    std::error_code ec;
    if (!fs::is_directory(d, ec)) return false;
    FILE* f = open_file(ustr(d / "session.json"), "rb");
    if (!f) return false;
    char buf[256] = {};
    const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;
    return std::strstr(buf, FORMAT_TAG) != nullptr;
}

int64_t dir_bytes(const fs::path& d) {
    int64_t s = 0;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(d, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code e2;
        if (it->is_regular_file(e2)) {
            const auto sz = it->file_size(e2);
            if (!e2) s += (int64_t)sz;
        }
    }
    return s;
}

bool append_file(const std::string& path, const void* data, size_t n) {
    FILE* f = open_file(path, "ab");
    if (!f) return false;
    const bool ok = fwrite(data, 1, n, f) == n;
    return std::fclose(f) == 0 && ok;
}

const char* class_short(const Track& t) {
    if (t.clutter || t.cls == TargetClass::Clutter) return "clutter";
    if (t.leaf) return "leaf";
    return class_name(t.cls);
}

// Вырезка: окно side = crop * k вокруг (cx, cy), прижатое к краям кадра, уменьшенное в k раз (среднее k x k)
struct Crop {
    int id = 0;
    int x0 = 0, y0 = 0, k = 1;   // угол окна в кадре (полном, с учётом ox/oy) и во сколько раз уменьшено
    float tx = 0, ty = 0;        // где была цель (полный кадр)
    std::vector<uint8_t> px;     // crop x crop
};

void cut(const FieldFrame& f, int crop, int k, float cx, float cy, Crop& c) {
    const int side = crop * k;
    int x0 = (int)std::lround(cx) - side / 2, y0 = (int)std::lround(cy) - side / 2;
    x0 = std::max(0, std::min(x0, f.w - side));
    y0 = std::max(0, std::min(y0, f.h - side));
    c.x0 = x0 + f.ox;
    c.y0 = y0 + f.oy;
    c.k = k;
    c.px.resize((size_t)crop * crop);
    if (k == 1 && side <= f.w && side <= f.h) {   // окно целиком в кадре: строки как есть
        for (int y = 0; y < crop; ++y) std::memcpy(c.px.data() + (size_t)y * crop, f.gray + (size_t)(y0 + y) * f.stride + x0, (size_t)crop);
        return;
    }
    // уменьшение: среднее 2x2 отсчётов внутри блока k x k (при k = 4 — через пиксель: вчетверо меньше чтений)
    const int a0 = k / 4, a1 = k > 1 ? (3 * k) / 4 : 0;
    for (int y = 0; y < crop; ++y) {
        const int ya = std::max(0, std::min(f.h - 1, y0 + y * k + a0));
        const int yb = std::max(0, std::min(f.h - 1, y0 + y * k + a1));
        const uint8_t* ra = f.gray + (size_t)ya * f.stride;
        const uint8_t* rb = f.gray + (size_t)yb * f.stride;
        uint8_t* o = c.px.data() + (size_t)y * crop;
        for (int x = 0; x < crop; ++x) {
            const int xa = std::max(0, std::min(f.w - 1, x0 + x * k + a0));
            const int xb = std::max(0, std::min(f.w - 1, x0 + x * k + a1));
            o[x] = (uint8_t)((ra[xa] + ra[xb] + rb[xa] + rb[xb] + 2) / 4);
        }
    }
}

}  // namespace

const char* FieldLog::trackFields() {
    return "[\"id\",\"x\",\"y\",\"vx\",\"vy\",\"cls\",\"flags\",\"bird\",\"weight\",\"hits\",\"miss\",\"contrast\",\"size\"]";
}

struct FieldLog::Impl {
    FieldLogConfig cfg;
    int w = 0, h = 0;
    double fps = 14;
    fs::path root, dir;
    std::string dirU8, source, settings;
    std::time_t started = 0;

    // ---- поток записи ----
    std::thread th;
    std::mutex qm;
    std::condition_variable qcv;
    struct Job {
        std::function<void()> run;
        size_t bytes;
    };
    std::deque<Job> q;
    size_t qBytes = 0;
    bool stop = false;
    static constexpr size_t Q_MAX = 64u << 20;   // столько ждёт записи, больше — новые вырезки отбрасываются
    // ---- место на диске (меняется в потоке записи) ----
    std::atomic<int64_t> used{0};      // все журналы в root
    std::atomic<bool> lowSpace{false}, diskFull{false};
    std::atomic<int> dropped{0};
    std::chrono::steady_clock::time_point spaceAt{};

    // ---- состояние кадров (под m) ----
    mutable std::mutex m;
    FieldLogStats st;
    std::string framesBuf;
    double flushT = -1e9;
    bool havePrev = false;
    int64_t prevIdx = 0;
    double prevT = 0, firstT = -1;
    int prevLock = 0;
    LockState prevState = LockState::None;
    bool prevManual = false;
    int prevMiss = 0;
    int eventN = 0;
    struct TrackInfo {
        TargetClass decided = TargetClass::Unknown;
        bool extBird = false;
        double clsT = -1e9, seenT = 0;
        float x = 0, y = 0;
        int k = 1;
    };
    std::map<int, TrackInfo> info;
    struct RingFrame {
        int64_t f;
        double t;
        std::vector<Crop> crops;
    };
    std::deque<RingFrame> ring;
    struct ClipFrame {
        int64_t f;
        double t;
        Crop c;
    };
    struct Clip {
        std::string name;
        int id = 0;
        double startT = 0, endT = 0;
        int64_t lastF = -1;
        float x = 0, y = 0;   // где цель была последний раз (полный кадр)
        int k = 1;
        std::vector<int> events;
        std::vector<ClipFrame> frames;
    };
    std::vector<Clip> clips;
    std::deque<double> clipTimes;
    double snapT = -1e9, overviewT = -1e9;

    // ---------------- запись ----------------
    void push(std::function<void()> f, size_t bytes, bool droppable) {
        {
            std::lock_guard<std::mutex> lk(qm);
            if (droppable && qBytes + bytes > Q_MAX) {
                ++dropped;
                return;
            }
            q.push_back({std::move(f), bytes});
            qBytes += bytes;
        }
        qcv.notify_one();
    }
    void writer() {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(qm);
                qcv.wait(lk, [&] { return stop || !q.empty(); });
                if (q.empty()) return;
                j = std::move(q.front());
                q.pop_front();
                qBytes -= j.bytes;
            }
            j.run();
            checkSpace(false);
        }
    }
    void wrote(int64_t n) { used += n; }
    bool writeAppend(const fs::path& p, const std::string& s) {
        if (s.empty()) return true;
        const bool ok = append_file(ustr(p), s.data(), s.size());
        if (ok) wrote((int64_t)s.size());
        return ok;
    }
    bool writeWhole(const fs::path& p, const void* data, size_t n) {
        std::error_code ec;
        const auto old = fs::exists(p, ec) ? fs::file_size(p, ec) : 0;
        if (!ec && old) used -= (int64_t)old;   // файл заменяется
        const bool ok = write_file_replace(ustr(p), "wb", [&](FILE* f) { return fwrite(data, 1, n, f) == n; });
        if (ok) wrote((int64_t)n);
        return ok;
    }

    // Место: предел журналов (max_mb) и свободное на диске (min_free_mb). Вызывается в потоке записи
    void checkSpace(bool force) {
        const auto now = std::chrono::steady_clock::now();
        if (force || now - spaceAt > std::chrono::seconds(10)) {
            spaceAt = now;
            std::error_code ec;
            const auto sp = fs::space(root, ec);
            if (!ec) {
                const double freeMb = (double)sp.available / (1 << 20);
                lowSpace = freeMb < cfg.min_free_mb;
                if (freeMb < cfg.min_free_mb / 4) diskFull = true;
            }
        }
        const int64_t lim = (int64_t)(cfg.max_mb * (1 << 20));
        while (used > lim && !diskFull)
            if (!prune()) diskFull = true;   // удалить не вышло (файлы заняты, нет прав) — писать некуда
    }
    // Освободить место: вырезки и снимки старых выездов, старые выезды, вырезки этого; иначе — полон.
    // false — ничего не удалилось
    bool prune() {
        const int64_t before = used;
        std::vector<fs::path> others;
        std::error_code ec;
        for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec))
            if (it->path() != dir && is_session_dir(it->path())) others.push_back(it->path());
        std::sort(others.begin(), others.end());   // имена — дата и время: старые первыми
        for (const auto& d : others)
            for (const char* sub : {"clips", "snap"}) {
                const fs::path p = d / sub;
                if (!fs::exists(p, ec)) continue;
                const int64_t b = dir_bytes(p);
                fs::remove_all(p, ec);
                used -= b - (ec ? dir_bytes(p) : 0);
                return used < before;
            }
        if (!others.empty()) {
            const int64_t b = dir_bytes(others.front());
            fs::remove_all(others.front(), ec);
            used -= b - (ec ? dir_bytes(others.front()) : 0);
            return used < before;
        }
        std::vector<fs::path> mine;
        for (const char* sub : {"clips", "snap"})
            for (fs::directory_iterator it(dir / sub, ec), end; !ec && it != end; it.increment(ec))
                if (it->is_regular_file() && it->path().filename() != "index.jsonl") mine.push_back(it->path());
        if (mine.empty()) return false;
        // самые старые по имени (номер события или кадра), снимки и вырезки поровну
        std::sort(mine.begin(), mine.end(), [](const fs::path& a, const fs::path& b) {
            return a.filename() < b.filename();
        });
        for (size_t i = 0; i < std::min<size_t>(mine.size(), 8); ++i) {
            std::error_code e2;
            const auto sz = fs::file_size(mine[i], e2);
            if (!e2 && fs::remove(mine[i], e2)) used -= (int64_t)sz;
        }
        return used < before;
    }

    // ---------------- кадры ----------------
    std::string eventLine(const char* type, double t, int64_t f, const std::string& rest) {
        char b[160];
        snprintf(b, sizeof(b), "{\"n\":%d,\"type\":\"%s\",\"f\":%lld,\"t\":%.3f", ++eventN, type, (long long)f, t);
        ++st.events;
        return std::string(b) + rest + "}\n";
    }
    void writeEvent(const std::string& line) {
        push([this, line] { writeAppend(dir / "events.jsonl", line); }, line.size(), false);
    }
    // minor — вырезка второстепенного события (решение по цели без захвата, отметка извне): ей — не больше
    // половины предела в минуту, чтобы срывы и смены захвата всегда успевали записаться
    bool clipAllowed(double t, bool minor) {
        while (!clipTimes.empty() && t - clipTimes.front() > 60) clipTimes.pop_front();
        const int lim = minor ? cfg.clips_per_min / 2 : cfg.clips_per_min;
        if (lowSpace || diskFull || (int)clipTimes.size() >= lim) {
            ++st.clipsSkipped;
            return false;
        }
        return true;
    }
    // Вырезка вокруг трека id: уже пишется — продлить, иначе начать с кадров кольца. "" — не пишется
    std::string clipFor(int id, double t, int ev, bool minor = false) {
        if (!id || cfg.crop_px <= 0) return "";
        for (auto& c : clips)
            if (c.id == id) {
                c.endT = std::min(c.startT + cfg.max_clip_s, std::max(c.endT, t + cfg.post_s));
                c.events.push_back(ev);
                return c.name;
            }
        if (!clipAllowed(t, minor)) return "";
        Clip c;
        char nm[48];
        snprintf(nm, sizeof(nm), "%06d_%d", ev, id);
        c.name = nm;
        c.id = id;
        c.startT = t;
        c.endT = t + cfg.post_s;
        c.events.push_back(ev);
        for (const auto& rf : ring) {
            if (rf.t < t - cfg.pre_s - 1e-6) continue;
            for (const auto& cr : rf.crops)
                if (cr.id == id) {
                    c.frames.push_back({rf.f, rf.t, cr});
                    c.lastF = rf.f;
                    c.x = cr.tx, c.y = cr.ty, c.k = cr.k;
                }
        }
        if (c.frames.empty()) {
            auto it = info.find(id);
            if (it == info.end()) return "";   // трека не видели — вырезать негде
            c.x = it->second.x, c.y = it->second.y, c.k = it->second.k;
        }
        clipTimes.push_back(t);
        clips.push_back(std::move(c));
        ++st.clips;
        return clips.back().name;
    }
    void finishClip(Clip& c) {
        if (c.frames.empty()) return;
        auto shared = std::make_shared<Clip>(std::move(c));
        size_t bytes = 0;
        for (const auto& fr : shared->frames) bytes += fr.c.px.size();
        const int crop = cfg.crop_px, q = cfg.jpeg_q;
        const double fpsv = fps;
        push(
            [this, shared, crop, q, fpsv] {
                if (lowSpace || diskFull) return;
                std::string mj;
                std::string meta = "{\"track\":" + std::to_string(shared->id) + ",\"size\":" + std::to_string(crop) +
                                   ",\"fps\":" + std::to_string(fpsv) + ",\"events\":[";
                for (size_t i = 0; i < shared->events.size(); ++i)
                    meta += (i ? "," : "") + std::to_string(shared->events[i]);
                meta += "],\"fields\":[\"f\",\"t\",\"x0\",\"y0\",\"k\",\"tx\",\"ty\"],\"frames\":[\n";
                for (size_t i = 0; i < shared->frames.size(); ++i) {
                    const auto& fr = shared->frames[i];
                    const auto jpg = encode_jpeg_gray(fr.c.px.data(), crop, crop, crop, q);
                    mj.append((const char*)jpg.data(), jpg.size());
                    char b[160];
                    snprintf(b, sizeof(b), "%s[%lld,%.3f,%d,%d,%d,%.1f,%.1f]", i ? ",\n" : "", (long long)fr.f, fr.t,
                             fr.c.x0, fr.c.y0, fr.c.k, fr.c.tx, fr.c.ty);
                    meta += b;
                }
                meta += "]}\n";
                std::error_code ec;
                fs::create_directories(dir / "clips", ec);
                writeWhole(dir / "clips" / upath(shared->name + ".mjpeg"), mj.data(), mj.size());
                writeWhole(dir / "clips" / upath(shared->name + ".json"), meta.data(), meta.size());
            },
            bytes, true);
    }
    // Снимок всего кадра (уменьшенный): копия кадра — в поток записи, там уменьшение и JPEG
    std::string snap(const FieldFrame& f, const char* why) {
        if (cfg.snap_px <= 0 || !f.gray || lowSpace || diskFull) return "";
        const int k = std::max(1, (f.w + cfg.snap_px - 1) / cfg.snap_px);
        const int sw = f.w / k, sh = f.h / k;
        if (sw < 8 || sh < 8) return "";
        // уменьшаем сразу (кадр после возврата из frame() не наш): среднее 2x2 отсчётов на блок k x k —
        // 4 чтения на пиксель снимка при любом k (на 4608 x 2592 ~1 мс на ПК, единицы мс на Pi)
        auto img = std::make_shared<std::vector<uint8_t>>((size_t)sw * sh);
        const int a0 = k / 4, a1 = k > 1 ? (3 * k) / 4 : 0;
        for (int y = 0; y < sh; ++y) {
            const uint8_t* ra = f.gray + (size_t)(y * k + a0) * f.stride;
            const uint8_t* rb = f.gray + (size_t)(y * k + a1) * f.stride;
            uint8_t* o = img->data() + (size_t)y * sw;
            for (int x = 0; x < sw; ++x) {
                const int xa = x * k + a0, xb = x * k + a1;
                o[x] = (uint8_t)((ra[xa] + ra[xb] + rb[xa] + rb[xb] + 2) / 4);
            }
        }
        char nm[48];
        snprintf(nm, sizeof(nm), "%08lld.jpg", (long long)f.index);
        const std::string name = nm;
        const int q = cfg.jpeg_q;
        const int ox = f.ox, oy = f.oy;
        const std::string meta = "{\"f\":" + std::to_string(f.index) + ",\"k\":" + std::to_string(k) + ",\"ox\":" +
                                 std::to_string(ox) + ",\"oy\":" + std::to_string(oy) + ",\"why\":" + jstr(why) + "}\n";
        push(
            [this, img, sw, sh, q, name, meta] {
                if (lowSpace || diskFull) return;
                const auto jpg = encode_jpeg_gray(img->data(), sw, sh, sw, q);
                std::error_code ec;
                fs::create_directories(dir / "snap", ec);
                writeWhole(dir / "snap" / upath(name), jpg.data(), jpg.size());
                writeAppend(dir / "snap" / "index.jsonl", meta);
            },
            img->size(), true);
        ++st.snaps;
        return name;
    }
    int zoomFor(const Track& t) const {
        float sz = std::max(t.bw, t.bh);
        if (!(sz > 0)) sz = t.expectedSize();
        const float need = 4.f * sz;   // цель — не больше четверти вырезки
        return need > 2.f * cfg.crop_px ? 4 : need > cfg.crop_px ? 2 : 1;
    }

    void frame(const FieldFrame& f) {
        static const std::vector<Track> none;
        const std::vector<Track>& tr = f.tracks ? *f.tracks : none;
        const double t = f.t;
        // перемотка, повтор ролика, перезапуск камеры: события до и после не связаны
        if (havePrev && (f.index < prevIdx || t < prevT)) reset(t, f.index, "back");
        // пропуск больше 2 с (камера стояла, перемотка вперёд): вырезки не тянуть через дыру
        else if (havePrev && t - prevT > 2.0) {
            for (auto& c : clips) finishClip(c);
            clips.clear();
            ring.clear();
            writeEvent(eventLine("gap", t, f.index, ",\"from\":" + std::to_string(prevIdx) + ",\"s\":" + std::to_string(t - prevT)));
        }
        const double dt = havePrev ? std::min(1.0, std::max(0.0, t - prevT)) : 0.0;
        if (firstT < 0) firstT = t;
        ++st.frames;
        if (prevLock) st.lockS += dt;
        if (prevState == LockState::Hard) st.hardS += dt;

        // строка кадра
        if (!diskFull) {
            char b[256];
            int n = snprintf(b, sizeof(b), "{\"f\":%lld,\"t\":%.3f", (long long)f.index, t);
            std::string line(b, (size_t)n);
            if (f.lockId) {
                n = snprintf(b, sizeof(b), ",\"L\":%d,\"S\":%d,\"v\":%d%s", f.lockId, (int)f.lockState, f.verify,
                             f.manual ? ",\"m\":1" : "");
                line.append(b, (size_t)n);
            }
            if (f.camDx != 0 || f.camDy != 0) {
                n = snprintf(b, sizeof(b), ",\"c\":[%.2f,%.2f]", f.camDx, f.camDy);
                line.append(b, (size_t)n);
            }
            bool first = true;
            for (const auto& x : tr) {
                if (!cfg.all_tracks && x.id != f.lockId) continue;
                const int flags = (x.solid ? 1 : 0) | (x.clutter ? 2 : 0) | (x.leaf ? 4 : 0) | (x.ghost ? 8 : 0) |
                                  (x.extBird ? 16 : 0) | (x.followed ? 32 : 0) | (x.hovering ? 64 : 0) |
                                  (x.anchored ? 128 : 0);
                float sz = std::max(x.bw, x.bh);
                if (!(sz > 0)) sz = x.expectedSize();
                n = snprintf(b, sizeof(b), "%s[%d,%.1f,%.1f,%.2f,%.2f,%d,%d,%.2f,%.2f,%d,%d,%.1f,%.1f]",
                             first ? ",\"T\":[" : ",", x.id, x.x + (float)f.ox, x.y + (float)f.oy, x.vx, x.vy, (int)x.cls,
                             flags, x.birdScore, x.lockW, x.hits, x.miss, x.contrast, sz);
                line.append(b, (size_t)n);
                first = false;
            }
            if (!first) line += "]";
            line += "}\n";
            framesBuf += line;
        }

        // кого держать в кольце вырезок: захваченную и первые цели очереди
        std::vector<int> want;
        if (f.lockId) want.push_back(f.lockId);
        if (f.queue) {
            for (int id : *f.queue)
                if ((int)want.size() < cfg.ring_tracks + 1 && id != f.lockId) want.push_back(id);
        } else {
            std::vector<const Track*> by;
            for (const auto& x : tr)
                if (!x.clutter && x.id != f.lockId) by.push_back(&x);
            std::sort(by.begin(), by.end(), [](const Track* a, const Track* b) { return a->lockW > b->lockW; });
            for (const Track* x : by)
                if ((int)want.size() < cfg.ring_tracks + 1) want.push_back(x->id);
        }
        auto findTrack = [&](int id) -> const Track* {
            for (const auto& x : tr)
                if (x.id == id) return &x;
            return nullptr;
        };
        for (const auto& x : tr) {
            auto& ti = info[x.id];
            ti.seenT = t;
            ti.x = x.x + (float)f.ox;
            ti.y = x.y + (float)f.oy;
            ti.k = zoomFor(x);
        }
        if (f.gray && cfg.crop_px > 0) {
            RingFrame rf{f.index, t, {}};
            for (int id : want)
                if (const Track* x = findTrack(id)) {
                    Crop c;
                    c.id = id;
                    c.tx = x->x + (float)f.ox;
                    c.ty = x->y + (float)f.oy;
                    cut(f, cfg.crop_px, zoomFor(*x), x->x, x->y, c);
                    rf.crops.push_back(std::move(c));
                }
            ring.push_back(std::move(rf));
            while (!ring.empty() && ring.front().t < t - cfg.pre_s - 1e-6) ring.pop_front();
        }

        // события захвата
        const int a = prevLock, bId = f.lockId;
        auto clsOf = [&](int id) -> std::string {
            const Track* x = findTrack(id);
            return x ? jstr(class_short(*x)) : "null";
        };
        auto clipsJson = [](const std::vector<std::string>& v) {
            std::string s = ",\"clips\":[";
            bool first = true;
            for (const auto& c : v)
                if (!c.empty()) s += (first ? "" : ",") + jstr(c), first = false;
            return s + "]";
        };
        auto emit = [&](const char* type, const std::string& rest, std::vector<int> clipIds, bool withSnap, bool minor = false) {
            const int ev = eventN + 1;
            std::vector<std::string> names;
            for (int id : clipIds) names.push_back(clipFor(id, t, ev, minor));
            std::string extra = rest + clipsJson(names);
            if (withSnap && t - snapT >= 2.0) {
                const std::string s = snap(f, type);
                if (!s.empty()) {
                    snapT = t;
                    extra += ",\"snap\":" + jstr(s);
                }
            }
            writeEvent(eventLine(type, t, f.index, extra));
        };
        auto pos = [&](int id) {
            const Track* x = findTrack(id);
            char b[96];
            if (x) snprintf(b, sizeof(b), ",\"x\":%.1f,\"y\":%.1f,\"miss\":%d", x->x + (float)f.ox, x->y + (float)f.oy, x->miss);
            else {
                auto it = info.find(id);
                if (it == info.end()) return std::string();
                snprintf(b, sizeof(b), ",\"x\":%.1f,\"y\":%.1f", it->second.x, it->second.y);
            }
            return std::string(b);
        };
        if (!a && bId) {
            ++st.locks;
            emit("lock", ",\"id\":" + std::to_string(bId) + ",\"cls\":" + clsOf(bId) + pos(bId) + (f.manual ? ",\"manual\":1" : ""),
                 {bId}, true);
        } else if (a && !bId) {
            ++st.lost;
            const Track* x = findTrack(a);
            const char* why = !x ? "track_gone" : x->clutter ? "clutter" : x->leaf ? "leaf"
                              : x->cls == TargetClass::Bird || x->extBird ? "bird" : "released";
            emit("lost", ",\"id\":" + std::to_string(a) + ",\"why\":\"" + why + "\",\"cls\":" + clsOf(a) + pos(a) +
                             ",\"was\":" + std::to_string((int)prevState),
                 {a}, true);
        } else if (a && bId && a != bId) {
            ++st.switches;
            const Track* x = findTrack(a);
            const char* why = f.manual && !prevManual ? "manual" : !x ? "track_gone"
                              : x->cls == TargetClass::Bird || x->extBird ? "bird" : "weight";
            emit("switch", ",\"from\":" + std::to_string(a) + ",\"id\":" + std::to_string(bId) + ",\"why\":\"" + why +
                               "\",\"cls\":" + clsOf(bId) + ",\"from_cls\":" + clsOf(a) + pos(bId) +
                               ",\"was\":" + std::to_string((int)prevState),
                 {a, bId}, true);
        } else if (a && a == bId) {
            if (prevState != LockState::Hard && f.lockState == LockState::Hard) {
                ++st.hard;
                emit("hard", ",\"id\":" + std::to_string(a) + ",\"cls\":" + clsOf(a) + pos(a), {a}, false);
            } else if (prevState == LockState::Hard && f.lockState != LockState::Hard) {
                emit("hard_off", ",\"id\":" + std::to_string(a) + ",\"cls\":" + clsOf(a) + pos(a), {a}, false);
            }
        }
        // захваченная цель ушла на прогноз (не видна) дольше полсекунды
        const Track* lk = bId ? findTrack(bId) : nullptr;
        const int coastN = std::max(2, (int)std::lround(0.5 * fps));
        const int miss = lk ? lk->miss : 0;
        if (bId && bId == a && miss >= coastN && prevMiss < coastN)
            emit("coast", ",\"id\":" + std::to_string(bId) + ",\"cls\":" + clsOf(bId) + pos(bId), {bId}, false);
        prevMiss = miss;

        // решение дрон/птица: у захваченной и первых целей очереди. Вырезка — у захваченной и при смене
        // решения (был дрон — стал птицей и наоборот); первое решение по цели без захвата — только строкой
        for (const auto& x : tr) {
            auto& ti = info[x.id];
            const bool top = std::find(want.begin(), want.end(), x.id) != want.end();
            const TargetClass c = x.clutter ? TargetClass::Unknown : x.cls;
            if ((c == TargetClass::Drone || c == TargetClass::Bird) && c != ti.decided) {
                const TargetClass was = ti.decided;
                ti.decided = c;
                if (top && t - ti.clsT >= 1.0) {
                    ti.clsT = t;
                    const bool locked = x.id == bId, flip = was != TargetClass::Unknown;
                    char b[160];
                    snprintf(b, sizeof(b), ",\"id\":%d,\"cls\":\"%s\",\"from_cls\":\"%s\",\"bird\":%.2f,\"locked\":%d",
                             x.id, class_name(c), class_name(was), x.birdScore, locked ? 1 : 0);
                    std::vector<int> ids;
                    if (locked || flip) ids.push_back(x.id);
                    emit("cls", std::string(b) + pos(x.id), ids, false, !locked);
                }
            }
            if (x.extBird && !ti.extBird) {
                ti.extBird = true;
                emit("bird_ext", ",\"id\":" + std::to_string(x.id) + pos(x.id), {x.id}, false, x.id != bId);
            }
        }

        // вырезки, которые пишутся: кадр за кадром до конца
        for (auto& c : clips) {
            if (f.index <= c.lastF || !f.gray) continue;
            const Track* x = findTrack(c.id);
            Crop cr;
            cr.id = c.id;
            if (x) c.x = x->x + (float)f.ox, c.y = x->y + (float)f.oy, c.k = zoomFor(*x);
            cr.tx = c.x, cr.ty = c.y;
            cut(f, cfg.crop_px, c.k, c.x - (float)f.ox, c.y - (float)f.oy, cr);
            c.frames.push_back({f.index, t, std::move(cr)});
            c.lastF = f.index;
        }
        for (size_t i = 0; i < clips.size();)
            if (t >= clips[i].endT) {
                finishClip(clips[i]);
                clips.erase(clips.begin() + (long)i);
            } else ++i;

        if (cfg.overview_every_s > 0 && t - overviewT >= cfg.overview_every_s) {
            overviewT = t;
            snap(f, "periodic");
        }
        // забыть давно пропавшие треки
        if (st.frames % 64 == 0) {
            for (auto it = info.begin(); it != info.end();) {
                if (t - it->second.seenT > 30) it = info.erase(it);
                else ++it;
            }
        }

        if (t - flushT >= cfg.flush_s || framesBuf.size() > (256u << 10)) flushFrames(t);
        havePrev = true;
        prevIdx = f.index;
        prevT = t;
        prevLock = bId;
        prevState = f.lockState;
        prevManual = f.manual;
    }
    void reset(double t, int64_t idx, const std::string& why) {
        for (auto& c : clips) finishClip(c);
        clips.clear();
        ring.clear();
        info.clear();
        writeEvent(eventLine("reset", t, idx, ",\"from\":" + std::to_string(prevIdx) + ",\"why\":" + jstr(why)));
        havePrev = false;
        prevLock = 0;
        prevState = LockState::None;
        prevManual = false;
        prevMiss = 0;
    }
    void flushFrames(double t) {
        flushT = t;
        if (framesBuf.empty()) return;
        std::string s;
        s.swap(framesBuf);
        const size_t n = s.size();
        push([this, s = std::move(s)] { if (!diskFull) writeAppend(dir / "frames.jsonl", s); }, n, false);
    }

    std::string sessionJson(bool final) const {
        char b[512];
        std::string s = "{\"format\":\"" + std::string(FORMAT_TAG) + "\",\"version\":1";
        s += ",\"started\":" + jstr(iso_time(started, false)) + ",\"started_utc\":" + jstr(iso_time(started, true));
        s += ",\"source\":" + jstr(source);
        snprintf(b, sizeof(b), ",\"w\":%d,\"h\":%d,\"fps\":%.3f,\"crop\":%d,\"pre_s\":%.2f,\"post_s\":%.2f,\"max_mb\":%.1f", w, h,
                 fps, cfg.crop_px, cfg.pre_s, cfg.post_s, cfg.max_mb);
        s += b;
        s += ",\"track_fields\":" + std::string(trackFields());
        s += ",\"classes\":[\"unknown\",\"drone\",\"bird\",\"clutter\"]";
        s += ",\"flags\":{\"solid\":1,\"clutter\":2,\"leaf\":4,\"ghost\":8,\"ext_bird\":16,\"followed\":32,\"hovering\":64,\"anchored\":128}";
        s += ",\"lock_states\":[\"none\",\"tracking\",\"hard\"]";
        if (!settings.empty()) s += ",\"settings\":" + settings;
        if (final) {
            s += ",\"ended\":" + jstr(iso_time(std::time(nullptr), false));
            snprintf(b, sizeof(b),
                     ",\"summary\":{\"frames\":%lld,\"duration_s\":%.1f,\"lock_s\":%.1f,\"hard_s\":%.1f,\"locks\":%d,"
                     "\"lost\":%d,\"switches\":%d,\"hard\":%d,\"events\":%d,\"clips\":%d,\"clips_skipped\":%d,"
                     "\"snaps\":%d,\"dropped\":%d,\"disk_full\":%s}",
                     (long long)st.frames, firstT >= 0 ? prevT - firstT : 0.0, st.lockS, st.hardS, st.locks, st.lost, st.switches, st.hard, st.events,
                     st.clips, st.clipsSkipped, st.snaps, dropped.load(), diskFull ? "true" : "false");
            s += b;
        }
        return s + "}\n";
    }
};

FieldLog::FieldLog() = default;
FieldLog::~FieldLog() { close(); }

bool FieldLog::open(const FieldLogConfig& cfg, int w, int h, double fps, const std::string& source,
                    const std::string& settingsJson, std::string& err) {
    close();
    if (cfg.root.empty()) {
        err = "не задана папка журналов";
        return false;
    }
    auto p = std::make_unique<Impl>();
    p->cfg = cfg;
    p->cfg.crop_px = std::max(0, std::min(cfg.crop_px, 1024));
    p->cfg.ring_tracks = std::max(0, std::min(cfg.ring_tracks, 32));
    p->cfg.pre_s = std::max(0.f, std::min(cfg.pre_s, 30.f));
    p->cfg.post_s = std::max(0.f, std::min(cfg.post_s, 30.f));
    p->cfg.max_clip_s = std::max(p->cfg.post_s, cfg.max_clip_s);
    p->w = w;
    p->h = h;
    p->fps = fps > 0 ? fps : 14;
    p->source = source;
    p->settings = settingsJson;
    p->started = std::time(nullptr);
    p->root = upath(cfg.root);
    std::error_code ec;
    fs::create_directories(p->root, ec);
    if (!fs::is_directory(p->root, ec)) {
        err = "не создать папку " + cfg.root;
        return false;
    }
    std::string name = cfg.session.empty() ? stamp_name(p->started) : cfg.session;
    p->dir = p->root / upath(name);
    for (int i = 2; fs::exists(p->dir, ec) && i < 1000; ++i) p->dir = p->root / upath(name + "-" + std::to_string(i));
    fs::create_directories(p->dir, ec);
    const std::string sj = p->sessionJson(false);
    if (ec || !p->writeWhole(p->dir / "session.json", sj.data(), sj.size())) {
        err = "не записать в " + ustr(p->dir);
        return false;
    }
    p->dirU8 = ustr(p->dir);
    int64_t used = 0;
    for (fs::directory_iterator it(p->root, ec), end; !ec && it != end; it.increment(ec))
        if (is_session_dir(it->path())) used += dir_bytes(it->path());
    p->used = used;
    p->checkSpace(true);
    p->th = std::thread([q = p.get()] { q->writer(); });
    p_ = std::move(p);
    {
        std::lock_guard<std::mutex> lk(p_->m);
        p_->writeEvent(p_->eventLine("start", 0, 0, ",\"source\":" + jstr(source)));
    }
    return true;
}

bool FieldLog::isOpen() const { return p_ != nullptr; }

void FieldLog::frame(const FieldFrame& f) {
    if (!p_) return;
    std::lock_guard<std::mutex> lk(p_->m);
    p_->frame(f);
}

void FieldLog::note(const std::string& type, int id, const std::string& text, bool clip) {
    if (!p_) return;
    std::lock_guard<std::mutex> lk(p_->m);
    Impl& p = *p_;
    std::string clipName;
    if (clip && id) clipName = p.clipFor(id, p.prevT, p.eventN + 1, id != p.prevLock);
    std::string rest = ",\"id\":" + std::to_string(id) + ",\"text\":" + jstr(text) + ",\"what\":" + jstr(type);
    if (!clipName.empty()) rest += ",\"clips\":[" + jstr(clipName) + "]";
    p.writeEvent(p.eventLine("note", p.prevT, p.prevIdx, rest));
}

void FieldLog::reset(const std::string& why) {
    if (!p_) return;
    std::lock_guard<std::mutex> lk(p_->m);
    if (p_->havePrev) p_->reset(p_->prevT, p_->prevIdx, why);
}

void FieldLog::close() {
    if (!p_) return;
    {
        std::lock_guard<std::mutex> lk(p_->m);
        Impl& p = *p_;
        for (auto& c : p.clips) p.finishClip(c);
        p.clips.clear();
        p.flushFrames(p.prevT);
        p.writeEvent(p.eventLine("stop", p.prevT, p.prevIdx, ""));
        const std::string sj = p.sessionJson(true);
        p.push([q = &p, sj] { q->writeWhole(q->dir / "session.json", sj.data(), sj.size()); }, sj.size(), false);
    }
    {
        std::lock_guard<std::mutex> lk(p_->qm);
        p_->stop = true;
    }
    p_->qcv.notify_all();
    if (p_->th.joinable()) p_->th.join();
    p_.reset();
}

const std::string& FieldLog::dir() const {
    static const std::string none;
    return p_ ? p_->dirU8 : none;
}

FieldLogStats FieldLog::stats() const {
    if (!p_) return {};
    std::lock_guard<std::mutex> lk(p_->m);
    FieldLogStats s = p_->st;
    s.dropped = p_->dropped;
    s.usedMb = (double)p_->used / (1 << 20);
    s.diskFull = p_->diskFull;
    return s;
}

}  // namespace dronolok
