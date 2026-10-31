// Журнал полевых испытаний: события захвата, вырезки до и после события, снимки, предел места.
// Кадры синтетические (тёмная точка на градиенте неба), треки и захват задаются напрямую.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "dronolok/fieldlog.hpp"

using namespace dronolok;
namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(cond, ...)                         \
    do {                                         \
        if (!(cond)) {                           \
            ++failures;                          \
            std::printf("FAIL: " __VA_ARGS__);   \
            std::printf("\n");                   \
        }                                        \
    } while (0)

static std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static int count(const std::string& s, const std::string& what) {
    int n = 0;
    for (size_t i = s.find(what); i != std::string::npos; i = s.find(what, i + 1)) ++n;
    return n;
}

static int jpegs(const std::string& s) {   // кадров MJPEG: маркеры начала SOI ... EOI
    int n = 0;
    for (size_t i = 0; i + 1 < s.size(); ++i)
        if ((uint8_t)s[i] == 0xFF && (uint8_t)s[i + 1] == 0xD8) ++n;
    return n;
}

static Track trk(int id, float x, float y, TargetClass c) {
    Track t;
    t.id = id;
    t.x = x;
    t.y = y;
    t.vx = 2;
    t.cls = c;
    t.solid = t.confirmed = true;
    t.hits = 10;
    t.lockW = (float)id;
    return t;
}

int main() {
    // кириллица в пути: на Windows папка журналов пользователя — «C:\\Файлы\\...»
    const fs::path root = fs::temp_directory_path() / fs::u8path(u8"dl_fieldlog_test_журнал");
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);

    // JPEG: заголовок и конец, размер разумный
    {
        std::vector<uint8_t> img(37 * 21);
        for (int i = 0; i < (int)img.size(); ++i) img[(size_t)i] = (uint8_t)(i * 7);
        const auto j = encode_jpeg_gray(img.data(), 37, 21, 37, 85);
        CHECK(j.size() > 200 && j[0] == 0xFF && j[1] == 0xD8 && j[j.size() - 2] == 0xFF && j[j.size() - 1] == 0xD9,
              "JPEG: %zu байт", j.size());
    }

    // чужая папка и старый выезд в root: старый удаляется при нехватке места, чужая — никогда
    fs::create_directories(root / "my_photos");
    { std::ofstream(root / "my_photos" / "session.json") << "{\"not\":\"ours\"}"; }
    { std::ofstream(root / "my_photos" / "big.bin") << std::string(300000, 'x'); }
    fs::create_directories(root / "20200101-000000" / "clips");
    { std::ofstream(root / "20200101-000000" / "session.json") << "{\"format\":\"dronolok-fieldlog\",\"version\":1}"; }
    { std::ofstream(root / "20200101-000000" / "clips" / "old.mjpeg") << std::string(600000, 'x'); }

    const int W = 640, H = 360;
    const double fps = 20;
    std::vector<uint8_t> img((size_t)W * H);
    FieldLogConfig cfg;
    cfg.root = root.u8string();
    cfg.max_mb = 0.5;           // 512 КБ на всё: старая вырезка (600 КБ) должна уйти
    cfg.min_free_mb = 0;
    cfg.crop_px = 64;
    cfg.pre_s = 1.0f;
    cfg.post_s = 1.0f;
    cfg.overview_every_s = 2.0f;
    cfg.snap_px = 320;
    FieldLog log;
    std::string err;
    CHECK(log.open(cfg, W, H, fps, "synthetic", "{\"thr\":10}", err), "open: %s", err.c_str());
    const std::string dir = log.dir();

    // 0-19: цель 1 без захвата; 20: захват 1; 40: жёсткий; 60: класс птица, жёсткий снят; 70: смена на 2;
    // 90: 1 пропал, захват 2 снят (трек 2 пропал); 120: конец
    std::vector<Track> tracks;
    for (int f = 0; f < 120; ++f) {
        const float x1 = 100 + 2.f * f, y1 = 120, x2 = 500 - 1.5f * f, y2 = 200;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) img[(size_t)y * W + x] = (uint8_t)(150 + y / 6);
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx) {
                img[(size_t)((int)y1 + dy) * W + (int)x1 + dx] = 40;
                img[(size_t)((int)y2 + dy) * W + (int)x2 + dx] = 60;
            }
        tracks.clear();
        if (f < 90) tracks.push_back(trk(1, x1, y1, f >= 60 ? TargetClass::Bird : f >= 15 ? TargetClass::Drone : TargetClass::Unknown));
        if (f >= 50 && f < 100) tracks.push_back(trk(2, x2, y2, TargetClass::Drone));
        FieldFrame ff;
        ff.index = f;
        ff.t = f / fps;
        ff.gray = img.data();
        ff.w = W, ff.h = H, ff.stride = W;
        ff.tracks = &tracks;
        ff.lockId = f >= 20 && f < 70 ? 1 : f >= 70 && f < 100 ? 2 : 0;
        ff.lockState = !ff.lockId ? LockState::None : f >= 40 && f < 60 ? LockState::Hard : LockState::Tracking;
        log.frame(ff);
        if (f == 80) log.note("slew", 2, "поворотка навелась", true);
    }
    const FieldLogStats st = log.stats();
    log.close();

    CHECK(st.frames == 120, "кадров %lld", (long long)st.frames);
    CHECK(st.locks == 1 && st.switches == 1 && st.lost == 1 && st.hard == 1, "захват %d смена %d срыв %d жёсткий %d",
          st.locks, st.switches, st.lost, st.hard);
    CHECK(st.lockS > 3.9 && st.lockS < 4.1, "в захвате %.2f с", st.lockS);
    const std::string ev = slurp(fs::u8path(dir) / "events.jsonl");
    for (const char* t : {"\"start\"", "\"lock\"", "\"hard\"", "\"hard_off\"", "\"switch\"", "\"lost\"", "\"cls\"", "\"note\"", "\"stop\""})
        CHECK(count(ev, std::string("\"type\":") + t) >= 1, "нет события %s", t);
    CHECK(ev.find("\"why\":\"bird\"") != std::string::npos, "смена цели после птицы: %s", ev.c_str());
    CHECK(ev.find("\"why\":\"track_gone\"") != std::string::npos, "срыв: трек пропал");
    const std::string fr = slurp(fs::u8path(dir) / "frames.jsonl");
    CHECK(count(fr, "\n") == 120, "строк кадров %d", count(fr, "\n"));
    CHECK(count(fr, "\"L\":1") == 50 && count(fr, "\"L\":2") == 30, "захват по кадрам");
    // вырезка захвата 1: 1 с до (20 кадров кольца, но цель видна с 0 — 21 кадр с текущим) и до 1 с после
    // (продлевается жёстким, сменой класса — пока не дольше max_clip_s)
    int clipFiles = 0;
    for (auto& e : fs::directory_iterator(fs::u8path(dir) / "clips"))
        if (e.path().extension() == ".mjpeg") {
            ++clipFiles;
            const std::string mj = slurp(e.path());
            const std::string meta = slurp(fs::path(e.path()).replace_extension(".json"));
            CHECK(jpegs(mj) == count(meta, "\n["), "кадров в %s: %d",
                  e.path().filename().string().c_str(), jpegs(mj));
            CHECK(jpegs(mj) >= 10, "короткая вырезка %s: %d", e.path().filename().string().c_str(), jpegs(mj));
        }
    CHECK(clipFiles >= 2, "вырезок %d", clipFiles);
    CHECK(fs::exists(fs::u8path(dir) / "snap" / "index.jsonl"), "нет снимков");
    const std::string ses = slurp(fs::u8path(dir) / "session.json");
    CHECK(ses.find("\"summary\"") != std::string::npos && ses.find("\"thr\":10") != std::string::npos, "session.json: %s",
          ses.c_str());
    CHECK(!fs::exists(root / "20200101-000000" / "clips"), "старые вырезки не удалены при нехватке места");
    CHECK(fs::exists(root / "my_photos" / "big.bin"), "удалена чужая папка");

    if (!getenv("DL_KEEP")) fs::remove_all(root, ec);
    if (failures) {
        std::printf("%d проверок не прошло\n", failures);
        return 1;
    }
    std::printf("журнал: все проверки прошли\n");
    return 0;
}
