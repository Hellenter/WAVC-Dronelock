// Разбор журнала полевых испытаний (dronolok/fieldlog.hpp): чтение папки выезда для вкладки «Журнал».
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace fieldview {

struct TrackRow {
    int id = 0;
    float x = 0, y = 0, vx = 0, vy = 0;
    int cls = 0, flags = 0;
    float bird = 0, weight = 0;
    int hits = 0, miss = 0;
    float contrast = 0, size = 0;
};

struct FrameRow {
    int64_t f = 0;
    double t = 0;
    int lock = 0, state = 0, verify = 0;
    bool manual = false;
    float cdx = 0, cdy = 0;
    uint32_t trackOff = 0, trackN = 0;   // треки кадра в Session::tracks
};

struct Event {
    int n = 0;
    std::string type;
    int64_t f = 0;
    double t = 0;
    int id = 0, from = 0, was = 0;
    bool manual = false, locked = false;
    float x = -1, y = -1, bird = -1;
    int miss = -1;
    std::string why, cls, fromCls, text, what, snap, source;
    std::vector<std::string> clips;
};

struct ClipFrame {
    int64_t f = 0;
    double t = 0;
    int x0 = 0, y0 = 0, k = 1;
    float tx = 0, ty = 0;
};

struct Clip {
    std::string name;
    int track = 0, size = 0;
    std::vector<int> events;
    std::vector<ClipFrame> frames;
    std::vector<std::vector<uint8_t>> px;   // кадры size x size, после load_clip
    bool loaded = false, missing = false;   // missing — файла нет (удалён при нехватке места)
    std::string err;
};

struct Snap {
    int64_t f = 0;
    int k = 1, ox = 0, oy = 0;
    std::string file, why;
};

struct Session {
    std::string dir, name;
    std::string source, started, ended;
    int w = 0, h = 0, crop = 160;
    double fps = 14;
    std::string settings;   // как в session.json, одной строкой
    bool finished = false;  // есть итоги (журнал закрыт как положено, а не оборвался)
    std::vector<FrameRow> frames;
    std::vector<TrackRow> tracks;
    std::vector<Event> events;
    std::vector<Snap> snaps;
    std::map<std::string, Clip> clips;
    // посчитано по кадрам
    double duration = 0, lockS = 0, hardS = 0;
    int locks = 0, lost = 0, switches = 0, hard = 0, coast = 0, resets = 0, gaps = 0;
    int droneDecisions = 0, birdDecisions = 0, flips = 0;   // по событиям cls
    int64_t bytes = 0, clipBytes = 0, snapBytes = 0;
    int badLines = 0;

    // строка кадра с номером f или ближайшая до него (-1 — нет)
    int frameAt(int64_t f) const;
    // ближайшая по времени
    int frameAtTime(double t) const;
    const TrackRow* track(const FrameRow& fr, int id) const;
    // снимок всего кадра не позже f (и не раньше, чем за maxBack кадров), -1 — нет
    int snapNear(int64_t f, int64_t maxBack = -1) const;
};

// Папки выездов в root (с session.json от журнала), новые первыми. Если root — сама папка выезда, то она одна
std::vector<std::string> list_sessions(const std::string& root);
bool load(const std::string& dir, Session& s, std::string& err);
// Кадры вырезки (MJPEG через FFmpeg)
bool load_clip(const Session& s, Clip& c);
// Серый снимок (JPEG)
bool load_image(const std::string& path, std::vector<uint8_t>& px, int& w, int& h, std::string& err);

// По-русски: тип события, его описание одной строкой, класс
const char* type_ru(const std::string& type);
std::string describe(const Event& e);
const char* class_ru(int cls);
const char* why_ru(const std::string& why);
// группы событий для фильтра: 0 захват/срыв/смена, 1 жёсткий, 2 дрон/птица, 3 на прогнозе, 4 прочее
int group(const std::string& type);

}  // namespace fieldview
