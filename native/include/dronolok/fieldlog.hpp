// Журнал полевых испытаний: что видел и решал алгоритм на выезде, без полной записи видео.
//
// В папке журналов (FieldLogConfig::root) на каждый запуск — своя папка <ГГГГММДД-ЧЧММСС>:
//   session.json   — когда, с какой камеры или видео, размер кадра, частота, настройки; при закрытии — итоги
//   frames.jsonl   — строка на обработанный кадр: захват, его состояние, сдвиг камеры, подтверждённые треки
//                    (поля трека перечислены в session.json, "track_fields")
//   events.jsonl   — события: захват, смена цели, срыв, жёсткий захват и его снятие, решение дрон/птица,
//                    отметки извне (поворотка, оператор)
//   clips/         — вырезки вокруг цели за pre_s до события и post_s после (кадры MJPEG одного размера,
//                    <номер>_<трек>.mjpeg, и где они взяты в кадре — <номер>_<трек>.json)
//   snap/          — снимок всего кадра в уменьшенном виде в момент срыва или смены цели и раз в
//                    overview_every_s (<кадр>.jpg)
// Всё пишется отдельным потоком: кадр обработки не ждёт диска. Место на диске ограничено max_mb на все
// журналы в root: сначала удаляются вырезки и снимки старых выездов, потом старые выезды целиком, потом
// вырезки текущего; строки треков перестают писаться последними.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "dronolok/detector.hpp"

namespace dronolok {

struct FieldLogConfig {
    std::string root;            // папка журналов (создаётся)
    std::string session;         // имя папки запуска; "" — дата и время начала
    double max_mb = 2048;        // предел места на все журналы в root, МБ
    double min_free_mb = 300;    // на диске свободно меньше — вырезки и снимки не пишутся
    float pre_s = 2.f;           // вырезка: секунд до события...
    float post_s = 2.f;          // ...и после
    float max_clip_s = 12.f;     // события на той же цели продлевают вырезку, но не дольше
    int crop_px = 160;           // сторона вырезки, пикс. (крупная цель — вырезка в 2 или 4 раза больше, уменьшенная до crop_px)
    int ring_tracks = 4;         // кроме захваченной, держать вырезки стольких первых целей очереди
    int jpeg_q = 85;             // качество JPEG вырезок и снимков
    int snap_px = 1280;          // снимок всего кадра: не шире стольких пикс. (0 — снимков нет)
    float overview_every_s = 60.f;   // снимок всего кадра раз в столько секунд (0 — только у событий)
    int clips_per_min = 20;      // не больше вырезок в минуту (частые срывы на плохой цели)
    bool all_tracks = true;      // треки каждого кадра; false — только захваченный
    float flush_s = 1.f;         // строки кадров сбрасываются на диск не реже (пропало питание — теряется не больше)
};

// Кадр для журнала: изображение (серое, как у детектора), треки и захват — после Tracker::update
struct FieldFrame {
    int64_t index = 0;           // номер кадра (камеры или файла)
    double t = 0;                // время кадра от начала, с
    const uint8_t* gray = nullptr;   // nullptr — без вырезок и снимков
    int w = 0, h = 0, stride = 0;
    int ox = 0, oy = 0;          // кадр — окно полного кадра с этим смещением (камера ведёт цель); треки — в координатах окна
    const std::vector<Track>* tracks = nullptr;
    const std::vector<int>* queue = nullptr;   // очередь целей (Tracker::lockQueue); nullptr — по весу треков
    int lockId = 0;
    LockState lockState = LockState::None;
    int verify = 0;              // кадров подряд цель — дрон (Tracker::verifyFrames)
    bool manual = false;         // захват выбран вручную
    float camDx = 0, camDy = 0;  // сдвиг картинки за кадр
};

struct FieldLogStats {
    int64_t frames = 0;
    int events = 0, clips = 0, snaps = 0;
    int locks = 0, lost = 0, switches = 0, hard = 0;
    int clipsSkipped = 0;        // не записаны: предел в минуту, место на диске
    int dropped = 0;             // очередь записи переполнена (диск не успевает)
    double lockS = 0, hardS = 0; // сколько секунд был захват и жёсткий захват
    double usedMb = 0;           // сколько занимают все журналы в root
    bool diskFull = false;       // строки треков больше не пишутся
};

class FieldLog {
public:
    FieldLog();
    ~FieldLog();   // close()
    FieldLog(const FieldLog&) = delete;
    FieldLog& operator=(const FieldLog&) = delete;

    // w x h — полный кадр, fps — частота камеры. source — откуда кадры (путь к видео или имя камеры),
    // settingsJson — настройки алгоритма как JSON-объект (или ""), попадают в session.json как есть
    bool open(const FieldLogConfig& cfg, int w, int h, double fps, const std::string& source,
              const std::string& settingsJson, std::string& err);
    bool isOpen() const;
    void frame(const FieldFrame& f);
    // Отметка извне: type — короткое слово ("slew", "engaged", "operator", ...), id — трек (0 — нет),
    // text — пояснение. clip — записать вырезку вокруг трека id
    void note(const std::string& type, int id, const std::string& text, bool clip = false);
    // Треки начинаются заново (перемотка видео, перезапуск трекера): дописать вырезки, забыть прошлый
    // захват (его пропажа — не срыв). why — пояснение в событии "reset"
    void reset(const std::string& why);
    void close();
    const std::string& dir() const;   // папка этого запуска
    FieldLogStats stats() const;

    // Поля трека в строке frames.jsonl, по порядку
    static const char* trackFields();

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// Серый JPEG (baseline, одна компонента) без внешних библиотек: для вырезок и снимков журнала
std::vector<uint8_t> encode_jpeg_gray(const uint8_t* px, int w, int h, int stride, int quality = 85);

}  // namespace dronolok
