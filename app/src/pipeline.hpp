// Конвейер: поток чтения/декодирования -> очередь -> поток детекции и трекинга -> последний
// результат для интерфейса. Так же устроено и на устройстве (захват, обработка, вывод
// в разных потоках), поэтому замеры отражают реальную картину.
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "dronolok/detector.hpp"
#include "gt_eval.hpp"
#include "video_source.hpp"

// Пиксели кадра неизменяемы после декодирования: при предзагрузке один и тот же
// кадр подаётся по кругу без копирования (как DMA-буферы камеры).
struct Frame {
    std::vector<uint8_t> px;
    int w = 0, h = 0;
    int64_t index = 0;        // номер кадра в файле
    double decodeMs = 0;
};
using FramePtr = std::shared_ptr<Frame>;

// Кадр в очереди: сам кадр + когда и в каком проходе он "пришёл с камеры"
struct Packet {
    FramePtr frame;
    int loop = 0;
    bool loopStart = false;
    double readyAt = 0;
};
struct BufPool;

enum class Pace { Realtime, MaxSpeed };
enum class Source { Stream, Preload };

struct Settings {
    dronolok::DetectorConfig det;
    dronolok::TrackerConfig trk;
    // источник кадров
    Pace pace = Pace::Realtime;
    double targetFps = 0;      // 0 = FPS видео
    Source source = Source::Stream;
    int preloadFrames = 100;
    int procW = 0, procH = 0;  // 0 = исходный размер
    int decoderThreads = 0;    // 0 = авто
    bool loop = true;
    // имитация слабого железа: обработка кадра занимает не меньше emulateMs
    double emulateMs = 0;
    int learnMaskFrames = 25;  // авто-маска неба по первым кадрам (0 = выкл.)
    // просмотр
    double speed = 1.0;        // скорость воспроизведения в режиме "как камера": 0.25 = в 4 раза медленнее
    // алгоритм поверх det/trk; "авто" решается по размеру кадра (см. effective())
    int scales = 0;            // 0 = авто, иначе 1..4 (DetectorConfig::scales)
    int motion = -1;           // компенсация движения камеры: -1 авто, 0 выкл., 1 вкл.
    bool focus = false;        // пониженный порог возле прогноза треков (эксперимент)
    int zones = -1;            // зоны кадра небо/смешанные/земля: -1 авто, 0 выкл., 1 вкл.
};

// Что реально включено для кадра w x h. Авто: кадр до ~2 Мп (Full HD и меньше) —
// обычно съёмка с рук и близкий крупный дрон: 3 масштаба и учёт движения камеры.
// 8-16 Мп — неподвижная камера и дальние точки: 1 масштаб, без оценки движения.
struct Effective {
    int scales;
    bool motion;
    bool zones;
};
Effective effective(const Settings& s, int w, int h);

struct Result {
    FramePtr frame;
    int loop = 0;
    std::vector<dronolok::Candidate> cands, clutter, noise;
    std::vector<dronolok::Track> tracks;
    dronolok::DetectTiming dt;
    double trackMs = 0, procMs = 0, latencyMs = 0;
    std::vector<uint8_t> mask;
    int tilesX = 0, tilesY = 0, tilePx = 0;
    bool learning = false;
    std::vector<dronolok::GtObj> gt;
    float gsx = 1, gsy = 1;
    int seq = 0;
    // захват и прогноз
    int lockId = 0;                              // захваченная цель (Track::id), 0 — нет
    float camDx = 0, camDy = 0, camQ = 0;        // сдвиг картинки за кадр и уверенность оценки
    bool camValid = false, motionOn = false;
    double motionMs = 0;
    int scales = 1;
    std::vector<dronolok::FocusArea> focus;      // где детектор искал с пониженным порогом
    // долгая память статичных помех (обновляется раз в 15 кадров)
    bool staticOn = false;
    std::vector<dronolok::TrackPoint> staticCells;
    int staticCellPx = 8;
    std::vector<uint8_t> zones;   // по тайлам (tilesX x tilesY): 0 земля, 1 смешанная, 2 небо
};

// Одна строка журнала метрик (на каждый обработанный кадр)
struct FrameRec {
    double t;          // момент окончания обработки, мс от начала замера
    float decodeMs, detectMs, responseMs, selectMs, trackMs, procMs, latencyMs, motionMs;
    int64_t index;
    int cands, drones, birds, unknown, clutter, noise;
    int lockId;
    float camDx, camDy;
};

struct Stat {
    double mean = 0, p50 = 0, p95 = 0, max = 0;
};

struct Metrics {
    long processed = 0, dropped = 0, decoded = 0;
    double procFps = 0, decodeFps = 0, sourceFps = 0;
    Stat decode, detect, response, select, track, proc, latency, motion;
    std::vector<float> histDecode, histDetect, histTrack, histProc;   // последние кадры, для графика
    int queue = 0, queueCap = 0;
    std::string backend, status;
    int threads = 0;
    dronolok::EvalStats eval;
    bool haveGt = false;
    double elapsedS = 0;
    double lockPct = 0;   // доля кадров (из последних 300) с захватом
    std::string staticMsg;
};

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    bool open(const std::string& path, std::string& err);
    void close();
    bool isOpen() const { return open_; }
    const std::string& path() const { return path_; }
    VideoInfo info() const;

    void setSettings(const Settings& s);
    Settings settings() const;
    void setPaused(bool p);
    bool paused() const { return paused_; }
    void step();                         // один кадр на паузе
    void relearnMask();
    // Память статичных помех лежит рядом с видео (<видео>.static.bin): загружается при
    // открытии и сохраняется при закрытии, если включена
    std::string staticPath() const { return path_ + ".static.bin"; }
    void staticSave() { staticCmd_ = 1; }
    void staticClear() { staticCmd_ = 2; }
    void resetStats();

    std::shared_ptr<const Result> latest() const;
    // Последние обработанные кадры для просмотра назад на паузе (номера seq подряд).
    // Держим не больше HIST_MAX кадров и HIST_BYTES памяти.
    std::shared_ptr<const Result> at(int seq) const;   // nullptr, если кадр уже вытеснен
    bool historyRange(int& first, int& last) const;
    Metrics metrics(int histLen = 300) const;
    bool exportCsv(const std::string& path) const;
    std::string reportJson() const;

private:
    void startThreads();
    void stopThreads();
    void decodeLoop();
    void processLoop();
    FramePtr allocFrame(int w, int h);
    void push(Packet p, bool realtime);

    std::string path_;
    bool open_ = false;
    VideoInfo info_;
    dronolok::GroundTruth gt_;
    bool haveGt_ = false;

    mutable std::mutex sm_;
    Settings settings_;
    int settingsVer_ = 0;

    std::thread dec_, proc_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> paused_{false};
    std::atomic<int> steps_{0};
    std::atomic<bool> relearn_{true};
    std::atomic<int> staticCmd_{0};
    std::string staticMsg_;

    // очередь кадров
    mutable std::mutex qm_;
    std::condition_variable qcv_, qspace_;
    std::deque<Packet> q_;
    const size_t qcap_ = 3;

    // свободные буферы кадров, чтобы не выделять 16 МБ на каждый кадр
    std::shared_ptr<BufPool> pool_;

    mutable std::mutex rm_;
    std::shared_ptr<const Result> latest_;
    std::deque<std::shared_ptr<const Result>> hist_;
    size_t histBytes_ = 0;
    std::vector<FrameRec> log_;
    long dropped_ = 0, decoded_ = 0;
    std::deque<double> decodeTimes_;   // моменты выдачи кадров декодером
    std::string backend_, status_;
    int threads_ = 0;
    dronolok::EvalStats eval_;
    double t0_ = 0;
};

double now_ms();
