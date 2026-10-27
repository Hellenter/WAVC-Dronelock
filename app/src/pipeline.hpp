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
};

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
};

// Одна строка журнала метрик (на каждый обработанный кадр)
struct FrameRec {
    double t;          // момент окончания обработки, мс от начала замера
    float decodeMs, detectMs, responseMs, selectMs, trackMs, procMs, latencyMs;
    int64_t index;
    int cands, drones, birds, unknown, clutter, noise;
};

struct Stat {
    double mean = 0, p50 = 0, p95 = 0, max = 0;
};

struct Metrics {
    long processed = 0, dropped = 0, decoded = 0;
    double procFps = 0, decodeFps = 0, sourceFps = 0;
    Stat decode, detect, response, select, track, proc, latency;
    std::vector<float> histDecode, histDetect, histTrack, histProc;   // последние кадры, для графика
    int queue = 0, queueCap = 0;
    std::string backend, status;
    int threads = 0;
    dronolok::EvalStats eval;
    bool haveGt = false;
    double elapsedS = 0;
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
    void resetStats();

    std::shared_ptr<const Result> latest() const;
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

    // очередь кадров
    mutable std::mutex qm_;
    std::condition_variable qcv_, qspace_;
    std::deque<Packet> q_;
    const size_t qcap_ = 3;

    // свободные буферы кадров, чтобы не выделять 16 МБ на каждый кадр
    std::shared_ptr<BufPool> pool_;

    mutable std::mutex rm_;
    std::shared_ptr<const Result> latest_;
    std::vector<FrameRec> log_;
    long dropped_ = 0, decoded_ = 0;
    std::deque<double> decodeTimes_;   // моменты выдачи кадров декодером
    std::string backend_, status_;
    int threads_ = 0;
    dronolok::EvalStats eval_;
    double t0_ = 0;
};

double now_ms();
