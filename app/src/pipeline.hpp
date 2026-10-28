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
#include "dronolok/scene.hpp"
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
    bool seekStart = false;   // первый кадр после перемотки: треки начинаются заново
    bool force = false;       // обработать и на паузе (перемотка на паузе)
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
    // память помех из файла рядом с видео: загружать при открытии и сохранять при закрытии.
    // В замерах выключено: зацикленный ролик повторяет те же позиции целей, и память,
    // накопленная прошлыми запусками, искажала бы результат
    bool staticFile = true;
    // маска неба из файла рядом с видео (<видео>.mask.bin) вместо обучения на каждом старте;
    // первые learnMaskFrames кадров она сверяется с видимым, при расхождении учится заново
    bool maskFile = true;
    // трекер считает в секундах: скорости, ворота и выдержки пересчитываются под частоту
    // кадров (dronolok::tracker_for_fps; повторы кадров записи экрана не считаются)
    bool trkSeconds = true;
    // эмуляция устройства: обработка кадра растягивается в slowdown раз (ядро устройства
    // медленнее ядра ПК во столько раз, см. calibrate_ms_per_mp)
    double slowdown = 1.0;
    // авто-упрощение: держать обработку в темпе камеры (dronolok::LoadGovernor)
    bool governor = false;
    // повторы кадра (запись экрана 60 к/с с ролика 25-30 к/с) не обрабатывать заново.
    // Выключено по умолчанию: на записях экрана повтор даёт тусклой цели лишнюю попытку
    // найтись, и захват в среднем держится лучше
    bool skipDuplicates = false;
    // Камера ведёт цель («виртуальная поворотка»): трекер видит окно кадра followFrac от
    // размера, и окно, как камера на подвесе, поворачивается за захваченной целью — цель
    // стоит почти на месте, фон едет навстречу. Сдвиг окна известен точно и передаётся
    // трекеру как поворот камеры. Проверка трекера для камеры на поворотке до самой поворотки
    bool follow = false;
    float followFrac = 0.5f;      // окно — доля ширины и высоты кадра
    float followGain = 0.3f;      // доля ошибки до центра, выбираемая за кадр (плюс скорость цели)
    float followMaxPan = 0.05f;   // предел поворота за кадр, доля ширины окна
    bool followVerified = true;   // поворачиваться только за проверенной целью (жёсткий захват)
    // Подстройка под сцену (dronolok::SceneAnalyzer): раз в 0.2 с оценивается свет, шум, облака,
    // ветер у кромки, осадки, дымка и окружение, и пороги, полярность, строгость подтверждения
    // подстраиваются относительно выставленных здесь. Резкая смена света — фон ячеек заново;
    // камеру сдвинули — маска неба заново
    bool adapt = true;
    // журнал треков (замеры): по строке на подтверждённый трек каждого кадра, CSV
    std::string trackLog;
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
    int govLevel = -1;            // ступень авто-упрощения, -1 — выключено
    bool dup = false;             // повтор прошлого кадра: результат взят от него
    dronolok::LockState lockState = dronolok::LockState::None;
    int verify = 0, verifyNeed = 0;   // кадров подряд цель — дрон, сколько нужно для жёсткого захвата
    bool autoThr = false;         // порог по шуму
    float noiseLvl = 0, thr = 0;  // шум отклика неба и порог, с которым искали на этом кадре
    int fullEvery = 1, coarseEvery = 1;   // что сейчас стоит в детекторе (с учётом упрощения)
    // сцена и подстройка под неё
    bool adapt = false;
    dronolok::SceneState scene;
    std::string sceneWhy;         // что подстроено и почему ("" — как выставлено)
    // камера ведёт цель: frame — окно исходного кадра srcW x srcH со смещением fx, fy
    bool follow = false;
    int fx = 0, fy = 0, srcW = 0, srcH = 0;
};

// Одна строка журнала метрик (на каждый обработанный кадр)
struct FrameRec {
    double t;          // момент окончания обработки, мс от начала замера
    float decodeMs, detectMs, responseMs, selectMs, trackMs, procMs, latencyMs, motionMs;
    int64_t index;
    int cands, drones, birds, unknown, clutter, noise;
    int lockId;
    float camDx, camDy;
    int govLevel;
    bool dup;          // повтор кадра: не обрабатывался, во времена обработки не входит
};

// Отметки на шкале видео, по номеру кадра (биты)
enum Mark : uint8_t {
    MARK_DONE = 1,      // кадр обработан
    MARK_TARGET = 2,    // есть цель (не помеха), класс неясен
    MARK_DRONE = 4,
    MARK_BIRD = 8,
    MARK_LOCK = 16,     // есть захват
    MARK_GT = 32,       // по разметке на кадре есть дрон
    MARK_MISS = 64,     // ...и он не найден
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
    // Перемотка на кадр frame: чтение продолжается оттуда, треки начинаются заново.
    // На паузе этот кадр обрабатывается и показывается.
    void seek(int64_t frame);
    int64_t length() const;              // кадров на шкале (0 — неизвестно)
    void marks(std::vector<uint8_t>& out) const;
    int findInHistory(int64_t frame) const;   // seq последнего обработанного кадра frame или -1
    void relearnMask();
    // Память статичных помех лежит рядом с видео (<видео>.static.bin): загружается при
    // открытии и сохраняется при закрытии, если включена
    std::string staticPath() const { return path_ + ".static.bin"; }
    std::string maskPath() const { return path_ + ".mask.bin"; }
    void staticSave() { staticCmd_ = 1; }
    void staticClear() { staticCmd_ = 2; }
    void resetStats();
    // Разметка <видео>.gt.json: заменить на ходу (после правки в приложении); пустая — убрать
    std::string gtPath() const { return path_ + ".gt.json"; }
    void setGroundTruth(const dronolok::GroundTruth& g);
    // Вырезки целей из истории кадров с подписью по разметке — набор для настройки классификатора.
    // Возвращает число вырезок, msg — что и куда сохранено
    int exportCrops(const std::string& dir, std::string& msg) const;

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
    std::shared_ptr<const dronolok::GroundTruth> gt_;   // под rm_: разметку можно заменить на ходу
    bool haveGt_ = false;

    mutable std::mutex sm_;
    Settings settings_;
    int settingsVer_ = 0;

    std::thread dec_, proc_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> paused_{false};
    std::atomic<int> steps_{0};
    std::atomic<bool> relearn_{true};
    std::atomic<bool> maskForce_{false};   // "заново": учить, даже если есть файл
    std::atomic<int> staticCmd_{0};
    std::atomic<int64_t> seekReq_{-1};
    std::atomic<int64_t> preN_{0};       // кадров в предзагрузке (0 — читаем файл)
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
    std::vector<uint8_t> marks_;
    long dropped_ = 0, decoded_ = 0;
    std::deque<double> decodeTimes_;   // моменты выдачи кадров декодером
    std::string backend_, status_;
    int threads_ = 0;
    dronolok::EvalStats eval_;
    double t0_ = 0;
};

double now_ms();
