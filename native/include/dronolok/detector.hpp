// Дронолок: нативное ядро детектора точечных целей (C++ API).
//
// Алгоритм тот же, что в detector/detector.py (морфология линейными ядрами по
// 4 направлениям после min/max-пулинга 2x2, поиск пиков по тайлам), но все
// стадии слиты в один потоковый проход по блокам кадра, которые лежат в кэше,
// и посчитаны SIMD-ядрами (AVX2/SSE2 на x86, NEON на ARM).
#pragma once
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace dronolok {

// Набор инструкций CPU-бэкенда. Auto = лучший из доступных.
enum class Isa : int { Auto = 0, Scalar = 1, SSE2 = 2, AVX2 = 3, NEON = 4 };
const char* isa_name(Isa isa);
std::vector<Isa> available_isas();   // поддержанные и сборкой, и процессором
int hardware_threads();

struct DetectorConfig {
    int k = 7;                  // макс. размер цели после пулинга, нечётный
    int pool = 2;               // 1 = полный размер, 2 = пулинг 2x2
    float thr = 10.f;           // мин. контраст цели, уровни яркости
    float clutter_k = 3.f;      // во сколько раз отклик выше среднего в тайле
    int tile = 32;              // тайл для оценки клаттера и поиска максимумов
    float max_density = 0.012f; // доля "горячих" пикселей в тайле, выше — текстура
    bool dark = true, bright = true;
    int max_cands = 300;        // на полярность
    // Масштабы. 1: цели примерно до 7 пикс. (при pool 2; крупнее тайл похож на текстуру
    // и отбрасывается). 2, 3, 4 добавляют проходы по кадру, уменьшенному в 2, 4, 8 раз
    // (min/max-пулинг): цели до ~28, ~56, ~112 пикс. — близкий дрон, ролик низкого
    // разрешения. Все вместе стоят примерно +35% времени.
    int scales = 1;
    float coarse_thr_k = 3.f;   // на грубых масштабах порог выше: крупная близкая цель контрастна
    float focus_k = 0.6f;       // возле прогноза трека (setFocus) порог = thr * focus_k
    // Экономия: весь кадр (основной масштаб) и грубые масштабы просматриваются раз в
    // full_every / coarse_every кадров, в остальные — только тайлы вокруг прогнозов
    // треков (setRoi). Цель, которую уже ведём, не теряется, новая находится с
    // задержкой до N-1 кадров. 1 — каждый кадр целиком.
    int full_every = 1;
    int coarse_every = 2;       // крупная цель за кадр смещается мало, а грубые масштабы недёшевы
    // ресурсы
    int threads = 0;            // 0 = все аппаратные потоки
    std::vector<int> cpus;      // привязка рабочих потоков к ядрам (пусто = как решит ОС)
    Isa isa = Isa::Auto;
    int job_rows = 4;           // размер задания для потока, в тайлах;
    int job_cols = 0;           // 0 = подобрать под кэш платформы (x86: 16, ARM: 8)
};

struct Candidate {
    float x, y;        // в пикселях исходного кадра
    float contrast;    // отклик, уровни яркости
    float snr;         // отклик / (средний отклик в тайле + 1)
    int polarity;      // -1 тёмная цель, +1 яркая
    int level = 0;     // масштаб, на котором найдена: размер цели до k*pool*2^level пикс.
    bool weak = false; // прошла только по пониженному порогу возле прогноза трека
};

// Область, где ждём цель (прогноз трека): там порог ниже, см. DetectorConfig::focus_k
struct FocusArea {
    float x, y, r;     // пиксели кадра
    int level = 0;     // масштаб, на котором ведётся цель (для setRoi)
};

struct DetectTiming {
    double total_ms = 0;     // весь detect()
    double response_ms = 0;  // пулинг + морфология + статистика тайлов (бэкенд)
    double select_ms = 0;    // выбор кандидатов
    int jobs = 0, jobs_skipped = 0;   // задания бэкенда, пропущенные по маске
    bool partial = false;    // основной масштаб — только вокруг треков (full_every)
    bool coarsePartial = false;
};

class Backend;
struct TileStats;

class Detector {
public:
    explicit Detector(const DetectorConfig& cfg = DetectorConfig());
    ~Detector();
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    // gray: u8, w x h, stride байт между строками
    void detect(const uint8_t* gray, int w, int h, int stride, std::vector<Candidate>& out);

    // Камера стационарна: тайлы, которые постоянно "текстурные" (лес, здания,
    // земля), исключаются из поиска. Между begin и end вызывается detect() по кадрам.
    void beginMaskLearning();
    void endMaskLearning(float frac = 0.3f);
    bool learningMask() const { return learning_; }
    void clearMask();

    // Прогнозы треков на этот кадр. Действует до следующего вызова.
    void setFocus(const std::vector<FocusArea>& areas) { focus_ = areas; }
    // Где искать в кадры между полными просмотрами (Tracker::roi)
    void setRoi(const std::vector<FocusArea>& areas) { roi_ = areas; }
    const std::vector<uint8_t>& tileMask() const { return mask_; }   // tilesY x tilesX, 1 = искать
    int tilesX() const { return tw_; }
    int tilesY() const { return th_; }
    int tileSizePx() const { return cfg_.tile * cfg_.pool; }        // тайл в пикселях кадра

    // Меняет параметры. Пороги применяются сразу; число потоков, привязка,
    // набор инструкций и геометрия (k, pool, tile) пересоздают бэкенд.
    void reconfigure(const DetectorConfig& cfg);
    const DetectorConfig& config() const { return cfg_; }
    const DetectTiming& timing() const { return timing_; }
    std::string backendName() const;
    int threads() const;

private:
    void selectTiles(const TileStats& ts, int tw, int th, int tile, int factor, int pol, int lvl,
                     bool fineGrid, std::vector<Candidate>& out);
    void markFocus(int tw, int th, int tilePx);

    DetectorConfig cfg_;
    std::unique_ptr<Backend> backend_;
    DetectTiming timing_;
    int w_ = 0, h_ = 0, tw_ = 0, th_ = 0;
    std::vector<uint8_t> mask_;
    bool learning_ = false;
    std::vector<float> texSum_;
    int texN_ = 0;
    std::vector<FocusArea> focus_, roi_;
    std::vector<uint8_t> focusTiles_, roiMask_;
    long frameNo_ = 0;
    void roiMask(int tw, int th, int tilePx, const uint8_t* base);
    struct Stats;
    std::unique_ptr<Stats> stats_;
};

// ---------------- трекер и классификатор ----------------

enum class TargetClass : uint8_t {
    Unknown = 0,   // подтверждённый трек, но истории ещё мало для решения
    Drone = 1,     // плавное движение
    Bird = 2,      // рваное движение, взмахи (меняется контраст)
    Clutter = 3,   // статичная помеха: провода, просветы листвы
};
const char* class_name(TargetClass c);

struct TrackerConfig {
    // ассоциация: прогноз фильтра Калмана (постоянная скорость) и ворота по его ошибке
    float gate = 40.f;          // макс. радиус ассоциации, пиксели кадра
    float gate_sigma = 4.f;     // ворота = столько СКО ошибки прогноза...
    float gate_min = 4.f;       // ...но не меньше, пикс.
    float q_accel = 4.f;        // манёвренность цели: СКО ускорения, пикс./кадр^2 (птицы рвут до 4)
    float r_meas = 1.5f;        // ошибка измерения позиции, пикс. (на грубых масштабах больше)
    int confirm = 3;            // подтверждение: confirm попаданий из window кадров
    int window = 5;
    int max_miss = 12;          // подтверждённый трек держится на прогнозе до стольких кадров (не дольше, чем было попаданий)
    int tentative_miss = 2;     // неподтверждённый — столько
    float mover_px = 16.f;      // трек, ушедший по сцене дальше, — движущаяся цель, а не помеха
    float dup_px = 8.f;         // ближе к существующему треку новый трек не рождается
    float lock_switch = 1.5f;   // захват переходит на другую цель, если она весомее во столько раз
    int clutter_cell = 8;       // ячейка карты статичного клаттера
    float clutter_decay = 0.97f;
    float clutter_level = 0.35f;
    // Долгая память статичных помех (катушки и изоляторы ЛЭП, фонари, антенны) для
    // неподвижной камеры. Ячейка копит отклики, которые так никуда и не полетели: трек,
    // умерший на месте, или помеха карты клаттера. Набрала static_hits (с забыванием за
    // ~static_frames кадров) — там больше не рождаются цели. Летящая цель, проходящая
    // мимо такой точки, не теряется. static_frames = 0 — выключено.
    int static_frames = 3000;
    float static_hits = 8.f;
    // классификатор по движению
    int cls_window = 16;        // сколько последних попаданий смотреть
    int cls_min_hits = 8;       // меньше — класс Unknown
    float bird_resid = 2.2f;    // СКО отклонения от плавной траектории, пикс., выше — птица
    float bird_contrast_cv = 0.35f;  // вариация контраста (взмахи крыльев)
    float bird_speed = 0.f;     // >0: быстрее (пикс./кадр) — тоже признак птицы
};

struct TrackPoint { float x, y; };

struct Track {
    int id = 0;
    float x = 0, y = 0;      // позиция на кадре; при пропуске — прогноз
    float vx = 0, vy = 0;    // скорость на кадре, пикс./кадр (с учётом движения камеры)
    int polarity = -1;
    float contrast = 0;
    int miss = 0;            // кадров подряд без попадания (> 0 — трек держится на прогнозе)
    int hits = 0;            // всего попаданий
    int age = 0;             // кадров с рождения
    bool confirmed = false;  // хоть раз был подтверждён
    bool solid = false;      // подтверждён сейчас (на пропусках держится прежнее значение)
    bool clutter = false;    // стоит в "горячей" ячейке карты клаттера и никуда не летал
    float sigma = 0;         // СКО ошибки прогноза позиции, пикс.
    float score = 0;         // "весомость" цели для захвата: контраст, стаж, не помеха
    float travel = 0;        // как далеко ушёл по сцене от точки рождения, пикс.
    TargetClass cls = TargetClass::Unknown;
    float birdScore = 0;     // сглаженная "птичность", 0..1
    // признаки классификатора
    float speed = 0, resid = 0, contrastCv = 0;
    std::deque<TrackPoint> path;      // последние 30 позиций на кадре
    // служебное
    int det = -1;            // индекс детекции этого кадра или -1
    std::deque<bool> hist;
    struct Obs { int t; float x, y, c; int level; };
    std::deque<Obs> obs;     // попадания в координатах сцены
    float vsx = 0, vsy = 0;  // скорость относительно сцены
    float p00 = 0, p01 = 0, p11 = 0;   // ковариация Калмана (одинакова для x и y)
    float bx = 0, by = 0;    // точка рождения в координатах сцены
    int level = 0;
    bool everMover = false;
    std::vector<uint32_t> pend;   // ячейки, где был, пока не полетел: в память, если так и не полетит
};

class Tracker {
public:
    Tracker(int w, int h, const TrackerConfig& cfg = TrackerConfig());
    // Кадр: детекции и сдвиг картинки относительно прошлого кадра (GlobalMotion, 0 для
    // неподвижной камеры). Возвращает подтверждённые треки, в том числе идущие на прогнозе.
    const std::vector<Track>& update(const std::vector<Candidate>& dets, float camDx = 0, float camDy = 0);
    // Где ждать подтверждённые цели на следующем кадре (до update): для Detector::setFocus
    void focus(float camDx, float camDy, std::vector<FocusArea>& out) const;
    // Где ждать все живые треки (и новые, и на прогнозе) на следующем кадре: для
    // Detector::setRoi — между полными просмотрами кадра ищем только там
    void roi(float camDx, float camDy, std::vector<FocusArea>& out) const;
    // Следующий кадр детектор смотрел только вокруг треков (DetectTiming::partial):
    // карта клаттера в этот кадр не остывает, иначе помехи вне треков "забываются"
    void setPartial(bool p) { partial_ = p; }
    const std::vector<Track>& confirmed() const { return out_; }
    // Захваченная цель: самая весомая из подтверждённых, держится, пока трек жив и его
    // не превзошёл другой в lock_switch раз. 0 — захвата нет.
    int lockedId() const { return lockId_; }
    const std::vector<Candidate>& clutterDets() const { return clutterDets_; }
    // кандидаты, не попавшие ни в клаттер, ни в подтверждённый трек
    const std::vector<Candidate>& noiseDets() const { return noiseDets_; }
    void resetTracks();
    // Память статичных помех: сохранить/загрузить (файл привязан к размеру кадра), забыть,
    // и центры запомненных ячеек — для отображения
    bool saveStatic(const std::string& path) const;
    bool loadStatic(const std::string& path);
    void clearStatic();
    void staticCells(std::vector<TrackPoint>& out) const;
    int staticCellPx() const { return cfg_.clutter_cell; }
    TrackerConfig& config() { return cfg_; }
    int frame() const { return frame_; }

private:
    bool hot(float x, float y) const;
    float staticAt(size_t i) const;
    void staticAdd(size_t i);
    bool staticHot(float x, float y) const;
    void classify(Track& t) const;
    void shiftHeat(float dx, float dy);
    float measVar(int level) const;
    int coastLimit(const Track& t) const;

    TrackerConfig cfg_;
    int gw_, gh_;
    std::vector<float> heat_, heatTmp_;
    float heatAccX_ = 0, heatAccY_ = 0;
    std::vector<float> stat_;   // память статичных помех: значение на кадр statT_
    std::vector<int> statT_;    // (затухание считается лениво, при обращении)
    float gx_ = 0, gy_ = 0;   // накопленный сдвиг камеры
    std::vector<Track> tracks_, out_;
    std::vector<Candidate> clutterDets_, noiseDets_;
    int nextId_ = 1;
    int frame_ = 0;
    int lockId_ = 0, lostFor_ = 0;
    bool partial_ = false;
    float lastX_ = 0, lastY_ = 0, lastVx_ = 0, lastVy_ = 0;
};

// Сдвиг картинки между соседними кадрами (камера в руке или на поворотном устройстве).
// Сопоставление градиентов на уменьшенной копии кадра, от грубого уровня к точному.
struct MotionEstimate {
    float dx = 0, dy = 0;     // куда сдвинулась картинка, пикс. кадра
    float quality = 0;        // 0..1: насколько уверенно (на ровном небе — низко)
    bool valid = false;
};

class GlobalMotion {
public:
    MotionEstimate estimate(const uint8_t* gray, int w, int h, int stride);
    void reset() { prev_.clear(); }
    double lastMs() const { return ms_; }

private:
    struct Img { std::vector<uint8_t> px; int w = 0, h = 0; };
    std::vector<Img> prev_, cur_;
    int f_ = 1, w_ = 0, h_ = 0;
    double ms_ = 0;
};

}  // namespace dronolok
