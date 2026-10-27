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
};

struct DetectTiming {
    double total_ms = 0;     // весь detect()
    double response_ms = 0;  // пулинг + морфология + статистика тайлов (бэкенд)
    double select_ms = 0;    // выбор кандидатов
    int jobs = 0, jobs_skipped = 0;   // задания бэкенда, пропущенные по маске
};

class Backend;

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
    void selectPolarity(int pol, std::vector<Candidate>& out);

    DetectorConfig cfg_;
    std::unique_ptr<Backend> backend_;
    DetectTiming timing_;
    int w_ = 0, h_ = 0, tw_ = 0, th_ = 0;
    std::vector<uint8_t> mask_;
    bool learning_ = false;
    std::vector<float> texSum_;
    int texN_ = 0;
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
    float gate = 40.f;          // радиус ассоциации, пиксели кадра
    int confirm = 3;            // подтверждение: confirm попаданий из window кадров
    int window = 5;
    int max_miss = 5;
    int clutter_cell = 8;       // ячейка карты статичного клаттера
    float clutter_decay = 0.97f;
    float clutter_level = 0.35f;
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
    float x = 0, y = 0, vx = 0, vy = 0;
    int polarity = -1;
    float contrast = 0;
    int miss = 0;
    int hits = 0;            // всего попаданий
    int age = 0;             // кадров с рождения
    bool clutter = false;    // стоит в "горячей" ячейке карты клаттера
    TargetClass cls = TargetClass::Unknown;
    float birdScore = 0;     // сглаженная "птичность", 0..1
    // признаки классификатора
    float speed = 0, resid = 0, contrastCv = 0;
    std::deque<TrackPoint> path;      // последние 30 позиций
    // служебное
    int det = -1;            // индекс детекции этого кадра (среди не-клаттерных)
    std::deque<bool> hist;
    struct Obs { int t; float x, y, c; };
    std::deque<Obs> obs;
};

class Tracker {
public:
    Tracker(int w, int h, const TrackerConfig& cfg = TrackerConfig());
    // Возвращает подтверждённые треки (как Tracker.update в Python)
    const std::vector<Track>& update(const std::vector<Candidate>& dets);
    const std::vector<Track>& confirmed() const { return out_; }
    const std::vector<Candidate>& clutterDets() const { return clutterDets_; }
    // кандидаты, не попавшие ни в клаттер, ни в подтверждённый трек
    const std::vector<Candidate>& noiseDets() const { return noiseDets_; }
    void resetTracks();
    TrackerConfig& config() { return cfg_; }
    int frame() const { return frame_; }

private:
    bool hot(float x, float y) const;
    void classify(Track& t) const;

    TrackerConfig cfg_;
    int gw_, gh_;
    std::vector<float> heat_;
    std::vector<Track> tracks_, out_;
    std::vector<Candidate> clutterDets_, noiseDets_;
    int nextId_ = 1;
    int frame_ = 0;
};

}  // namespace dronolok
