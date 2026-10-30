// Мультитест: папка с видео прогоняется с несколькими вариантами настроек, по каждому
// прогону считаются метрики (цели, захват, скорость, а при <видео>.gt.json — точность по
// разметке), и складывается отчёт: report.html (смотреть самому), results.json и
// results.csv (машиночитаемо, для разбора). Работает и из окна (вкладка «Мультитест»),
// и без окна: dronolok_app --multitest папка ... (см. run_cli).
#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "pipeline.hpp"

namespace mt {

// Что меняется от варианта к варианту: настройки конвейера и доля разрешения кадра
// (размер обработки у каждого видео свой, поэтому доля, а не пиксели)
struct Setup {
    Settings s;
    float scale = 1.f;
};

// Настройка, которую мультитест умеет перебирать
struct Param {
    const char* key;      // имя в командной строке и в JSON
    const char* name;     // по-русски, как в приложении
    const char* group;    // вкладка или раздел приложения
    bool flag;            // да/нет (значения 0 и 1)
    const char* values;   // что перебирать по умолчанию
    double (*get)(const Setup&);
    void (*set)(Setup&, double);
};
const std::vector<Param>& params();
const Param* find_param(const std::string& key);
std::string fmt_value(const Param& p, double v);
// "8, 10, 14" или "вкл, выкл" -> числа; false — не разобрать
bool parse_values(const Param& p, const std::string& text, std::vector<double>& out);

struct Choice {
    std::string key;
    std::vector<double> values;
};

enum class Mode {
    OneAtATime,   // база + каждая настройка по отдельности (остальные как в базе)
    Grid,         // все сочетания выбранных значений
};
const int GRID_MAX = 256;

struct Variant {
    std::string name;                                   // "база", "Порог по шуму: вкл"
    std::vector<std::pair<std::string, double>> changes;   // ключ -> значение (отличия от базы)
    Setup setup;
};
// Варианты: первым всегда база. warn — что пришлось урезать или пропустить
std::vector<Variant> make_variants(const Setup& base, const std::vector<Choice>& ch, Mode mode, std::string& warn);
int count_variants(const Setup& base, const std::vector<Choice>& ch, Mode mode);

// Перенести в dst всё, что относится к алгоритму (а ресурсы, подачу кадров и вид оставить)
void apply_algo(Settings& dst, const Settings& src);

bool is_video_file(const std::string& name);
std::vector<std::string> list_videos(const std::string& folder);   // по алфавиту, без вложенных папок

struct Config {
    std::string folder;                 // папка с видео
    std::vector<std::string> videos;    // пусто — все видео из folder
    std::string outDir;                 // пусто — folder/multitest-ГГГГММДД-ЧЧММСС
    Setup base;
    std::vector<Choice> choices;
    Mode mode = Mode::OneAtATime;
    long maxFrames = 0;      // кадров с каждого видео, 0 — всё видео
    int warmup = 25;         // первые кадры (обучение маски неба) в точность не входят
    float radius = 12;       // цель найдена, если трек ближе, пикс. кадра обработки
    int parallel = 1;        // прогонов одновременно
    bool realtime = false;   // как камера: кадры в темпе видео, что не успели — пропускаются
    bool emulate = false;    // оставить имитацию устройства (замедление); иначе — во всю силу ПК
    bool maskFile = false;   // брать маску неба из файла (.mask.bin) — иначе учится в каждом прогоне
    std::string tracksDir;   // журнал треков каждого прогона: <папка>/<видео>.<вариант>.csv (пусто — нет)
    // все сдвиги такта: у камеры реже видео (step, cam_fps) снимки зависят от того, с какого кадра файла
    // пошёл такт. Прогон по каждому сдвигу (шаг 4 — 4 прогона, 14 к/с из 60 — 5), в отчёте среднее и разброс
    bool allPhases = false;
};

struct VideoMeta {
    std::string path, file;
    int w = 0, h = 0;
    double fps = 0;
    long frames = 0;
    bool gt = false;
    bool noDrones = false;   // по разметке в ролике нет ни одной цели
};

// Итог одного прогона «видео x вариант». Отрицательное значение — нет данных
struct RunResult {
    int video = 0, variant = 0;
    int phase = -1;   // сдвиг такта этого прогона (Config::allPhases); -1 — сдвиг из настроек варианта
    bool done = false;
    std::string error;
    double frames = 0;
    double wallS = 0, fps = 0;
    double detectMs = 0, detectP95 = 0, procMs = 0, procP95 = 0;
    // без разметки
    double targetsPerFrame = 0, targetFramesPct = 0;
    double tracks = 0, droneIds = 0, birdIds = 0;
    double lockPct = 0, hardPct = 0;
    double lockAcq = 0;   // сколько раз захват брал новую цель
    double skyPct = -1; // доля кадра, где детектор ищет (не закрыто маской неба); -1 — маски нет
    double skyDarkPct = -1; // ...тёмные цели (с тайлами, закрытыми только светлой текстурой)
    // по разметке
    bool gt = false;
    double evalFrames = 0, gtDroneFrames = 0;
    double recallDrone = -1, recallBird = -1, droneAsDrone = -1;
    double falsePerFrame = -1;
    double falseTracks = -1;
    double lockOnDrone = -1, hardOnDrone = -1, falseLock = -1;
    double firstDetectS = -1, firstLockS = -1;
    double score = -1;
    // среднее по сдвигам такта (merge_phases): сколько сдвигов и разброс по ним — у каждой метрики из
    // metrics() худшее и лучшее значение (меньшее и большее; отрицательное — нет данных)
    int phases = 1;
    std::vector<double> lo, hi;
};

// Метрика для таблиц отчёта: как называть, как печатать и что считать лучше
struct Metric {
    const char* key;
    const char* name;
    const char* unit;
    int better;     // +1 больше — лучше, -1 меньше — лучше, 0 — смотреть по смыслу
    bool gtOnly;    // только при разметке
    int digits;
    double (*get)(const RunResult&);
    const char* help;
};
const std::vector<Metric>& metrics();
// значение метрики прогона; false — нет данных (прогон не готов, нет разметки или метрика не посчитана)
bool metric_value(const Metric& m, const RunResult& r, double& v);

// Сколько разных сдвигов такта у камеры реже видео: шаг 4 — 4, 14 к/с из 60 — 5 (шаги 4 и 5); 1 — кадры подряд
int phase_count(const Settings& s, double videoFps);
// Прогоны по сдвигам такта -> один итог на «видео x вариант»: среднее по сдвигам и разброс (lo, hi).
// Итог готов, когда готовы все его сдвиги; ошибка любого сдвига — ошибка итога
std::vector<RunResult> merge_phases(const std::vector<RunResult>& runs);

class Runner {
public:
    ~Runner();
    bool start(const Config& c, std::string& err);
    void stop();                  // остановить; отчёт по уже сделанному остаётся
    bool running() const { return running_; }

    struct Status {
        int done = 0, total = 0;
        double frac = 0;          // с учётом кадров текущих прогонов
        std::string current;
        double elapsedS = 0, etaS = -1;
        bool finished = false, stopped = false;
        std::string outDir, html, error;
    };
    Status status() const;
    // снимок для интерфейса; res — по одному итогу на «видео x вариант» (при всех сдвигах — среднее)
    void snapshot(std::vector<VideoMeta>& v, std::vector<Variant>& var, std::vector<RunResult>& res) const;

private:
    void work();
    void writeReport();

    Config cfg_;
    std::vector<VideoMeta> videos_;
    std::vector<Variant> variants_;
    mutable std::mutex m_;
    std::vector<RunResult> res_;
    std::vector<std::thread> th_;
    std::thread joiner_;
    std::atomic<bool> running_{false}, stop_{false};
    std::atomic<int> next_{0}, done_{0};
    std::vector<std::atomic<long>*> live_;   // кадров у прогонов в работе (по потокам)
    std::vector<int> liveJob_;
    std::string error_, html_;
    double t0_ = 0, t1_ = 0;
    bool finished_ = false;
};

// Отчёт: report.html, results.json, results.csv в dir. runs — все прогоны (при всех сдвигах такта —
// каждый сдвиг отдельно, отчёт сам сводит их в среднее). false — не удалось записать
bool write_report(const std::string& dir, const Config& c, const std::vector<VideoMeta>& videos,
                  const std::vector<Variant>& variants, const std::vector<RunResult>& runs, bool finished,
                  double elapsedS, std::string& err);

// dronolok_app --multitest ...
int run_cli(int argc, char** argv);

}  // namespace mt
