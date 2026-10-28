// Дронолок: нативное ядро детектора точечных целей (C++ API).
//
// Алгоритм тот же, что в detector/detector.py (морфология линейными ядрами по
// 4 направлениям после min/max-пулинга 2x2, поиск пиков по тайлам), но все
// стадии слиты в один потоковый проход по блокам кадра, которые лежат в кэше,
// и посчитаны SIMD-ядрами (AVX2/SSE2 на x86, NEON на ARM).
#pragma once
#include <algorithm>
#include <cmath>
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
    // Порог по шуму (CFAR, как в радарах и ИК-поиске): уровень шума отклика оценивается
    // по медиане тайлов неба на каждом кадре, порог следующего кадра = noise_k * шум, но
    // не ниже thr_min и не выше thr_max. Порог "горячих" пикселей (текстура, маска неба)
    // масштабируется так же. Чистое небо — порог ниже, видны тусклые цели; шумная матрица
    // (вечер, высокий ISO) — выше, кадр не засыпает ложными. false — постоянный thr.
    bool auto_thr = false;
    float noise_k = 12.f;       // шум 2 уровня (типичная матрица днём) -> порог ~8
    float thr_min = 5.f, thr_max = 40.f;
    // Экономия: весь кадр (основной масштаб) и грубые масштабы просматриваются раз в
    // full_every / coarse_every кадров, в остальные — только тайлы вокруг прогнозов
    // треков (setRoi). Цель, которую уже ведём, не теряется, новая находится с
    // задержкой до N-1 кадров. 1 — каждый кадр целиком.
    int full_every = 1;
    // Зоны кадра по тайлам основного масштаба: небо — гладкие тайлы, связанные с верхним
    // краем кадра через гладкие; земля — текстура и всё, что ею отрезано от неба (снег,
    // стены, асфальт под линией деревьев); смешанные — небо рядом с землёй (кроны, крыши,
    // горизонт). На земле не ищем, на смешанных порог можно поднять (mixed_k), а цель там
    // подтверждается, только когда сдвинулась (Tracker). Считается каждый кадр, поэтому
    // годится и для съёмки с рук; для неподвижной камеры землю и так убирает маска неба.
    bool zones = false;
    float mixed_k = 1.f;        // на ваших роликах главное — правило "подтверждать, когда полетела", порог можно не поднимать
    int coarse_every = 2;       // крупная цель за кадр смещается мало, а грубые масштабы недёшевы
    // --- Неподвижная камера: ячейки 16x16 пикс. кадра (8x8 после пулинга) ---
    // Второй кандидат в тайле: самая сильная ячейка тайла вдали от первого — дрон рядом с
    // бликом, птицей или другим дроном. 1 — как раньше, один на тайл. Пара дронов в 20 пикс.:
    // полнота 65% -> 99%, ложных не прибавилось; цена ~0.5 мс на 16 Мп (4 потока).
    int per_tile = 2;
    // Фон по ячейкам у земли (неподвижная камера с маской неба). Маска неба не расширяется на
    // тайл запаса, а текстурные тайлы у самой кромки леса, которые раньше выбрасывались
    // целиком, смотрятся по ячейкам: у каждой ячейки там скользящие квантили (медиана и 90%)
    // её максимального отклика за ~bg_frames кадров, цель — только если отклик выше
    // медиана + bg_k * СКО. Шум неба для общего порога считается без этих тайлов.
    // Дальние дроны в 1-12% высоты над лесом: полнота 16% -> 52%, ложных не прибавилось.
    // В чистом небе свой порог ячейки не нужен: шум там ровный, общий порог точнее (с ним
    // же на всё небо дальние теряются: 52% -> 6%). Ячейки вокруг треков фон не учат.
    bool bg_map = true;
    float bg_k = 3.f;
    // Линия горизонта в маске неба (неподвижная камера): в каждом столбце тайлов всё ниже
    // кромки — первого сверху текстурного тайла, под которым текстурных больше половины, —
    // земля, даже если там попадаются гладкие тайлы (трава, поляна, дорога, просветы между
    // стволами). Дрон прилетает из-за горизонта и бывает над кромкой крон и крыш, ниже неё
    // почти никогда. Облако в небе кромкой не считается: под ним снова чистое небо.
    bool skyline = true;
    // Маска уточняется по ячейкам 16x16 (нужен bg_map): тайлы земли 64x64 у самой кромки
    // считаются бэкендом, и у каждой их ячейки копится, как часто в ней есть текстура (листва,
    // ветки). Ячейка, где за fine_frames кадров текстуры почти не бывает (< fine_tex) и над
    // которой тоже небо, — небо: просветы между кронами, небо над низкой крышей. Там ищем по
    // ячейкам с порогом по их фону, как у кромки. Маска так детализируется со временем, пока
    // работает камера; меняется маска тайлов — копится заново.
    bool fine_mask = true;
    int fine_frames = 150;
    float fine_tex = 0.05f;
    int bg_frames = 250;
    // Накопление слабой цели по кадрам (track-before-detect на ячейках, фон ячеек тогда
    // учится по всему небу): A = tbd_decay * max3x3(A) + max(0, z - tbd_z0), z — превышение
    // отклика ячейки над её фоном в СКО. Ячейка, накопившая tbd_thr, даёт кандидата, даже
    // если её отклик ниже порога (но не ниже tbd_min порога). Пока не окупается: на 16 Мп
    // дальних 19% -> 28%, на сценах 4 Мп прироста нет, а ложных тревог +10..70 в минуту;
    // максимум по ячейке 16x16 сам шумит, дрон с контрастом 7-12 не виден и так. +4.7 мс на 16 Мп.
    bool tbd = false;
    float tbd_decay = 0.7f, tbd_thr = 7.f, tbd_z0 = 1.5f, tbd_min = 0.5f;
    // Полоса у горизонта (неподвижная камера с маской неба): band_px пикс. кадра над кромкой
    // земли смотрятся каждый кадр и с порогом band_k * порог; небо выше — раз в upper_every
    // кадров, между ними только вокруг треков. Дальняя цель — низко над горизонтом, близкая
    // высоко, крупная и контрастная. band_px = 0 — выключено. 16 Мп, полоса 512 пикс.:
    // upper_every 2 — потерь нет; 3 — на 24% быстрее, высокие цели 97% -> 90%. band_k < 1
    // не помогает: в полосе прибавляются ложные, а не дальние.
    int band_px = 0;
    float band_k = 1.f;
    int upper_every = 1;
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
    bool edge = false; // найдена в смешанной зоне (DetectorConfig::zones)
    // Суммарный контраст пятна: сумма отклонений от фона в окне вокруг точки. У цели
    // в 1-4 пикс. пиковый контраст почти не меняется, а суммарный следует за площадью
    // силуэта, то есть за взмахами крыльев.
    float energy = 0;
    float margin = 0;  // во сколько раз отклик выше действующего порога (0 — неизвестно: не из Detector)
    bool tbd = false;  // найдена накоплением по кадрам (DetectorConfig::tbd), отклик ниже порога
    bool second = false;   // второй кандидат в тайле (DetectorConfig::per_tile)
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
    bool partial = false;    // основной масштаб — только вокруг треков (full_every) или полоса у горизонта
    bool upperSkipped = false;   // небо выше полосы у горизонта в этот кадр не смотрели (upper_every)
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
    // Забыть фон ячеек (DetectorConfig::bg_map) и оценку шума: освещение резко сменилось
    // (облако закрыло солнце), старый фон только мешает (см. SceneAnalyzer)
    void resetBackground();
    // Маска неба в файле: неподвижной камере не нужно учить её заново при каждом запуске
    // (в эти кадры захват ложный). loadMask сразу включает сохранённую маску и первые verify
    // кадров параллельно учит маску заново; если новая отличается больше чем на 10% тайлов
    // (камеру сдвинули, повернули), берётся новая (maskReplaced). Файл привязан к размеру
    // кадра, пулингу и тайлу — чужой не загрузится.
    bool saveMask(const std::string& path) const;
    bool loadMask(const std::string& path, int verify = 25);
    bool verifyingMask() const { return verifyLeft_ > 0; }
    bool maskReplaced() const { return maskReplaced_; }

    // Прогнозы треков на этот кадр. Действует до следующего вызова.
    void setFocus(const std::vector<FocusArea>& areas) { focus_ = areas; }
    // Где искать в кадры между полными просмотрами (Tracker::roi)
    void setRoi(const std::vector<FocusArea>& areas) { roi_ = areas; }
    const std::vector<uint8_t>& tileMask() const { return mask_; }   // tilesY x tilesX, 1 = искать
    // зоны последнего кадра (tilesY x tilesX): 0 земля, 1 смешанная, 2 небо; пусто, если выключены
    const std::vector<uint8_t>& zones() const { return zones_; }
    int tilesX() const { return tw_; }
    int tilesY() const { return th_; }
    int tileSizePx() const { return cfg_.tile * cfg_.pool; }        // тайл в пикселях кадра

    // Меняет параметры. Пороги применяются сразу; число потоков, привязка,
    // набор инструкций и геометрия (k, pool, tile) пересоздают бэкенд.
    void reconfigure(const DetectorConfig& cfg);
    const DetectorConfig& config() const { return cfg_; }
    const DetectTiming& timing() const { return timing_; }
    // Оценка шума отклика (средний отклик тайла неба, медиана) и действующий порог
    float noiseLevel() const { return noise_; }
    float threshold() const { return cfg_.auto_thr && thrEff_ > 0 ? thrEff_ : cfg_.thr; }
    std::string backendName() const;
    int threads() const;

private:
    void selectTiles(const TileStats& ts, int tw, int th, int tile, int factor, int pol, int lvl,
                     bool fineGrid, std::vector<Candidate>& out);
    void selectTbd(const TileStats& ts, int pol, std::vector<Candidate>& out);
    void updateBg(const TileStats& ts, int pol);
    void updateBand();
    float bgThr(int pol, size_t ci) const;
    bool bgWarm(int pol, size_t ci) const;
    void markExcl();
    void markFocus(int tw, int th, int tilePx);

    DetectorConfig cfg_;
    std::unique_ptr<Backend> backend_;
    DetectTiming timing_;
    int w_ = 0, h_ = 0, tw_ = 0, th_ = 0;
    std::vector<uint8_t> mask_;
    bool learning_ = false;
    std::vector<float> texSum_;
    int texN_ = 0;
    int verifyLeft_ = 0;
    bool maskReplaced_ = false;
    long maskVer_ = 0, bandVer_ = -1;
    std::vector<uint8_t> band_, nearGround_;   // тайлы полосы у горизонта; тайлы рядом с маской
    // уточнение маски по ячейкам (fine_mask): тайлы земли рядом с небом, маска для бэкенда
    // с ними, доля кадров с текстурой и число замеров по ячейкам, ячейки-небо в тайлах земли
    std::vector<uint8_t> edgeG_, compMask_, fineSky_, fineBase_;
    std::vector<float> fineTex_;
    std::vector<uint16_t> fineN_;
    void updateFine();
public:
    // тайлы полосы у горизонта (tilesY x tilesX, 1 — полоса), пусто — выключено или нет маски
    const std::vector<uint8_t>& horizonBand() const { return band_; }
    // уточнённая маска по ячейкам 16x16 пикс. кадра (tilesY*tile/8 x tilesX*tile/8): 1 — ячейка
    // в тайле земли, которая оказалась небом; пусто — уточнения нет
    const std::vector<uint8_t>& fineSky() const { return fineSky_; }
    int cellsPerTile() const { return cfg_.tile / 8; }
    // ячейка в точке кадра (x, y): максимум отклика в этом кадре, фон (среднее, СКО) и
    // накопленное A; false — ячеек нет (для отладки и вывода в приложении)
    bool cellInfo(int pol, float x, float y, float& cmax, float& mu, float& sd, float& A) const;
private:
    void buildMask(float frac, std::vector<uint8_t>& out) const;
    std::vector<FocusArea> focus_, roi_;
    std::vector<uint8_t> focusTiles_, roiMask_, zones_;
    std::vector<int> fill_;
    void computeZones();
    long frameNo_ = 0;
    float noise_ = 0, thrEff_ = 0;
    std::vector<float> noiseTmp_;
    // "горячий" пиксель (оценка текстуры тайла): 0.6 порога; с порогом по шуму — не ниже
    // уровня, при котором чистое небо даёт столько же горячих, сколько при thr 10 и шуме 2
    // (иначе низкий порог превращает небо в "текстуру" и маска выключает весь кадр)
    float hotLevel() const {
        const float h = threshold() * 0.6f;
        return cfg_.auto_thr && thrEff_ > 0 ? std::max(h, 8.5f * noise_) : h;
    }
    void updateNoise();
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
    // Подтверждение по накопленной "уверенности" (последовательный тест отношения
    // правдоподобия, как в track-before-detect): попадание добавляет margin^2 - 1 + ev_hit
    // (отклик у самого порога — почти ничего, вдвое выше — много), пропуск отнимает ev_miss.
    // Трек выдаётся, когда набрал ev_confirm. Шум, проскочивший у порога, и случайные
    // совпадения 3 из 5 так не подтверждаются, а тусклая, но устойчивая цель — да,
    // за несколько кадров. Позволяет держать порог детектора ниже. 0 — выключено.
    float ev_confirm = 3.f;
    float ev_hit = 0.f, ev_miss = 1.f;
    float mover_px = 16.f;      // трек, ушедший по сцене дальше, — движущаяся цель, а не помеха
    // родившемуся у кромки крон и крыш (Candidate::edge) — дальше в edge_mover_k раз: листва
    // на ветру "ходит" по кроне на десяток пикселей и иначе выдаётся целью
    float edge_mover_k = 3.f;
    float dup_px = 8.f;         // ближе к существующему треку новый трек не рождается
    float lock_switch = 1.5f;   // захват переходит на другую цель, если она весомее во столько раз
    // Кого захватывать из нескольких целей. Дрон обычно мелкий и медленный, самолёт и вертолёт —
    // контрастнее и быстрее. lock_slow > 0: вес для захвата делится на 1 + скорость / lock_slow
    // (пикс./кадр при 25 к/с), растёт с пробегом до 64 пикс., у зависшей цели — x0.05, и зависшая
    // не держит жёсткий захват. hard_switch > 0: и жёсткий захват переходит на цель, весомее в
    // hard_switch раз и хотя бы вдвое медленнее (иначе захваченный первым самолёт держится до
    // конца; на просто более яркую цель жёсткий захват по-прежнему не уходит). lock_cmax > 0:
    // контраст в весе не больше lock_cmax (на записях War Thunder не помогло). 0 — выкл.:
    // захват на самой контрастной цели. Записи War Thunder: дроны в захвате 31% -> 47%,
    // самолёты и вертолёты 53% -> 45%, ложный захват 26% кадров, как и было.
    float lock_cmax = 0.f;
    float lock_slow = 1.f;
    float hard_switch = 2.f;
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
    // Взмахи крыльев: суммарный контраст (Candidate::energy) за последние flap_window
    // кадров приближается синусоидой с периодом 2..16 кадров (при 25 кадр/с это 1.5-12.5 Гц;
    // более частые взмахи складываются в эти же частоты). Птица — если синусоида объясняет
    // долю дисперсии не меньше flap_r2 и её размах не меньше flap_depth от среднего.
    // flap_window = 0 — выключено.
    int flap_window = 48;       // до 64
    float flap_r2 = 0.45f;
    float flap_depth = 0.3f;
    // Птица, махавшая крыльями, остаётся птицей ещё flap_hold кадров: ласточки и
    // воробьиные машут пачками и планируют между ними
    int flap_hold = 50;
    // Зависание: цель, которая летала, а теперь стоит (скорость по сцене меньше hover_speed
    // пикс./кадр), — дрон; из птиц так умеет только пустельга. 0 — выключено.
    float hover_speed = 0.3f;
    // Стая: попутчик машущей птицы (ближе flock_px, скорость отличается меньше чем на
    // flock_dv от его скорости) — тоже птица. Рой дронов крыльями не машет и так не
    // помечается. 0 — выключено.
    float flock_px = 150.f;
    float flock_dv = 0.35f;
    // Захват крупной цели целиком: вокруг захваченной цели на каждом кадре ищется её силуэт
    // (measure_silhouette по кадру делает вызывающий код, см. lockSearch). Прицел — в центре
    // силуэта, а не на кончике крыла или хвосте; отклики внутри силуэта (крыло, хвост,
    // подвес) — части той же цели: другие треки на них не рождаются и захват не уводят.
    // Силуэт ведёт цель и там, где детектор её не видит (крупнее его масштабов).
    bool silhouette = true;
    float silh_max = 400.f;     // предел полуразмера окна поиска силуэта, пикс.
    // Ожидание размера: видимый размер захваченной цели (по силуэту) ведётся вместе со
    // скоростью его изменения — дрон приближается (растёт) или удаляется (уменьшается).
    // По ожидаемому размеру на следующем кадре: окно поиска силуэта, масштаб детектора
    // вокруг цели, ворота сопоставления; силуэт, не похожий по размеру на ожидаемый
    // (соседнее облако, крона, птица рядом), не принимается. Большой дрон, проходящий
    // мимо камеры, сначала маленький, потом крупный, потом снова маленький — захват
    // ждёт этого, а не теряет цель на переходе.
    bool size_predict = true;
    float size_rate_max = 0.08f;   // предел скорости изменения размера, доля за кадр
    float size_tol = 1.8f;         // силуэт принимается, если размер в size_tol раз от ожидаемого
    // Жёсткий захват: захваченная цель verify_frames кадров подряд классифицирована как дрон —
    // захват больше не переходит на другие цели, цель держится на прогнозе и силуэте до
    // hard_miss кадров, а детектор смотрит в основном вокруг неё (весь кадр — раз в
    // hard_full_every кадров, см. hardFullEvery). Снимается, если цель пропала дольше
    // hard_miss кадров или оказалась птицей. false — захват как обычно.
    bool hard_lock = true;
    int verify_frames = 25;
    int hard_miss = 60;
    int hard_full_every = 6;
    // Наложения на кадр (надписи, логотипы, OSD, грязь на объективе): отклик стоит в одной
    // точке кадра (СКО меньше overlay_px) overlay_hits попаданий подряд, пока камера ушла на
    // overlay_cam пикс. — это не цель, а наложение: в долгую память (она в координатах кадра
    // и для наложений учится и при движущейся камере). overlay_hits = 0 — выключено.
    int overlay_hits = 16;
    // Цель на мачте или столбе (светильник, антенна, изолятор): у захваченной цели
    // вверх или вниз от силуэта уходит длинная тонкая линия в большинстве замеров —
    // помеха, захват снимается. При съёмке с рук такая цель "летит" из-за неточной
    // оценки поворота камеры и иначе проходит проверку на дрона
    bool anchor_check = true;
    float overlay_px = 1.f;
    float overlay_cam = 24.f;
};

// Время в секундах. Числа выше подобраны для камеры 25 кадр/с: счётчики кадров — это время
// при 25 к/с, скорости, ускорения и ворота — пикс. за кадр при 25 к/с. tracker_for_fps
// пересчитывает их под настоящую частоту камеры так, чтобы в секундах и пикс./с всё осталось
// как было: камера 16 Мп даёт ~10 к/с (цель смещается за кадр в 2.5 раза дальше, удержание
// 12 кадров — уже 1.2 с вместо 0.5), запись экрана — 60. Счётчики попаданий (confirm,
// cls_min_hits, static_hits) и расстояния (mover_px, dup_px) не меняются.
TrackerConfig tracker_for_fps(const TrackerConfig& c, float fps, float ref_fps = 25.f);

struct TrackPoint { float x, y; };

// Силуэт цели вокруг точки (measure_silhouette): связная область темнее/светлее фона
struct Silhouette {
    bool valid = false;
    float x = 0, y = 0;    // центр, взвешенный по контрасту, пикс. кадра
    float w = 0, h = 0;    // габарит, пикс.
    float contrast = 0;    // средний контраст силуэта
    int area = 0;          // площадь, пикс.
    // от силуэта вверх или вниз тянется длинная тонкая линия (мачта, столб, ствол) —
    // это светильник на мачте или что-то на конструкции, а не летящий дрон
    bool attached = false;
};
// Силуэт объекта, который ждём в точке (x, y) кадра gray: r — ожидаемый радиус (0 —
// неизвестен), polarity -1 тёмный / +1 светлый. Окно растёт, пока силуэт упирается в его
// край, до полуразмера maxR; крупнее — не цель (земля, крона, облако), valid = false.
Silhouette measure_silhouette(const uint8_t* gray, int w, int h, int stride, float x, float y, float r, int polarity,
                              float minContrast = 8.f, float maxR = 400.f);

// Состояние захвата
enum class LockState : uint8_t {
    None = 0,      // захвата нет
    Tracking = 1,  // цель захвачена, проверяем, дрон ли это
    Hard = 2,      // жёсткий захват: подтверждено, что дрон (TrackerConfig::hard_lock)
};

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
    float lockW = 0;         // вес для выбора захвата: score с учётом lock_cmax и lock_slow
    float evidence = 0;      // накопленная уверенность, см. TrackerConfig::ev_confirm
    float travel = 0;        // как далеко ушёл по сцене от точки рождения, пикс.
    TargetClass cls = TargetClass::Unknown;
    float birdScore = 0;     // сглаженная "птичность", 0..1
    // признаки классификатора
    float speed = 0, resid = 0, contrastCv = 0;
    float flapR2 = 0;        // доля дисперсии суммарного контраста, объяснённая взмахами
    float flapPeriod = 0;    // период взмахов (видимый, с учётом наложения частот), кадров
    float flapDepth = 0;     // размах пульсации к среднему
    bool hovering = false;   // летал, а теперь висит на месте
    bool flock = false;      // летит в стае с машущей птицей
    bool overlay = false;    // наложение на кадр: стоит в кадре, пока камера движется
    bool anchored = false;   // на мачте или столбе (см. Silhouette::attached): не дрон
    // ожидание размера (захват): логарифм видимого размера и его изменение за кадр
    float ls = 0, dls = 0;
    uint8_t lsN = 0;         // замеров размера, 0 — размер не ведётся
    uint8_t sizeRej = 0;     // кадров подряд силуэт не похож размером на ожидаемый
    // ожидаемый размер (наибольшая сторона силуэта) через ahead кадров, 0 — не ведётся
    float expectedSize(int ahead = 1) const { return lsN ? std::exp(ls + (float)ahead * dls) : 0.f; }
    uint8_t silN = 0, attN = 0;   // замеров силуэта у захвата и сколько из них "на мачте"
    bool everLocked = false; // был захвачен: в долгую память помех не попадает
    float bw = 0, bh = 0;    // габарит силуэта (захваченная цель), 0 — не измерен
    std::deque<TrackPoint> path;      // последние 30 позиций на кадре
    // служебное
    int det = -1;            // индекс детекции этого кадра или -1
    std::deque<bool> hist;
    struct Obs { int t; float x, y, c; int level; float e; float fx, fy; };   // fx, fy — в координатах кадра
    std::deque<Obs> obs;     // попадания в координатах сцены
    float vsx = 0, vsy = 0;  // скорость относительно сцены
    float p00 = 0, p01 = 0, p11 = 0;   // ковариация Калмана (одинакова для x и y)
    float bx = 0, by = 0;    // точка рождения в координатах сцены
    int level = 0;
    bool everMover = false;
    bool edgeBorn = false;   // родился в смешанной зоне: выдаётся, только когда полетит
    int flapT = -1000000;    // кадр, когда последний раз были видны взмахи
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
    LockState lockState() const { return !lockId_ ? LockState::None : hard_ ? LockState::Hard : LockState::Tracking; }
    // сколько кадров подряд захваченная цель — дрон (до verify_frames — проверка)
    int verifyFrames() const { return verify_; }
    // Жёсткий захват: детектору смотреть весь кадр раз в столько кадров (иначе 1)
    int hardFullEvery() const { return hard_ ? std::max(1, cfg_.hard_full_every) : 1; }
    // Где искать силуэт захваченной цели на следующем кадре (до update): точка, ожидаемый
    // радиус, полярность. false — захвата нет или силуэт выключен
    bool lockSearch(float camDx, float camDy, float& x, float& y, float& r, int& polarity) const;
    // Силуэт захваченной цели на этом кадре — учитывается в следующем update()
    void setLockSilhouette(const Silhouette& s) { lockSilh_ = s; }
    // При движущейся камере долгая память учится только наложениям на кадр (см. overlay_hits)
    void setMovingCamera(bool m) { movingCam_ = m; }
    // Камера на поворотном устройстве ведёт захваченную цель: цель стоит почти на месте
    // кадра, а фон едет навстречу её движению (сдвиг камеры — в update как camDx/camDy, от
    // датчиков поворотки или GlobalMotion). Захваченная цель тогда не считается наложением
    // на кадр (см. overlay_hits) — она и должна стоять в кадре, пока сцена уезжает
    void setFollowing(bool f) { following_ = f; }
    bool following() const { return following_; }
    // Куда вести камеру: захваченная цель на этом кадре (x, y — пикс. кадра) и её скорость
    // по сцене (пикс./кадр). Поворотка, сдвинувшая кадр на vx, vy плюс долю ошибки до
    // центра, держит цель в центре. false — захвата нет
    bool aim(float& x, float& y, float& vx, float& vy) const;
    const std::vector<Candidate>& clutterDets() const { return clutterDets_; }
    // кандидаты, не попавшие ни в клаттер, ни в подтверждённый трек
    const std::vector<Candidate>& noiseDets() const { return noiseDets_; }
    void resetTracks();
    // Память статичных помех: сохранить/загрузить (файл привязан к размеру кадра), забыть,
    // и центры запомненных ячеек — для отображения
    bool saveStatic(const std::string& path) const;
    bool loadStatic(const std::string& path);
    void clearStatic();
    // снимок памяти помех и возврат к нему (повторный просмотр ролика — как первый)
    void getStatic(std::vector<float>& out) const;
    void setStatic(const std::vector<float>& v);
    void staticCells(std::vector<TrackPoint>& out) const;
    int staticCellPx() const { return cfg_.clutter_cell; }
    TrackerConfig& config() { return cfg_; }
    int frame() const { return frame_; }

private:
    bool hot(float x, float y) const;
    float staticAt(size_t i) const;
    void staticAdd(size_t i);
    bool staticHot(float x, float y) const;
    bool silhouetteFor(const Track& t) const;
    void classify(Track& t) const;
    void shiftHeat(float dx, float dy);
    float measVar(int level) const;
    float evidenceOf(const Candidate& d) const;
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
    int verify_ = 0;
    bool hard_ = false;
    Silhouette lockSilh_;
    bool movingCam_ = false;
    bool following_ = false;
    bool partial_ = false;
    float lastX_ = 0, lastY_ = 0, lastVx_ = 0, lastVy_ = 0;
    float lastLs_ = 0, lastDls_ = 0;   // ожидание размера захвата: передаётся треку, подхватившему цель
    uint8_t lastLsN_ = 0;
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
