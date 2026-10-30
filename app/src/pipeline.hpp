// Конвейер: поток чтения/декодирования -> очередь -> поток детекции и трекинга -> последний
// результат для интерфейса. Так же устроено и на устройстве (захват, обработка, вывод
// в разных потоках), поэтому замеры отражают реальную картину.
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
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
    ColorPlanes color;        // цвет (выборка U, V), если он есть в видео
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
    // частота камеры, 0 = FPS видео. Ниже частоты видео — эмуляция медленной камеры: кадры файла
    // между её снимками пропускаются (60 к/с, камера 14 — каждый 4-й-5-й по времени), и цель за
    // кадр смещается, как у настоящей камеры. Раньше все кадры подавались реже, как замедленное
    // видео, а трекер считал на заданной частоте — смещения за кадр вчетверо меньше ожидаемых
    // (60 к/с при "14": захват на дроне 86 -> 63%, жёсткий 64 -> 40%). Выше частоты видео — кадры подаются чаще
    double targetFps = 0;
    // камера реже видео: брать каждый frameStep-й кадр файла (60 к/с, шаг 4 — камера 15 к/с).
    // Номера кадров — файла, поэтому разметка <видео>.gt.json подходит без пересчёта;
    // трекер и темп камеры — на частоте видео / шаг
    int frameStep = 1;
    // ...и с какого кадра файла считать такт камеры (шаг 4, сдвиг 1 — кадры 1, 5, 9, ...). На 14-15 к/с
    // итог по ролику сильно зависит от того, какие кадры попали (wt17: сдвиг на кадр — на дроне 81 -> 71%),
    // поэтому сравнивать настройки на медленной камере лучше по нескольким сдвигам
    int framePhase = 0;
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
    bool sceneShift = true;    // авто на крупном кадре: сдвиг камеры — для координат сцены трекера
    bool panDetect = true;     // авто на крупном кадре: плавный поворот камеры — без маски неба
    bool panMotion = false;    // ...и с учётом сдвига в прогнозе трекера
    // авто на крупном кадре: сдвиг камеры меряется от кадра ~1/15 с назад (GlobalMotion::setLong) —
    // на 60 к/с медленный поворот по соседнему кадру занижен втрое (wt19 жёсткий 31 -> 88%)
    bool motionLong = true;
    bool focus = false;        // пониженный порог возле прогноза треков (эксперимент)
    int zones = -1;            // зоны кадра небо/смешанные/земля: -1 авто, 0 выкл., 1 вкл.
    // память помех из файла рядом с видео: загружать при открытии и сохранять при закрытии.
    // В замерах выключено: зацикленный ролик повторяет те же позиции целей, и память,
    // накопленная прошлыми запусками, искажала бы результат
    bool staticFile = true;
    // маска неба из файла рядом с видео (<видео>.mask.bin) вместо обучения на каждом старте;
    // первые learnMaskFrames кадров она сверяется с видимым, при расхождении учится заново
    bool maskFile = true;
    // чистый старт: накопленное с прошлых запусков (память помех, маска неба из файлов рядом
    // с видео) не читается и не пишется — видео проверяется так, будто открыто впервые
    bool fresh = false;
    bool useStaticFile() const { return staticFile && !fresh; }
    bool useMaskFile() const { return maskFile && !fresh; }
    // трекер считает в секундах: скорости, ворота и выдержки пересчитываются под частоту
    // кадров (dronolok::tracker_for_fps; повторы кадров записи экрана не считаются)
    bool trkSeconds = true;
    // настоящий интервал кадра в трекер: камера реже видео идёт неровным тактом (14 к/с из 60 —
    // шагами 4 и 5 кадров), а пропущенный кадр — двойной интервал. Прогноз трека — на это время,
    // как у камеры с ровным тактом (Tracker::setFrameDt). При шаге кадров целым числом ничего не меняет
    bool frameDt = true;
    // окна маски неба (обучение, уточнение у кромки, доращивание, перепроверка), оценка сдвига
    // камеры и признак панорамы — тоже в секундах: числа кадров подобраны на 60 к/с и
    // пересчитываются под частоту камеры (dronolok::detector_for_fps, GlobalMotion::setFps)
    bool fpsScale = true;
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
    std::vector<uint8_t> maskDark;  // закрытые маской тайлы, где ищутся тёмные цели (Detector::polarityOpen)
    std::vector<uint8_t> maskBright;  // ...и светлые
    std::vector<uint8_t> band;      // полоса у горизонта (Detector::horizonBand), пусто — выключена
    std::vector<uint8_t> fineSky;   // ячейки 16x16 неба в тайлах земли (уточнение маски), см. Detector::fineSky
    int cellsPerTile = 0;
    int tilesX = 0, tilesY = 0, tilePx = 0;
    bool learning = false;
    std::vector<dronolok::GtObj> gt;
    float gsx = 1, gsy = 1;
    int seq = 0;
    // захват и прогноз
    int lockId = 0;                              // захваченная цель (Track::id), 0 — нет
    std::vector<int> lockQueue;                  // очередь целей по весомости (Tracker::lockQueue)
    bool manualLock = false;                     // захват выбран вручную
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
    // Кадры не из файла, а от генератора (эмулятор камер): gen заполняет px (w x h), index и
    // возвращает false, когда кадров больше не будет. Вызывается в потоке чтения; может ждать.
    // name — подпись вместо пути к файлу (файлы рядом с видео не читаются и не пишутся)
    bool openGenerator(const VideoInfo& info, std::function<bool(Frame&)> gen, const std::string& name);
    void close();
    bool isOpen() const { return open_; }
    const std::string& path() const { return path_; }
    VideoInfo info() const;

    void setSettings(const Settings& s);
    Settings settings() const;
    // частота кадров, которые видит алгоритм: камеры (или видео), делённая на шаг прореживания
    double cameraFps(const Settings& s) const;
    // кадров файла на кадр камеры (шаг frameStep и частота камеры ниже частоты видео), 1 — все
    double fileStep(const Settings& s) const;
    // кадр файла index пропускается: между снимками медленной камеры
    bool skipFrame(const Settings& s, int64_t index) const;
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
    // Маска неба снаружи (панорамная маска 360° эмулятора при известном повороте камеры): ставится
    // со следующего кадра без обучения; треки — заново (вид сменился), смену сцены hold кадров не ловим
    void setExternalMask(std::vector<uint8_t> m, int tw, int th, int hold) {
        std::lock_guard<std::mutex> lk(extM_);
        extMask_ = std::move(m);
        extTw_ = tw;
        extTh_ = th;
        extHold_ = hold;
        extNew_ = true;
    }
    // Память статичных помех лежит рядом с видео (<видео>.static.bin): загружается при
    // открытии и сохраняется при закрытии, если включена
    std::string staticPath() const { return path_ + ".static.bin"; }
    std::string maskPath() const { return path_ + ".mask.bin"; }
    void staticSave() { staticCmd_ = 1; }
    void staticClear() { staticCmd_ = 2; }
    // Очередь целей: захват на следующую по весомости (по кругу) или на выбранный трек.
    // Ручной выбор держится, пока цель жива (Tracker::selectLock)
    void nextTarget() { lockCmd_ = -1; }
    // Подсказка извне (точная камера видит цель, а широкая её потеряла или ведёт слабо): здесь
    // детектор ищет с пониженным порогом, как вокруг прогноза своих треков, и не пропускает это
    // место между полными просмотрами кадра. Действует на кадры, обработанные после вызова; пусто — нет
    void setCue(const std::vector<dronolok::FocusArea>& a) {
        std::lock_guard<std::mutex> lk(cueM_);
        cue_ = a;
    }
    // Другая камера рассмотрела трек id вблизи и видит птицу (Tracker::markBird) — со следующего кадра
    void markBird(int id) {
        std::lock_guard<std::mutex> lk(cueM_);
        birds_.push_back(id);
    }
    void selectTarget(int id) { lockCmd_ = id; }
    // забыть накопленное в этом запуске (память помех, маску неба), файлы не трогать
    void forget() {
        staticCmd_ = 3;
        relearnMask();
    }
    void resetStats();
    // Разметка <видео>.gt.json: заменить на ходу (после правки в приложении); пустая — убрать
    std::string gtPath() const { return path_ + ".gt.json"; }
    void setGroundTruth(const dronolok::GroundTruth& g);
    // Вырезки целей из истории кадров с подписью по разметке — набор для настройки классификатора.
    // Возвращает число вырезок, msg — что и куда сохранено
    int exportCrops(const std::string& dir, std::string& msg) const;

    // Каждый обработанный кадр, по порядку, в потоке обработки (мультитест считает по ним
    // свои метрики). Задавать до open(); вызывается под внутренней блокировкой — из f
    // к самому Pipeline не обращаться
    void setObserver(std::function<void(const Result&)> f) { observer_ = std::move(f); }
    // Лёгкий опрос хода без копирования метрик
    struct Progress {
        long processed = 0, decoded = 0;
        int queue = 0;
        std::string status;
    };
    Progress progress() const;

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
    std::function<bool(Frame&)> gen_;   // кадры от генератора вместо файла
    VideoInfo info_;
    std::shared_ptr<const dronolok::GroundTruth> gt_;   // под rm_: разметку можно заменить на ходу
    bool haveGt_ = false;

    mutable std::mutex sm_;
    Settings settings_;
    std::mutex cueM_;
    std::vector<dronolok::FocusArea> cue_;
    std::vector<int> birds_;
    int settingsVer_ = 0;

    std::thread dec_, proc_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> paused_{false};
    std::atomic<int> steps_{0};
    std::atomic<bool> relearn_{true};
    std::atomic<bool> maskForce_{false};   // "заново": учить, даже если есть файл
    std::mutex extM_;
    std::vector<uint8_t> extMask_;
    int extTw_ = 0, extTh_ = 0, extHold_ = 0;
    bool extNew_ = false;
    std::atomic<int> staticCmd_{0};
    std::atomic<int> lockCmd_{0};   // -1 — следующая цель очереди, > 0 — id выбранного трека
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
    std::function<void(const Result&)> observer_;
};

double now_ms();
