// Эмулятор двух камер: неподвижная широкая (16 Мп, 120°, ~14 к/с) находит цели нашим конвейером
// и указывает их точной камере 640x480 на поворотке; точная наводится с ограничением скорости
// поворота, находит цель в окне поиска и дальше ведёт её сама. Мир, камеры и поворотка —
// синтетические, алгоритм — настоящий: Pipeline для широкой, Detector + Tracker для точной.
//
// Время модельное и общее для всех устройств. Шаг — кадр точной камеры; кадр широкой
// обрабатывается целиком (конвейер ждём), а её сообщение о цели доходит до поворотки через
// handoffLatency. Поэтому результат не зависит от скорости ПК и повторяется при том же seed.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "dronolok/detector.hpp"
#include "dronolok/geometry.hpp"
#include "dronolok/gimbal.hpp"
#include "mountsim.hpp"
#include "pipeline.hpp"
#include "scenelink.hpp"
#include "silhouette.hpp"

namespace twocam {

using dronolok::CameraModel;
using dronolok::Pose;
using dronolok::TargetReport;
using dronolok::Vec3;

constexpr int PANO_W = 1440, PANO_H = 360;   // 0.25° на ячейку: азимут 0..360, угол места -20..70
constexpr float PANO_STEP = 0.25f, PANO_EL0 = -20.f;

enum class SwitchMode : int { Manual = 0, Queue = 1, Nearest = 2 };

struct Config {
    // мир: пусто — синтетическое небо; иначе 3D-сцена Godot (путь к sim/scenes/*.json), см. scenelink.hpp.
    // Цели тогда летают в сцене (числа, дальность и доли типов — ниже), облака и видимость — из сцены.
    std::string scene, sector, godot, simProject;
    uint64_t renderAffinity = 0;   // ядра для отрисовки сцены, 0 — любые
    // Панорамная маска 360°: маска неба широкой копится по азимуту и углу места; после поворота
    // широкой маска берётся из панорамы (если она покрывает новый вид), а не учится заново
    bool panoMask = true;
    // сцена
    int drones = 3, birds = 1;
    float rangeMin = 100, rangeMax = 750;    // м
    float speedMin = 6, speedMax = 22;       // м/с
    float droneSize = 0.5f, birdSize = 0.35f;   // м: размах квадрокоптера (средний) и птицы (не используется)
    // доли типов дронов: квадрокоптер (0.3–0.8 м при размере 0.5), летающее крыло (1.8–3.5 м, 25–45 м/с),
    // самолёт (1.5–4 м, 15–30 м/с)
    float quadShare = 0.5f, deltaShare = 0.25f, planeShare = 0.25f;
    float psf = 0.6f;   // пятно объектива, СКО в пикселях
    float clouds = 0.35f;                    // 0..1
    float noise = 2.0f;                      // СКО шума датчика, уровни яркости
    float visibility = 6000;                 // м: контраст падает как exp(-R/visibility)
    uint32_t seed = 1;
    // широкая камера
    int wideW = 4608, wideH = 3456;
    float wideFov = 120, wideFps = 14;
    bool wideFisheye = false;
    float wideAz = 0, wideEl = 32;   // куда смотрит ось
    // точная камера на поворотке
    int narrowW = 640, narrowH = 480;
    float narrowFov = 5, narrowFps = 60;   // поле точной: 5° — зум x4 (основной у Maxim), 20° — без зума
    float maxRate = 30, maxAccel = 150;   // град/с, град/с²
    int narrowScales = 2;                 // масштабов детектора точной
    float narrowThrMin = 8;               // порог по шуму не ниже этого: в облаках оценка шума мала, и порог 5 даёт ложные треки
    bool narrowAutoThr = true;            // порог детектора точной по шуму кадра (CFAR): при шуме 3 зум x4 ведёт 84% времени вместо 18%
    float encoderDeg = 0.01f;             // дискретность датчиков угла
    float handoffLatency = 0.06f;         // от кадра широкой до команды поворотке, с
    // поворотка как на железе (MountSim): кадр точной доходит до алгоритма с задержкой, команда до двигателя —
    // тоже, двигатель шаговый (микрошаг, пропуски), передача с люфтом, камера на упругом креплении.
    // mountIdeal — как до 31.10: команда исполняется мгновенно, положение известно точно (бит в бит)
    bool mountIdeal = false;
    int mountCtl = 1;                     // 0 — прежний закон (ошибка + скорость цели), 1 — плавный (фильтр цели, рывок)
    float maxJerk = 3000;                 // град/с³
    float mountKp = 8, mountKv = 32;      // плавный: скорость к ошибке и ускорение к ошибке скорости, 1/с
    float camLatency = 0.033f;            // от середины выдержки до результата алгоритма: считывание, передача, обработка, с
    float cmdLatency = 0.010f;            // от команды Pi до движения двигателя, с
    float cmdJitter = 0.002f;             // разброс задержки команды, с
    float exposure = 0.001f;              // выдержка точной, с: смаз цели, пока поворотка и цель движутся по-разному
    float stepDeg = 0.01125f;             // микрошаг на выходе передачи: 1.8° / 16 / 10, град
    float backlash = 0.05f;               // люфт передачи, град
    bool backlashComp = true;             // плавный: контроллер выбирает люфт при смене направления
    float resonanceHz = 12, damping = 0.15f;   // крепление камеры: собственная частота и затухание
    float slipEvery = 0;                  // пропуск шагов в среднем раз в столько секунд (0 — нет)
    int slipSteps = 8;                    // ...на столько микрошагов
    int encoder = 0;                      // 0 — датчиков угла нет (положение по выданным шагам), 1 — на оси (encoderDeg)
    int poseModel = 1;                    // без датчиков: 0 — по шагам, 1 — по профилю, 2 — профиль через модель крепления
    float modelHz = 12, modelDamping = 0.15f;   // модель крепления у контроллера (по замеру на железе)
    bool wideBias = true;                 // без датчиков: уход нуля поворотки — по широкой (обе видят цель)
    // наведение
    SwitchMode switchMode = SwitchMode::Manual;
    float dwell = 6;          // сколько вести цель, прежде чем перейти к следующей (авто), с
    bool dronesOnly = true;   // не наводиться на цели, которые широкая считает птицами
    float simSpeed = 1;       // во сколько раз быстрее реального времени; 0 — как успевает ПК
    double stopAt = 0;        // остановиться на этом модельном времени, с (замеры); 0 — не останавливаться
    // алгоритм точной (для сравнения: что даёт каждая доработка)
    bool encoderMotion = true;   // сдвиг картинки — по датчикам угла поворотки, иначе оценка по кадру
    bool feedForward = true;     // поворотка ведёт со скоростью цели, иначе только по ошибке
    bool windowFocus = true;     // в окне поиска порог ниже и берётся ближайший к прогнозу трек
    bool wideCheck = true;       // трек точной сверяется с прогнозом широкой на этот момент (0.4–0.5°), а не с её
                                 // последним сообщением с допуском 1.5°: соседний дрон в окне не берётся
    bool lockFocus = true;       // и вокруг своей цели при ведении порог ниже: слабая цель не рвётся на куски
    float minHold = 0.08f;       // трек в окне берётся, если держится столько секунд (не шум на один-два кадра)
    float giveUp = 3;            // столько секунд точная ищет и не может удержать цель — цель «не подтверждена»
    float coolDown = 10;         // столько секунд к неподтверждённой цели не возвращаться (авто-режимы)
    // обмен между камерами: точная видит цель крупнее, чаще и без лишних пикселей
    bool narrowSilhouette = true;   // точная ведёт крупную цель по силуэту (центр), а не по частям: не скачет с крыла на хвост
    float silhMinPx = 8;            // ...если силуэт не меньше стольких пикселей (мелкая цель на части не разваливается)
    bool narrowHold = true;    // точная, уверенная, что это дрон, держит цель сама, когда широкая её теряет или
                               // её трек уходит на другое; как только широкая снова видит цель — передаёт ей её трек
    bool narrowVeto = true;    // точная отсеивает цель, которую сама видит птицей (взмахи крыльев)
    bool vetoToWide = true;    // ...и сообщает широкой: её трек (и новые треки на прогнозе отсеянной) — птица
    bool wideCue = true;       // точная сообщает широкой, где цель: там у широкой порог ниже
    float confirmT = 0.5f;     // точная уверена в дроне: ведёт цель столько секунд и её классификатор — «дрон»
    float vetoT = 0.6f;        // столько секунд подряд птица у точной — цель отсеивается
    // птица по силуэту: у точной (особенно с зумом) цель — несколько пикселей и больше, взмахи меняют
    // площадь и размах силуэта с частотой 3–12 Гц, а дрон (квадрокоптер, крыло, самолёт) жёсткий
    bool narrowShape = true;
    float shapeWin = 0.6f;     // окно, с: 2–4 взмаха (лаг до половины окна — от 3.3 Гц)
    float shapeAmp = 0.04f;    // размах колебаний «массы» силуэта (контраст x площадь) к среднему, СКО: птица 0.8 м — 0.07, дрон до 0.03
    float shapeR = 0.75f;      // автокорреляция на периоде взмаха: птица 0.85–0.95, дрон по шуму до 0.6
    int shapeHits = 6;         // столько оценок подряд «машет» — птица (после подтверждения дрона — вдвое больше)
    bool nextOnBird = true;    // отсеяв птицу, точная сразу берёт следующую цель широкой (и в ручном режиме, где
                               // сама берёт только жёсткий захват широкой), а не ждёт, пока широкая подтвердит дрон
    bool vetoManual = true;    // и цель, выбранную вручную, отсеивает — но только уверенно (вдвое дольше)
    float vetoCool = 30;       // к отсеянной цели и к целям широкой на её прогнозе не возвращаться, с
    float narrowLostHold = 2;  // цель есть только у точной: столько секунд искать её по своему прогнозу
};

// Тело в мире: прямолинейное движение (у птиц — с покачиванием и взмахами)
struct Body {
    int id = 0;
    bool bird = false;
    Shape shape = Shape::Quad;
    Vec3 p0, v;          // м, м/с; камеры в начале координат
    double t0 = 0;       // когда появилось
    float size = 0.5f;   // м
    float flapHz = 0, phase = 0;
    bool wobble = true;    // покачивание птицы (в 3D-сцене траекторию задаёт сцена)
    bool hidden = false;   // в 3D-сцене: заслонено от установки домом или деревом
    Vec3 at(double t) const;
};

enum class NState : int { Idle = 0, Slew = 1, Acquire = 2, Track = 3, Lost = 4 };
const char* state_name(NState s);

// Цель, которую видит широкая камера (её трек), с последним сообщением о ней
struct WideTarget {
    int id = 0;
    TargetReport rep;
    int cls = 0;
    bool hard = false;
    int truth = 0;   // ближайшее тело по правде (для оценки), 0 — ни одного рядом
    int verdict = 0; // что сказала точная: 1 — дрон (подтвердила), 2 — птица (отсеяна)
};

struct Event {
    double t;
    std::string text;
};

struct Stats {
    double simT = 0;
    int wideFrames = 0, narrowFrames = 0;
    int assignments = 0, acquired = 0, wrongLocks = 0, losses = 0;
    int wideSwaps = 0;                        // трек широкой, указанный точной, перескочил на другое тело
    int unconfirmed = 0;                      // точная не смогла удержать цель за giveUp и отказалась от неё
    double trackTime = 0, trackCorrect = 0;   // с: точная ведёт цель, и это назначенное тело
    double idleTime = 0, falseTarget = 0;     // с: нечего вести; назначена цель широкой, которой нет (шум, облако)
    double slewSum = 0, acqSum = 0;           // с: до цели в кадре точной и до захвата, сумма по назначениям
    int slewN = 0, acqN = 0;
    std::vector<float> errLock, errAxis, errWide;   // град: захват точной, ось точной, сообщение широкой — от правды
    // обмен между камерами
    double narrowOnly = 0, narrowOnlyCorrect = 0;   // с: цель есть только у точной (широкая потеряла или ушла с неё)
    double trackDrone = 0;                    // с: точная ведёт настоящий дрон (любой, не птицу и не пустое место)
    int kept = 0;                             // точная не бросила цель, когда широкая её потеряла или ушла с неё
    int rebinds = 0;                          // широкая снова видит цель, которую держала точная, — трек передан
    int confirmed = 0;                        // точная подтвердила дрон
    int vetoBird = 0, vetoDrone = 0;          // точная отсеяла как птицу: по правде птица / дрон (ошибка)
    int vetoNone = 0;                         // ...и там по правде ничего нет (помеха)
    double trackBird = 0;                     // с: точная ведёт птицу
    int confirmedBird = 0;                    // точная назвала дроном птицу (ошибка)
    double vetoDelay = 0;                     // с: от захвата птицы точной до отсева, сумма по отсеянным птицам
    int vetoShape = 0;                        // из отсеянных — по силуэту (взмахи)
    int nextAfterBird = 0;                    // переходов на следующую цель сразу после отсева птицы
    int partJumps = 0;                        // захват точной перешёл на соседний трек той же цели (часть крупной)
    double wideMs = 0, narrowMs = 0;          // среднее время обработки кадра
    // поворотка: насколько ровно и точно она ведёт цель
    double jitterSq = 0;  int jitterN = 0;    // дрожание цели на кадре точной (вторая разность положения), пикс.
    double motionSq = 0;  int motionN = 0;    // ошибка сдвига кадра, которым трекер вычитает поворот, пикс.
    double poseSq = 0;    int poseN = 0;      // ошибка положения поворотки, которое знает алгоритм, град
    double blurSum = 0;   int blurN = 0;      // смаз цели за выдержку, пикс.
    double centerSq = 0;  int centerN = 0;    // цель от центра кадра точной при ведении, пикс.
    MountSim::Stats mount;                    // ускорение камеры, рывок профиля, пропуски шагов
    float biasAz = 0, biasEl = 0;             // поправка ухода нуля по широкой, град
    float slipAz = 0, slipEl = 0;             // на самом деле пропущено, град
    double etaSum = 0, etaErr = 0; int etaN = 0;   // прогноз времени поворота к цели и его ошибка, с
    double renderWideMs = 0, renderNarrowMs = 0;   // отрисовка кадров, сумма, мс
};

// Что показать в интерфейсе: копия последнего состояния
struct View {
    std::shared_ptr<const Result> wide;   // кадр и треки широкой
    std::vector<uint8_t> narrow;          // кадр точной
    int nw = 0, nh = 0;
    std::vector<dronolok::Track> ntracks;
    int nlock = 0;
    NState state = NState::Idle;
    int assigned = 0;                     // id трека широкой
    bool nConf = false, nOnly = false;    // точная подтвердила дрон; цель только у точной
    float nWu = -1, nWv = -1;             // где точная видит цель, на кадре широкой
    std::vector<std::pair<float, float>> vetoW;   // отсеянные точной (прогноз), на кадре широкой
    std::vector<WideTarget> targets;
    Pose axis, cmd;
    Pose believed;                        // где поворотка по своему знанию (шаги или датчик)
    float accAz = 0, accEl = 0;           // ускорение профиля
    int narrowLagFrames = 0;              // кадров точной в пути (задержка камеры)
    Pose widePose;                        // куда смотрит широкая
    float rateAz = 0, rateEl = 0;
    dronolok::SearchWindow win;
    bool winOn = false;
    struct TruthPt {
        float wu, wv, nu, nv;
        bool inNarrow, bird;
        int id;
    };
    std::vector<TruthPt> truth;
    std::vector<std::pair<float, float>> footprint;   // контур кадра точной на широкой
    std::deque<Event> events;
    Stats st;
    // панорамная маска: PANO_W x PANO_H ячеек по PANO_STEP°, угол места от PANO_EL0; 0 — не видели,
    // 1 — небо (искать), 2 — закрыто маской
    std::vector<uint8_t> pano;
    float panoKnown = 0;                  // доля тайлов текущего вида, покрытых панорамой
    bool scanning = false;                // идёт обзор 360°
    bool running = false, paused = false;
    std::string status;
};

class Emulator {
public:
    Emulator();
    ~Emulator();
    void start(const Config& c);
    void stop();
    bool running() const { return running_; }
    void setPaused(bool p) { paused_ = p; }
    bool paused() const { return paused_; }
    // переключение целей: следующая в очереди или конкретный трек широкой
    void nextTarget() { cmd_ = -1; }
    void selectTarget(int wideId) { cmd_ = wideId; }
    void setSwitchMode(SwitchMode m) { switchMode_ = (int)m; }
    void setSpeed(float s) { speed_ = s; }
    // повернуть широкую камеру (вступает с ближайшего её кадра); поворотка точной не трогается
    // обзор 360°: широкая поворачивается шагами по полкадра, на каждом шаге маска учится и
    // ложится в панораму, в конце — обратно
    void startScan() { scanReq_ = true; }
    void setWidePose(float az, float el) {
        wantAz_ = az;
        wantEl_ = el;
        ++poseReq_;
    }
    View view() const;
    // без интерфейса: прогнать seconds модельного времени как можно быстрее
    Stats runBlocking(const Config& c, double seconds, bool verbose);
    double dumpAt_ = 0, dumpLen_ = 0;   // отладка (--dump t,len): кадры точной в PGM

private:
    void loop();
    void spawn(Body& b, double t, bool first);
    void prepareWide();   // фон синтетического неба под текущий поворот широкой
    bool sceneRender(double t, int cam, const Pose& p, int W, int H, std::vector<uint8_t>& out);
    void renderWide(double t, std::vector<uint8_t>& out);
    // pa, pb — положение камеры в начале и в конце выдержки ex (смаз цели); без них — без смаза
    void renderNarrow(double t, const Pose& p, std::vector<uint8_t>& out, const Pose* pa = nullptr,
                      const Pose* pb = nullptr, double ex = 0);
    void drawBodies(const CameraModel& m, const Pose& p, double t, uint8_t* px, int w, int h, const Pose* pa = nullptr,
                    const Pose* pb = nullptr, double ex = 0);
    float background(double az, double el, double t, bool fine) const;
    void onWide(double t, const Result& r);
    void stepNarrow(double t, double dt);
    void stepNarrowReal(double t, double dt);   // поворотка как на железе (MountSim)
    void processNarrow(double t, double dt, const Pose& truth, const Pose& enc, double tNow);
    void controlReal(double t);
    void mountFeed(double t);                   // отметка точной — фильтру цели, невязка с широкой — уходу нуля
    void choose(double t);
    void assign(double t, int wideId, const char* why);
    void rebind(double t);
    bool vetoed(double t, const TargetReport& r) const;
    void event(double t, const std::string& s);
    int nearestBody(double t, double az, double el, double maxDeg) const;

    Config c_;
    std::thread th_;
    std::atomic<bool> running_{false}, stop_{false}, paused_{false};
    std::atomic<int> cmd_{0}, switchMode_{1};
    std::atomic<float> speed_{1};
    std::atomic<float> wantAz_{0}, wantEl_{0};
    std::atomic<int> poseReq_{0};
    std::atomic<bool> scanReq_{false};
    // панорамная маска
    std::vector<uint8_t> pano_;
    int panoTw_ = 0, panoTh_ = 0, panoTile_ = 0;   // тайлы маски широкой (из её результата)
    int sincePose_ = 0;          // кадров широкой с последнего поворота
    bool lastLearning_ = true;
    bool scan_ = false;
    int scanStep_ = 0;
    float scanAz0_ = 0, scanEl0_ = 0;
    void panoLearn(const Result& r);
    float panoMaskFor(std::vector<uint8_t>& m) const;
    int poseSeen_ = 0;
    // 3D-сцена: отдельный процесс Godot
    bool sceneMode_ = false;
    std::unique_ptr<scene::Link> link_;
    scene::Options linkOpt_;
    std::string linkHello_;
    int linkRestarts_ = 0;
    std::vector<scene::Body> sbodies_;
    mutable std::mutex vm_;
    View view_;

    // мир
    std::mt19937 rng_;
    std::vector<Body> bodies_;
    int nextBodyId_ = 1;
    CameraModel wideCam_, narrowCam_;
    Pose widePose_;
    std::vector<float> wideBase_;      // небо и земля без облаков, по пикселю
    std::vector<uint8_t> wideGround_;  // 1 — земля
    std::vector<float> wideAz_, wideEl_;   // направления узлов сетки облаков (каждые CLOUD_STEP пикс.)
    int cgW_ = 0, cgH_ = 0;
    std::vector<int16_t> noiseTab_;    // гауссов шум, x8
    std::vector<int16_t> wideSky_, wideNoise_;   // фон с облаками и шум кадра широкой, x8
    double wideSkyT_ = -1;
    uint32_t noiseSeed_ = 1;

    // широкая: конвейер и ожидание его результата
    Pipeline pipe_;
    std::mutex wm_;
    std::condition_variable wcv_;
    std::vector<uint8_t> wideFrame_;
    bool wideReady_ = false;   // кадр отрисован, генератор может забрать
    int64_t wideIdx_ = -1, wideDone_ = -1;
    std::shared_ptr<const Result> wideRes_;
    std::deque<std::pair<double, std::vector<WideTarget>>> inbox_;   // сообщения в пути
    std::vector<WideTarget> targets_;   // последнее дошедшее сообщение
    std::vector<int> visited_;          // уже обслуженные цели (авто: по очереди)
    std::vector<std::pair<int, double>> cool_;   // не подтверждённые точной цели широкой и до какого времени
    std::vector<std::pair<TargetReport, double>> veto_;   // отсеянные точной (где были) и до какого времени
    std::vector<int> wideBirds_;        // треки широкой, которым точная сказала «птица»

    // точная
    struct Mount {
        double az = 0, el = 10, vaz = 0, vel = 0;
    } mount_;
    Pose cmdPose_;
    double ffAz_ = 0, ffEl_ = 0;
    std::unique_ptr<dronolok::Detector> ndet_;
    std::unique_ptr<dronolok::Tracker> ntrk_;
    dronolok::GlobalMotion ngm_;
    std::vector<dronolok::Candidate> ncands_;
    std::vector<uint8_t> nframe_;
    Pose prevEnc_;
    bool havePrev_ = false;
    NState state_ = NState::Idle;
    int assigned_ = 0;       // трек широкой
    int assignedBody_ = 0;   // тело по правде (оценка)
    int nlock_ = 0;          // трек точной
    double tAssign_ = 0, tInView_ = -1, tState_ = 0, tLastSeen_ = 0, tTrackStart_ = 0;
    double tSearch_ = -1;    // с какого момента точная ищет цель и не может удержать; -1 — держит
    bool gotThis_ = false;   // назначенную цель точная уже брала (захват считаем раз на назначение)
    double tGot_ = 0;        // когда точная впервые взяла назначенную цель
    bool manualPick_ = false;   // цель выбрал оператор: точная отсеивает её, только если уверена вдвое
    bool afterBird_ = false;    // точная только что отсеяла птицу: берём следующую цель и в ручном режиме
    bool nextPick_ = false;     // назначенная взята так («следующая после птицы»): пропала — ищем следующую
    bool nConf_ = false;     // точная уверена, что ведёт дрон
    bool nOnly_ = false;     // цель есть только у точной: трек широкой assigned_ уже не она (или пропал)
    double tBird_ = -1;      // с какого момента точная видит в цели птицу
    std::deque<float> shapeE_;   // «масса» силуэта захваченной цели по кадрам (птица по силуэту)
    int shapeId_ = 0, shapeBirdN_ = 0;
    float shapeHz_ = 0;
    bool shapeRigid_ = false;    // окно набрано, силуэт не машет
    int detachId_ = 0;       // трек широкой, от которого точная только что отказалась (ушёл с цели)
    double tDetach_ = -10;
    double tWideSeen_ = 0;   // когда широкая последний раз сообщала о назначенной цели
    float lastU_ = 0, lastV_ = 0, lastSz_ = 0;   // захват точной на прошлом кадре: где и какого размера, пикс.
    TargetReport nrep_;      // последнее положение по точной
    bool haveNrep_ = false;
    // поворотка как на железе
    MountSim msim_;
    dronolok::GimbalConfig gcfg_;
    dronolok::GimbalFilter gf_;   // цель для поворотки: по отметкам точной (или широкой, пока точная не видит)
    dronolok::WideBias wbias_;
    int gfAssignN_ = -1;           // фильтр начат на этом назначении (номер по счёту)
    double gfWideT_ = -1, biasT_ = -1;
    struct NFrame {
        double t = 0, ready = 0;   // середина выдержки; когда кадр дойдёт до алгоритма
        Pose truth, bel;           // где камера на самом деле и где по знанию поворотки
        std::vector<uint8_t> px;
    };
    std::deque<NFrame> nq_;        // кадры точной в пути
    Pose prevTruth_;
    int jitBody_ = 0, jitK_ = 0;
    double jitT_ = -1, jitU_[2] = {0, 0}, jitV_[2] = {0, 0};
    double etaT_ = -1, etaPred_ = 0;   // прогноз времени поворота к назначенной цели
    dronolok::SearchWindow win_;
    bool winOn_ = false;
    Stats st_;
    std::deque<Event> events_;
};

// CLI: dronolok_app --twocam [параметры]
int run_cli(int argc, char** argv);

}  // namespace twocam
