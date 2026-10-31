// Эмулятор двух камер: неподвижная широкая (16 Мп, 120°, ~14 к/с) находит цели нашим конвейером
// и указывает их точной камере 640x480 на поворотке; точная наводится с ограничением скорости
// поворота, находит цель в окне поиска и дальше ведёт её сама. Мир, камеры и поворотка —
// синтетические, алгоритм — настоящий: Pipeline для широкой, Detector + Tracker для точной.
//
// Широких может быть несколько (разные сектора): у каждой свой конвейер, их цели объединяет
// dronolok::Fusion (multicam.hpp) в цели установки с общими номерами — точная берёт цели от всех
// и ведёт дрон через стык секторов без переназначения. Развёртывание: одно устройство на все
// широкие (кадры в такт, объединение там же) или устройство на камеру — сообщения по сети
// (кодек netmsg.hpp, по желанию настоящий UDP), задержки, камеры не в такт, сверка часов.
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
#include "dronolok/multicam.hpp"
#include "dronolok/netmsg.hpp"
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

// Широкая камера установки: где стоит и как установлена. Размер кадра, поле и частота — общие (wideW...)
struct WideCam {
    float az = 120, el = 32;            // куда смотрит ось, по паспорту (у первой камеры — wideAz, wideEl)
    float x = 0, y = 0, z = 0;          // м: положение относительно поворотки точной (восток, север, верх)
    float errAz = 0, errEl = 0, errRoll = 0;   // ошибка установки, градусы: на самом деле ось смотрит так
    float phase = 0;                    // с: кадры позже первой камеры на столько (устройства не в такт)
    float clock = 0;                    // с: часы её устройства спешат на столько (до сверки часов)
};

// Где работают широкие: все на одном устройстве (кадры в такт, объединение целей там же) или каждая на
// своём — цели идут устройству поворотки по сети, объединяет оно
enum class Deploy : int { OneDevice = 0, PerCamera = 1 };

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
    WideCam wide0;                   // первая: положение, ошибка установки, часы (az, el — выше)
    std::vector<WideCam> moreWide;   // вторая и следующие широкие
    int wideCount() const { return 1 + (int)moreWide.size(); }
    // несколько широких
    Deploy deploy = Deploy::OneDevice;
    float netLatency = 0.004f;       // с: сеть между устройствами в одну сторону (поворотка -> камера; камера -> поворотка — в handoffLatency)
    float netJitter = 0.003f;        // с: разброс задержки (очереди), в среднем
    bool clockSync = true;           // устройство поворотки сверяет часы с камерами (иначе время кадров — по часам камер)
    bool netUdp = false;             // сообщения камер — по настоящему UDP на 127.0.0.1 (проверка обмена)
    bool fusion = true;              // объединять цели камер (выключено — у каждой камеры свои цели и номера)
    bool fusionCalib = true;         // калибровка широких по точной
    float calibInit = 0.5f;          // неуверенность установки широких до калибровки, СКО, градусы
    float checkBias = 3;             // «точная ведёт не то»: неуверенность калибровки — постоянный сдвиг, не шум;
                                     // допуск на неё — столько её СКО (а не 6, как на шум сообщения)
    bool crossCue = true;            // камере — где ждать цель из соседнего сектора
    float seamShare = 0;             // доля целей, которые появляются у стыка секторов и летят через него
    // точная камера на поворотке
    int narrowW = 640, narrowH = 480;
    float narrowFov = 5, narrowFps = 60;   // поле точной: 5° — зум x4 (основной у Maxim), 20° — без зума
    float maxRate = 30, maxAccel = 150;   // град/с, град/с²
    int narrowScales = 2;                 // масштабов детектора точной
    float narrowThrMin = 8;               // порог по шуму не ниже этого: в облаках оценка шума мала, и порог 5 даёт ложные треки
    bool narrowAutoThr = true;            // порог детектора точной по шуму кадра (CFAR): при шуме 3 зум x4 ведёт 84% времени вместо 18%
    float encoderDeg = 0.01f;             // дискретность датчиков угла
    float handoffLatency = 0.06f;         // от кадра широкой до команды поворотке, с
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
    int id = 0;          // номер цели установки (одна широкая — номер её трека)
    TargetReport rep;    // в системе поворотки
    int cls = 0;
    bool hard = false;
    int cam = 0, local = 0;   // от какой широкой сообщение и номер её трека
    uint32_t seenBy = 1;      // какие широкие видят (бит на камеру)
    float bias = 0;           // часть rep.sigma — неуверенность калибровки камеры (сдвиг, одинаковый у всех целей), градусы
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
    double renderWideMs = 0, renderNarrowMs = 0;   // отрисовка кадров, сумма, мс
    // несколько широких: цель перешла к другой камере — с тем же номером / с новым
    int handoffOk = 0, handoffBroken = 0;
    int renumbered = 0;                       // та же камера потеряла цель и нашла под новым номером
    double dupTime = 0;                       // с: у одной цели больше одного номера (камеры не объединились)
    int seamTrack = 0, seamKept = 0;          // точная вела цель, когда та перешла к другой камере; из них не потеряла за 1 с
    int fusionCreated = 0, fusionHandoffs = 0, fusionReacquired = 0, fusionMerged = 0;
    std::vector<float> calibDeg, calibSigma;  // по камерам: остаточная ошибка позы (правда), неуверенность калибровки
    std::vector<int> calibPairs;
    std::vector<int> camFrames;               // по камерам: кадров обработано
    double clockErrMs = 0;                    // наибольшая ошибка сверки часов, мс
    int netMsgs = 0, netBytes = 0;            // сообщений и байт по сети (устройство на камеру)
};

// Что показать в интерфейсе: копия последнего состояния
struct View {
    // по широким камерам: кадр и треки, номера целей установки у треков, проекции на её кадр
    struct WideView {
        std::shared_ptr<const Result> res;
        Pose pose;                                      // по паспорту
        std::vector<std::pair<int, int>> gid;           // трек камеры -> цель установки
        std::vector<std::pair<float, float>> footprint; // контур кадра точной
        std::vector<std::pair<float, float>> vetoW;     // отсеянные точной
        struct Pt {
            float u, v;
            int id;
            bool bird;
        };
        std::vector<Pt> truth;
        float nWu = -1, nWv = -1;                       // где цель видит точная
        float calibSigma = 0, calibErr = 0;             // калибровка: неуверенность и правда, градусы
        int calibPairs = 0;
        bool primary = false;                           // от неё сейчас сообщения о назначенной цели
    };
    std::vector<WideView> wides;
    std::shared_ptr<const Result> wide;   // кадр и треки первой широкой (= wides[0].res)
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
    void setWidePose(float az, float el) { setWidePose(0, az, el); }
    void setWidePose(int cam, float az, float el) {
        std::lock_guard<std::mutex> lk(poseM_);
        poseWant_.push_back({cam, az, el});
        ++poseReq_;
    }
    View view() const;
    // без интерфейса: прогнать seconds модельного времени как можно быстрее
    Stats runBlocking(const Config& c, double seconds, bool verbose);
    double dumpAt_ = 0, dumpLen_ = 0;   // отладка (--dump t,len): кадры точной в PGM

private:
    struct WideUnit;   // широкая камера: конвейер, кадры, маска (twocam.cpp)
    struct Msg {       // сообщение в пути между устройствами
        double at = 0;   // когда дойдёт (модельное время)
        int kind = 0;    // 0 цели установки -> поворотке, 1 цели камеры -> поворотке (по сети), 2 подсказка камере,
                         // 3 «птица» камере, 4 точная -> объединению (одно устройство), 5 сверка часов
        int cam = 0;
        std::string data;
        std::vector<WideTarget> snap;
        TargetReport rep;
        int id = 0;
        bool flag = false;
        std::vector<std::pair<int, int>> merges;   // слияния целей (старый номер -> новый)
    };
    void loop();
    void spawn(Body& b, double t, bool first);
    void prepareWide(WideUnit& w);   // фон синтетического неба под текущий поворот широкой
    bool sceneRender(double t, int cam, const Pose& p, int W, int H, std::vector<uint8_t>& out);
    void renderWide(WideUnit& w, double t, std::vector<uint8_t>& out);
    void renderNarrow(double t, const Pose& p, std::vector<uint8_t>& out);
    void drawBodies(const CameraModel& m, const Pose& p, const Vec3& pos, double t, uint8_t* px, int w, int h);
    float background(double az, double el, double t, bool fine) const;
    void onWide(WideUnit& w, double t, const Result& r);
    void post(Msg m);                        // в очередь сообщений по времени доставки
    void deliver(double t);                  // дошедшие сообщения
    void fusedSnapshot(double t, std::vector<WideTarget>& out);   // цели установки сейчас (для поворотки)
    void setTargets(double t, std::vector<WideTarget> v);         // поворотка получила цели установки
    void markWideBird(double t, int id);     // точная: цель id — птица; её трекам у широких сказать
    void wideStats(double t);                // переходы целей между камерами (по правде)
    double calibTruthDeg(const WideUnit& w) const;   // ошибка направления по камере с калибровкой (правда), градусы
    int nearestBodyFrom(double t, const Vec3& pos, const Vec3& d, double maxDeg) const;
    void stepNarrow(double t, double dt);
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
    void panoLearn(const WideUnit& w, const Result& r);
    float panoMaskFor(const WideUnit& w, std::vector<uint8_t>& m) const;
    int poseSeen_ = 0;
    std::mutex poseM_;
    struct PoseCmd {
        int cam;
        float az, el;
    };
    std::vector<PoseCmd> poseWant_;
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
    CameraModel narrowCam_;
    std::vector<int16_t> noiseTab_;    // гауссов шум, x8
    uint32_t noiseSeed_ = 1;

    // широкие: у каждой свой конвейер; цели объединяет fusion_
    std::vector<std::unique_ptr<WideUnit>> wu_;
    dronolok::Fusion fusion_;
    std::deque<Msg> net_;               // сообщения в пути (по времени доставки)
    std::mt19937 netRng_;
    dronolok::net::UdpSocket udpTx_, udpRx_;
    std::vector<WideTarget> targets_;   // последнее дошедшее сообщение
    int primaryCam_ = -1;               // от какой камеры сообщения о назначенной цели
    int uiCalibN_ = 0;
    struct BodySeen {
        int body, gid, cam;
        double t;
    };
    std::vector<BodySeen> bodySeen_;    // оценка переходов между камерами: последний номер и камера у тела
    std::vector<std::pair<double, int>> seamCheck_;   // точная вела тело на стыке: проверить через 1 с
    double lastSnapT_ = -1;
    // смещение точной от прогноза широкой в начале захвата (калибровка не сошлась): по нему — сползла ли точная
    double offRefT_ = -1, offX_ = 0, offY_ = 0;
    int offN_ = 0;
    std::vector<int> visited_;          // уже обслуженные цели (авто: по очереди)
    std::vector<std::pair<int, double>> cool_;   // не подтверждённые точной цели широкой и до какого времени
    std::vector<std::pair<TargetReport, double>> veto_;   // отсеянные точной (где были) и до какого времени


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
    dronolok::SearchWindow win_;
    bool winOn_ = false;
    Stats st_;
    std::deque<Event> events_;
};

// CLI: dronolok_app --twocam [параметры]
// Широкие камеры установки: count штук, оси второй и следующих — через step° по азимуту от первой (wideAz),
// угол места — wideEl. Ошибки установки — СКО installErr° по номеру сцены (у первой тоже); устройство на
// камеру: кадры k-й позже первой на phase*k с (phase < 0 — вразнобой), часы спешат на ±clockErr с.
// Положения камер (x, y, z) не трогаются.
void layout_wides(Config& c, int count, float step, float installErr, float clockErr = 0.05f, float phase = -1);

int run_cli(int argc, char** argv);

}  // namespace twocam
