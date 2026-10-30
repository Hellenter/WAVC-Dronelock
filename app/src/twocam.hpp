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
#include "pipeline.hpp"
#include "silhouette.hpp"

namespace twocam {

using dronolok::CameraModel;
using dronolok::Pose;
using dronolok::TargetReport;
using dronolok::Vec3;

enum class SwitchMode : int { Manual = 0, Queue = 1, Nearest = 2 };

struct Config {
    // сцена
    int drones = 3, birds = 1;
    float rangeMin = 250, rangeMax = 1200;   // м
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
    float narrowFov = 20, narrowFps = 60;
    float maxRate = 30, maxAccel = 150;   // град/с, град/с²
    int narrowScales = 2;                 // масштабов детектора точной
    float narrowThrMin = 8;               // порог по шуму не ниже этого: в облаках оценка шума мала, и порог 5 даёт ложные треки
    bool narrowAutoThr = true;            // порог детектора точной по шуму кадра (CFAR): при шуме 3 зум x4 ведёт 84% времени вместо 18%
    float encoderDeg = 0.01f;             // дискретность датчиков угла
    float handoffLatency = 0.06f;         // от кадра широкой до команды поворотке, с
    // наведение
    SwitchMode switchMode = SwitchMode::Queue;
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
    bool wideCue = true;       // точная сообщает широкой, где цель: там у широкой порог ниже
    float confirmT = 0.5f;     // точная уверена в дроне: ведёт цель столько секунд и её классификатор — «дрон»
    float vetoT = 0.6f;        // столько секунд подряд птица у точной — цель отсеивается
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
    int partJumps = 0;                        // захват точной перешёл на соседний трек той же цели (часть крупной)
    double wideMs = 0, narrowMs = 0;          // среднее время обработки кадра
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
    View view() const;
    // без интерфейса: прогнать seconds модельного времени как можно быстрее
    Stats runBlocking(const Config& c, double seconds, bool verbose);
    double dumpAt_ = 0, dumpLen_ = 0;   // отладка (--dump t,len): кадры точной в PGM

private:
    void loop();
    void spawn(Body& b, double t, bool first);
    void renderWide(double t, std::vector<uint8_t>& out);
    void renderNarrow(double t, const Pose& p, std::vector<uint8_t>& out);
    void drawBodies(const CameraModel& m, const Pose& p, double t, uint8_t* px, int w, int h);
    float background(double az, double el, double t, bool fine) const;
    void onWide(double t, const Result& r);
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
    bool manualPick_ = false;   // цель выбрал оператор: точная её не отсеивает
    bool nConf_ = false;     // точная уверена, что ведёт дрон
    bool nOnly_ = false;     // цель есть только у точной: трек широкой assigned_ уже не она (или пропал)
    double tBird_ = -1;      // с какого момента точная видит в цели птицу
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
int run_cli(int argc, char** argv);

}  // namespace twocam
