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
    float droneSize = 0.5f, birdSize = 0.35f;   // м
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
    float encoderDeg = 0.01f;             // дискретность датчиков угла
    float handoffLatency = 0.06f;         // от кадра широкой до команды поворотке, с
    // наведение
    SwitchMode switchMode = SwitchMode::Queue;
    float dwell = 6;          // сколько вести цель, прежде чем перейти к следующей (авто), с
    bool dronesOnly = true;   // не наводиться на цели, которые широкая считает птицами
    float simSpeed = 1;       // во сколько раз быстрее реального времени; 0 — как успевает ПК
    // алгоритм точной (для сравнения: что даёт каждая доработка)
    bool encoderMotion = true;   // сдвиг картинки — по датчикам угла поворотки, иначе оценка по кадру
    bool feedForward = true;     // поворотка ведёт со скоростью цели, иначе только по ошибке
    bool windowFocus = true;     // в окне поиска порог ниже и берётся ближайший к прогнозу трек
};

// Тело в мире: прямолинейное движение (у птиц — с покачиванием и взмахами)
struct Body {
    int id = 0;
    bool bird = false;
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
};

struct Event {
    double t;
    std::string text;
};

struct Stats {
    double simT = 0;
    int wideFrames = 0, narrowFrames = 0;
    int assignments = 0, acquired = 0, wrongLocks = 0, losses = 0;
    double trackTime = 0, trackCorrect = 0;   // с: точная ведёт цель, и это назначенное тело
    double idleTime = 0, falseTarget = 0;     // с: нечего вести; назначена цель широкой, которой нет (шум, облако)
    double slewSum = 0, acqSum = 0;           // с: до цели в кадре точной и до захвата, сумма по назначениям
    int slewN = 0, acqN = 0;
    std::vector<float> errLock, errAxis, errWide;   // град: захват точной, ось точной, сообщение широкой — от правды
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
