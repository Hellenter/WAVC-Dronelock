// 3D-сцена (Godot, sim/godot) как источник кадров эмулятора двух камер. Godot работает отдельным
// процессом на своих ядрах, связь — TCP на 127.0.0.1: эмулятор просит кадр камеры на момент t при
// повороте (az, el), сцена отрисовывает его и отвечает серым кадром и положением целей.
// Время модельное: сцена рисует ровно тот момент, о котором спросили, сколько бы ни шла отрисовка,
// поэтому скорость отрисовки и скорость алгоритма друг на друга не влияют.
//
// Протокол (sim/godot/sim.gd, --live <порт>): от нас строка JSON — сначала hello с камерами и
// целями (вторая и следующие широкие — "wides": [[аз, у.м., x, y, z], ...], камеры 2, 3...),
// затем {"t", "cam", "az", "el"[, "roll"]} на каждый кадр, {"quit": 1} в конце. От сцены — строка
// "FRAME w h n k", k строк целей "id bird shape size x y z vx vy vz hidden" (от камер: восток, север, верх; м и м/с)
// и n байт кадра (серый, w x h).
#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace scene {

struct Body {
    int id = 0;
    bool bird = false;
    int shape = 1;      // 0 птица, 1 квадрокоптер, 2 летающее крыло, 3 самолёт (как twocam::Shape)
    float size = 0.5f;  // м
    double x = 0, y = 0, z = 0, vx = 0, vy = 0, vz = 0;
    bool hidden = false;   // заслонена от установки (дом, дерево)
};

struct Options {
    std::string godot;     // путь к Godot 4.3
    std::string project;   // папка sim/godot
    std::string scene;     // сцена .json
    std::string sector;    // имя сектора или пусто
    uint64_t affinity = 0; // ядра для отрисовки, 0 — любые
};

class Link {
public:
    ~Link() { stop(); }
    // запустить Godot, дождаться связи и передать hello (JSON без перевода строки)
    bool start(const Options& o, const std::string& hello, std::string& err);
    // кадр камеры cam (0 широкая, 1 точная, 2... — вторая и следующие широкие) на момент t; roll — поворот
    // кадра вокруг оси (ошибка установки широкой)
    bool render(double t, int cam, double az, double el, std::vector<uint8_t>& px, int& w, int& h,
                std::vector<Body>& bodies, std::string& err, double roll = 0);
    void stop();
    bool running() const { return proc_ != nullptr; }
    // флаг остановки эмулятора: ожидание Godot и кадра прерывается сразу, а не через 2 минуты
    void setCancel(const std::atomic<bool>* c) { cancel_ = c; }

private:
    bool cancelled() const { return cancel_ && cancel_->load(); }
    bool sendAll(const char* p, size_t n);
    bool waitReadable();
    bool recvLine(std::string& line);
    bool recvAll(uint8_t* p, size_t n);

    const std::atomic<bool>* cancel_ = nullptr;
    void* proc_ = nullptr;
    intptr_t lsock_ = -1, sock_ = -1;
    std::string buf_;   // принятое сверх прочитанной строки
    int64_t frames_ = 0;
};

// Где лежит симулятор: папка sim рядом с программой (сборка) или в репозитории (../sim от build-*)
// и Godot в ней (Godot.exe, godot.x86_64) или в PATH. base — папка программы.
bool find_sim(const std::string& base, std::string& simDir, std::string& godot);
std::vector<std::string> list_scenes(const std::string& simDir);   // имена файлов sim/scenes/*.json
// сектора сцены: имена и середины по азимуту
struct Sector {
    std::string name;
    float mid = 0, from = 0, to = 0;
};
std::vector<Sector> scene_sectors(const std::string& path);

}  // namespace scene
