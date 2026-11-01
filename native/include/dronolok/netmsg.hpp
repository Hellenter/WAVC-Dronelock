// Обмен между устройствами установки по сети: широкие камеры на своих устройствах (Pi на камеру)
// шлют цели устройству поворотки, где их объединяет Fusion (multicam.hpp); оно в ответ шлёт камерам
// подсказки (где ждать цель) и «это птица» (точная рассмотрела трек).
//
// Транспорт — UDP, одна датаграмма — одна строка текста (читается и из Python: split по ';' и ' ').
// Потеря датаграммы не страшна: следующее сообщение камеры через 1/14 с несёт всё заново.
//
//   DLW1 cam seq t sent n;id az el vaz vel sigma sigma_v cls hard size conf;...   цели камеры cam в её системе
//                                                                                  (t — время кадра по часам камеры)
//   DLC1 cam t n;az el r;...       подсказка камере cam: где (её система, градусы) ждать цель, r — радиус, градусы
//   DLB1 cam local                 трек local камеры cam — птица (Pipeline::markBird)
//   DLP1 seq t1                    сверка часов: запрос (t1 — часы того, кто спрашивает)
//   DLQ1 seq t1 t2 t3              ответ: t2 — когда пришёл запрос, t3 — когда ушёл ответ (часы отвечающего)
//
// Часы: время кадра в сообщении — по часам камеры; устройство поворотки сверяет часы с каждой камерой
// (ClockSync, как NTP: смещение по самому быстрому обмену из последних) и переводит время в свои часы.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "dronolok/geometry.hpp"

namespace dronolok {
namespace net {

constexpr int MAX_DATAGRAM = 1400;   // без фрагментации в обычной сети

struct WideMsg {
    int cam = 0;
    uint32_t seq = 0;
    double t = 0;      // время кадра, часы камеры
    double sent = 0;   // когда отправлено, часы камеры
    std::vector<TargetReport> reps;
};
struct CueMsg {
    int cam = 0;
    double t = 0;
    struct Area {
        float az, el, r;
    };
    std::vector<Area> areas;
};
struct BirdMsg {
    int cam = 0, local = 0;
};
struct PingMsg {
    uint32_t seq = 0;
    double t1 = 0;
};
struct PongMsg {
    uint32_t seq = 0;
    double t1 = 0, t2 = 0, t3 = 0;
};

// Целей в сообщении не больше стольких, чтобы строка уложилась в датаграмму (первые — важнее)
std::string encode(const WideMsg& m, int maxTargets = 18);
std::string encode(const CueMsg& m);
std::string encode(const BirdMsg& m);
std::string encode(const PingMsg& m);
std::string encode(const PongMsg& m);
// тип сообщения: 'W', 'C', 'B', 'P', 'Q' или 0 — не наше
char kind(const std::string& s);
bool decode(const std::string& s, WideMsg& m);
bool decode(const std::string& s, CueMsg& m);
bool decode(const std::string& s, BirdMsg& m);
bool decode(const std::string& s, PingMsg& m);
bool decode(const std::string& s, PongMsg& m);

// Сверка часов с одним устройством по обменам DLP1/DLQ1. offset(): часы устройства минус свои;
// своё время события = время по часам устройства - offset().
class ClockSync {
public:
    // t1, t4 — свои часы (запрос ушёл, ответ пришёл), t2, t3 — часы устройства
    void add(double t1, double t2, double t3, double t4);
    bool valid() const { return n_ > 0; }
    double offset() const { return off_; }
    double delay() const { return rtt_; }   // круговая задержка лучшего из последних обменов
    int samples() const { return n_; }

private:
    struct S {
        double off, rtt;
    };
    std::vector<S> last_;
    double off_ = 0, rtt_ = 0;
    int n_ = 0;
};

// UDP-сокет (Linux, Windows)
class UdpSocket {
public:
    UdpSocket();
    ~UdpSocket();
    UdpSocket(const UdpSocket&) = delete;
    UdpSocket& operator=(const UdpSocket&) = delete;
    // port 0 — любой свободный (узнать: port())
    bool bind(const std::string& host, int port, std::string& err);
    bool sendTo(const std::string& host, int port, const std::string& data);
    // ждать датаграмму до timeoutMs (0 — не ждать); false — ничего не пришло
    bool recv(std::string& out, int timeoutMs, std::string* fromHost = nullptr, int* fromPort = nullptr);
    int port() const { return port_; }
    void close();
    bool open() const { return fd_ >= 0; }

private:
    intptr_t fd_ = -1;
    int port_ = 0;
};

}  // namespace net
}  // namespace dronolok
