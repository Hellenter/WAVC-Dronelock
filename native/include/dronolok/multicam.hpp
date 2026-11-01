// Несколько широких камер с разными секторами — одна установка.
//
// Каждая широкая камера находит цели сама (свой детектор и трекер) и сообщает их в своей системе:
// направление из своего положения по своей (паспортной) позе, номер трека — свой. Объединение (Fusion)
// переводит сообщения в общую систему поворотки точной камеры (азимут/угол место из начала отсчёта),
// сопоставляет треки разных камер одной цели и даёт каждой цели установки свой номер, общий для
// всех камер. Дрон перелетает из сектора одной камеры в сектор другой — номер цели тот же: точная,
// которая его ведёт, не переназначается и не поворачивается заново.
//
//  * перекрытие секторов: две камеры видят одну цель — это одна цель (сопоставление по прогнозу
//    с допуском по СКО); сообщения о ней — от одной камеры, пока цель не подойдёт к краю её кадра
//    (тогда — от соседней): у точной не скачет направление между камерами;
//  * промежуток между секторами или заслонение: цель, которую не видит ни одна камера, ждёт по
//    прогнозу coastS секунд (точная, которая её ведёт, продлевает прогноз своими наблюдениями) —
//    новый трек соседней камеры на прогнозе получает её номер;
//  * подсказка камере: где и когда цель войдёт в её поле — там у неё порог ниже (Pipeline::setCue);
//  * калибровка: поза широкой по паспорту установки неточна (полградуса — вся ширина окна поиска
//    точной с зумом x4). По парам «широкая — точная» на одной цели оценивается малый поворот
//    камеры к поворотке; пока пар мало, неуверенность калибровки добавляется к СКО сообщений,
//    и окно поиска точной шире.
//
// Положение камеры (pos, м) — относительно поворотки: направления пересчитываются с поправкой на
// параллакс по дальности rangeGuess, а неуверенность дальности (rangeMin..rangeMax) добавляется к СКО.
//
// Время — общие часы установки (на разных устройствах — после сверки часов, см. netmsg.hpp).
// Сообщения камер могут приходить не по порядку: всё считается по времени кадра.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "dronolok/geometry.hpp"

namespace dronolok {

// Широкая камера установки
struct RigCamera {
    CameraModel model;
    Pose pose;    // куда смотрит ось — по паспорту установки (калибровка уточняет)
    Vec3 pos;     // м: восток, север, верх относительно поворотки точной
};

struct FusionConfig {
    float gateDeg = 0.3f;        // допуск сопоставления: столько градусов...
    float gateSigma = 4.f;       // ...плюс столько СКО прогноза цели и сообщения
    float gateMaxDeg = 8.f;      // но не больше этого (цель долго в щели: прогноз расплылся)
    float coastS = 4.f;          // цель, которую не видит ни одна камера, ждёт по прогнозу столько секунд
    // ...а если по прогнозу она в щели между секторами (вне кадров всех камер, видеть её некому) — до
    // coastGapS секунд, пока неуверенность прогноза меньше coastGapSigma градусов (медленную цель щель в
    // 15° держит дольше 4 с)
    float coastGapS = 15.f, coastGapSigma = 6.f;
    float coastEmergeS = 3.f;    // вышла из щели в кадр камеры — столько секунд, чтобы её там нашли
    float dropS = 0.6f;          // трек камеры не приходил столько — связь с целью снимается; камера молчит столько — её цели не видны
    float mergeS = 0.25f;        // две цели с разных камер совпадают столько — это одна (младшая сливается в старшую)
    float edgeDeg = 6.f;         // цель ближе к краю кадра своей камеры — сообщения от соседней, если у той цель дальше от края
    float keepS = 3.f;           // после перехода к другой камере столько секунд держатся класс и жёсткий захват прежней
    float rangeMin = 60, rangeMax = 2000;   // м: дальности целей (для параллакса камер не в начале отсчёта)
    float rangeGuess = 300;      // м: дальность для пересчёта направления
    // калибровка широких по точной
    bool calib = true;
    float calibInitDeg = 0.5f;   // неуверенность позы камеры до калибровки, СКО
    float calibMaxDeg = 2.f;     // пары с невязкой больше — выбросы
    float calibHalfLife = 400.f; // пар: старые пары забываются (камеру могли сдвинуть)
    float calibPairMaxDt = 0.1f; // с: сообщение точной не старше этого от кадра широкой
    // пара — только если трек широкой на той же цели, что точная: невязка не больше calibGateSigma СКО
    // (поправки, сообщения и точной; не меньше calibGateMinDeg), скорости совпадают (calibVelDeg °/с + 3 СКО),
    // у широкой в этом кадре отклик, а не прогноз (СКО не больше calibSigmaPx её пикселей). Трек широкой,
    // сползший на помеху, иначе уводит калибровку на доли градуса
    float calibGateSigma = 4.f, calibGateMinDeg = 0.15f;
    float calibVelDeg = 1.f;
    float calibSigmaPx = 3.f;
    // камеру сдвинули: столько пар подряд (от двух разных целей) не проходят по невязке, а по скорости
    // совпадают — калибровка начинается заново от паспорта
    int calibRelearn = 40;
    // подсказка камере: цель, которой она пока не видит, по прогнозу уже в её кадре (вход из соседнего сектора)
    bool cue = true;
    float cueMinPx = 12, cueMaxPx = 90;   // радиус области, пикс. её кадра
    // одна камера: номера целей = номера её треков, сообщения без пересчёта (поведение без объединения)
    bool passthrough1 = true;
    // без объединения (для сравнения): у каждой камеры свои цели, номер = камера * 100000 + номер трека
    bool independent = false;
};

// Цель установки
struct FusedTarget {
    int id = 0;             // номер цели установки
    TargetReport rep;       // в системе поворотки, rep.id = id
    int cam = -1;           // камера, от которой сообщение (-1 — цель ждёт по прогнозу)
    int local = 0;          // номер трека этой камеры
    uint32_t seenBy = 0;    // камеры, которые видят цель в своём последнем сообщении (бит k — камера k)
    int cls = 0;            // 0 неизвестно, 1 дрон, 2 птица, 3 помеха
    bool hard = false;      // жёсткий захват у какой-то камеры (или недавно у прежней)
    int rank = 1000;        // место в очереди камеры (меньше — важнее)
    double born = 0, seen = 0;   // когда появилась, когда последний раз видна камерой
    double refSeen = -1;    // когда последний раз видна точной
    int handoffs = 0;       // переходов от камеры к камере
};

// Калибровка камеры: малый поворот w (рад) — направление в системе поворотки d' = d + w x d
struct CameraCalib {
    double w[3] = {0, 0, 0};
    double A[9] = {0}, b[3] = {0};   // накопленные нормальные уравнения (без априорной части)
    double pairs = 0;                // пар с забыванием
    double rmsDeg = 0;               // невязка пар после поправки
    double sigmaDeg = 0;             // неуверенность поправки, СКО направления
    int used = 0, rejected = 0;
    int streak = 0, streakId = 0;    // пар подряд не по невязке и номер цели первой из них
    bool streakMulti = false;        // среди них были разные цели
    int relearned = 0;               // раз начиналась заново
};

// Подсказка камере: где (пиксели её кадра) ждать цель
struct CueArea {
    float u = 0, v = 0, r = 0;
    int id = 0;   // номер цели установки
};

class Fusion {
public:
    Fusion() = default;
    Fusion(const std::vector<RigCamera>& cams, const FusionConfig& c) { reset(cams, c); }
    void reset(const std::vector<RigCamera>& cams, const FusionConfig& c);
    int cameras() const { return (int)cams_.size(); }
    const RigCamera& camera(int k) const { return cams_[k]; }
    const FusionConfig& config() const { return c_; }
    bool passthrough() const { return (c_.passthrough1 && cams_.size() == 1) || c_.independent; }
    // камеру повернули (на её поворотке или руками): новая поза по паспорту, калибровка остаётся
    void setPose(int k, const Pose& p) { cams_[k].pose = p; }

    // Сообщение камеры k о кадре t: её цели в её системе (make_report от её модели и позы, rep.id —
    // номер её трека), по порядку важности (первая — захваченная камерой)
    void update(int k, double t, const std::vector<TargetReport>& reps);
    // Точная ведёт цель id (сообщение в системе поворотки): прогноз цели между секторами — по ней,
    // а если confident (точная уверена, что это дрон, и держит его) — пары для калибровки широких
    void observeReference(int id, const TargetReport& r, bool confident);

    // Цели, которые видит хоть одна камера в последнем своём сообщении, по важности
    std::vector<FusedTarget> targets() const;
    // Цель по номеру, в том числе ждущая по прогнозу; nullptr — нет такой
    const FusedTarget* find(int id) const;
    // Номер цели после слияний (цель слилась с другой — номер той); 0 — цели нет
    int resolve(int id) const;
    int localOf(int id, int k) const;    // номер трека камеры k у цели id; 0 — камера её не видит
    int globalOf(int k, int local) const;
    // Где камере k ждать к моменту t цель, которой она пока не видит (вход из соседнего сектора)
    std::vector<CueArea> cues(int k, double t) const;

    // сообщение камеры k (её система) -> в систему поворотки, с калибровкой и параллаксом
    TargetReport toRef(int k, const TargetReport& r) const;
    // направление в системе поворотки -> пиксель камеры k (её паспортная поза с калибровкой)
    bool projectTo(int k, double az, double el, double& u, double& v) const;

    const CameraCalib& calib(int k) const { return cal_[k]; }
    void setCalib(int k, const CameraCalib& c) { cal_[k] = c; }
    // калибровка текстом (строка на камеру): сохранить и загрузить при следующем запуске
    std::string saveCalib() const;
    bool loadCalib(const std::string& s);

    struct Counters {
        int created = 0;    // новых целей
        int handoffs = 0;   // цель перешла к другой камере с тем же номером
        int reacquired = 0; // цель, ждавшая по прогнозу, снова найдена (той же или другой камерой)
        int merged = 0;     // две цели оказались одной
    } counters;
    // слияния целей (старый номер -> новый): забрать и учесть у себя (назначенная цель, очереди)
    std::vector<std::pair<int, int>> takeMerges() {
        std::vector<std::pair<int, int>> m;
        m.swap(merges_);
        return m;
    }

private:
    struct Obs {
        int local = 0;
        TargetReport cam;   // в системе камеры
        TargetReport ref;   // в системе поворотки
        double t = -1;      // время кадра
        int rank = 1000;
        double bornT = 0;   // когда трек камеры впервые связан с целью
        double edge = 0;    // до края кадра камеры, градусы
    };
    struct Tgt {
        FusedTarget f;
        std::vector<Obs> obs;     // по камерам
        int primary = -1;
        TargetReport last;        // последнее известное (система поворотки)
        TargetReport refRep;      // последнее от точной
        bool haveRef = false;
        double lastHardT = -1e9, lastClsT = -1e9;
        int lastCls = 0;
        int lastCam = -1;         // последняя камера, от которой были сообщения
        bool refConfident = false;
        double gapT = -1e9;       // когда последний раз была по прогнозу в щели между секторами
        double nearT = -1;        // с какого момента совпадает с другой целью (слияние); не near — макрос windows.h
        int nearId = 0;
    };
    struct Map {
        int cam, local, id;
        double lastT;
    };
    Tgt* get(int id);
    const Tgt* get(int id) const;
    bool listed(const Tgt& g, int k) const;   // камера k видит цель в своём последнем сообщении
    bool listedAny(const Tgt& g) const;
    const TargetReport& best(const Tgt& g, double t) const;   // по чему прогнозировать цель
    void choosePrimary(Tgt& g);
    void mergeInto(Tgt& keep, Tgt& gone);
    void calibPair(int k, int id, const TargetReport& wideCam, const TargetReport& ref);
    double calibSigma(int k, const Vec3& d) const;
    Vec3 refPoint(int k, const Vec3& dcam, double R) const;   // точка в системе поворотки

    FusionConfig c_;
    std::vector<RigCamera> cams_;
    std::vector<CameraCalib> cal_;
    std::vector<double> camT_;     // время последнего сообщения камеры
    double now_ = 0;
    std::vector<Tgt> tg_;
    std::vector<Map> map_;
    std::vector<std::pair<int, int>> alias_;   // слившиеся: старый -> новый
    std::vector<std::pair<int, int>> merges_;
    std::vector<std::vector<FusedTarget>> pt_;   // без объединения: последнее сообщение каждой камеры как есть
    int nextId_ = 1;
};

}  // namespace dronolok
