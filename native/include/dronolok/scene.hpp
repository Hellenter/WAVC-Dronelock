// Дронолок: оценка сцены и погоды и подстройка параметров под неё.
//
// Раз в ~0.2 с по разреженной выборке кадра (~60 тыс. точек: на 16 Мп каждый 16-й пиксель
// каждой 16-й строки, блоки 4x4 точки) считаются: яркость неба и земли, шум матрицы, облачность, качание кромки
// леса на ветру, осадки, дымка, пересвет, тип того, что ниже горизонта (лес, город, поле),
// и резкие перемены (облако закрыло солнце, камеру сдвинули). Цена ~0.3 мс на 16 Мп за
// анализ, то есть меньше 0.1 мс на кадр.
//
// По оценке scene_profile() подстраивает настройки детектора и трекера относительно
// базовых (тех, что выставлены пользователем): порог по шуму, полярность, строгость
// подтверждения, сколько должна пролететь цель у кромки, память неподвижных помех.
// Подстройка плавная (по сглаженной оценке), поэтому не дёргается от кадра к кадру.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "dronolok/detector.hpp"

namespace dronolok {

enum class Light : uint8_t { Day = 0, Dusk = 1, Night = 2 };
enum class Terrain : uint8_t { Unknown = 0, Open = 1, Forest = 2, Urban = 3 };
const char* light_name(Light l);
const char* terrain_name(Terrain t);

struct SceneConfig {
    float interval_s = 0.2f;    // анализ раз в столько секунд
    // Шаг выборки, пикс. (0 — по размеру кадра: ~60 тыс. выборок, 4 Мп — 8, 16 Мп — 16);
    // блок — 4x4 выборки
    int step = 0;
    float tau_s = 3.f;          // сглаживание погоды (ветер, облака, осадки), с
    float light_jump = 0.2f;    // яркость неба изменилась на столько (доля) за ~1 с — смена освещения
};

struct SceneState {
    bool ready = false;         // хоть один анализ (до этого — профиль = база)
    int analyses = 0;
    // освещение
    float skyLum = 0;           // медиана яркости неба
    float groundLum = 0;        // медиана яркости земли
    float glare = 0;            // доля пересвеченного неба (>= 250)
    float noise = 0;            // СКО шума матрицы на небе, уровни яркости
    Light light = Light::Day;
    // погода (сглаженные)
    float clouds = 0;           // 0..1: доля неба с облачной неоднородностью
    float wind = 0;             // качание кромки: сдвиг крон за 0.2 с, пикс. (0 — штиль)
    float precip = 0;           // осадки: точечных вспышек на 1000 выборок неба
    float haze = 0;             // 0..1: дымка, кромка почти не отличается от неба
    // окружение
    float skyFrac = 0;          // доля кадра под небом
    float hv = 0;               // доля градиентов земли строго по горизонтали/вертикали (город)
    float skylineFlat = 0;      // доля соседних столбцов с одинаковой высотой кромки
    float skylineSd = 0;        // разброс высоты кромки, пикс. кадра
    Terrain terrain = Terrain::Unknown;
    // перемены (только на кадре, где замечены)
    bool lightJump = false;     // освещение резко изменилось: фон ячеек выучить заново
    bool moved = false;         // сцена сдвинулась (камеру повернули): маску неба — заново
    int events = 0;             // всего перемен
    std::string lastEvent;      // описание последней перемены
    int lastEventFrame = -1;
    std::vector<int16_t> skyline;   // кромка по столбцам блоков (номер ряда блока), -1 — нет земли
};

class SceneAnalyzer {
public:
    explicit SceneAnalyzer(const SceneConfig& cfg = SceneConfig()) : cfg_(cfg) {}
    // Каждый кадр (fps — частота камеры, для интервала анализа). true — был анализ.
    // movingCamera: камера движется — погоду у кромки и сдвиг сцены не оцениваем.
    bool update(const uint8_t* gray, int w, int h, int stride, float fps, bool movingCamera = false);
    const SceneState& state() const { return st_; }
    void reset();
    int blockPx() const { return 4 * step_; }   // блок кадра, пикс. (для skyline)

private:
    SceneConfig cfg_;
    SceneState st_;
    int w_ = 0, h_ = 0, bw_ = 0, bh_ = 0, step_ = 4, frame_ = 0, last_ = -1000000;
    std::vector<uint8_t> prev_, cur_;          // выборка кадра
    std::vector<float> mean_, act_, rough_;    // по блокам
    std::vector<uint16_t> out_;                // точечные выбросы во времени по блокам
    std::vector<uint8_t> sky_;                 // 1 — блок неба
    std::vector<float> ref_;                   // земля на опорном кадре (сдвиг сцены)
    std::vector<int16_t> refLine_;
    std::vector<int16_t> lastLine_;           // кромка на прошлом анализе (дрожь кромки)
    float lumFast_ = 0, lumSlow_ = 0;
    int movedRun_ = 0;
    bool havePrev_ = false;
    std::vector<float> tmp_;
};

// Настройки под сцену: base — то, что выставил пользователь (трекер — уже под частоту
// кадров). Пока оценки нет, возвращается база. why — коротко, что и почему поменялось.
void scene_profile(const SceneState& s, const DetectorConfig& base, const TrackerConfig& baseT, DetectorConfig& d,
                   TrackerConfig& t, std::string* why = nullptr);

// Профиль сменил то, от чего зависит маска неба (порог по шуму, полярность): маску лучше
// выучить заново — выученная со старым порогом в сумерках закрывает всё небо
bool scene_mask_stale(const DetectorConfig& was, const DetectorConfig& now);

}  // namespace dronolok
