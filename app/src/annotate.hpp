// Разметка целей на своих видео прямо в приложении. Объект (дрон или птица) задаётся
// опорными точками на отдельных кадрах, между ними положение считается по прямой —
// достаточно ставить точку раз в 10-30 кадров и на поворотах. Сохраняется в
// <видео>.gt.json того же вида, что пишет detector/synth_sky.py (по кадрам), плюс
// опорные точки в поле "objects", чтобы разметку можно было дальше править.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "gt_eval.hpp"

struct AnnObject {
    int id = 0;
    bool bird = false;
    std::map<int64_t, std::pair<float, float>> keys;   // кадр -> x, y в пикселях видео
};

struct Annotation {
    int w = 0, h = 0;          // размер кадра видео
    double fps = 25;
    std::vector<AnnObject> objs;
    bool editable = true;      // false — разметка без опорных точек (синтетика), править нельзя
    // В ролике нет ни дронов, ни птиц: любая цель и любой захват — ложные. Сохраняется как
    // разметка из пустых кадров, поэтому её понимают и качество в приложении, и мультитест
    bool noDrones = false;

    // Загрузить из <видео>.gt.json. Нет файла — пустая разметка; есть, но без "objects" — не правится
    void load(const std::string& path, int vw, int vh, double vfps);
    bool save(const std::string& path, int64_t frames) const;
    // Положение объекта на кадре: опорная точка или по прямой между соседними. false — объекта нет
    static bool pos(const AnnObject& o, int64_t f, float& x, float& y);
    void frameObjects(int64_t f, std::vector<dronolok::GtObj>& out) const;
    dronolok::GroundTruth toGroundTruth(int64_t frames) const;
    AnnObject* find(int id);
    int add(bool bird);   // новый объект, возвращает id
};
