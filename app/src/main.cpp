// Дронолок: тестовое приложение для ПК. Загружает видео, прогоняет его через нативный
// детектор и показывает, что алгоритм считает дроном, птицей, помехой и шумом, вместе
// с метриками скорости. Ресурсы можно урезать (потоки, ядра, SIMD, потолок CPU,
// разрешение, искусственное замедление), чтобы примерить алгоритм к слабому железу.
//
//   dronolok_app [видео]
//   dronolok_app --bench видео [--frames N] [--threads N] [--isa avx2] [--scales 0..4] [--motion 0|1]
//                [--focus] [--camera] [--slowdown K] [--governor] [--profile 0..3] [--json out.json]
//   dronolok_app --stress видео [--seconds 30] [--seed N]
//   dronolok_app --multitest папка [--vary ключ[=знач,...]]... [--frames N] [--out папка]  (--help, --list)
#include <SDL.h>
#if defined(_WIN32)
#include <SDL_syswm.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "annotate.hpp"
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"
#include "multitest.hpp"
#include "twocam.hpp"
#include "dronolok/fileutil.hpp"
#include "dronolok/tuning.hpp"
#include "pipeline.hpp"
#include "platform.hpp"

using namespace dronolok;

namespace {

const ImU32 COL_DRONE = IM_COL32(255, 70, 70, 255);
const ImU32 COL_BIRD = IM_COL32(70, 170, 255, 255);
const ImU32 COL_UNKNOWN = IM_COL32(255, 210, 0, 255);
const ImU32 COL_CLUTTER = IM_COL32(170, 170, 170, 220);
const ImU32 COL_NOISE = IM_COL32(210, 210, 210, 150);
const ImU32 COL_GT = IM_COL32(0, 255, 120, 220);
const ImU32 COL_LOCK = IM_COL32(255, 60, 230, 255);
const ImU32 COL_FOCUS = IM_COL32(90, 170, 255, 110);
const ImU32 COL_STATIC = IM_COL32(190, 190, 190, 230);   // как помеха, но крестиком

ImU32 class_color(const Track& t) {
    if (t.clutter) return COL_CLUTTER;
    switch (t.cls) {
        case TargetClass::Drone: return COL_DRONE;
        case TargetClass::Bird: return COL_BIRD;
        case TargetClass::Clutter: return COL_CLUTTER;
        default: return COL_UNKNOWN;
    }
}

const char* class_ru(const Track& t) {
    if (t.clutter) return "помеха";
    switch (t.cls) {
        case TargetClass::Drone: return "дрон";
        case TargetClass::Bird: return "птица";
        case TargetClass::Clutter: return "помеха";
        default: return "цель?";
    }
}

const char* isa_ru(Isa i) {
    switch (i) {
        case Isa::Auto: return "Авто (лучший)";
        case Isa::Scalar: return "Скалярный (без SIMD)";
        case Isa::SSE2: return "SSE2 (128 бит, как NEON)";
        case Isa::AVX2: return "AVX2 (256 бит)";
        case Isa::NEON: return "NEON (128 бит)";
    }
    return "?";
}

struct ResOpt {
    const char* label;
    float scale;   // > 0: доля исходного размера
    int w, h;      // иначе — фиксированный размер
};
const ResOpt RES_OPTS[] = {
    {"Исходное", 1.f, 0, 0},
    {"75%", 0.75f, 0, 0},
    {"50%", 0.5f, 0, 0},
    {"25%", 0.25f, 0, 0},
    {"16 Мп (4608x3456)", 0, 4608, 3456},
    {"12 Мп (4000x3000)", 0, 4000, 3000},
    {"8 Мп (3264x2448)", 0, 3264, 2448},
    {"Full HD (1920x1080)", 0, 1920, 1080},
};

// Устройства для имитации. мс/Мп — эталонный замер calibrate_ms_per_mp (1 поток, 128 бит).
// Для Raspberry Pi — оценки по типовой скорости ядер относительно x86 (Xeon 2.8 ГГц даёт ~0.6):
// A76 2.4 ГГц ~2.5x медленнее, A72 1.8 ГГц ~6x, A53 1.2 ГГц (в порядке очереди, слабая память) ~15x.
// Точное число даёт dronolok_bench --calibrate на самом устройстве.
struct DevicePreset {
    const char* name;
    float msPerMp;   // 0 — этот ПК без ограничений
    int cores;
    int profile;     // рекомендуемый профиль алгоритма (Profile), -1 — не менять
};
const DevicePreset DEVICES[] = {
    {"Этот ПК (без ограничений)", 0, 0, -1},
    {"Raspberry Pi 5 (4 ядра A76, 2.4 ГГц)", 1.5f, 4, (int)Profile::Balanced},
    {"Raspberry Pi 4 (4 ядра A72, 1.8 ГГц)", 3.5f, 4, (int)Profile::Economy},
    {"Raspberry Pi 3 (4 ядра A53, 1.2 ГГц)", 9.f, 4, (int)Profile::Minimal},
    {"Своё устройство (4 ядра)", 3.f, 4, -1},
};
const int N_DEVICES = (int)(sizeof(DEVICES) / sizeof(DEVICES[0]));

struct App {
    Pipeline pipe;
    // режим: 0 — видео, 1 — эмулятор двух камер
    int appMode = 0;
    twocam::Emulator emu;
    twocam::Config emuCfg;
    twocam::View emuView;
    SDL_Texture *emuWTex = nullptr, *emuNTex = nullptr;
    int emuWTw = 0, emuWTh = 0, emuNTw = 0, emuNTh = 0;
    int64_t emuWSeq = -1;
    int emuRes = 0, emuSpeed = 2;
    bool emuTruth = true;
    Settings s;
    // устройство и профиль
    int devIdx = 0, profIdx = 1;
    float devMs[N_DEVICES] = {};
    double pcMs = 0.6;     // калибровка этого ПК, мс/Мп
    std::string err, note;
    // ресурсы
    std::vector<char> cpuOn;
    bool pinWorkers = false;
    int cpuCap = 100, priority = 0, resIdx = 0;
    // отображение
    bool showNoise = true, showClutter = true, showTracks = true, showPaths = true, showIds = true;
    bool showMask = false, showUpper = false, showGt = true, showCrops = true, showPred = true, showFocus = false;
    int viewSeq = -1;      // кадр из истории на паузе (Result::seq), -1 — последний обработанный
    // разметка целей
    bool annotate = false;
    Annotation ann;
    int annCur = 0;        // выбранный объект (AnnObject::id)
    AnnObject annUndo;     // последний удалённый целиком — «вернуть»
    bool annHasUndo = false;
    std::string annMsg;
    // разметка по цели: после щелчка по дрону точки ставятся сами, пока он виден — по треку
    // алгоритма, а где трека нет (трекер сбросил треки, дрон принят за помеху) — по картинке
    bool annAuto = true;
    int annStep = 10;          // точка раз в столько кадров
    bool annFollow = false;    // разметка идёт за целью
    int annFollowId = 0;       // трек (Track::id), по которому она сейчас идёт; 0 — по картинке
    int annFollowObj = 0;      // объект разметки
    int annFollowSeq = 0;      // последний просмотренный результат (Result::seq)
    int64_t annFollowKey = 0;  // кадр последней поставленной точки
    int64_t annSeenFrame = -1; // последний кадр, где цель была найдена
    float annSeenX = 0, annSeenY = 0;   // где, в пикселях видео
    std::vector<float> annTpl; // как цель выглядит: окрестность, яркость без среднего и нормированная
    int annByTrack = 0, annByImage = 0;   // кадров, где цель нашлась по треку и по картинке
    bool annDirty = false;     // точки добавлены, файл ещё не записан
    double annSavedAt = 0;
    // шкала видео
    std::vector<uint8_t> marks;
    int64_t scrub = -1;    // куда тянут ползунок шкалы
    double scrubAt = 0;
    int displayMode = 1;   // 0 каждый кадр, 1 до 30 FPS, 2 до 10 FPS, 3 не показывать
    float zoom = 1.f, panX = 0, panY = 0;
    // текстуры
    SDL_Renderer* ren = nullptr;
    void* owner = nullptr;   // окно приложения для диалога выбора файла (HWND)
    SDL_Texture* tex = nullptr;
    int texW = 0, texH = 0, texDs = 1, texSeq = -1;
    double texAt = 0;
    std::shared_ptr<const Result> shown;   // кадр, который сейчас на текстуре
    SDL_Texture* cropTex = nullptr;
    int maxTexW = 4096, maxTexH = 4096;
    // загрузка CPU
    double cpuT = 0, cpuWall = 0, cpuPct = 0;
    // мультитест
    mt::Runner mt;
    char mtFolder[1024] = "";
    struct MtSel {
        bool on = false;
        char vals[128] = "";
    };
    std::vector<MtSel> mtSel;   // по mt::params()
    int mtMode = 0, mtFrames = 0, mtParallel = 1;
    bool mtRealtime = false, mtEmulate = false;
    std::vector<std::string> mtVideos;
    int mtGt = 0;
    std::string mtMsg, mtScanned;
    bool mtOpen = false;
};

void apply_resources(App& a) {
    uint64_t mask = 0;
    std::vector<int> cpus;
    for (size_t i = 0; i < a.cpuOn.size() && i < 64; ++i)
        if (a.cpuOn[i]) {
            mask |= 1ull << i;
            cpus.push_back((int)i);
        }
    if (!mask) {   // хотя бы одно ядро
        a.cpuOn[0] = 1;
        mask = 1;
        cpus = {0};
    }
    plat::set_process_affinity(mask);
    // маска процесса действует на все уже запущенные потоки; при закреплении за ядрами
    // новый список cpus пересоздаёт пул детектора
    a.s.det.cpus = a.pinWorkers ? cpus : std::vector<int>();
}

void set_resolution(App& a);

void apply_profile(App& a, int p) {
    const AlgoBlocks b = profile_blocks((Profile)p);
    TrackerConfig t = a.s.trk;
    apply_blocks(b, a.s.det, t);
    a.s.trk.static_frames = t.static_frames;
    a.s.scales = b.scales;   // 0 — авто по размеру кадра
    a.s.motion = b.motion ? -1 : 0;
    a.s.zones = b.zones ? -1 : 0;
    a.s.learnMaskFrames = b.skyMask ? 25 : 0;
    a.resIdx = b.resolution >= 0.99f ? 0 : b.resolution >= 0.74f ? 1 : b.resolution >= 0.49f ? 2 : 3;
    set_resolution(a);
    a.profIdx = p;
}

void apply_device(App& a, int i) {
    const DevicePreset& d = DEVICES[i];
    a.devIdx = i;
    if (d.msPerMp <= 0) {   // весь ПК
        a.s.det.threads = 0;
        a.s.det.isa = Isa::Auto;
        std::fill(a.cpuOn.begin(), a.cpuOn.end(), 1);
        a.pinWorkers = false;
        a.s.slowdown = 1;
    } else {
        // 128-битные векторы, как NEON у Raspberry Pi
        a.s.det.isa = Isa::Scalar;
        for (Isa x : available_isas())
            if (x == Isa::SSE2 || x == Isa::NEON) a.s.det.isa = x;
        const int n = std::min<int>(d.cores, (int)a.cpuOn.size());
        a.s.det.threads = n;
        for (size_t k = 0; k < a.cpuOn.size(); ++k) a.cpuOn[k] = (int)k < n;
        a.pinWorkers = true;
        a.s.slowdown = std::max(1.0, a.devMs[i] / std::max(1e-3, a.pcMs));
    }
    a.s.emulateMs = 0;
    apply_resources(a);
    if (d.profile >= 0) apply_profile(a, d.profile);
}

void set_resolution(App& a) {
    const ResOpt& r = RES_OPTS[a.resIdx];
    const VideoInfo vi = a.pipe.info();
    if (r.scale > 0) {
        a.s.procW = r.scale == 1.f ? 0 : std::max(64, (int)(vi.w * r.scale) & ~1);
        a.s.procH = r.scale == 1.f ? 0 : std::max(64, (int)(vi.h * r.scale) & ~1);
    } else {
        a.s.procW = r.w;
        a.s.procH = r.h;
    }
}

void open_video(App& a, const std::string& path) {
    a.err.clear();
    a.shown.reset();
    a.texSeq = -1;
    if (!a.pipe.open(path, a.err)) return;
    set_resolution(a);
    a.pipe.setSettings(a.s);
    a.zoom = 1.f;
    a.panX = a.panY = 0;
    a.viewSeq = -1;
    const VideoInfo vi = a.pipe.info();
    a.ann.load(a.pipe.gtPath(), vi.w, vi.h, vi.fps);
    a.annCur = a.ann.objs.empty() ? 0 : a.ann.objs[0].id;
    a.annMsg.clear();
    a.annFollow = false;
    a.annFollowId = 0;
    a.annDirty = false;
    a.annHasUndo = false;
}

// разметку поменяли: сохранить в <видео>.gt.json и сразу считать по ней качество
void ann_changed(App& a) {
    const int64_t n = a.pipe.length();
    if (!a.ann.save(a.pipe.gtPath(), n)) {
        a.annMsg = "не удалось сохранить " + a.pipe.gtPath();
        return;
    }
    a.pipe.setGroundTruth(a.ann.toGroundTruth(n));
    int keys = 0;
    for (const auto& o : a.ann.objs) keys += (int)o.keys.size();
    a.annMsg = "сохранено: объектов " + std::to_string(a.ann.objs.size()) + ", точек " + std::to_string(keys);
}

// удалить путь объекта целиком (все его точки на всех кадрах); «вернуть» восстанавливает
void ann_delete(App& a, int id) {
    auto it = std::find_if(a.ann.objs.begin(), a.ann.objs.end(), [&](const AnnObject& o) { return o.id == id; });
    if (it == a.ann.objs.end()) return;
    if (a.annFollow && a.annFollowObj == id) {
        a.annFollow = false;   // разметка шла за ним — остановить, не дописывая последнюю точку
        a.annFollowId = 0;
        a.annDirty = false;
    }
    a.annUndo = *it;
    a.annHasUndo = true;
    a.ann.objs.erase(it);
    if (a.annCur == id) a.annCur = a.ann.objs.empty() ? 0 : a.ann.objs.back().id;
    ann_changed(a);
    a.annMsg = "путь #" + std::to_string(id) + " удалён целиком (точек " + std::to_string(a.annUndo.keys.size()) +
               ") — «вернуть» отменит";
}

void ann_undo(App& a) {
    if (!a.annHasUndo) return;
    if (a.ann.find(a.annUndo.id)) a.annUndo.id = a.ann.add(a.annUndo.bird);   // номер успели занять
    else a.ann.objs.push_back(a.annUndo);
    if (AnnObject* o = a.ann.find(a.annUndo.id)) o->keys = a.annUndo.keys;
    a.annCur = a.annUndo.id;
    a.annHasUndo = false;
    ann_changed(a);
}

// ------------------------------------------------------------ разметка по треку

// координаты цели результата r — в пикселях видео
static void track_to_video(const App& a, const Result& r, const Track& t, float& x, float& y) {
    const VideoInfo vi = a.pipe.info();
    const int W = r.follow ? r.srcW : r.frame->w, H = r.follow ? r.srcH : r.frame->h;
    x = (t.x + (float)r.fx) * (W > 0 ? (float)vi.w / W : 1.f);
    y = (t.y + (float)r.fy) * (H > 0 ? (float)vi.h / H : 1.f);
}

// ---- поиск цели по картинке: окрестность (2R+1)^2 кадра обработки, нормированная корреляция
const int TPL_R = 7;

static bool tpl_grab(const Frame& f, float x, float y, std::vector<float>& out) {
    const int cx = (int)std::lround(x), cy = (int)std::lround(y), R = TPL_R;
    if (cx - R < 0 || cy - R < 0 || cx + R >= f.w || cy + R >= f.h || f.px.size() < (size_t)f.w * f.h) return false;
    out.resize((2 * R + 1) * (2 * R + 1));
    double m = 0;
    for (int j = -R, k = 0; j <= R; ++j)
        for (int i = -R; i <= R; ++i, ++k) m += out[k] = f.px[(size_t)(cy + j) * f.w + cx + i];
    m /= out.size();
    double n = 0;
    for (float& v : out) {
        v -= (float)m;
        n += v * v;
    }
    if (n < out.size() * 4.0) return false;   // ровное место (СКО < 2): искать нечего
    const float inv = (float)(1 / std::sqrt(n));
    for (float& v : out) v *= inv;
    return true;
}

// лучшее совпадение с шаблоном в пределах ±S пикс. от (x, y); score — корреляция -1..1
static bool tpl_match(const Frame& f, const std::vector<float>& tpl, float& x, float& y, int S, float& score) {
    const int R = TPL_R, cx0 = (int)std::lround(x), cy0 = (int)std::lround(y);
    if (tpl.empty() || f.px.size() < (size_t)f.w * f.h) return false;
    std::vector<float> p;
    float best = -2;
    int bx = 0, by = 0;
    for (int dy = -S; dy <= S; ++dy)
        for (int dx = -S; dx <= S; ++dx) {
            if (!tpl_grab(f, (float)(cx0 + dx), (float)(cy0 + dy), p)) continue;
            float c = 0;
            for (size_t k = 0; k < p.size(); ++k) c += p[k] * tpl[k];
            c -= 0.002f * (float)(dx * dx + dy * dy) / (float)(S * S + 1);   // при равенстве — ближе
            if (c > best) {
                best = c;
                bx = cx0 + dx;
                by = cy0 + dy;
            }
        }
    if (best < -1) return false;
    x = (float)bx;
    y = (float)by;
    score = best;
    return true;
}

// пиксели кадра обработки <-> пиксели видео (у кадра из поворотки — свои координаты, там только по треку)
static bool frame_scale(const App& a, const Result& r, float& kx, float& ky) {
    const VideoInfo vi = a.pipe.info();
    if (r.follow || vi.w <= 0 || vi.h <= 0) return false;
    kx = (float)r.frame->w / vi.w;
    ky = (float)r.frame->h / vi.h;
    return true;
}

// щелчок мимо цели алгоритма: прилипнуть к самой заметной точке рядом (дрон — пятно темнее
// или светлее фона), чтобы промах мышью на пару пикселей экрана не сдвинул всю разметку
static void snap_to_spot(const Frame& f, float& x, float& y, int R) {
    const int cx = (int)std::lround(x), cy = (int)std::lround(y);
    if (cx - R - 1 < 0 || cy - R - 1 < 0 || cx + R + 1 >= f.w || cy + R + 1 >= f.h || f.px.size() < (size_t)f.w * f.h)
        return;
    auto P = [&](int i, int j) { return (float)f.px[(size_t)j * f.w + i]; };
    double m = 0;
    for (int j = cy - R; j <= cy + R; ++j)
        for (int i = cx - R; i <= cx + R; ++i) m += P(i, j);
    m /= (2 * R + 1) * (2 * R + 1);
    float best = 0;
    for (int j = cy - R; j <= cy + R; ++j)
        for (int i = cx - R; i <= cx + R; ++i) {
            float v = 0;   // 3x3 — чтобы не прилипать к одиночному шумному пикселю
            for (int b = -1; b <= 1; ++b)
                for (int c = -1; c <= 1; ++c) v += P(i + c, j + b);
            v = std::fabs(v / 9 - (float)m);
            if (v > best + 1e-3f) {
                best = v;
                x = (float)i;
                y = (float)j;
            }
        }
}

static void ann_grab(App& a, const Result& r) {
    float kx, ky;
    std::vector<float> t;
    if (frame_scale(a, r, kx, ky) && tpl_grab(*r.frame, a.annSeenX * kx, a.annSeenY * ky, t)) a.annTpl = t;
}

// начать с точки (x, y) в пикселях видео на кадре r; t — цель алгоритма под ней, если есть
void ann_follow_start(App& a, const Result& r, const Track* t, float x, float y) {
    a.annFollow = true;
    a.annFollowId = t ? t->id : 0;
    a.annFollowObj = a.annCur;
    a.annFollowSeq = r.seq;
    a.annFollowKey = r.frame->index;
    a.annSeenFrame = r.frame->index;
    a.annSeenX = x;
    a.annSeenY = y;
    a.annByTrack = a.annByImage = 0;
    a.annTpl.clear();
    ann_grab(a, r);
    if (!t && a.annTpl.empty()) {
        a.annFollow = false;
        a.annMsg = "точка поставлена; вести нечего: рядом нет цели алгоритма, а на картинке ровное место";
        return;
    }
    a.annMsg = "разметка идёт за целью: пустите видео (пробел), где ошибётся — пауза и щелчок по дрону";
}

void ann_follow_stop(App& a, const char* why) {
    if (!a.annFollow) return;
    // последняя точка — где цель видели в последний раз, чтобы объект не тянулся по прямой дальше
    if (AnnObject* ob = a.ann.find(a.annFollowObj))
        if (a.annSeenFrame > a.annFollowKey) {
            ob->keys.erase(ob->keys.upper_bound(a.annFollowKey), ob->keys.lower_bound(a.annSeenFrame));
            ob->keys[a.annSeenFrame] = {a.annSeenX, a.annSeenY};
            a.annDirty = true;
        }
    a.annFollow = false;
    a.annFollowId = 0;
    if (a.annDirty) ann_changed(a);
    a.annDirty = false;
    if (why) a.annMsg = why;
}

// Трек алгоритма рядом с местом, где цель видели: трекер сбросил треки (маска неба выучена
// заново, сцена поменялась) или захват перешёл на новый трек того же дрона — у цели новый
// номер. Захват — первый кандидат
static const Track* ann_follow_reacquire(const App& a, const Result& r) {
    const VideoInfo vi = a.pipe.info();
    const int64_t gap = std::max<int64_t>(1, r.frame->index - a.annSeenFrame);
    // около 0.8% ширины кадра (20 пикс. на 2560) плюс запас на полёт за пропущенные кадры;
    // шире нельзя — рядом с дроном полно помех, и разметка уползла бы на них
    const float lim = 0.008f * (float)std::max(vi.w, 640) + 2.f * (float)std::min<int64_t>(gap, 60);
    const Track* best = nullptr;
    float bd = lim;
    for (const auto& t : r.tracks) {
        if (t.miss > 0) continue;
        float x, y;
        track_to_video(a, r, t, x, y);
        float d = std::hypot(x - a.annSeenX, y - a.annSeenY);
        if (t.id == r.lockId) d *= 0.5f;
        if (d < bd) {
            bd = d;
            best = &t;
        }
    }
    return best;
}

// Каждый кадр интерфейса: пройти по новым результатам и поставить точки, пока цель видна.
// Сначала — трек алгоритма (тот же или новый номер рядом), нет его — поиск по картинке около
// прошлого места. Не нашлась около секунды — пауза там, где её потеряли: щёлкнуть по дрону снова
void ann_follow_update(App& a) {
    if (!a.annFollow) return;
    if (!a.annotate || !a.ann.find(a.annFollowObj)) {
        ann_follow_stop(a, nullptr);
        return;
    }
    int first, last;
    if (!a.pipe.historyRange(first, last)) return;
    AnnObject* ob = a.ann.find(a.annFollowObj);
    const double fps = a.pipe.info().fps;
    const int64_t grace = std::max<int64_t>(5, (int64_t)(fps > 0 ? fps : 30));
    for (int seq = std::max(first, a.annFollowSeq + 1); seq <= last; ++seq) {
        auto r = a.pipe.at(seq);
        if (!r) continue;
        a.annFollowSeq = seq;
        const int64_t idx = r->frame->index;
        if (idx <= a.annSeenFrame) {   // перемотали назад или видео пошло сначала
            ann_follow_stop(a, "разметка за целью остановлена: видео перемотали");
            return;
        }
        if (r->dup) continue;   // повтор прошлого кадра записи экрана
        const Track* t = nullptr;
        if (a.annFollowId)
            for (const auto& x : r->tracks)
                if (x.id == a.annFollowId && x.miss == 0) t = &x;
        if (t) {   // тот же трек не должен уйти далеко от прошлого места (перескок на помеху)
            float x, y;
            track_to_video(a, *r, *t, x, y);
            const VideoInfo vi = a.pipe.info();
            const float lim = 0.02f * (float)std::max(vi.w, 640) + 2.f * (float)std::min<int64_t>(idx - a.annSeenFrame, 60);
            if (std::hypot(x - a.annSeenX, y - a.annSeenY) > lim) t = nullptr;
        }
        if (!t) t = ann_follow_reacquire(a, *r);
        bool found = false;
        if (t) {
            track_to_video(a, *r, *t, a.annSeenX, a.annSeenY);
            a.annFollowId = t->id;
            ann_grab(a, *r);   // как цель выглядит сейчас — на случай, если трек дальше пропадёт
            ++a.annByTrack;
            found = true;
        } else {
            a.annFollowId = 0;
            float kx, ky;
            if (!a.annTpl.empty() && frame_scale(a, *r, kx, ky)) {
                const int64_t gap = std::min<int64_t>(idx - a.annSeenFrame, 10);
                float x = a.annSeenX * kx, y = a.annSeenY * ky, sc = 0;
                const int S = std::min(24, 4 + 2 * (int)gap);
                if (tpl_match(*r->frame, a.annTpl, x, y, S, sc) && sc > 0.6f) {
                    a.annSeenX = x / kx;
                    a.annSeenY = y / ky;
                    std::vector<float> nt;   // шаблон понемногу следует за видом цели
                    if (tpl_grab(*r->frame, x, y, nt)) {
                        double n = 0;
                        for (size_t k = 0; k < nt.size(); ++k) n += (a.annTpl[k] = 0.9f * a.annTpl[k] + 0.1f * nt[k]) * a.annTpl[k];
                        if (n > 0)
                            for (float& v : a.annTpl) v /= (float)std::sqrt(n);
                    }
                    ++a.annByImage;
                    found = true;
                }
            }
        }
        if (!found) {
            if (idx - a.annSeenFrame <= grace) continue;   // дрон мог мигнуть: ждём до секунды
            char b[160];
            snprintf(b, sizeof(b), "цель потеряна после кадра %lld: найдите дрон и щёлкните по нему снова",
                     (long long)a.annSeenFrame);
            ann_follow_stop(a, b);
            a.pipe.setPaused(true);
            return;
        }
        a.annSeenFrame = idx;
        if (idx - a.annFollowKey >= a.annStep) {
            // пройденный участок — заново: старые точки на нём (прошлый проход, ошибка) убрать
            ob->keys.erase(ob->keys.upper_bound(a.annFollowKey), ob->keys.lower_bound(idx));
            ob->keys[idx] = {a.annSeenX, a.annSeenY};
            a.annFollowKey = idx;
            a.annDirty = true;
        }
    }
    // файл — не чаще раза в секунду: на нём пересчитывается качество
    if (a.annDirty && now_ms() - a.annSavedAt > 1000) {
        ann_changed(a);
        a.annMsg = "разметка идёт за целью: " + a.annMsg;
        a.annDirty = false;
        a.annSavedAt = now_ms();
    }
}

// ------------------------------------------------------------ плеер: пауза, покадрово, скорость

void toggle_pause(App& a) {
    const bool p = !a.pipe.paused();
    a.pipe.setPaused(p);
    if (!p) a.viewSeq = -1;
}

// назад — по уже обработанным кадрам (история), трекер не откатывается
void step_back(App& a) {
    a.pipe.setPaused(true);
    int first, last;
    if (!a.pipe.historyRange(first, last)) return;
    a.viewSeq = std::max(first, (a.viewSeq >= 0 ? a.viewSeq : last) - 1);
}

// вперёд — по истории, а с последнего кадра — обработать следующий
void step_forward(App& a) {
    int first, last;
    if (a.pipe.paused() && a.viewSeq >= 0 && a.pipe.historyRange(first, last) && a.viewSeq < last) {
        a.viewSeq = a.viewSeq + 1 >= last ? -1 : a.viewSeq + 1;
        return;
    }
    a.pipe.setPaused(true);
    a.viewSeq = -1;
    a.pipe.step();
}

void set_speed(App& a, double sp) {
    a.s.speed = std::min(2.0, std::max(0.05, sp));
    a.s.pace = Pace::Realtime;   // замедление имеет смысл только в темпе камеры
    a.pipe.setSettings(a.s);
}

// перемотка: на паузе кадр из истории (уже обработан, с разметкой) показывается сразу,
// иначе чтение продолжается с этого кадра и треки начинаются заново
void seek_to(App& a, int64_t frame, bool preferHistory) {
    frame = std::max<int64_t>(0, frame);
    if (preferHistory && a.pipe.paused()) {
        const int seq = a.pipe.findInHistory(frame);
        if (seq >= 0) {
            int first, last;
            a.viewSeq = a.pipe.historyRange(first, last) && seq == last ? -1 : seq;
            return;
        }
    }
    a.viewSeq = -1;
    a.pipe.seek(frame);
}

std::string fmt_time(double s) {
    char b[32];
    const int m = (int)(s / 60);
    snprintf(b, sizeof(b), "%d:%04.1f", m, s - m * 60);
    return b;
}

// ------------------------------------------------------------ текстура кадра

bool ensure_texture(App& a, int w, int h) {
    int ds = 1;
    while ((w / ds > a.maxTexW || h / ds > a.maxTexH) && ds < 8) ds *= 2;
    const int tw = (w / ds) & ~1, th = (h / ds) & ~1;
    if (a.tex && tw == a.texW && th == a.texH) return true;
    if (a.tex) SDL_DestroyTexture(a.tex);
    SDL_SetYUVConversionMode(SDL_YUV_CONVERSION_JPEG);   // серый в полном диапазоне 0..255
    a.tex = SDL_CreateTexture(a.ren, SDL_PIXELFORMAT_IYUV, SDL_TEXTUREACCESS_STREAMING, tw, th);
    if (!a.tex) return false;
    SDL_SetTextureScaleMode(a.tex, SDL_ScaleModeLinear);
    a.texW = tw;
    a.texH = th;
    a.texDs = ds;
    // цветовые плоскости U и V = 128 (серый) пишутся один раз
    void* px;
    int pitch;
    if (SDL_LockTexture(a.tex, nullptr, &px, &pitch) == 0) {
        uint8_t* p = (uint8_t*)px;
        memset(p, 0, (size_t)pitch * th);
        memset(p + (size_t)pitch * th, 128, (size_t)(pitch / 2) * (th / 2) * 2);
        SDL_UnlockTexture(a.tex);
    }
    return true;
}

void upload_frame(App& a, const Frame& f) {
    if (!ensure_texture(a, f.w, f.h)) return;
    void* px;
    int pitch;
    if (SDL_LockTexture(a.tex, nullptr, &px, &pitch) != 0) return;
    uint8_t* dst = (uint8_t*)px;
    const int ds = a.texDs;
    for (int y = 0; y < a.texH; ++y) {
        uint8_t* o = dst + (size_t)y * pitch;
        if (ds == 1) {
            memcpy(o, f.px.data() + (size_t)y * f.w, a.texW);
        } else {   // текстура меньше кадра: минимум по блоку, чтобы тёмные точки не пропали
            for (int x = 0; x < a.texW; ++x) {
                uint8_t m = 255;
                for (int yy = 0; yy < ds; ++yy) {
                    const uint8_t* s = f.px.data() + (size_t)(y * ds + yy) * f.w + x * ds;
                    for (int xx = 0; xx < ds; ++xx) m = std::min(m, s[xx]);
                }
                o[x] = m;
            }
        }
    }
    SDL_UnlockTexture(a.tex);
}

// ------------------------------------------------------------ вырезки целей

const int CROP_SRC = 40, CROP_ZOOM = 5, CROP_N = 6, CROP_PX = CROP_SRC * CROP_ZOOM;

std::vector<const Track*> top_targets(const Result& r) {
    std::vector<const Track*> v;
    for (const auto& t : r.tracks)
        if (!t.clutter) v.push_back(&t);
    auto rank = [&](const Track* t) {
        return t->id == r.lockId ? -1 : t->cls == TargetClass::Drone ? 0 : t->cls == TargetClass::Unknown ? 1 : 2;
    };
    std::sort(v.begin(), v.end(), [&](const Track* a, const Track* b) {
        return rank(a) != rank(b) ? rank(a) < rank(b) : a->contrast > b->contrast;
    });
    if (v.size() > (size_t)CROP_N) v.resize(CROP_N);
    return v;
}

void upload_crops(App& a, const Result& r, const std::vector<const Track*>& targets) {
    if (!a.cropTex) {
        a.cropTex = SDL_CreateTexture(a.ren, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, CROP_PX,
                                      CROP_PX * CROP_N);
        if (!a.cropTex) return;
        SDL_SetTextureScaleMode(a.cropTex, SDL_ScaleModeNearest);
    }
    void* px;
    int pitch;
    if (SDL_LockTexture(a.cropTex, nullptr, &px, &pitch) != 0) return;
    const Frame& f = *r.frame;
    for (int i = 0; i < CROP_N; ++i) {
        for (int y = 0; y < CROP_PX; ++y) {
            uint32_t* o = (uint32_t*)((uint8_t*)px + (size_t)(i * CROP_PX + y) * pitch);
            if (i >= (int)targets.size()) {
                for (int x = 0; x < CROP_PX; ++x) o[x] = 0xFF202020;
                continue;
            }
            const int x0 = std::min(std::max(0, (int)targets[i]->x - CROP_SRC / 2), f.w - CROP_SRC);
            const int y0 = std::min(std::max(0, (int)targets[i]->y - CROP_SRC / 2), f.h - CROP_SRC);
            const uint8_t* s = f.px.data() + (size_t)(y0 + y / CROP_ZOOM) * f.w + x0;
            for (int x = 0; x < CROP_PX; ++x) o[x] = 0xFF000000u | 0x010101u * s[x / CROP_ZOOM];
        }
    }
    SDL_UnlockTexture(a.cropTex);
}

// ------------------------------------------------------------ график

void draw_chart(const Metrics& m, float budgetMs, ImVec2 size) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p0 = ImGui::GetCursorScreenPos(), p1(p0.x + size.x, p0.y + size.y);
    dl->AddRectFilled(p0, p1, IM_COL32(25, 27, 32, 255));
    // масштаб по обработке кадра и бюджету; декодирование идёт в своём потоке и
    // может быть намного медленнее (16 Мп H.264 программно), поэтому его выбросы обрезаются
    std::vector<float> all = m.histProc;
    float ymax = 1;
    if (!all.empty()) {
        std::sort(all.begin(), all.end());
        ymax = all[(size_t)(0.98 * (all.size() - 1))] * 1.4f;
    }
    if (budgetMs > 0) ymax = std::max(ymax, budgetMs * 1.15f);
    ymax = std::max(ymax, 1.f);
    // сетка: шаг 1-2-5, не больше 4 линий
    float step = 0.1f;
    for (int i = 0; ymax / step > 4; ++i) step *= (i % 3 == 1) ? 2.5f : 2.f;
    for (float y = step; y < ymax; y += step) {
        const float py = p1.y - y / ymax * size.y;
        dl->AddLine(ImVec2(p0.x, py), ImVec2(p1.x, py), IM_COL32(60, 64, 72, 255));
        char b[32];
        snprintf(b, sizeof(b), "%g мс", y);
        dl->AddText(ImVec2(p0.x + 3, py - ImGui::GetFontSize()), IM_COL32(140, 140, 150, 255), b);
    }
    auto line = [&](const std::vector<float>& v, ImU32 c) {
        if (v.size() < 2) return;
        const size_t n = v.size(), cap = 300;
        std::vector<ImVec2> pts;
        for (size_t i = 0; i < n; ++i)
            pts.push_back(ImVec2(p1.x - (float)(n - 1 - i) / (cap - 1) * size.x,
                                 p1.y - std::min(v[i], ymax) / ymax * size.y));
        dl->AddPolyline(pts.data(), (int)pts.size(), c, 0, 1.5f);
    };
    if (budgetMs > 0) {
        const float py = p1.y - budgetMs / ymax * size.y;
        for (float x = p0.x; x < p1.x; x += 8) dl->AddLine(ImVec2(x, py), ImVec2(x + 4, py), IM_COL32(255, 90, 90, 200));
    }
    line(m.histDecode, IM_COL32(150, 150, 160, 255));
    line(m.histProc, IM_COL32(255, 170, 60, 255));
    line(m.histDetect, IM_COL32(90, 220, 120, 255));
    ImGui::Dummy(size);
    ImGui::TextColored(ImVec4(0.35f, 0.86f, 0.47f, 1), "детекция");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1, 0.67f, 0.24f, 1), "кадр");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.63f, 1), "декодер");
    if (budgetMs > 0) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1, 0.35f, 0.35f, 1), "бюджет");
    }
}

// ------------------------------------------------------------ оформление

const float FONT_BASE = 16.f;   // размер шрифта интерфейса до масштаба экрана
const ImVec4 TXT_OK(0.40f, 0.90f, 0.50f, 1), TXT_WARN(1, 0.80f, 0.30f, 1), TXT_BAD(1, 0.42f, 0.42f, 1);
const ImVec4 TXT_DIM(0.60f, 0.60f, 0.63f, 1), TXT_HEAD(0.62f, 0.76f, 1, 1);
const ImVec4 ACC(0.20f, 0.45f, 0.85f, 1), ACC_HOV(0.27f, 0.53f, 0.93f, 1), ACC_ACT(0.16f, 0.38f, 0.75f, 1);

// крупнее или мельче обычного шрифта на время блока
struct FontSize {
    explicit FontSize(float k) { ImGui::PushFont(nullptr, FONT_BASE * k); }
    ~FontSize() { ImGui::PopFont(); }
    FontSize(const FontSize&) = delete;
    FontSize& operator=(const FontSize&) = delete;
};

void push_accent() {
    ImGui::PushStyleColor(ImGuiCol_Button, ACC);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ACC_HOV);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ACC_ACT);
}
void pop_accent() { ImGui::PopStyleColor(3); }

void help_marker(const char* t) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28);
        ImGui::TextUnformatted(t);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

// заголовок группы настроек
void heading(const char* t) {
    ImGui::Spacing();
    FontSize f(1.1f);
    ImGui::PushStyleColor(ImGuiCol_Text, TXT_HEAD);
    ImGui::SeparatorText(t);
    ImGui::PopStyleColor();
}

// подпись над полем, само поле — на всю ширину следующей строкой
void label(const char* t, const char* help = nullptr) {
    ImGui::TextUnformatted(t);
    if (help) help_marker(help);
    ImGui::SetNextItemWidth(-FLT_MIN);
}

// ряд кнопок-переключателей на всю ширину; выбранная подсвечена. true — выбрали другую
bool segmented(const char* id, const char* const* labels, int n, int& cur, float h = 0,
               const char* const* tips = nullptr) {
    ImGui::PushID(id);
    const float sp = ImGui::GetStyle().ItemSpacing.x * 0.5f;
    const float w = (ImGui::GetContentRegionAvail().x - sp * (n - 1)) / n;
    bool ch = false;
    for (int i = 0; i < n; ++i) {
        if (i) ImGui::SameLine(0, sp);
        const bool on = cur == i;
        if (on) push_accent();
        if (ImGui::Button(labels[i], ImVec2(w, h)) && !on) {
            cur = i;
            ch = true;
        }
        if (on) pop_accent();
        if (tips && tips[i]) ImGui::SetItemTooltip("%s", tips[i]);
    }
    ImGui::PopID();
    return ch;
}

// вкладка, у которой содержимое прокручивается отдельно от того, что над ней
template <class F>
void tab(const char* name, F&& body) {
    if (!ImGui::BeginTabItem(name)) return;
    ImGui::PushID(name);
    ImGui::BeginChild("body", ImVec2(0, 0));
    body();
    ImGui::EndChild();
    ImGui::PopID();
    ImGui::EndTabItem();
}

// ------------------------------------------------------------ панель управления

void panel_annotate(App& a) {
    if (!a.pipe.isOpen()) {
        ImGui::TextDisabled("сначала откройте видео");
        return;
    }
    if (!a.ann.editable) {
        ImGui::TextDisabled("разметка из synth_sky.py: только просмотр");
        return;
    }
    {
        FontSize f(1.1f);
        ImGui::Checkbox("Размечать на видео", &a.annotate);
    }
    help_marker("Своя разметка, чтобы приложение считало полноту и ложные цели на ваших роликах, а шкала "
                "показывала, где дрон пропущен. Поставьте паузу и щёлкните по дрону: на этом кадре появится "
                "опорная точка (рядом с целью алгоритма она прилипает к ней). Между точками положение считается "
                "по прямой, поэтому точку достаточно ставить раз в 10-30 кадров и на поворотах; объект есть "
                "от первой до последней точки. Правый щелчок — убрать точку на этом кадре. Сохраняется сразу "
                "в <видео>.gt.json, рядом с видео.");
    ImGui::TextWrapped("Пауза, щелчок по дрону — опорная точка, дальше точки ставятся сами, пока дрон виден. "
                       "Ошиблись — пауза и щелчок по дрону: поправит и поведёт дальше. Правый щелчок — убрать точку, "
                       "Delete или «x» у объекта — весь путь.");
    if (!a.annotate) {
        ann_follow_stop(a, nullptr);
        if (a.ann.noDrones) ImGui::TextDisabled("ролик отмечен: дронов нет");
        else if (!a.ann.objs.empty()) ImGui::TextDisabled("размечено объектов: %d", (int)a.ann.objs.size());
        return;
    }
    ImGui::BeginDisabled(!a.ann.objs.empty());
    if (ImGui::Checkbox("В ролике нет дронов", &a.ann.noDrones)) ann_changed(a);
    ImGui::EndDisabled();
    help_marker("Для роликов, где дрона нет совсем (пустое небо, только птицы и самолёты вдалеке): тогда любая цель "
                "и любой захват считаются ложными, и мультитест сравнивает варианты и на таком ролике. Доступно, "
                "пока в разметке нет объектов.");
    if (a.ann.noDrones) {
        ImGui::TextDisabled("размечено: дронов нет — любой захват ложный");
        if (!a.annMsg.empty()) ImGui::TextDisabled("%s", a.annMsg.c_str());
        return;
    }
    ImGui::Checkbox("Вести точки по цели", &a.annAuto);
    help_marker("Щелчок по дрону (или кнопка «вести точки по захвату») — и дальше, пока видео идёт, точки ставятся "
                "сами раз в N кадров: по треку алгоритма, а где трека нет (трекер сбросил треки, дрон принят за "
                "помеху, захват перешёл на новый номер) — по картинке, по виду дрона около прошлого места. Дрон не "
                "нашёлся больше секунды — видео встаёт на паузу: щёлкните по нему снова. Ушло на другой объект — "
                "пауза, стрелкой влево назад к ошибке и щелчок по дрону: точки пойдут заново от этого кадра, "
                "старые на пройденном участке заменятся.");
    if (a.annAuto) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::SliderInt("##annstep", &a.annStep, 2, 60, "раз в %d кадр.");
    }
    if (a.annFollow) {
        if (a.annFollowId)
            ImGui::TextColored(TXT_OK, "идёт за целью #%d, кадр %lld", a.annFollowId, (long long)a.annSeenFrame);
        else
            ImGui::TextColored(TXT_OK, "идёт по картинке, кадр %lld", (long long)a.annSeenFrame);
        ImGui::SameLine();
        if (ImGui::SmallButton("стоп")) ann_follow_stop(a, "разметка за целью остановлена");
        if (a.annByImage > 0)
            ImGui::TextDisabled("кадров по треку %d, по картинке %d", a.annByTrack, a.annByImage);
    } else if (a.annAuto) {
        // без щелчка: начать с захваченной цели на показанном кадре
        const std::shared_ptr<const Result> sh = a.shown;
        const Track* lk = nullptr;
        if (sh && !sh->follow)
            for (const auto& t : sh->tracks)
                if (t.id == sh->lockId) lk = &t;
        ImGui::BeginDisabled(!lk);
        if (ImGui::Button(lk ? "вести точки по захвату" : "вести точки по захвату (захвата нет)", ImVec2(-FLT_MIN, 0)) &&
            lk) {
            if (!a.ann.find(a.annCur)) a.annCur = a.ann.add(false);
            float x, y;
            track_to_video(a, *sh, *lk, x, y);
            a.ann.find(a.annCur)->keys[sh->frame->index] = {x, y};
            ann_changed(a);
            ann_follow_start(a, *sh, lk, x, y);
        }
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Точка на этом кадре ставится в захват, дальше — сами по нему. Захват ушёл на "
                              "другой объект — пауза, щелчок по дрону поправит точку и поведёт дальше от неё");
    }
    const float bw = (ImGui::GetContentRegionAvail().x - 2 * ImGui::GetStyle().ItemSpacing.x) / 3;
    if (ImGui::Button("+ дрон", ImVec2(bw, 0))) a.annCur = a.ann.add(false);
    ImGui::SameLine();
    if (ImGui::Button("+ птица", ImVec2(bw, 0))) a.annCur = a.ann.add(true);
    ImGui::SameLine();
    ImGui::BeginDisabled(!a.ann.find(a.annCur));
    if (ImGui::Button("удалить путь", ImVec2(bw, 0))) ann_delete(a, a.annCur);
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Удалить выбранный объект со всеми точками на всех кадрах (клавиша Delete; на видео — "
                          "Shift + правый щелчок по кружку объекта). «Вернуть» под списком отменит");
    {
        const std::shared_ptr<const Result> sh = a.shown;
        AnnObject* ob = a.ann.find(a.annCur);
        ImGui::BeginDisabled(!ob || !sh || ob->keys.empty() || ob->keys.rbegin()->first <= sh->frame->index);
        if (ImGui::Button("стереть точки дальше этого кадра", ImVec2(-FLT_MIN, 0)) && ob && sh) {
            ann_follow_stop(a, nullptr);
            ob->keys.erase(ob->keys.upper_bound(sh->frame->index), ob->keys.end());
            ann_changed(a);
        }
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Если трек ушёл с дрона на другой объект: отмотайте к месту, где он ещё на дроне, "
                              "сотрите точки дальше и щёлкните по дрону снова");
    }
    heading("Объекты");
    int del = 0;
    for (const auto& o : a.ann.objs) {
        char b[128];
        ImGui::PushID(o.id);
        if (ImGui::SmallButton("x")) del = o.id;
        ImGui::SetItemTooltip("удалить путь #%d целиком", o.id);
        ImGui::PopID();
        ImGui::SameLine();
        if (o.keys.empty()) snprintf(b, sizeof(b), "#%d %s: точек нет##o%d", o.id, o.bird ? "птица" : "дрон", o.id);
        else
            snprintf(b, sizeof(b), "#%d %s: точек %d, кадры %lld-%lld##o%d", o.id, o.bird ? "птица" : "дрон",
                     (int)o.keys.size(), (long long)o.keys.begin()->first, (long long)o.keys.rbegin()->first, o.id);
        if (ImGui::Selectable(b, a.annCur == o.id)) a.annCur = o.id;
    }
    if (del) ann_delete(a, del);
    if (a.ann.objs.empty()) ImGui::TextDisabled("щёлкните по дрону на кадре — появится объект #1");
    if (a.annHasUndo) {
        char b[96];
        snprintf(b, sizeof(b), "вернуть удалённый путь #%d (точек %d)", a.annUndo.id, (int)a.annUndo.keys.size());
        if (ImGui::Button(b, ImVec2(-FLT_MIN, 0))) ann_undo(a);
    }
    if (!a.annMsg.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, TXT_DIM);
        ImGui::TextWrapped("%s", a.annMsg.c_str());
        ImGui::PopStyleColor();
    }
}

// главное, всегда на виду: видео, устройство, профиль, темп камеры
void panel_main(App& a, bool& changed) {
    const VideoInfo vi = a.pipe.info();
    const float bigH = ImGui::GetFrameHeight() * 1.9f;
    {
        FontSize f(1.25f);
        push_accent();
        const char* cap = plat::video_dialog_open() ? "Выбор файла открыт..."
                          : a.pipe.isOpen()          ? "Открыть другое видео..."
                                                     : "Открыть видео...";
        if (ImGui::Button(cap, ImVec2(-FLT_MIN, bigH))) plat::open_video_dialog(a.owner);
        pop_accent();
    }
    ImGui::SetItemTooltip("Или перетащите файл в окно");
    {
        const mt::Runner::Status st = a.mt.status();
        char cap[96];
        if (a.mt.running()) snprintf(cap, sizeof(cap), "Мультитест: %d из %d прогонов###mt", st.done, st.total);
        else snprintf(cap, sizeof(cap), "Мультитест: подобрать настройки на папке видео...###mt");
        if (ImGui::Button(cap, ImVec2(-FLT_MIN, 0))) a.mtOpen = !a.mtOpen;
        ImGui::SetItemTooltip("Прогнать каждое видео из папки с разными настройками и получить отчёт");
    }
    if (!a.err.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, TXT_BAD);
        ImGui::TextWrapped("%s", a.err.c_str());
        ImGui::PopStyleColor();
    }
    if (a.pipe.isOpen()) {
        std::string name = a.pipe.path();
        size_t sl = name.find_last_of("/\\");
        if (sl != std::string::npos) name = name.substr(sl + 1);
        {
            FontSize f(1.1f);
            ImGui::TextWrapped("%s", name.c_str());
        }
        ImGui::TextDisabled("%dx%d, %.2f к/с, %s%s%s", vi.w, vi.h, vi.fps, vi.codec.c_str(),
                            vi.frames ? (", ~" + std::to_string(vi.frames) + " кадров").c_str() : "",
                            vi.rotation ? (", повёрнуто на " + std::to_string(vi.rotation) + "°").c_str() : "");
    } else {
        ImGui::TextDisabled("или перетащите файл в окно");
    }

    heading("Устройство");
    {
        FontSize f(1.15f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x * 1.5f,
                                                               ImGui::GetStyle().FramePadding.y * 1.6f));
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::BeginCombo("##device", DEVICES[a.devIdx].name)) {
            for (int i = 0; i < N_DEVICES; ++i)
                if (ImGui::Selectable(DEVICES[i].name, a.devIdx == i)) {
                    apply_device(a, i);
                    changed = true;
                }
            ImGui::EndCombo();
        }
        ImGui::PopStyleVar();
    }
    ImGui::SetItemTooltip("Имитация устройства на этом ПК: столько же ядер, векторы 128 бит (как NEON), а обработка "
                          "каждого кадра растягивается во столько раз, во сколько ядро устройства медленнее ядра ПК. "
                          "Вместе с устройством ставится подходящий профиль алгоритма. Тонкая настройка — на "
                          "вкладке «Железо».");
    if (a.devIdx > 0) {
        ImGui::TextDisabled("%d ядра, 128 бит, обработка медленнее этого ПК в %.1f раза", DEVICES[a.devIdx].cores,
                            a.s.slowdown);
        if (a.devMs[a.devIdx] < a.pcMs)
            ImGui::TextColored(TXT_WARN, "ядро ПК медленнее устройства: точная имитация невозможна");
    } else {
        ImGui::TextDisabled("все ядра и SIMD этого ПК");
    }

    heading("Профиль алгоритма");
    static const char* profs[] = {"Полный", "Баланс", "Эконом", "Минимум"};
    static const char* tips[] = {
        "Всё включено: каждый кадр целиком, 3 масштаба. Для мощного ПК.",
        "Как по умолчанию: хороший компромисс скорости и полноты.",
        "Для Raspberry Pi 4-5: весь кадр через кадр, 2 масштаба.",
        "Для Raspberry Pi 3: только точки и тёмные цели, кадр вдвое меньше, весь кадр раз в 3.",
    };
    int pi = a.profIdx;
    {
        FontSize f(1.1f);
        if (segmented("profile", profs, 4, pi, bigH * 0.75f, tips)) {
            apply_profile(a, pi);
            changed = true;
        }
    }
    if (a.profIdx < 0) ImGui::TextColored(TXT_WARN, "свой набор: блоки изменены на вкладке «Алгоритм»");
    {
        FontSize f(1.1f);
        changed |= ImGui::Checkbox("Держать темп камеры", &a.s.governor);
    }
    help_marker("Авто-упрощение. Если кадр обрабатывается дольше, чем приходит следующий с камеры, алгоритм "
                "упрощается ступенями: реже крупные масштабы, реже весь кадр, меньше масштабов. Когда запас "
                "появляется, возвращается обратно. Ступень видна справа на вкладке «Цели» и в CSV. Это же есть в "
                "ядре для устройства (dronolok::LoadGovernor).");
}

void tab_video(App& a, bool& changed) {
    const VideoInfo vi = a.pipe.info();
    heading("Подача кадров");
    static const char* paces[] = {"Как камера", "Максимально быстро"};
    static const char* paceTips[] = {
        "Кадры приходят с частотой видео (или заданной). Если обработка не успевает, лишние кадры "
        "выбрасываются, как у настоящей камеры. Счётчик пропусков — справа.",
        "Без пропусков и пауз: замер предельной скорости всего конвейера.",
    };
    int pace = a.s.pace == Pace::Realtime ? 0 : 1;
    if (segmented("pace", paces, 2, pace, 0, paceTips)) {
        a.s.pace = pace == 0 ? Pace::Realtime : Pace::MaxSpeed;
        changed = true;
    }
    if (a.s.pace == Pace::Realtime) {
        float fps = (float)a.s.targetFps;
        label("частота камеры, к/с");
        if (ImGui::SliderFloat("##fps", &fps, 0, 120, fps <= 0 ? "как в видео" : "%.0f")) {
            a.s.targetFps = fps;
            changed = true;
        }
        help_marker("Ниже частоты видео — камера реже видео: из файла берётся кадр на каждом такте камеры "
                    "(14 к/с из 60 — через 4-5 кадров), видео идёт с обычной скоростью, и цель за кадр "
                    "смещается дальше, как у настоящей камеры 16 Мп (10-15 к/с). Выше частоты видео — видео "
                    "ускоряется. Замедлить видео, не пропуская кадров, — \"скорость видео\".");
        label("камера реже видео: каждый N-й кадр");
        if (ImGui::SliderInt("##fstep", &a.s.frameStep, 1, 8, a.s.frameStep <= 1 ? "все кадры" : "каждый %d-й")) {
            a.s.frameStep = std::max(1, a.s.frameStep);
            changed = true;
        }
        help_marker("Эмуляция медленной камеры: из видео 60 к/с с шагом 4 алгоритм видит 15 к/с (камера 16 Мп — "
                    "10-15 к/с), и цель за кадр смещается вчетверо дальше. Разметка видео подходит без пересчёта. "
                    "Складывается с \"частотой камеры\": 30 к/с и каждый 2-й — 15 к/с.");
        float sp = (float)a.s.speed;
        label("скорость видео", "Замедление: кадры подаются реже, каждый обрабатывается. Кнопки x0.1-x1 есть и "
                                "под видео. Клавиши + и -.");
        if (ImGui::SliderFloat("##speed", &sp, 0.05f, 2.f, "x%.2f", ImGuiSliderFlags_Logarithmic)) set_speed(a, sp);
    }
    heading("Источник кадров");
    static const char* srcs[] = {"Читать файл", "Предзагрузка"};
    static const char* srcTips[] = {
        "Кадры декодируются по ходу проигрывания.",
        "Кадры заранее декодируются в память и подаются по кругу без копирования, как сырые буферы камеры. "
        "Тогда декодирование видео не мешает замеру.",
    };
    int src = a.s.source == Source::Stream ? 0 : 1;
    bool srcCh = segmented("src", srcs, 2, src, 0, srcTips);
    if (src == 1) {
        label("кадров в памяти");
        srcCh |= ImGui::SliderInt("##preload", &a.s.preloadFrames, 10, 1000);
        const int w = a.s.procW ? a.s.procW : vi.w, h = a.s.procH ? a.s.procH : vi.h;
        ImGui::TextDisabled("памяти: ~%.0f МБ", (double)w * h * a.s.preloadFrames / (1024.0 * 1024.0));
    }
    if (srcCh) {
        a.s.source = src == 0 ? Source::Stream : Source::Preload;
        changed = true;
    }
    heading("Проигрывание");
    changed |= ImGui::Checkbox("Зациклить", &a.s.loop);
    if (ImGui::Checkbox("Чистый старт (без накопленного опыта)", &a.s.fresh)) {
        a.pipe.setSettings(a.s);
        if (a.s.fresh) {   // забыть уже накопленное и пройти видео с начала
            a.pipe.forget();
            seek_to(a, 0, false);
        }
    }
    help_marker("Видео проверяется так, будто открыто впервые: память статичных помех (.static.bin) и маска неба "
                "(.mask.bin), накопленные прошлыми запусками рядом с видео, не загружаются и не перезаписываются, "
                "а на каждом новом проходе зацикленного видео маска неба учится заново. Включение забывает уже "
                "накопленное и начинает видео сначала. Для честной оценки настроек; мультитест всегда идёт так.");
    changed |= ImGui::Checkbox("Пропускать повторы кадра", &a.s.skipDuplicates);
    help_marker("Запись экрана 60 к/с с ролика 25-30 к/с повторяет каждый кадр 2-3 раза. Включено — повтор не "
                "обрабатывается, трекер видит ролик с его настоящей частотой, как с камеры 25-30 к/с. Выключено — "
                "повторы обрабатываются как новые кадры (у тусклой цели больше попыток найтись, но скорость цели "
                "скачет: то стоит, то прыгнула вдвое).");
    int first, last;
    if (a.pipe.paused() && a.pipe.historyRange(first, last) && last > first) {
        int cur = a.viewSeq >= 0 ? a.viewSeq : last;
        char fmt[64];
        snprintf(fmt, sizeof(fmt), cur == last ? "последний" : "%d кадров назад", last - cur);
        label("история обработанных кадров",
              "Последние обработанные кадры с разметкой (до 600 кадров и 1 ГБ памяти). Трекер при этом не "
              "откатывается: вперёд с последнего кадра идёт обработка следующего.");
        if (ImGui::SliderInt("##history", &cur, first, last, fmt)) a.viewSeq = cur >= last ? -1 : cur;
    }
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, TXT_DIM);
    ImGui::TextWrapped("Клавиши: пробел — пауза, стрелки — кадр назад/вперёд, Shift+стрелки — на секунду, "
                       "Home — в начало, +/- — скорость.");
    ImGui::PopStyleColor();
}

void tab_hardware(App& a, bool& changed) {
    heading("Скорость устройства");
    if (a.devIdx > 0) {
        label("мс на 1 Мп на устройстве",
              "Сколько ядро устройства тратит на 1 Мп (1 поток, 128 бит). Сейчас — оценка; точное число даёт "
              "dronolok_bench --calibrate, запущенный на самом устройстве.");
        if (ImGui::InputFloat("##devms", &a.devMs[a.devIdx], 0, 0, "%.2f")) {
            a.devMs[a.devIdx] = std::max(0.01f, a.devMs[a.devIdx]);
            a.s.slowdown = std::max(1.0, a.devMs[a.devIdx] / std::max(1e-3, a.pcMs));
            changed = true;
        }
        ImGui::TextDisabled("этот ПК: %.2f мс/Мп, устройство медленнее в %.1f раза", a.pcMs, a.s.slowdown);
    } else {
        ImGui::TextDisabled("этот ПК: %.2f мс/Мп (1 поток, 128 бит)", a.pcMs);
        ImGui::TextDisabled("выберите устройство сверху, чтобы имитировать Pi");
    }

    heading("Процессор");
    const int hw = plat::logical_cpus();
    int th = a.s.det.threads > 0 ? a.s.det.threads : hw;
    label("потоков детектора");
    if (ImGui::SliderInt("##threads", &th, 1, std::max(hw, 16))) {
        a.s.det.threads = th;
        changed = true;
    }
    label("набор инструкций (SIMD)", "На Raspberry Pi 3/5 есть только NEON (128 бит), по ширине это как SSE2. "
                                     "AVX2 — 256 бит, есть на большинстве ПК.");
    if (ImGui::BeginCombo("##simd", isa_ru(a.s.det.isa))) {
        std::vector<Isa> isas = available_isas();
        isas.insert(isas.begin(), Isa::Auto);
        for (Isa i : isas)
            if (ImGui::Selectable(isa_ru(i), a.s.det.isa == i)) {
                a.s.det.isa = i;
                changed = true;
            }
        ImGui::EndCombo();
    }
    ImGui::TextUnformatted("ядра процесса");
    help_marker("Процесс (детектор, декодер, интерфейс) работает только на отмеченных логических ядрах. "
                "Например, 4 ядра — как у Raspberry Pi.");
    bool cpuCh = false;
    for (int i = 0; i < (int)a.cpuOn.size() && i < 64; ++i) {
        if (i % 8) ImGui::SameLine();
        char lbl[8];
        snprintf(lbl, sizeof(lbl), "%d##c", i);
        bool on = a.cpuOn[i] != 0;
        ImGui::PushID(i);
        if (ImGui::Checkbox(lbl, &on)) {
            a.cpuOn[i] = on;
            cpuCh = true;
        }
        ImGui::PopID();
    }
    if (ImGui::SmallButton("все")) {
        std::fill(a.cpuOn.begin(), a.cpuOn.end(), 1);
        cpuCh = true;
    }
    for (int n : {1, 2, 4}) {
        if (n >= (int)a.cpuOn.size()) continue;
        ImGui::SameLine();
        char lbl[32];
        snprintf(lbl, sizeof(lbl), "первые %d", n);
        if (ImGui::SmallButton(lbl)) {
            for (size_t i = 0; i < a.cpuOn.size(); ++i) a.cpuOn[i] = (int)i < n;
            cpuCh = true;
        }
    }
    cpuCh |= ImGui::Checkbox("Закрепить потоки детектора за ядрами", &a.pinWorkers);
    if (cpuCh) {
        apply_resources(a);
        changed = true;
    }
    int dth = a.s.decoderThreads;
    label("потоков декодера");
    if (ImGui::SliderInt("##decth", &dth, 0, hw, dth == 0 ? "авто" : "%d")) {
        a.s.decoderThreads = dth;
        changed = true;
    }
    ImGui::BeginDisabled(!plat::cpu_cap_supported());
    label("потолок загрузки CPU",
          "Жёсткое ограничение загрузки всем процессом, в процентах от всех ядер ПК (Windows Job Object). "
          "Имитирует более медленный процессор.");
    if (ImGui::SliderInt("##cpucap", &a.cpuCap, 5, 100, a.cpuCap >= 100 ? "нет" : "%d%%")) plat::set_cpu_cap(a.cpuCap);
    ImGui::EndDisabled();
    static const char* prios[] = {"низкий", "ниже среднего", "обычный", "выше среднего", "высокий"};
    int pr = a.priority + 2;
    label("приоритет процесса");
    if (ImGui::Combo("##prio", &pr, prios, 5)) {
        a.priority = pr - 2;
        plat::set_process_priority(a.priority);
    }

    heading("Кадр");
    label("разрешение обработки", "Кадр масштабируется перед детектором. 16 Мп — растянуть ролик до размера "
                                  "кадра широкоугольной камеры (как --upscale в run_demo.py).");
    if (ImGui::BeginCombo("##res", RES_OPTS[a.resIdx].label)) {
        for (int i = 0; i < (int)(sizeof(RES_OPTS) / sizeof(RES_OPTS[0])); ++i)
            if (ImGui::Selectable(RES_OPTS[i].label, a.resIdx == i)) {
                a.resIdx = i;
                set_resolution(a);
                changed = true;
            }
        ImGui::EndCombo();
    }
    float emu = (float)a.s.emulateMs;
    label("не быстрее, мс на кадр",
          "Обработка кадра занимает не меньше заданного времени. Подставьте время, которое dronolok_bench "
          "показал на Raspberry Pi, и посмотрите, сколько кадров будет теряться в режиме «как камера».");
    if (ImGui::SliderFloat("##emu", &emu, 0, 200, emu <= 0 ? "выкл." : "%.0f мс")) {
        a.s.emulateMs = emu;
        changed = true;
    }
}

void tab_algorithm(App& a, bool& changed) {
    const VideoInfo vi = a.pipe.info();
    DetectorConfig& d = a.s.det;
    TrackerConfig& t = a.s.trk;
    bool blk = false;   // блок поменяли вручную — профиль "свой"
    ImGui::TextWrapped("Блоки, из которых собран профиль. Любой можно переключить — профиль станет «свой».");
    heading("Подстройка под сцену");
    blk |= ImGui::Checkbox("Подстраиваться под сцену и погоду", &a.s.adapt);
    help_marker("Раз в 0.2 с по кадру оцениваются свет (день, сумерки, ночь), шум матрицы, облачность, качание "
                "кромки леса на ветру, осадки, дымка и что ниже горизонта (лес, город, поле). Под них "
                "подстраиваются выставленные здесь настройки: в сумерки и в дымку — порог по шуму, ночью — только "
                "светлые цели (огни), в снегопад — только тёмные, в осадки — строже подтверждение, в ветер цель у "
                "крон должна пролететь дальше, а в штиль выдаётся раньше, в городе быстрее запоминаются антенны и "
                "фонари. Облако закрыло солнце — фон ячеек учится заново; камеру повернули — маска неба заново. "
                "Синтетика: сумерки 0% -> 28% дронов, ночь 0% -> 20%, снег 45 тыс. ложных в минуту -> 0. Что "
                "сейчас подстроено — справа, вкладка «Цели».");
    heading("Крупные цели и движение камеры");
    static const char* scl[] = {"авто", "1: до ~7 пикс.", "2: до ~28 пикс.", "3: до ~56 пикс.", "4: до ~112 пикс."};
    label("размер цели (масштабов)",
          "Сколько масштабов просматривать. 1 — только точки (дальний дрон), каждый следующий — кадр, уменьшенный "
          "ещё в 2 раза, для близкой крупной цели. Каждый добавляет ~20-35% времени. Авто: до 2 Мп — 3, больше — 1.");
    blk |= ImGui::Combo("##scales", &a.s.scales, scl, 5);
    static const char* mos[] = {"авто", "нет", "да"};
    int mo = a.s.motion + 1;
    label("учитывать движение камеры",
          "Съёмка с рук или с поворотного устройства: сдвиг всей картинки между кадрами оценивается (~1-2 мс) и "
          "прогноз целей переносится вместе с ним, иначе захват срывается при каждом движении камеры. Авто: до "
          "2 Мп — учитывать.");
    if (segmented("motion", mos, 3, mo)) {
        a.s.motion = mo - 1;
        blk = true;
    }
    int zo = a.s.zones + 1;
    label("зоны кадра (небо, земля, кромки)",
          "Кадр делится на небо, землю (кроны, дома, земля) и смешанные тайлы неба рядом с землёй. В смешанных "
          "цель выдаётся, только когда полетела, — так пропадают ложные цели на кромках крон, крыш и столбов. "
          "Авто: вместе с учётом движения камеры.");
    if (segmented("zones", mos, 3, zo)) {
        a.s.zones = zo - 1;
        blk = true;
    }
    if (a.pipe.isOpen()) {
        const Effective ef = effective(a.s, a.s.procW ? a.s.procW : vi.w, a.s.procH ? a.s.procH : vi.h);
        ImGui::TextDisabled("сейчас: масштабов %d, камера %s, зоны %s", ef.scales, ef.motion ? "учитывается" : "нет",
                            ef.zones ? "да" : "нет");
    }
    blk |= ImGui::Checkbox("тёмные цели", &d.dark);
    ImGui::SameLine();
    blk |= ImGui::Checkbox("яркие цели", &d.bright);
    help_marker("Каждая полярность — отдельный проход морфологии. Дрон на фоне неба почти всегда темнее неба, "
                "поэтому на слабом железе можно оставить только тёмные: вдвое быстрее.");

    blk |= ImGui::Checkbox("Трекер в секундах", &a.s.trkSeconds);
    help_marker("Скорости, ворота и выдержки трекера настроены на 25 новых кадров в секунду. Включено — они "
                "пересчитываются под частоту камеры (повторы кадров в записи экрана не считаются). Камера 16 Мп "
                "на 10 к/с: птиц находится 77% вместо 51%, дронов, принятых за птиц, нет.");
    blk |= ImGui::Checkbox("Маска неба и поворот камеры в секундах", &a.s.fpsScale);
    help_marker("Окна маски неба (обучение, уточнение у кромки леса, доращивание по помехам, перепроверка "
                "облаков), оценка поворота камеры и признак панорамы подобраны на 60 к/с. Включено — они "
                "пересчитываются под частоту камеры, и в секундах всё как на 60 к/с. Выключено на 15 к/с: "
                "уточнение маски у кромки копится 10 с вместо 2.5, а медленный поворот, при котором часть кадра "
                "стоит (своя техника, наложения записи экрана), считается \"камера стоит\" — маска не поспевает "
                "за картинкой.");
    heading("Камера ведёт цель");
    changed |= ImGui::Checkbox("виртуальная поворотка", &a.s.follow);
    help_marker("Проверка захвата для камеры на поворотном устройстве, которая следует за целью. Алгоритм видит "
                "не весь кадр, а окно, и окно, как камера на подвесе, поворачивается за захваченной целью: цель "
                "стоит почти в центре, фон едет навстречу её движению. Поворот окна трекер знает точно — как "
                "показания датчиков поворотки. На экране — то, что видит такая камера.");
    if (a.s.follow) {
        label("поле зрения", "Окно — эта доля ширины и высоты кадра: узкое поле камеры на поворотке.");
        float fr = a.s.followFrac * 100.f;
        if (ImGui::SliderFloat("##ffrac", &fr, 20.f, 100.f, "%.0f%% кадра")) {
            a.s.followFrac = fr / 100.f;
            changed = true;
        }
        label("резкость поворота", "Какую долю отклонения цели от центра поворотка выбирает за кадр (сверх "
                                   "скорости цели). Больше — цель ближе к центру, но поворот дёрганее.");
        changed |= ImGui::SliderFloat("##fgain", &a.s.followGain, 0.05f, 1.f, "%.2f");
        label("предел поворота", "Сколько окно может повернуться за кадр, доля его ширины.");
        changed |= ImGui::SliderFloat("##fpan", &a.s.followMaxPan, 0.005f, 0.2f, "%.3f");
        changed |= ImGui::Checkbox("только за проверенным дроном", &a.s.followVerified);
        help_marker("Поворачиваться только при жёстком захвате (цель проверена как дрон). Иначе окно уходит за "
                    "первой попавшейся помехой, и настоящая цель оказывается за краем узкого поля зрения.");
    }

    heading("Неподвижная камера");
    label("кадров на маску неба", "Тайлы, где по первым кадрам видна текстура (деревья, дома), дальше не "
                                  "просматриваются. При учёте движения камеры маска не строится.");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.7f);
    blk |= ImGui::SliderInt("##mask", &a.s.learnMaskFrames, 0, 100, a.s.learnMaskFrames ? "%d" : "выкл.");
    ImGui::SameLine();
    if (ImGui::Button("заново", ImVec2(-FLT_MIN, 0))) a.pipe.relearnMask();
    blk |= ImGui::Checkbox("Маска из файла", &a.s.maskFile);
    help_marker("Маска неба лежит рядом с видео (.mask.bin): на старте она закрывает лес сразу, а не через 25 "
                "кадров обучения (ложных треков в первые 3 с — в 6 раз меньше). Пока идут эти кадры, маска "
                "сверяется с видимым; если камеру сдвинули и больше 10% тайлов не совпало — учится заново и "
                "перезаписывается. «Заново» учит, не глядя в файл.");
    bool pt2 = d.per_tile > 1;
    if (ImGui::Checkbox("Два кандидата в тайле", &pt2)) {
        d.per_tile = pt2 ? 2 : 1;
        blk = true;
    }
    help_marker("Кроме самого сильного отклика тайла 64x64 берётся ещё один, из ячейки 16x16 подальше от первого: "
                "дрон рядом с бликом, птицей или вторым дроном. Пара дронов в 20 пикс.: 65% -> 99%. Цена ~0.5 мс на "
                "16 Мп.");
    blk |= ImGui::Checkbox("Всё ниже горизонта — земля", &d.skyline);
    help_marker("Маска неба: в каждом столбце всё ниже кромки леса или крыш считается землёй, даже гладкие места "
                "(трава, поляна, дорога, просветы между стволами). Дрон прилетает из-за горизонта и бывает над "
                "кромкой, ниже почти никогда. Записи War Thunder: ложных целей на кадр 8 -> ~1. Действует со "
                "следующего обучения маски.");
    blk |= ImGui::Checkbox("Маска уточняется по ячейкам", &d.fine_mask);
    help_marker("Тайлы земли 64x64 у самой кромки делятся на ячейки 16x16: ячейка, где несколько секунд "
                "подряд не бывает листвы и веток и над которой небо, становится небом (просветы между кронами, "
                "небо над низкой крышей). Маска так детализируется со временем, пока работает камера. Нужен "
                "фон по ячейкам у кромки.");
    blk |= ImGui::Checkbox("Маска дорастает, где копятся помехи", &d.grow_mask);
    help_marker("Где маска легла неточно (дальняя трава, поле у горизонта, просвет между домами), копятся "
                "помехи. Тайл неба, где их за пару секунд в несколько раз больше среднего по небу этой сцены, "
                "закрывается, если прирастает к маске: земля хотя бы с двух сторон и над ним не чистое небо. "
                "Ряд тайлов над кромкой крон и крыш так не закрывается, вокруг захваченной цели тоже. Только "
                "для неподвижной камеры.");
    blk |= ImGui::Checkbox("Маска открывает облака", &d.mask_recheck);
    help_marker("Облака с рваной кромкой при обучении текстурны, как кроны, и маска закрывала их — в облачную "
                "погоду до половины неба. Закрытые так тайлы раз в 10 кадров перепроверяются: у кроны и крыши "
                "отклик сильный по всему тайлу, у облака слабый. Слабый тайл, выше которого в столбце нет земли, "
                "за несколько секунд открывается; цель в нём подтверждается, только когда полетела.");
    {
        bool pm = d.pol_mask > 0;
        if (ImGui::Checkbox("Облака закрыты только для светлых целей", &pm)) {
            d.pol_mask = pm ? 1 : 0;
            blk = true;
        }
    }
    help_marker("Облака дают светлую текстуру (яркие клочья на небе), а кроны, дома и трава — и тёмную, и светлую. "
                "Тайл, который маска закрыла только из-за светлой текстуры, остаётся в поиске тёмных целей: дрон "
                "на фоне облаков тёмный. На экране такие тайлы закрашены едва заметно. Цель там подтверждается, когда "
                "полетела; если помехи копятся, тайл закрывается совсем. Действует со следующего обучения маски.");
    blk |= ImGui::Checkbox("Облака по цвету", &d.sky_color);
    help_marker("Облака у солнца дают и светлую, и тёмную текстуру (яркие клочья и голубые просветы между ними), и "
                "маска закрывала их целиком, как кроны. По цвету кадра такой тайл — небо: не зелёный (листва, трава), "
                "не темнее открытого неба этой сцены, без глубоких теней (у крон и домов они есть) и не стоит на "
                "земле. Он остаётся в поиске целей (по фону своих ячеек, цель подтверждается, когда полетела), а "
                "доращивание маски его не закрывает. Нужно цветное видео; действует со следующего обучения маски.");
    if (d.sky_color) {
        blk |= ImGui::Checkbox("Мягкие облака и у кромки леса", &d.sky_soft);
        help_marker("Облако, касающееся крон, стоит на земле, и правило выше его не открывало. В облаке яркость меняется "
                    "плавно: перепад на 1 пиксель — около четверти перепада на 4 (у листвы, веток и леса в дымке — "
                    "половина и больше). Такой тайл цвета неба — облако и у самой кромки леса, линия горизонта через "
                    "него не идёт. Гребень дальнего холма в дымке так же мягок, но лежит полосой в один тайл между "
                    "гладким небом и лесом — он остаётся закрытым. Над тёмной верхушкой близкого дерева облако линию "
                    "горизонта всё же поднимает: просвет между ними закрыт, иначе при малейшем сдвиге камеры туда "
                    "заходят ветки.");
        blk |= ImGui::Checkbox("Облака по цвету — и для светлых целей", &d.sky_bright);
        help_marker("Выкл.: облако по цвету открыто только для тёмных целей (дрон на фоне облака тёмный), на экране "
                    "такие тайлы закрашены едва заметно.");
    }
    blk |= ImGui::Checkbox("Фон по ячейкам у кромки леса", &d.bg_map);
    help_marker("Маска неба не расширяется на тайл запаса: тайлы у самой кромки леса смотрятся по ячейкам 16x16, "
                "и у каждой ячейки там свой порог по её шуму за ~10 с (листва, марево). Дальние дроны низко над "
                "лесом: 16% -> 52%, ложных не больше.");
    blk |= ImGui::Checkbox("Накопление слабых целей (опыт)", &d.tbd);
    help_marker("Track-before-detect: ячейка, где отклик чуть выше её фона кадр за кадром, даёт кандидата и ниже "
                "порога. Пока не окупается: на 16 Мп дальних 19% -> 28%, но на других сценах прироста нет, а ложных "
                "+10..70 в минуту; +4.7 мс на 16 Мп.");
    label("полоса у горизонта, пикс.",
          "Полоса над кромкой земли (по маске неба) смотрится каждый кадр, небо выше — раз в N кадров и вокруг "
          "уже найденных целей. Дальние дроны — низко над горизонтом, близкие — выше, крупнее и контрастнее. "
          "16 Мп, полоса 512: раз в 2 — без потерь, раз в 3 — на 24% быстрее, высокие цели 97% -> 90%.");
    blk |= ImGui::SliderInt("##band", &d.band_px, 0, 1024, d.band_px > 0 ? "%d" : "выкл.");
    if (d.band_px > 0) {
        label("небо выше полосы раз в, кадров");
        blk |= ImGui::SliderInt("##upev", &d.upper_every, 1, 4, d.upper_every > 1 ? "%d" : "каждый");
    }
    bool stOn = t.static_frames > 0;
    if (ImGui::Checkbox("Память статичных помех", &stOn)) {
        t.static_frames = stOn ? 3000 : 0;
        blk = true;
    }
    help_marker("Места, где цели раз за разом появляются и никуда не летят (катушки и изоляторы ЛЭП, фонари, "
                "антенны), запоминаются, и там больше не рождаются цели. Дрон, пролетающий мимо такого места, не "
                "теряется. Память лежит рядом с видео (.static.bin). При учёте движения камеры не ведётся.");
    if (stOn) {
        ImGui::Indent();
        label("запомнить после, откликов", "Меньше — быстрее учится, но зависший на одном месте дрон тоже может "
                                           "стать «помехой».");
        changed |= ImGui::SliderFloat("##sthits", &t.static_hits, 3, 40, "%.0f");
        label("забывать за, кадров");
        changed |= ImGui::SliderInt("##stframes", &t.static_frames, 300, 30000, "%d", ImGuiSliderFlags_Logarithmic);
        if (ImGui::SmallButton("сохранить память")) a.pipe.staticSave();
        ImGui::SameLine();
        if (ImGui::SmallButton("забыть всё")) a.pipe.staticClear();
        ImGui::Unindent();
    }

    heading("Экономия");
    label("весь кадр раз в, кадров",
          "В остальные кадры детектор смотрит только вокруг уже найденных целей. Ведомая цель не теряется, новая "
          "находится с задержкой до N-1 кадров. На 16 Мп раз в 2 кадра — минус ~35% времени.");
    blk |= ImGui::SliderInt("##fullev", &d.full_every, 1, 4, d.full_every > 1 ? "%d" : "каждый");
    label("крупные масштабы раз в, кадров",
          "Уменьшенные копии кадра (поиск крупной цели) целиком — раз в N кадров, а между ними только если крупная "
          "цель уже ведётся. 16 Мп, 3 масштаба: 7.4 мс -> 5.7 (раз в 2) -> 4.8 (раз в 4).");
    blk |= ImGui::SliderInt("##coarseev", &d.coarse_every, 1, 4, d.coarse_every > 1 ? "%d" : "каждый");
    if (blk) {
        a.profIdx = -1;
        changed = true;
    }
}

void tab_fine(App& a, bool& changed) {
    DetectorConfig& d = a.s.det;
    TrackerConfig& t = a.s.trk;
    if (ImGui::Button("Вернуть по умолчанию", ImVec2(-FLT_MIN, 0))) {
        // ресурсы (потоки, SIMD, ядра) не трогаем, блоки — по текущему профилю
        DetectorConfig nd;
        nd.threads = d.threads;
        nd.isa = d.isa;
        nd.cpus = d.cpus;
        d = nd;
        t = TrackerConfig();
        a.s.focus = false;
        apply_profile(a, a.profIdx >= 0 ? a.profIdx : (int)Profile::Balanced);
        changed = true;
    }
    ImGui::SetItemTooltip("Пороги, трекер и классификатор — как по умолчанию; блоки — по выбранному профилю.");

    heading("Порог обнаружения");
    {
        FontSize f(1.05f);
        changed |= ImGui::Checkbox("Порог по шуму", &d.auto_thr);
    }
    help_marker("Порог считается на каждом кадре от шума неба (как в радарах): чистое небо — порог ниже и видны "
                "тусклые цели, шумная матрица вечером — выше и меньше ложных. Текущие шум и порог — справа на "
                "вкладке «Цели».");
    if (d.auto_thr) {
        label("во сколько раз выше шума");
        changed |= ImGui::SliderFloat("##noisek", &d.noise_k, 4, 30, "%.1f");
        label("пределы порога");
        changed |= ImGui::DragFloatRange2("##thrrange", &d.thr_min, &d.thr_max, 0.5f, 2, 80, "от %.0f", "до %.0f");
    } else {
        label("порог контраста", "Минимальная разница яркости цели и фона, уровни 0..255.");
        changed |= ImGui::SliderFloat("##thr", &d.thr, 3, 40, "%.0f");
    }
    label("выше клаттера, раз");
    changed |= ImGui::SliderFloat("##clutk", &d.clutter_k, 1, 8, "%.1f");
    float dens = d.max_density * 100;
    label("текстура тайла, %", "Тайл, где откликов больше этой доли, считается текстурой (листва, рябь облаков) "
                               "и пропускается.");
    if (ImGui::SliderFloat("##dens", &dens, 0.2f, 5, "%.1f")) {
        d.max_density = dens / 100;
        changed = true;
    }
    changed |= ImGui::Checkbox("крупная цель — не текстура", &d.compact_tex);
    help_marker("Близкий контрастный дрон (5+ пикс.) сам даёт много откликов, и его тайл выглядел текстурой — "
                "цель пропадала. С галочкой тайл, где все отклики собраны в одно пятно, текстурой не считается. "
                "Синтетика: дрон 6 пикс. с контрастом 30 находится в 93% кадров вместо 29%.");
    changed |= ImGui::Checkbox("сглаживание 3x3 (тусклые точки)", &d.smooth);
    help_marker("Кадр сглаживается перед поиском (согласованный фильтр для точки 2-4 пикс.): шум падает втрое, "
                "тусклая точка — меньше чем вдвое, порог вдвое ниже. Синтетика: нужный контраст 14 -> 9. Резкие "
                "точки в 1 пикс. (запись игры) видны хуже — проверьте на своей камере.");
    changed |= ImGui::Checkbox("положение точнее пикселя", &d.subpixel);
    help_marker("Центр цели уточняется по пятну, а не по клетке 2x2: ошибка положения 0.8 -> 0.1-0.3 пикс. "
                "Нужно для наведения узкой камеры и поворотки.");
    label("порог крупных масштабов, раз", "Во сколько раз порог выше на уменьшенных масштабах: крупная близкая "
                                          "цель контрастна, а пятна облаков на них дают ложные цели.");
    changed |= ImGui::SliderFloat("##coarsek", &d.coarse_thr_k, 1, 6, "%.1f");
    int kk = d.k;
    label("макс. размер цели k, пикс.", "После пулинга (нечётное). При пулинге 2x2 цель до ~2k пикселей кадра.");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.6f);
    if (ImGui::SliderInt("##k", &kk, 3, 15)) {
        d.k = kk | 1;
        changed = true;
    }
    ImGui::SameLine();
    bool pool2 = d.pool == 2;
    if (ImGui::Checkbox("пулинг 2x2", &pool2)) {
        d.pool = pool2 ? 2 : 1;
        changed = true;
    }
    changed |= ImGui::Checkbox("Искать возле прогноза", &a.s.focus);
    help_marker("Эксперимент: там, где трек ожидается на следующем кадре, порог ниже, чтобы не терять бледнеющую "
                "цель. На ваших роликах пока чуть хуже, поэтому выключено.");
    if (a.s.focus) {
        label("порог там");
        changed |= ImGui::SliderFloat("##focusk", &d.focus_k, 0.2f, 1, "x%.2f");
    }

    heading("Трекер и захват");
    label("макс. радиус поиска, пикс.", "Где искать цель на следующем кадре: вокруг прогноза по её скорости. "
                                        "Окно растёт с неуверенностью прогноза, но не больше этого.");
    changed |= ImGui::SliderFloat("##gate", &t.gate, 5, 150, "%.0f");
    label("манёвренность цели", "Ускорение в фильтре Калмана, пикс./кадр². Меньше — прогноз жёстче, захват "
                                "плавно летящего дрона устойчивее; больше — не теряет резкие манёвры.");
    changed |= ImGui::SliderFloat("##qacc", &t.q_accel, 0.25f, 20, "%.2f", ImGuiSliderFlags_Logarithmic);
    label("держать на прогнозе, кадров", "Сколько кадров подряд цель может не находиться (ушла за провод, слилась "
                                         "с фоном), а трек и захват идут по прогнозу.");
    changed |= ImGui::SliderInt("##maxmiss", &t.max_miss, 1, 60);
    label("смена захвата, раз", "Захват переходит на другую цель, только если она весомее текущей во столько раз.");
    changed |= ImGui::SliderFloat("##lockswitch", &t.lock_switch, 1, 5, "%.1f");
    label("подтверждение, кадров");
    changed |= ImGui::SliderInt("##confirm", &t.confirm, 1, t.window);
    changed |= ImGui::Checkbox("крупная цель целиком (силуэт)", &t.silhouette);
    help_marker("У крупного БПЛА детектор находит части: крыло, хвост, подвес. Силуэт собирает их в один объект: "
                "захват держится за центр всей цели, а части не становятся отдельными целями.");
    if (t.silhouette) {
        label("силуэт до, пикс.", "Больше этого — уже не цель, а земля, крона или облако.");
        changed |= ImGui::SliderFloat("##silhmax", &t.silh_max, 50, 1000, "%.0f");
    }
    changed |= ImGui::Checkbox("ожидание размера", &t.size_predict);
    help_marker("Видимый размер захваченной цели ведётся вместе со скоростью его изменения: дрон приближается — "
                "растёт, удаляется — уменьшается. По ожидаемому размеру выбирается масштаб детектора вокруг цели, "
                "а силуэт, внезапно ставший намного больше ожидаемого (слился с облаком или кроной), не "
                "принимается. Размер и «приближается / удаляется» видны в статусе захвата.");
    bool quiet = t.quiet_frames > 0;
    if (ImGui::Checkbox("новая точка в спокойном небе — дрон", &quiet)) {
        t.quiet_frames = quiet ? 50 : 0;
        changed = true;
    }
    help_marker("Дрон, который висит или летит прямо на камеру, по кадру почти не движется, и обычно его гасит "
                "карта помех. С галочкой точка, появившаяся там, где 2 с не было ни одного отклика (камера "
                "стоит), выдаётся как зависший дрон. Рядом появилось что-то ещё (облако, листва) — как обычно.");
    label("у кромки — прямота полёта", "Цель, родившаяся у кромки крон, выдаётся, только если летит по прямой "
                                       "(смещение / длина пути). Листва ходит туда-сюда (~0.4), дрон летит "
                                       "прямо (0.9+). 0 — не проверять.");
    changed |= ImGui::SliderFloat("##edgestraight", &t.edge_straight, 0, 0.9f, t.edge_straight > 0 ? "%.2f" : "не проверять");
    changed |= ImGui::Checkbox("на мачте или столбе — не дрон", &t.anchor_check);
    help_marker("Если от захваченной цели вверх или вниз тянется длинная тонкая линия (мачта прожектора, столб, "
                "ствол), это светильник или что-то на конструкции: захват снимается, цель становится помехой. "
                "При съёмке с рук такая цель «летит» по кадру и иначе может пройти проверку на дрона.");

    heading("Жёсткий захват дрона");
    changed |= ImGui::Checkbox("включён", &t.hard_lock);
    help_marker("Когда захваченная цель достаточно долго подряд классифицируется как дрон, захват становится "
                "жёстким: «Захвачен дрон», на другие цели не переключается, дольше держится на прогнозе, "
                "детектор смотрит в основном вокруг неё. Снимается, если цель оказалась надписью на кадре или "
                "уверенно птицей.");
    if (t.hard_lock) {
        label("проверка: дрон подряд, кадров", "Сколько обработанных кадров подряд цель должна быть дроном (и уже "
                                                "пролететь по сцене), прежде чем захват станет жёстким. Больше — "
                                                "надёжнее, но дольше до жёсткого захвата.");
        changed |= ImGui::SliderInt("##verify", &t.verify_frames, 1, 120);
        label("держать на прогнозе, кадров", "Сколько кадров жёстко захваченный дрон может не находиться.");
        changed |= ImGui::SliderInt("##hardmiss", &t.hard_miss, 5, 300);
        label("весь кадр раз в, кадров", "Остальное время детектор смотрит только вокруг захваченного дрона: "
                                         "быстрее и не отвлекается.");
        changed |= ImGui::SliderInt("##hardfull", &t.hard_full_every, 1, 30);
    }
    bool slow = t.lock_slow > 0;
    if (ImGui::Checkbox("захват: сначала мелкий медленный дрон", &slow)) {
        t.lock_slow = slow ? 1.f : 0.f;
        t.hard_switch = slow ? 2.f : 0.f;
        changed = true;
    }
    help_marker("Из нескольких целей захват берёт ту, что медленнее и уже пролетела по сцене, а не самую "
                "контрастную: самолёт и вертолёт крупнее и быстрее дрона. Зависшая на месте точка уступает "
                "летящим, и жёсткий захват переходит на цель вдвое весомее и вдвое медленнее. Записи War Thunder: дроны в захвате "
                "31% -> 47%, самолёты и вертолёты 53% -> 45%. Выключено — захват на самой контрастной цели.");

    heading("Птица или дрон");
    label("отклонение от траектории, пикс.", "Средний разброс точек трека вокруг плавной траектории за последние "
                                             "16 попаданий. Дрон летит плавно (~1 пикс.), ласточка дёргается (3-10).");
    changed |= ImGui::SliderFloat("##birdresid", &t.bird_resid, 0.5f, 10, "%.1f");
    label("мигание контраста (взмахи)");
    changed |= ImGui::SliderFloat("##birdcv", &t.bird_contrast_cv, 0.05f, 1, "%.2f");
    label("скорость птицы, пикс./кадр");
    changed |= ImGui::SliderFloat("##birdspeed", &t.bird_speed, 0, 40, t.bird_speed > 0 ? "%.0f" : "не учитывать");
    label("попаданий до решения");
    changed |= ImGui::SliderInt("##clsmin", &t.cls_min_hits, 3, t.cls_window);
}

void tab_view(App& a) {
    heading("Видео на экране");
    static const char* modes[] = {"каждый кадр", "до 30 к/с", "до 10 к/с", "не показывать видео"};
    label("обновление картинки", "Отрисовка 16 Мп кадра тоже тратит CPU. Для чистого замера скорости выберите "
                                 "«не показывать».");
    ImGui::Combo("##dispmode", &a.displayMode, modes, 4);
    heading("Что рисовать на кадре");
    if (ImGui::BeginTable("show", 2, ImGuiTableFlags_SizingStretchSame)) {
        auto item = [](const char* name, bool* v, const char* help = nullptr) {
            ImGui::TableNextColumn();
            ImGui::Checkbox(name, v);
            if (help) help_marker(help);
        };
        item("цели и классы", &a.showTracks);
        item("траектории", &a.showPaths);
        item("номера целей", &a.showIds);
        item("прогноз движения", &a.showPred,
             "Линия — куда цель придёт за 10 кадров при нынешней скорости (у захвата и устоявшихся треков). "
             "Пунктирная рамка — трек идёт по прогнозу; у захвата круг — где цель может быть (3 сигмы).");
        item("шум", &a.showNoise);
        item("помехи", &a.showClutter);
        item("маска неба и зоны", &a.showMask);
        item("менее важное небо", &a.showUpper,
             "Фиолетовым — небо, где дрон бывает реже: дальний дрон держится низко над горизонтом, а верх неба "
             "нужен только близкой, крупной цели. С полосой у горизонта (вкладка «Алгоритм») это небо выше полосы, "
             "его детектор смотрит раз в несколько кадров; без полосы — верхние 30% неба над землёй в каждом "
             "столбце, они смотрятся каждый кадр, подсветка только показывает, где это.");
        item("пониженный порог", &a.showFocus);
        item("разметка (.gt.json)", &a.showGt);
        item("вырезки целей", &a.showCrops);
        ImGui::EndTable();
    }
}

// ------------------------------------------------------------ мультитест

void mt_scan(App& a) {
    a.mtScanned = a.mtFolder;
    a.mtVideos = a.mtScanned.empty() ? std::vector<std::string>() : mt::list_videos(a.mtScanned);
    a.mtGt = 0;
    for (const auto& v : a.mtVideos)
        if (FILE* f = open_file(v + ".gt.json", "rb")) {
            fclose(f);
            ++a.mtGt;
        }
}

mt::Setup mt_base(const App& a) {
    mt::Setup u;
    u.s = a.s;
    const ResOpt& r = RES_OPTS[a.resIdx];
    u.scale = r.scale > 0 ? r.scale : 1.f;
    return u;
}

std::vector<mt::Choice> mt_choices(const App& a, bool& bad) {
    std::vector<mt::Choice> ch;
    bad = false;
    const auto& P = mt::params();
    for (size_t i = 0; i < P.size() && i < a.mtSel.size(); ++i) {
        if (!a.mtSel[i].on) continue;
        mt::Choice c;
        c.key = P[i].key;
        if (!mt::parse_values(P[i], a.mtSel[i].vals, c.values)) {
            bad = true;
            continue;
        }
        ch.push_back(c);
    }
    return ch;
}

void mt_start(App& a) {
    mt::Config c;
    c.folder = a.mtScanned;
    c.videos = a.mtVideos;
    c.base = mt_base(a);
    bool bad;
    c.choices = mt_choices(a, bad);
    c.mode = a.mtMode ? mt::Mode::Grid : mt::Mode::OneAtATime;
    c.maxFrames = a.mtFrames;
    c.parallel = a.mtParallel;
    c.realtime = a.mtRealtime;
    c.emulate = a.mtEmulate;
    std::string err;
    if (!a.mt.start(c, err)) {
        a.mtMsg = err;
        return;
    }
    a.mtMsg.clear();
    // просмотр на паузе, чтобы не отнимать ядра у прогонов
    if (a.pipe.isOpen()) a.pipe.setPaused(true);
}

void mt_setup(App& a) {
    const auto& P = mt::params();
    const bool run = a.mt.running();
    ImGui::TextWrapped("Каждое видео из папки прогоняется с несколькими вариантами настроек, и складывается отчёт: "
                       "что лучше находит и держит дрона. База — текущие настройки приложения.");
    heading("Папка с видео");
    ImGui::BeginDisabled(run);
    const float bw = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2;
    if (ImGui::Button(plat::folder_dialog_open() ? "Выбор открыт..." : "Выбрать папку...", ImVec2(bw, 0)))
        plat::open_folder_dialog(a.owner);
    ImGui::SameLine();
    ImGui::BeginDisabled(!a.pipe.isOpen());
    if (ImGui::Button("Папка открытого видео", ImVec2(-FLT_MIN, 0))) {
        std::string d = a.pipe.path();
        const size_t sl = d.find_last_of("/\\");
        d = sl == std::string::npos ? "." : d.substr(0, sl);
        snprintf(a.mtFolder, sizeof(a.mtFolder), "%s", d.c_str());
    }
    ImGui::EndDisabled();
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##mtfolder", "путь к папке", a.mtFolder, sizeof(a.mtFolder));
    ImGui::EndDisabled();
    if (a.mtScanned != a.mtFolder) mt_scan(a);
    if (a.mtScanned.empty()) ImGui::TextDisabled("выберите папку");
    else if (a.mtVideos.empty()) ImGui::TextColored(TXT_WARN, "в папке нет видео");
    else {
        ImGui::TextDisabled("видео: %d, с разметкой (.gt.json): %d", (int)a.mtVideos.size(), a.mtGt);
        if (!a.mtGt)
            help_marker("Без разметки видно только, как меняются число целей, захват и смены захвата. Чтобы мультитест "
                        "считал точность (дрон найден, захват на дроне, ложный захват), разметьте ролики на вкладке "
                        "«Разметка».");
    }

    heading("Что менять");
    ImGui::BeginDisabled(run);
    static const char* modes[] = {"По одной", "Все сочетания"};
    static const char* modeTips[] = {
        "База и каждое значение каждой отмеченной настройки по отдельности, остальное — как в базе. Видно, что "
        "даёт каждая настройка.",
        "Все сочетания значений отмеченных настроек (не больше 256). Для 2-3 настроек, чтобы найти лучшее вместе.",
    };
    segmented("mtmode", modes, 2, a.mtMode, 0, modeTips);
    ImGui::TextDisabled("отметьте настройки и через запятую — значения");
    const char* groups[] = {"Главное", "Железо", "Алгоритм", "Видео", "Тонко"};
    for (const char* g : groups) {
        int on = 0;
        for (size_t i = 0; i < P.size(); ++i) on += !strcmp(P[i].group, g) && a.mtSel[i].on;
        char title[96];
        if (on) snprintf(title, sizeof(title), "%s (%d)###mtg%s", g, on, g);
        else snprintf(title, sizeof(title), "%s###mtg%s", g, g);
        if (!ImGui::CollapsingHeader(title, !strcmp(g, "Алгоритм") ? ImGuiTreeNodeFlags_DefaultOpen : 0)) continue;
        for (size_t i = 0; i < P.size(); ++i) {
            if (strcmp(P[i].group, g)) continue;
            ImGui::PushID((int)i);
            ImGui::Checkbox(P[i].name, &a.mtSel[i].on);
            if (a.mtSel[i].on) {
                ImGui::Indent();
                std::vector<double> v;
                const bool ok = mt::parse_values(P[i], a.mtSel[i].vals, v);
                if (!ok) ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.45f, 0.12f, 0.12f, 1));
                ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.6f);
                ImGui::InputText("##vals", a.mtSel[i].vals, sizeof(a.mtSel[i].vals));
                if (!ok) ImGui::PopStyleColor();
                ImGui::SameLine();
                const mt::Setup b = mt_base(a);
                if (strcmp(P[i].key, "profile")) ImGui::TextDisabled("сейчас %s", mt::fmt_value(P[i], P[i].get(b)).c_str());
                ImGui::Unindent();
            }
            ImGui::PopID();
        }
    }
    bool bad;
    const auto ch = mt_choices(a, bad);
    const int nv = mt::count_variants(mt_base(a), ch, a.mtMode ? mt::Mode::Grid : mt::Mode::OneAtATime);
    const int nvc = std::min(nv, a.mtMode ? mt::GRID_MAX + 1 : nv);
    ImGui::Text("вариантов %d x видео %d = %d прогонов", nvc, (int)a.mtVideos.size(), nvc * (int)a.mtVideos.size());
    if (bad) ImGui::TextColored(TXT_BAD, "не разобрать значения (подсвечены)");
    if (a.mtMode && nv > mt::GRID_MAX) ImGui::TextColored(TXT_WARN, "сочетаний больше %d: возьмутся первые", mt::GRID_MAX);

    heading("Прогон");
    label("кадров с каждого видео", "0 — видео целиком. Меньше — быстрее, но короткий кусок может не дойти до дрона.");
    ImGui::SliderInt("##mtframes", &a.mtFrames, 0, 3000, a.mtFrames ? "%d" : "всё видео");
    label("прогонов одновременно", "Несколько прогонов сразу делят ядра ПК: всё вместе быстрее (декодирование видео "
                                   "идёт параллельно), но время обработки кадра в отчёте выше.");
    ImGui::SliderInt("##mtpar", &a.mtParallel, 1, std::max(1, std::min(8, plat::logical_cpus())));
    ImGui::Checkbox("как камера: с пропусками кадров", &a.mtRealtime);
    help_marker("Кадры подаются в темпе видео, и что обработка не успела — пропускается, как на устройстве. Прогон "
                "длится столько же, сколько видео. Выключено — каждый кадр обрабатывается, так быстро, как может ПК.");
    ImGui::Checkbox("с имитацией устройства", &a.mtEmulate);
    help_marker("Оставить замедление выбранного устройства (Raspberry Pi). Имеет смысл вместе с «как камера» — тогда "
                "видно, сколько захвата теряется из-за пропущенных кадров. Иначе только дольше.");
    ImGui::TextDisabled("каждый прогон — чистый старт, без накопленного опыта");
    ImGui::SetItemTooltip("Память помех и маска неба, накопленные прошлыми запусками рядом с видео (.static.bin, "
                          ".mask.bin), не загружаются и не перезаписываются: варианты сравниваются честно");
    ImGui::EndDisabled();
    ImGui::Spacing();
    {
        FontSize f(1.15f);
        if (!run) {
            ImGui::BeginDisabled(a.mtVideos.empty() || bad);
            push_accent();
            if (ImGui::Button("Запустить мультитест", ImVec2(-FLT_MIN, ImGui::GetFrameHeight() * 1.6f))) mt_start(a);
            pop_accent();
            ImGui::EndDisabled();
        } else if (ImGui::Button("Остановить", ImVec2(-FLT_MIN, ImGui::GetFrameHeight() * 1.6f))) {
            a.mt.stop();
        }
    }
    if (!a.mtMsg.empty()) ImGui::TextColored(TXT_BAD, "%s", a.mtMsg.c_str());
}

// ход и итоги мультитеста
void mt_progress(App& a, bool& changed) {
    const bool run = a.mt.running();
    const mt::Runner::Status st = a.mt.status();
    if (st.total == 0) {
        ImGui::TextWrapped("Здесь будут ход прогона и итоги. Полный отчёт — report.html в папке multitest-<дата> "
                           "рядом с видео, а results.json оттуда можно прислать на разбор.");
        return;
    }
    if (st.total == 0) return;
    char ov[96];
    snprintf(ov, sizeof(ov), "%d из %d прогонов", st.done, st.total);
    ImGui::ProgressBar((float)st.frac, ImVec2(-FLT_MIN, 0), ov);
    if (run) {
        if (st.etaS >= 0) ImGui::TextDisabled("прошло %s, осталось ~%s", fmt_time(st.elapsedS).c_str(), fmt_time(st.etaS).c_str());
        if (!st.current.empty()) ImGui::TextWrapped("%s", st.current.c_str());
        if (st.stopped) ImGui::TextColored(TXT_WARN, "останавливается после текущих кадров...");
    } else {
        ImGui::TextColored(st.done == st.total ? TXT_OK : TXT_WARN, "%s за %s", st.done == st.total ? "готово" : "остановлен",
                           fmt_time(st.elapsedS).c_str());
    }
    if (!st.error.empty()) ImGui::TextColored(TXT_WARN, "%s", st.error.c_str());
    ImGui::BeginDisabled(st.html.empty());
    const float hw = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2;
    if (ImGui::Button("Открыть отчёт", ImVec2(hw, 0))) plat::open_in_browser(st.html);
    ImGui::SameLine();
    if (ImGui::Button("Открыть папку", ImVec2(-FLT_MIN, 0))) plat::open_in_browser(st.outDir);
    ImGui::EndDisabled();
    ImGui::TextDisabled("report.html — посмотреть, results.json — прислать на разбор");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", st.outDir.c_str());

    // короткая сводка: главная метрика, среднее по видео
    std::vector<mt::VideoMeta> vids;
    std::vector<mt::Variant> vars;
    std::vector<mt::RunResult> res;
    a.mt.snapshot(vids, vars, res);
    bool anyGt = false;
    for (const auto& r : res) anyGt |= r.done && r.gt;
    const char* key = anyGt ? "score" : "lock_pct";
    const mt::Metric* M = nullptr;
    for (const auto& m : mt::metrics())
        if (!strcmp(m.key, key)) M = &m;
    heading(anyGt ? "Итоги: оценка (по разметке)" : "Итоги: кадров с захватом, %");
    std::vector<std::pair<double, int>> rows;
    for (size_t k = 0; k < vars.size(); ++k) {
        double s = 0;
        int n = 0;
        for (const auto& r : res)
            if (r.variant == (int)k && r.done && (!M->gtOnly || r.gt) && M->get(r) >= 0) {
                s += M->get(r);
                ++n;
            }
        if (n) rows.push_back({s / n, (int)k});
    }
    if (rows.empty()) {
        ImGui::TextDisabled("пока нет готовых прогонов");
        return;
    }
    double base = -1;
    for (const auto& r : rows)
        if (r.second == 0) base = r.first;
    std::sort(rows.begin(), rows.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
    if (ImGui::BeginTable("mtres", 3, ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("вариант", ImGuiTableColumnFlags_WidthStretch, 3.f);
        ImGui::TableSetupColumn(anyGt ? "оценка" : "захват", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFontSize() * 7);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("взять").x + ImGui::GetStyle().FramePadding.x * 2);
        ImGui::TableHeadersRow();
        for (const auto& [v, k] : rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (k == 0) ImGui::TextColored(TXT_HEAD, "%s", vars[k].name.c_str());
            else ImGui::TextWrapped("%s", vars[k].name.c_str());
            ImGui::TableNextColumn();
            if (k == 0 || base < 0) ImGui::Text("%.1f", v);
            else ImGui::TextColored(!anyGt ? ImVec4(0.85f, 0.88f, 0.95f, 1) : v > base + 0.5 ? TXT_OK : v < base - 0.5 ? TXT_BAD : TXT_DIM,
                                    "%.1f (%+.1f)", v, v - base);
            ImGui::TableNextColumn();
            if (k != 0) {
                ImGui::PushID(k);
                if (ImGui::SmallButton("взять")) {
                    mt::apply_algo(a.s, vars[k].setup.s);
                    for (int i = 0; i < (int)(sizeof(RES_OPTS) / sizeof(RES_OPTS[0])); ++i)
                        if (RES_OPTS[i].scale > 0 && std::fabs(RES_OPTS[i].scale - vars[k].setup.scale) < 1e-3f) a.resIdx = i;
                    set_resolution(a);
                    a.profIdx = -1;
                    changed = true;
                }
                ImGui::SetItemTooltip("Поставить в приложении настройки этого варианта");
                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
}

// окно мультитеста поверх видео: слева что перебирать, справа ход и итоги
void window_multitest(App& a, ImVec2 pos, ImVec2 size) {
    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(size);
    if (!ImGui::Begin("Мультитест: подбор настроек на папке видео", &a.mtOpen,
                      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                          ImGuiWindowFlags_NoSavedSettings)) {
        ImGui::End();
        return;
    }
    bool changed = false;
    const float w = ImGui::GetContentRegionAvail().x;
    if (w > 700) {
        const float lw = std::min(w * 0.5f, ImGui::GetFontSize() * 30);
        ImGui::BeginChild("mtsetup", ImVec2(lw, 0));
        mt_setup(a);
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("mtprog", ImVec2(0, 0));
        mt_progress(a, changed);
        ImGui::EndChild();
    } else {
        ImGui::BeginChild("mtall", ImVec2(0, 0));
        mt_setup(a);
        heading("Ход и итоги");
        mt_progress(a, changed);
        ImGui::EndChild();
    }
    ImGui::End();
    if (changed) a.pipe.setSettings(a.s);
}

void panel_twocam(App& a);
void panel_controls(App& a) {
    static const char* modes[] = {"Видео", "Две камеры"};
    static const char* modeTips[] = {"Обработка видео и записей с камеры.",
                                     "Эмулятор: широкая камера находит цели и указывает их точной 640x480 на "
                                     "поворотке, точная захватывает и ведёт сама."};
    segmented("appmode", modes, 2, a.appMode, 0, modeTips);
    ImGui::Spacing();
    if (a.appMode == 1) {
        panel_twocam(a);
        return;
    }
    bool changed = false;
    panel_main(a, changed);
    ImGui::Spacing();
    ImGui::Spacing();
    if (ImGui::BeginTabBar("settings")) {
        tab("Видео", [&] { tab_video(a, changed); });
        tab("Железо", [&] { tab_hardware(a, changed); });
        tab("Алгоритм", [&] { tab_algorithm(a, changed); });
        tab("Тонко", [&] { tab_fine(a, changed); });
        tab("Вид", [&] { tab_view(a); });
        tab("Разметка", [&] { panel_annotate(a); });
        ImGui::EndTabBar();
    }
    if (changed) a.pipe.setSettings(a.s);
}

void row_stat(const char* name, const Stat& s) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::TextUnformatted(name);
    const double v[4] = {s.mean, s.p50, s.p95, s.max};
    for (int i = 0; i < 4; ++i) {
        ImGui::TableSetColumnIndex(i + 1);
        ImGui::Text(v[i] < 10 ? "%.2f" : "%.1f", v[i]);
    }
}

// плитка ключевой метрики: подпись, крупное значение, пояснение
void kpi(const char* title, const char* value, ImVec4 col, const char* sub, float w) {
    const ImGuiStyle& st = ImGui::GetStyle();
    const float h = ImGui::GetTextLineHeight() * 2 + FONT_BASE * 1.8f * st.FontScaleDpi + st.ItemSpacing.y * 2 +
                    st.WindowPadding.y * 2;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(32, 35, 42, 255));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5);
    ImGui::BeginChild(title, ImVec2(w, h), ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::TextDisabled("%s", title);
    {
        FontSize f(1.8f);
        ImGui::TextColored(col, "%s", value);
    }
    ImGui::TextDisabled("%s", sub);
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

// крупный статус захвата: на что сейчас наведена система
void lock_banner(const Result* r, const Metrics& m) {
    const Track* lk = nullptr;
    if (r)
        for (const auto& t : r->tracks)
            if (t.id == r->lockId) lk = &t;
    const ImGuiStyle& st = ImGui::GetStyle();
    const float h = FONT_BASE * 1.6f * st.FontScaleDpi + ImGui::GetTextLineHeight() + st.ItemSpacing.y +
                    st.WindowPadding.y * 2;
    const bool hard = lk && r->lockState == LockState::Hard;
    const ImVec4 bg = hard ? ImVec4(0.62f, 0.08f, 0.22f, 1) : lk ? ImVec4(0.45f, 0.10f, 0.40f, 1) : ImVec4(0.13f, 0.14f, 0.17f, 1);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, bg);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5);
    ImGui::BeginChild("lock", ImVec2(-FLT_MIN, h), ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    {
        FontSize f(1.6f);
        if (lk && hard) ImGui::TextColored(ImVec4(1, 0.85f, 0.98f, 1), "ЗАХВАЧЕН ДРОН  #%d", lk->id);
        else if (lk) ImGui::TextColored(ImVec4(1, 0.85f, 0.98f, 1), "ЗАХВАТ  #%d %s", lk->id, class_ru(*lk));
        else ImGui::TextColored(TXT_DIM, "Захвата нет");
    }
    // ожидание размера: видимый размер и приближается ли цель
    char sz[64] = "";
    if (lk && lk->lsN >= 3) {
        const float pct = 100.f * (std::exp(lk->dls) - 1.f);
        snprintf(sz, sizeof(sz), "  ·  %.0f пикс.%s", lk->expectedSize(0),
                 pct > 0.5f ? ", приближается" : pct < -0.5f ? ", удаляется" : "");
    }
    if (lk && lk->miss) ImGui::TextColored(TXT_WARN, "на прогнозе %d кадр.%s", lk->miss, sz);
    else if (lk && hard) ImGui::Text("проверен, на другие цели не отвлекается%s", sz);
    else if (lk && r->verifyNeed > 0) ImGui::Text("проверка, дрон ли это: %d из %d кадров%s", r->verify, r->verifyNeed, sz);
    else if (lk) ImGui::Text("цель ведётся%s", sz);
    else ImGui::TextDisabled("нет подтверждённой цели");
    char pct[48];
    snprintf(pct, sizeof(pct), "захват в %.0f%% кадров", m.lockPct);
    ImGui::SameLine(std::max(ImGui::GetItemRectMax().x - ImGui::GetWindowPos().x + 8, ImGui::GetWindowWidth() - st.WindowPadding.x -
                                                            ImGui::CalcTextSize(pct).x));
    ImGui::TextDisabled("%s", pct);
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void metrics_targets(App& a, const Metrics& m, const Result* r) {
    if (!r) {
        ImGui::TextDisabled("нет обработанного кадра");
        return;
    }
    int dr = 0, bi = 0, un = 0, cl = (int)r->clutter.size();
    for (const auto& t : r->tracks) {
        if (t.clutter) {
            ++cl;
            continue;
        }
        dr += t.cls == TargetClass::Drone;
        bi += t.cls == TargetClass::Bird;
        un += t.cls == TargetClass::Unknown;
    }
    heading("На кадре");
    if (ImGui::BeginTable("counts", 3, ImGuiTableFlags_SizingStretchSame)) {
        auto cell = [](ImU32 c, const char* name, int n) {
            ImGui::TableNextColumn();
            {
                FontSize f(1.4f);
                ImGui::TextColored(ImColor(c).Value, "%d", n);
            }
            ImGui::TextDisabled("%s", name);
        };
        cell(COL_DRONE, "дронов", dr);
        cell(COL_BIRD, "птиц", bi);
        cell(COL_UNKNOWN, "неясно", un);
        cell(COL_CLUTTER, "помех", cl);
        cell(COL_NOISE, "шум", (int)r->noise.size());
        cell(IM_COL32(150, 150, 160, 255), "кандидатов", (int)r->cands.size());
        ImGui::EndTable();
    }
    if (r->learning) ImGui::TextColored(TXT_WARN, "учится маска неба...");

    heading("Состояние алгоритма");
    if (r->motionOn) {
        if (r->camValid) ImGui::Text("камера: сдвиг %+.1f, %+.1f пикс./кадр", r->camDx, r->camDy);
        else ImGui::TextDisabled("камера: сдвиг не определён (нет текстуры)");
    } else {
        ImGui::TextDisabled("движение камеры не учитывается");
    }
    if (r->adapt && r->scene.ready) {
        const SceneState& sc = r->scene;
        ImGui::Text("сцена: %s, %s; неба %.0f%% кадра", light_name(sc.light), terrain_name(sc.terrain),
                    100 * sc.skyFrac);
        ImGui::TextDisabled("шум %.1f, облака %.0f%%, ветер у кромки %.1f пикс., осадки %.1f, дымка %.0f%%", sc.noise,
                            100 * sc.clouds, sc.wind, sc.precip, 100 * sc.haze);
        if (!r->sceneWhy.empty()) ImGui::TextWrapped("подстроено: %s", r->sceneWhy.c_str());
        if (!sc.lastEvent.empty()) ImGui::TextColored(TXT_WARN, "перемена: %s", sc.lastEvent.c_str());
    } else if (r->adapt) {
        ImGui::TextDisabled("сцена: оценивается...");
    }
    if (r->autoThr) ImGui::Text("шум неба %.2f, порог %.1f", r->noiseLvl, r->thr);
    else ImGui::Text("порог %.0f (постоянный)", r->thr);
    ImGui::TextDisabled("масштабов %d, весь кадр раз в %d%s", r->scales, r->fullEvery,
                        r->coarseEvery > 1 ? (", крупные раз в " + std::to_string(r->coarseEvery)).c_str() : "");
    if (r->govLevel >= 0)
        ImGui::TextColored(r->govLevel ? TXT_WARN : TXT_DIM, "авто-упрощение: ступень %d из %d — %s", r->govLevel,
                           LoadGovernor::MAX_LEVEL, LoadGovernor::describe(r->govLevel));
    if (a.s.slowdown > 1)
        ImGui::TextColored(TXT_WARN, "имитация: %s, обработка x%.1f", DEVICES[a.devIdx].name, a.s.slowdown);
    if (r->staticOn) ImGui::TextColored(ImColor(COL_STATIC).Value, "запомнено помех: %d", (int)r->staticCells.size());
    if (!m.staticMsg.empty()) ImGui::TextDisabled("%s", m.staticMsg.c_str());

    if (a.showCrops && a.cropTex) {
        heading("Цели крупно");
        const auto targets = top_targets(*r);
        if (targets.empty()) ImGui::TextDisabled("целей нет");
        const float sz = std::min(ImGui::GetContentRegionAvail().x / 2 - 8, 150.f);
        for (size_t i = 0; i < targets.size(); ++i) {
            const Track& t = *targets[i];
            ImGui::BeginGroup();
            const ImVec2 uv0(0, (float)i / CROP_N), uv1(1, (float)(i + 1) / CROP_N);
            const ImVec2 p = ImGui::GetCursorScreenPos();
            ImGui::Image((ImTextureID)(intptr_t)a.cropTex, ImVec2(sz, sz), uv0, uv1);
            ImGui::GetWindowDrawList()->AddRect(p, ImVec2(p.x + sz, p.y + sz), class_color(t), 0.0f, ImDrawFlags_None, 2.0f);
            ImGui::TextColored(ImColor(class_color(t)).Value, "#%d %s", t.id, class_ru(t));
            ImGui::TextDisabled("c=%.0f v=%.1f откл=%.1f", t.contrast, t.speed, t.resid);
            ImGui::EndGroup();
            if (i % 2 == 0) ImGui::SameLine();
        }
    }
}

void metrics_quality(App& a, const Metrics& m) {
    heading("Качество по разметке");
    if (!m.haveGt) {
        ImGui::TextWrapped("Разметки нет. Её можно сделать на вкладке «Разметка» слева или положить рядом с видео "
                           "файл <видео>.gt.json.");
    } else if (m.eval.frames == 0) {
        ImGui::TextDisabled(a.pipe.paused() ? "считается по обработанным кадрам с разметкой"
                                            : "считается после первого прохода видео");
    } else if (ImGui::BeginTable("quality", 2, ImGuiTableFlags_SizingStretchSame)) {
        auto cell = [](const char* name, const char* fmt, double v, ImVec4 c) {
            ImGui::TableNextColumn();
            {
                FontSize f(1.4f);
                ImGui::TextColored(c, fmt, v);
            }
            ImGui::TextDisabled("%s", name);
        };
        const double rd = 100 * m.eval.recall(0), fp = m.eval.falsePerFrame();
        cell("найдено дронов", "%.1f%%", rd, rd >= 90 ? TXT_OK : rd >= 70 ? TXT_WARN : TXT_BAD);
        cell("найдено птиц", "%.1f%%", 100 * m.eval.recall(1), TXT_DIM);
        cell("ложных треков на кадр", "%.2f", fp, fp <= 0.1 ? TXT_OK : fp <= 0.5 ? TXT_WARN : TXT_BAD);
        cell("класс дрона верный", "%.0f%%", 100 * m.eval.clsAccuracy(0), TXT_DIM);
        cell("класс птицы верный", "%.0f%%", 100 * m.eval.clsAccuracy(1), TXT_DIM);
        ImGui::EndTable();
    }

    heading("Сохранить");
    const float bw = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2;
    ImGui::BeginDisabled(!a.pipe.isOpen());
    if (ImGui::Button("CSV по кадрам", ImVec2(bw, 0))) {
        std::string p = a.pipe.path() + ".metrics.csv";
        a.note = a.pipe.exportCsv(p) ? "сохранено: " + p : "не удалось сохранить " + p;
    }
    ImGui::SameLine();
    if (ImGui::Button("Отчёт JSON", ImVec2(bw, 0))) {
        std::string p = a.pipe.path() + ".report.json";
        FILE* f = open_file(p, "w");
        if (f) {
            fprintf(f, "%s\n", a.pipe.reportJson().c_str());
            fclose(f);
        }
        a.note = f ? "сохранено: " + p : "не удалось сохранить " + p;
    }
    if (ImGui::Button("Вырезки целей", ImVec2(bw, 0))) {
        std::string dir = a.pipe.path();
        const size_t dot = dir.find_last_of('.'), sl = dir.find_last_of("/\\");
        if (dot != std::string::npos && (sl == std::string::npos || dot > sl)) dir.resize(dot);
        a.pipe.exportCrops(dir + "_вырезки", a.note);
    }
    ImGui::EndDisabled();
    help_marker("Цели из последних обработанных кадров (история, до 600 кадров) — картинки 48x48 и таблица "
                "crops.csv с признаками трека (скорость, отклонение от траектории, мигание контраста), классом, "
                "который дал алгоритм, и правдой по разметке. Набор для настройки классификатора дрон/птица "
                "и для обучения нейросети. Папка <видео>_вырезки рядом с видео.");
    if (!a.note.empty()) ImGui::TextWrapped("%s", a.note.c_str());
}

// очередь целей: несколько в кадре — по весомости (уверенности, что это дрон); оператор
// переключает захват на следующую (Tab) или выбирает щелчком по цели на кадре
void lock_queue(App& a, const Result* r) {
    if (!r || r->lockQueue.empty()) return;
    if (ImGui::SmallButton("Следующая цель [Tab]")) a.pipe.nextTarget();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Захват — на следующую цель очереди (по кругу). Выбранная вручную держится,\n"
                          "пока жива: более весомые её не перебивают. Выбрать цель — щелчок по ней на кадре.");
    ImGui::SameLine();
    if (r->manualLock) ImGui::TextColored(TXT_WARN, "выбрана вручную");
    else ImGui::TextDisabled("выбор автоматический");
    std::string line = "очередь:";
    for (size_t i = 0; i < r->lockQueue.size() && i < 6; ++i)
        for (const auto& t : r->tracks)
            if (t.id == r->lockQueue[i]) {
                char b[64];
                snprintf(b, sizeof(b), "%s %zu. #%d %s %.0f%s", i ? " ·" : "", i + 1, t.id, class_ru(t), t.lockW,
                         t.id == r->lockId ? " ◄" : "");
                line += b;
            }
    if (r->lockQueue.size() > 6) line += " …";
    ImGui::TextWrapped("%s", line.c_str());
}

// ------------------------------------------------------------ эмулятор двух камер

// серый кадр на свою текстуру (IYUV, как основная); ds — уменьшение минимумом по блоку
bool upload_gray(App& a, SDL_Texture*& tex, int& tw, int& th, const uint8_t* src, int w, int h, int ds) {
    const int nw = (w / ds) & ~1, nh = (h / ds) & ~1;
    if (!tex || nw != tw || nh != th) {
        if (tex) SDL_DestroyTexture(tex);
        SDL_SetYUVConversionMode(SDL_YUV_CONVERSION_JPEG);
        tex = SDL_CreateTexture(a.ren, SDL_PIXELFORMAT_IYUV, SDL_TEXTUREACCESS_STREAMING, nw, nh);
        if (!tex) return false;
        SDL_SetTextureScaleMode(tex, SDL_ScaleModeLinear);
        tw = nw;
        th = nh;
        void* px;
        int pitch;
        if (SDL_LockTexture(tex, nullptr, &px, &pitch) == 0) {
            memset((uint8_t*)px + (size_t)pitch * th, 128, (size_t)(pitch / 2) * (th / 2) * 2);
            SDL_UnlockTexture(tex);
        }
    }
    void* px;
    int pitch;
    if (SDL_LockTexture(tex, nullptr, &px, &pitch) != 0) return false;
    for (int y = 0; y < th; ++y) {
        uint8_t* o = (uint8_t*)px + (size_t)y * pitch;
        if (ds == 1) {
            memcpy(o, src + (size_t)y * w, tw);
            continue;
        }
        for (int x = 0; x < tw; ++x) {
            uint8_t m = 255;
            for (int yy = 0; yy < ds; ++yy) {
                const uint8_t* s = src + (size_t)(y * ds + yy) * w + x * ds;
                for (int xx = 0; xx < ds; ++xx) m = std::min(m, s[xx]);
            }
            o[x] = m;
        }
    }
    SDL_UnlockTexture(tex);
    return true;
}

void emu_start(App& a) {
    twocam::Config c = a.emuCfg;
    static const int res[][2] = {{4608, 3456}, {2304, 1728}};
    c.wideW = res[a.emuRes][0];
    c.wideH = res[a.emuRes][1];
    static const float speeds[] = {0.25f, 0.5f, 1, 2, 0};
    c.simSpeed = speeds[a.emuSpeed];
    a.pipe.setPaused(true);   // видео не отнимает процессор у эмулятора
    a.emu.start(c);
    a.emuWSeq = -1;
}

void panel_twocam(App& a) {
    twocam::Config& c = a.emuCfg;
    const float bigH = ImGui::GetFrameHeight() * 1.6f;
    {
        FontSize f(1.2f);
        push_accent();
        if (ImGui::Button(a.emu.running() ? "Перезапустить" : "Запустить эмулятор", ImVec2(-FLT_MIN, bigH))) emu_start(a);
        pop_accent();
    }
    ImGui::SetItemTooltip("Сцена, камеры и поворотка — с настройками ниже. Цели и шум — по номеру сцены.");
    if (a.emu.running()) {
        const float w = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2;
        if (ImGui::Button(a.emu.paused() ? "Продолжить" : "Пауза", ImVec2(w, 0))) a.emu.setPaused(!a.emu.paused());
        ImGui::SameLine();
        if (ImGui::Button("Остановить", ImVec2(w, 0))) a.emu.stop();
    }
    static const char* sp[] = {"x0.25", "x0.5", "x1", "x2", "макс"};
    label("Скорость времени", "x1 — как в жизни. «макс» — так быстро, как успевает ПК; результат тот же, "
                              "время в эмуляторе модельное.");
    if (segmented("emusp", sp, 5, a.emuSpeed)) {
        static const float speeds[] = {0.25f, 0.5f, 1, 2, 0};
        a.emu.setSpeed(speeds[a.emuSpeed]);
    }

    heading("Переключение целей");
    static const char* modes[] = {"Вручную", "По очереди", "Ближайшая"};
    static const char* mtips[] = {
        "Точная берёт цель, подтверждённую широкой как дрон, и держит её. Другую — щелчком по цели на "
        "широкой, кнопкой справа или Tab.",
        "Ведёт цель заданное время, потом переходит к следующей в очереди широкой (по весомости).",
        "Ведёт цель заданное время, потом переходит к той, до которой меньше поворачиваться."};
    int m = (int)c.switchMode;
    if (segmented("emumode", modes, 3, m, 0, mtips)) {
        c.switchMode = (twocam::SwitchMode)m;
        a.emu.setSwitchMode(c.switchMode);
    }
    if (ImGui::Button("Следующая цель (Tab)", ImVec2(-FLT_MIN, 0))) a.emu.nextTarget();
    label("Вести цель, с", "В авто-режимах: столько точная ведёт цель, прежде чем перейти к следующей.");
    ImGui::SliderFloat("##dwell", &c.dwell, 1, 30, "%.0f с");
    ImGui::Checkbox("Только дроны (птиц широкой пропускать)", &c.dronesOnly);

    heading("Точная камера на поворотке");
    label("Максимальная скорость поворота", "По каждой оси. Маленькая скорость — дольше до цели и легче потерять "
                                            "быструю близкую цель.");
    ImGui::SliderFloat("##rate", &c.maxRate, 5, 180, "%.0f °/с");
    label("Ускорение поворотки");
    ImGui::SliderFloat("##acc", &c.maxAccel, 20, 1000, "%.0f °/с²", ImGuiSliderFlags_Logarithmic);
    label("Зум (поле зрения 20° без зума)", "Зум сужает поле точной: ошибка захвата в градусах падает во столько же "
                                          "раз (x2 — 0.015°, x4 — 0.008° на эмуляторе), но захват дольше, и близкую "
                                          "быструю цель легче упустить из кадра. Дальние цели, которых не видит широкая, "
                                          "зум не вернёт: точную наводит широкая.");
    {
        static const char* zs[] = {"x1", "x2", "x4"};
        int z = c.narrowFov > 15 ? 0 : c.narrowFov > 7.5f ? 1 : 2;
        if (segmented("emuzoom", zs, 3, z)) c.narrowFov = 20.f / (1 << z);
    }
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::SliderFloat("##nfov", &c.narrowFov, 2, 40, "%.1f°");
    label("Частота кадров");
    ImGui::SliderFloat("##nfps", &c.narrowFps, 15, 240, "%.0f к/с");
    label("Задержка передачи цели", "От кадра широкой до команды поворотке: обработка, связь между Pi.");
    ImGui::SliderFloat("##lat", &c.handoffLatency, 0, 0.5f, "%.2f с");
    label("Дискретность датчиков угла");
    ImGui::SliderFloat("##enc", &c.encoderDeg, 0.001f, 0.1f, "%.3f°", ImGuiSliderFlags_Logarithmic);

    heading("Алгоритм точной");
    ImGui::Checkbox("Сдвиг кадра по датчикам угла", &c.encoderMotion);
    help_marker("Поворотка знает, на сколько повернулась: сдвиг картинки считается по углам точно и "
                "мгновенно. Выключено — оценка по самому кадру, как при съёмке с рук.");
    ImGui::Checkbox("Поворот со скоростью цели", &c.feedForward);
    help_marker("Поворотка сразу идёт со скоростью цели (по сообщению широкой или по треку точной) и "
                "с упреждением на кадр, а ошибку до центра добирает. Выключено — только по ошибке.");
    ImGui::Checkbox("Окно поиска от широкой", &c.windowFocus);
    help_marker("В окне, где по прогнозу должна быть цель, порог ниже, и берётся ближайший к прогнозу "
                "трек. Выключено — самый первый трек точной, где бы он ни был.");

    heading("Широкая камера");
    static const char* rs[] = {"4608x3456", "2304x1728"};
    label("Разрешение", "16 Мп — как настоящая. Половина — если ПК не успевает в реальном времени.");
    segmented("emures", rs, 2, a.emuRes);
    label("Поле зрения");
    ImGui::SliderFloat("##wfov", &c.wideFov, 60, 170, "%.0f°");
    ImGui::Checkbox("Рыбий глаз", &c.wideFisheye);
    label("Частота кадров");
    ImGui::SliderFloat("##wfps", &c.wideFps, 2, 30, "%.0f к/с");
    label("Угол места оси");
    ImGui::SliderFloat("##wel", &c.wideEl, 0, 60, "%.0f°");

    heading("Сцена");
    label("Дронов");
    ImGui::SliderInt("##dr", &c.drones, 0, 10);
    label("Птиц");
    ImGui::SliderInt("##bi", &c.birds, 0, 10);
    label("Дальность, м");
    ImGui::DragFloatRange2("##rng", &c.rangeMin, &c.rangeMax, 10, 50, 5000, "от %.0f", "до %.0f");
    label("Скорость целей, м/с");
    ImGui::DragFloatRange2("##spd", &c.speedMin, &c.speedMax, 0.5f, 0, 80, "от %.0f", "до %.0f");
    label("Размер дрона, м");
    ImGui::SliderFloat("##dsz", &c.droneSize, 0.1f, 3, "%.2f");
    label("Облачность");
    ImGui::SliderFloat("##cl", &c.clouds, 0, 1, "%.2f");
    label("Шум датчика");
    ImGui::SliderFloat("##ns", &c.noise, 0, 8, "%.1f");
    label("Видимость, м");
    ImGui::SliderFloat("##vis", &c.visibility, 500, 20000, "%.0f", ImGuiSliderFlags_Logarithmic);
    label("Номер сцены");
    int seed = (int)c.seed;
    if (ImGui::InputInt("##seed", &seed)) c.seed = (uint32_t)std::max(1, seed);
    ImGui::TextDisabled("Сцена, камеры и алгоритм применяются при перезапуске.");
    ImGui::Checkbox("Показывать правду (где цели на самом деле)", &a.emuTruth);
}

static ImU32 cls_col(int cls) {
    return cls == 1 ? COL_DRONE : cls == 2 ? COL_BIRD : cls == 3 ? COL_CLUTTER : COL_UNKNOWN;
}

void draw_twocam(App& a) {
    const twocam::View& v = a.emuView;
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, ImVec2(p0.x + avail.x, p0.y + avail.y), IM_COL32(18, 18, 20, 255));
    if (!v.wide || !a.emuWTex) {
        const char* msg = a.emu.running() ? "Готовим сцену..." : "Запустите эмулятор слева";
        const ImVec2 ts = ImGui::CalcTextSize(msg);
        dl->AddText(ImVec2(p0.x + (avail.x - ts.x) / 2, p0.y + (avail.y - ts.y) / 2), IM_COL32(170, 170, 180, 255), msg);
        ImGui::Dummy(avail);
        return;
    }
    const Frame& f = *v.wide->frame;
    // широкая сверху, точная снизу; на широком экране — рядом
    const float gap = 8, capH = ImGui::GetTextLineHeightWithSpacing();
    const bool side = avail.x > avail.y * 1.6f;
    ImVec2 wSz, nSz, wPos, nPos;
    if (side) {
        const float ww = (avail.x - gap) * 0.58f;
        const float ws = std::min(ww / f.w, (avail.y - capH) / f.h);
        wSz = ImVec2(f.w * ws, f.h * ws);
        const float nwid = avail.x - gap - wSz.x;
        const float ns = std::min(nwid / v.nw, (avail.y - capH) / v.nh);
        nSz = ImVec2(v.nw * ns, v.nh * ns);
        wPos = ImVec2(p0.x, p0.y + capH);
        nPos = ImVec2(p0.x + wSz.x + gap, p0.y + capH);
    } else {
        const float hTop = (avail.y - 2 * capH - gap) * 0.55f;
        const float ws = std::min(avail.x / f.w, hTop / f.h);
        wSz = ImVec2(f.w * ws, f.h * ws);
        const float hBot = avail.y - 2 * capH - gap - wSz.y;
        const float ns = std::min(avail.x / v.nw, hBot / v.nh);
        nSz = ImVec2(v.nw * ns, v.nh * ns);
        wPos = ImVec2(p0.x, p0.y + capH);
        nPos = ImVec2(p0.x, wPos.y + wSz.y + gap + capH);
    }
    const ImU32 txt = IM_COL32(200, 205, 215, 255);
    char cap[160];
    snprintf(cap, sizeof(cap), "Широкая %dx%d, %.0f°, %.0f к/с — щелчок по цели: навести точную", f.w, f.h,
             a.emuCfg.wideFov, a.emuCfg.wideFps);
    dl->AddText(ImVec2(wPos.x, wPos.y - capH), txt, cap);
    snprintf(cap, sizeof(cap), "Точная %dx%d, %.1f°: %s", v.nw, v.nh, a.emuCfg.narrowFov, twocam::state_name(v.state));
    dl->AddText(ImVec2(nPos.x, nPos.y - capH), txt, cap);

    // --- широкая
    const float ks = wSz.x / f.w;
    auto W = [&](float x, float y) { return ImVec2(wPos.x + x * ks, wPos.y + y * ks); };
    dl->AddImage((ImTextureID)(intptr_t)a.emuWTex, wPos, ImVec2(wPos.x + wSz.x, wPos.y + wSz.y));
    dl->PushClipRect(wPos, ImVec2(wPos.x + wSz.x, wPos.y + wSz.y), true);
    if (v.footprint.size() >= 3) {   // куда смотрит точная
        std::vector<ImVec2> poly;
        for (const auto& p : v.footprint) poly.push_back(W(p.first, p.second));
        dl->AddPolyline(poly.data(), (int)poly.size(), IM_COL32(90, 220, 255, 230), ImDrawFlags_Closed, 2.f);
    }
    if (a.emuTruth)
        for (const auto& t : v.truth)
            if (t.wu >= 0) {
                const ImVec2 q = W(t.wu, t.wv);
                dl->AddLine(ImVec2(q.x - 4, q.y), ImVec2(q.x + 4, q.y), COL_GT);
                dl->AddLine(ImVec2(q.x, q.y - 4), ImVec2(q.x, q.y + 4), COL_GT);
            }
    ImGuiIO& io = ImGui::GetIO();
    const bool clicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
                         ImGui::IsMouseHoveringRect(wPos, ImVec2(wPos.x + wSz.x, wPos.y + wSz.y));
    int pickId = 0;
    float pickD = 25;
    for (const auto& tr : v.wide->tracks) {
        if (tr.clutter) continue;
        const ImVec2 q = W(tr.x, tr.y);
        const bool as = tr.id == v.assigned;
        dl->AddCircle(q, as ? 13.f : 8.f, cls_col((int)tr.cls), 0, as ? 3.f : 1.5f);
        char id[16];
        snprintf(id, sizeof(id), "%d", tr.id);
        dl->AddText(ImVec2(q.x + 9, q.y - 16), cls_col((int)tr.cls), id);
        if (clicked) {
            const float d = std::hypot(io.MousePos.x - q.x, io.MousePos.y - q.y);
            if (d < pickD) {
                pickD = d;
                pickId = tr.id;
            }
        }
    }
    if (pickId) a.emu.selectTarget(pickId);
    dl->PopClipRect();

    // --- точная
    const float kn = nSz.x / v.nw;
    auto N = [&](float x, float y) { return ImVec2(nPos.x + x * kn, nPos.y + y * kn); };
    if (a.emuNTex) dl->AddImage((ImTextureID)(intptr_t)a.emuNTex, nPos, ImVec2(nPos.x + nSz.x, nPos.y + nSz.y));
    dl->PushClipRect(nPos, ImVec2(nPos.x + nSz.x, nPos.y + nSz.y), true);
    const ImVec2 c = N(v.nw * 0.5f, v.nh * 0.5f);   // ось точной
    dl->AddLine(ImVec2(c.x - 12, c.y), ImVec2(c.x + 12, c.y), IM_COL32(255, 255, 255, 120));
    dl->AddLine(ImVec2(c.x, c.y - 12), ImVec2(c.x, c.y + 12), IM_COL32(255, 255, 255, 120));
    if (v.winOn && v.win.visible && v.state != twocam::NState::Track)
        dl->AddCircle(N(v.win.u, v.win.v), v.win.radius * kn, IM_COL32(90, 220, 255, 200), 0, 2);
    if (a.emuTruth)
        for (const auto& t : v.truth)
            if (t.inNarrow) dl->AddCircle(N(t.nu, t.nv), 16, COL_GT, 0, 1);
    for (const auto& tr : v.ntracks) {
        if (tr.clutter) continue;
        const ImVec2 q = N(tr.x, tr.y);
        if (tr.id == v.nlock) {
            dl->AddRect(ImVec2(q.x - 18, q.y - 18), ImVec2(q.x + 18, q.y + 18), COL_LOCK, 0.f, 0, 3.f);
            dl->AddText(ImVec2(q.x + 20, q.y - 20), COL_LOCK, tr.cls == TargetClass::Drone ? "дрон" : tr.cls == TargetClass::Bird ? "птица" : "цель");
        } else {
            dl->AddCircle(q, 7, cls_col((int)tr.cls), 0, 1.5f);
        }
    }
    dl->PopClipRect();
    ImGui::Dummy(avail);
}

void panel_twocam_metrics(App& a) {
    const twocam::View& v = a.emuView;
    const twocam::Stats& s = v.st;
    const float tw = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2;
    char val[48], sub[96];
    const bool track = v.state == twocam::NState::Track;
    snprintf(val, sizeof(val), "%s", twocam::state_name(v.state));
    if (v.assigned) snprintf(sub, sizeof(sub), "цель %d широкой", v.assigned);
    else snprintf(sub, sizeof(sub), "целей у широкой: %d", (int)v.targets.size());
    kpi("Точная", val, track ? TXT_OK : v.assigned ? TXT_WARN : TXT_DIM, sub, tw);
    ImGui::SameLine();
    snprintf(val, sizeof(val), "%.0f%%", s.simT > 0 ? 100 * s.trackTime / s.simT : 0);
    snprintf(sub, sizeof(sub), "на назначенной %.0f%%", s.trackTime > 0 ? 100 * s.trackCorrect / s.trackTime : 0);
    kpi("Время ведения", val, ImVec4(0.85f, 0.88f, 0.95f, 1), sub, tw);
    snprintf(val, sizeof(val), "%.2f с", s.acqN ? s.acqSum / s.acqN : 0);
    snprintf(sub, sizeof(sub), "в кадре через %.2f с", s.slewN ? s.slewSum / s.slewN : 0);
    kpi("До захвата точной", val, ImVec4(0.85f, 0.88f, 0.95f, 1), sub, tw);
    ImGui::SameLine();
    auto med = [](std::vector<float> e) {
        if (e.size() > 3000) e.erase(e.begin(), e.end() - 3000);
        if (e.empty()) return 0.f;
        std::nth_element(e.begin(), e.begin() + e.size() / 2, e.end());
        return e[e.size() / 2];
    };
    const float el = med(s.errLock), ea = med(s.errAxis);
    snprintf(val, sizeof(val), "%.3f°", el);
    snprintf(sub, sizeof(sub), "%.1f пикс. точной, ось %.2f°", el / (a.emuCfg.narrowFov / std::max(1, v.nw)), ea);
    kpi("Ошибка захвата", val, ImVec4(0.85f, 0.88f, 0.95f, 1), sub, tw);
    ImGui::Text("ось: аз %.2f°, угол места %.2f°", v.axis.az, v.axis.el);
    ImGui::Text("скорость поворота: %.1f / %.1f °/с (предел %.0f)", v.rateAz, v.rateEl, a.emuCfg.maxRate);
    ImGui::Text("назначений %d, взяла %d, не ту %d, потерь %d", s.assignments, s.acquired, s.wrongLocks, s.losses);
    if (s.simT > 0)
        ImGui::Text("нечего вести %.0f%% времени, ложная цель широкой %.0f%%", 100 * s.idleTime / s.simT,
                    100 * s.falseTarget / s.simT);
    ImGui::TextDisabled("время %.1f с; широкая %.1f мс/кадр, точная %.2f мс/кадр", s.simT,
                        s.wideFrames ? s.wideMs / s.wideFrames : 0, s.narrowFrames ? s.narrowMs / s.narrowFrames : 0);
    if (!v.status.empty()) ImGui::TextColored(TXT_WARN, "%s", v.status.c_str());

    heading("Цели широкой");
    if (ImGui::BeginTable("emutg", 4, ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("цель");
        ImGui::TableSetupColumn("класс");
        ImGui::TableSetupColumn("аз / угол места");
        ImGui::TableSetupColumn("");
        ImGui::TableHeadersRow();
        for (const auto& t : v.targets) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text(t.id == v.assigned ? "%d  <" : "%d", t.id);
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(cls_col(t.cls)), "%s%s",
                               t.cls == 1 ? "дрон" : t.cls == 2 ? "птица" : "?", t.hard ? ", жёсткий" : "");
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.1f° / %.1f°", t.rep.az, t.rep.el);
            ImGui::TableSetColumnIndex(3);
            ImGui::PushID(t.id);
            if (t.id != v.assigned && ImGui::SmallButton("навести")) a.emu.selectTarget(t.id);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    heading("События");
    ImGui::BeginChild("emuev", ImVec2(0, 0));
    for (auto it = v.events.rbegin(); it != v.events.rend(); ++it) ImGui::TextWrapped("%6.2f  %s", it->t, it->text.c_str());
    ImGui::EndChild();
}

void panel_metrics(App& a, const Metrics& m, const Result* r) {
    const Settings s = a.pipe.settings();
    const double fpsSrc = (s.targetFps > 0 ? s.targetFps : m.sourceFps) / std::max(1, s.frameStep);
    const bool rt = s.pace == Pace::Realtime;
    const double dropPct = m.processed + m.dropped ? 100.0 * m.dropped / (m.processed + m.dropped) : 0;
    const double detFps = m.detect.mean > 0 ? 1000.0 / m.detect.mean : 0;

    // ключевые метрики: успевает ли алгоритм за камерой и чего это стоит
    const float tw = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2;
    char v[32], sub[64];
    snprintf(v, sizeof(v), "%.1f к/с", m.procFps);
    if (rt && fpsSrc > 0) snprintf(sub, sizeof(sub), "камера %.0f к/с", fpsSrc);
    else snprintf(sub, sizeof(sub), "максимально быстро");
    kpi("Обработка", v, m.processed == 0 ? TXT_DIM : dropPct < 2 ? TXT_OK : TXT_WARN, sub, tw);
    ImGui::SameLine();
    snprintf(v, sizeof(v), "%.1f%%", dropPct);
    snprintf(sub, sizeof(sub), "%ld из %ld кадров", m.dropped, m.processed + m.dropped);
    kpi("Пропущено", v, m.processed == 0 ? TXT_DIM : dropPct < 2 ? TXT_OK : dropPct < 10 ? TXT_WARN : TXT_BAD, sub, tw);
    snprintf(v, sizeof(v), "%.1f мс", m.detect.mean);
    snprintf(sub, sizeof(sub), "до %.0f к/с, потоков %d", detFps, m.threads);
    kpi("Детектор", v, ImVec4(0.85f, 0.88f, 0.95f, 1), sub, tw);
    ImGui::SameLine();
    snprintf(v, sizeof(v), "%.0f%%", a.cpuPct);
    snprintf(sub, sizeof(sub), "≈%.1f ядра, %.0f МБ", a.cpuPct / 100.0 * plat::logical_cpus(),
             plat::process_memory_mb());
    kpi("CPU", v, a.cpuPct > 90 ? TXT_WARN : ImVec4(0.85f, 0.88f, 0.95f, 1), sub, tw);
    lock_banner(r, m);
    lock_queue(a, r);
    if (!m.status.empty()) ImGui::TextColored(TXT_WARN, "%s", m.status.c_str());
    ImGui::Spacing();

    if (ImGui::BeginTabBar("metrics")) {
        tab("Время", [&] {
            if (r) ImGui::TextDisabled("кадр %dx%d (%.1f Мп), %s", r->frame->w, r->frame->h,
                                       r->frame->w * (double)r->frame->h / 1e6, m.backend.c_str());
            if (ImGui::BeginTable("times", 5, ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("мс", ImGuiTableColumnFlags_WidthStretch, 2.2f);
                ImGui::TableSetupColumn("сред.");
                ImGui::TableSetupColumn("p50");
                ImGui::TableSetupColumn("p95");
                ImGui::TableSetupColumn("макс");
                ImGui::TableHeadersRow();
                row_stat("декодирование", m.decode);
                row_stat("движение камеры", m.motion);
                row_stat("детекция", m.detect);
                row_stat("  морфология", m.response);
                row_stat("  выбор пиков", m.select);
                row_stat("трекинг", m.track);
                row_stat("кадр целиком", m.proc);
                row_stat("задержка", m.latency);
                ImGui::EndTable();
            }
            ImGui::Dummy(ImVec2(0, ImGui::GetFontSize()));
            draw_chart(m, rt ? (float)(1000.0 / std::max(0.1, fpsSrc)) : 0.f,
                       ImVec2(ImGui::GetContentRegionAvail().x, 130));
            ImGui::Spacing();
            if (ImGui::Button("Сбросить замер", ImVec2(-FLT_MIN, 0))) a.pipe.resetStats();
        });
        tab("Цели", [&] { metrics_targets(a, m, r); });
        tab("Качество", [&] { metrics_quality(a, m); });
        tab("Система", [&] {
            heading("Конвейер");
            ImGui::Text("кадров: %ld, пропущено: %ld (%.1f%%)", m.processed, m.dropped, dropPct);
            ImGui::Text("декодер: %.1f к/с, видео: %.1f к/с", m.decodeFps, m.sourceFps);
            ImGui::Text("очередь: %d / %d, время замера %.0f с", m.queue, m.queueCap, m.elapsedS);
            heading("Процесс");
            ImGui::Text("CPU: %.0f%% (≈ %.1f ядра из %d)", a.cpuPct, a.cpuPct / 100.0 * plat::logical_cpus(),
                        plat::logical_cpus());
            ImGui::Text("память: %.0f МБ", plat::process_memory_mb());
            ImGui::TextWrapped("%s", plat::cpu_model().c_str());
            ImGui::TextDisabled("этот ПК: %.2f мс/Мп (1 поток, 128 бит)", a.pcMs);
        });
        ImGui::EndTabBar();
    }
}

void draw_view(App& a, const Result* r) {
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, ImVec2(p0.x + avail.x, p0.y + avail.y), IM_COL32(18, 18, 20, 255));
    ImGui::InvisibleButton("view", ImVec2(std::max(1.f, avail.x), std::max(1.f, avail.y)));
    const bool hovered = ImGui::IsItemHovered();
    if (!r || !a.tex || a.displayMode == 3) {
        const char* msg = !a.pipe.isOpen() ? "Откройте видео или перетащите файл в окно"
                          : a.displayMode == 3 ? "Показ видео выключен (Отображение > обновление)"
                                               : "Ждём первый кадр...";
        const ImVec2 ts = ImGui::CalcTextSize(msg);
        dl->AddText(ImVec2(p0.x + (avail.x - ts.x) / 2, p0.y + (avail.y - ts.y) / 2), IM_COL32(170, 170, 180, 255), msg);
        return;
    }
    const Frame& f = *r->frame;
    const float fit = std::min(avail.x / f.w, avail.y / f.h);
    // масштаб колесом относительно курсора, перетаскивание мышью, двойной щелчок — вписать
    if (hovered && io.MouseWheel != 0) {
        const float old = a.zoom;
        a.zoom = std::min(64.f, std::max(1.f, a.zoom * std::pow(1.25f, io.MouseWheel)));
        const float k = a.zoom / old;
        const ImVec2 c(io.MousePos.x - (p0.x + avail.x / 2), io.MousePos.y - (p0.y + avail.y / 2));
        a.panX = c.x - (c.x - a.panX) * k;
        a.panY = c.y - (c.y - a.panY) * k;
    }
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0)) {
        a.panX += io.MouseDelta.x;
        a.panY += io.MouseDelta.y;
    }
    if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        a.zoom = 1;
        a.panX = a.panY = 0;
    }
    if (a.zoom <= 1.f) a.panX = a.panY = 0;
    const float sc = fit * a.zoom;
    const ImVec2 o(p0.x + avail.x / 2 + a.panX - f.w * sc / 2, p0.y + avail.y / 2 + a.panY - f.h * sc / 2);
    auto P = [&](float x, float y) { return ImVec2(o.x + x * sc, o.y + y * sc); };
    dl->PushClipRect(p0, ImVec2(p0.x + avail.x, p0.y + avail.y), true);
    // текстура может быть уменьшена (texDs): берём её целиком
    dl->AddImage((ImTextureID)(intptr_t)a.tex, P(0, 0), P((float)a.texW * a.texDs, (float)a.texH * a.texDs));
    if (a.showMask && r->zones.size() == (size_t)r->tilesX * r->tilesY)   // земля темнее, смешанные — жёлтым
        for (int ty = 0; ty < r->tilesY; ++ty)
            for (int tx = 0; tx < r->tilesX; ++tx) {
                const uint8_t z = r->zones[(size_t)ty * r->tilesX + tx];
                if (z == 2) continue;
                dl->AddRectFilled(P((float)tx * r->tilePx, (float)ty * r->tilePx),
                                  P((float)(tx + 1) * r->tilePx, (float)(ty + 1) * r->tilePx),
                                  z == 0 ? IM_COL32(60, 30, 0, 110) : IM_COL32(255, 210, 0, 45));
            }
    if (a.showMask && !r->mask.empty()) {
        for (int ty = 0; ty < r->tilesY; ++ty)
            for (int tx = 0; tx < r->tilesX; ++tx)
                if (!r->mask[(size_t)ty * r->tilesX + tx]) {
                    const size_t ti = (size_t)ty * r->tilesX + tx;
                    const bool dk = r->maskDark.size() == r->mask.size() && r->maskDark[ti];
                    if (dk && r->maskBright.size() == r->mask.size() && r->maskBright[ti]) continue;   // облако: ищутся все цели
                    if (dk) {
                        // облако: закрыто только для светлых целей, тёмные там ищутся — едва заметно
                        dl->AddRectFilled(P((float)tx * r->tilePx, (float)ty * r->tilePx),
                                          P((float)(tx + 1) * r->tilePx, (float)(ty + 1) * r->tilePx), IM_COL32(0, 60, 120, 28));
                        continue;
                    }
                    // тайл земли с уточнением: закрашиваем только ячейки, не ставшие небом
                    const int cpt = r->cellsPerTile, ctw = r->tilesX * cpt;
                    bool any = false;
                    if (cpt > 0 && r->fineSky.size() == (size_t)ctw * r->tilesY * cpt)
                        for (int cy = ty * cpt; cy < (ty + 1) * cpt && !any; ++cy)
                            for (int cx = tx * cpt; cx < (tx + 1) * cpt && !any; ++cx) any = r->fineSky[(size_t)cy * ctw + cx] != 0;
                    if (!any) {
                        dl->AddRectFilled(P((float)tx * r->tilePx, (float)ty * r->tilePx),
                                          P((float)(tx + 1) * r->tilePx, (float)(ty + 1) * r->tilePx), IM_COL32(0, 60, 120, 90));
                        continue;
                    }
                    const float c = (float)r->tilePx / cpt;
                    for (int cy = ty * cpt; cy < (ty + 1) * cpt; ++cy)
                        for (int cx = tx * cpt; cx < (tx + 1) * cpt; ++cx)
                            if (!r->fineSky[(size_t)cy * ctw + cx])
                                dl->AddRectFilled(P(cx * c, cy * c), P((cx + 1) * c, (cy + 1) * c), IM_COL32(0, 60, 120, 90));
                }
    }
    if (a.showUpper && r->tilesX > 0 && r->tilesY > 0) {   // менее важное небо — фиолетовым
        const int tw = r->tilesX, th = r->tilesY;
        const size_t n = (size_t)tw * th;
        const bool m = r->mask.size() == n, z = r->zones.size() == n, b = r->band.size() == n;
        const bool md = m && r->maskDark.size() == n;   // облако, где ищутся тёмные цели, — небо
        auto ground = [&](size_t i) { return m ? !r->mask[i] && !(md && r->maskDark[i]) : z && r->zones[i] == 0; };
        if (m || z)
            for (int x = 0; x < tw; ++x) {
                int hy = th;   // верх земли, связанной с низом кадра, в этом столбце
                while (hy > 0 && ground((size_t)(hy - 1) * tw + x)) --hy;
                for (int y = 0; y < hy; ++y) {
                    const size_t i = (size_t)y * tw + x;
                    // без полосы — верхние 30% неба столбца (по середине тайла): половина неба на
                    // экране выглядела как "не смотрим"
                    if (ground(i) || (b ? r->band[i] != 0 : 20 * y + 10 >= 6 * hy)) continue;
                    dl->AddRectFilled(P((float)x * r->tilePx, (float)y * r->tilePx),
                                      P((float)(x + 1) * r->tilePx, (float)(y + 1) * r->tilePx), IM_COL32(150, 60, 220, 60));
                }
            }
    }
    if (a.showNoise)
        for (const auto& c : r->noise) dl->AddCircleFilled(P(c.x, c.y), 2.f, COL_NOISE);
    if (a.showClutter)
        for (const auto& c : r->clutter) {
            const ImVec2 q = P(c.x, c.y);
            dl->AddRect(ImVec2(q.x - 6, q.y - 6), ImVec2(q.x + 6, q.y + 6), COL_CLUTTER);
        }
    if (a.showFocus)
        for (const auto& fa : r->focus) dl->AddCircle(P(fa.x, fa.y), fa.r * sc, COL_FOCUS, 0, 1.f);
    if (a.showClutter && r->staticOn)   // запомненные статичные помехи — косые крестики
        for (const auto& c : r->staticCells) {
            const ImVec2 q = P(c.x, c.y);
            const float h = std::max(3.f, 0.5f * r->staticCellPx * sc);
            dl->AddLine(ImVec2(q.x - h, q.y - h), ImVec2(q.x + h, q.y + h), COL_STATIC, 1.f);
            dl->AddLine(ImVec2(q.x - h, q.y + h), ImVec2(q.x + h, q.y - h), COL_STATIC, 1.f);
        }
    if (a.showGt && !a.annotate)
        for (const auto& g : r->gt) {
            const ImVec2 q = P(g.x * r->gsx, g.y * r->gsy);
            if (g.bird) dl->AddNgon(q, 18, COL_GT, 4, 1.5f);
            else dl->AddCircle(q, 18, COL_GT, 0, 1.5f);
        }
    const Track* hover = nullptr;
    if (a.showTracks)
        for (const auto& t : r->tracks) {
            if (t.clutter && !a.showClutter) continue;
            const ImU32 col = class_color(t);
            const ImVec2 q = P(t.x, t.y);
            const float h = t.clutter ? 8.f : 12.f;
            if (t.miss > 0) {   // цель не найдена, трек идёт по прогнозу — пунктир
                const ImVec2 c[4] = {ImVec2(q.x - h, q.y - h), ImVec2(q.x + h, q.y - h), ImVec2(q.x + h, q.y + h),
                                     ImVec2(q.x - h, q.y + h)};
                for (int k = 0; k < 4; ++k) {
                    const ImVec2 a0 = c[k], a1 = c[(k + 1) % 4];
                    for (int j = 0; j < 4; j += 2)
                        dl->AddLine(ImVec2(a0.x + (a1.x - a0.x) * j / 4, a0.y + (a1.y - a0.y) * j / 4),
                                    ImVec2(a0.x + (a1.x - a0.x) * (j + 1) / 4, a0.y + (a1.y - a0.y) * (j + 1) / 4), col, 2.f);
                }
                if (a.showPred && t.id == r->lockId) dl->AddCircle(q, std::max(h, std::min(3 * t.sigma * sc, 200.f)), COL_LOCK, 0, 1.f);
            } else {
                dl->AddRect(ImVec2(q.x - h, q.y - h), ImVec2(q.x + h, q.y + h), col, 0, 0, t.clutter ? 1.f : 2.f);
            }
            // прогноз — у захваченной цели и у устоявшихся треков, иначе на кадре рябит от линий
            if (!t.clutter && a.showPred && (t.id == r->lockId || t.hits >= 8) && t.vx * t.vx + t.vy * t.vy > 0.01f) {
                const ImVec2 e = P(t.x + 10 * t.vx, t.y + 10 * t.vy);
                dl->AddLine(q, e, col, 1.5f);
                dl->AddCircleFilled(e, 3.f, col);
            }
            if (t.id == r->lockId) {   // прицел захвата: уголки (по габариту силуэта) и перекрестие
                const float g = std::max(h + 9, 0.5f * std::max(t.bw, t.bh) * sc + 6), l = std::max(8.f, 0.25f * g);
                for (int sx = -1; sx <= 1; sx += 2)
                    for (int sy = -1; sy <= 1; sy += 2) {
                        const ImVec2 c(q.x + sx * g, q.y + sy * g);
                        dl->AddLine(c, ImVec2(c.x - sx * l, c.y), COL_LOCK, 2.5f);
                        dl->AddLine(c, ImVec2(c.x, c.y - sy * l), COL_LOCK, 2.5f);
                    }
                dl->AddLine(ImVec2(q.x - g - 10, q.y), ImVec2(q.x - h - 3, q.y), COL_LOCK, 1.5f);
                dl->AddLine(ImVec2(q.x + h + 3, q.y), ImVec2(q.x + g + 10, q.y), COL_LOCK, 1.5f);
                dl->AddLine(ImVec2(q.x, q.y - g - 10), ImVec2(q.x, q.y - h - 3), COL_LOCK, 1.5f);
                dl->AddLine(ImVec2(q.x, q.y + h + 3), ImVec2(q.x, q.y + g + 10), COL_LOCK, 1.5f);
                char b[64];
                const bool hard = r->lockState == dronolok::LockState::Hard;
                const char* what = hard ? "ЗАХВАЧЕН ДРОН" : "ЗАХВАТ";
                if (t.miss) snprintf(b, sizeof(b), "%s (прогноз %d)", what, t.miss);
                else if (!hard && r->verifyNeed > 0 && r->verify > 0)
                    snprintf(b, sizeof(b), "ЗАХВАТ: дрон? %d/%d", r->verify, r->verifyNeed);
                else snprintf(b, sizeof(b), "%s", what);
                dl->AddText(ImVec2(q.x - g, q.y + g + 2), COL_LOCK, b);
            }
            if (!t.clutter && a.showPaths && t.path.size() > 1) {
                std::vector<ImVec2> pts;
                for (const auto& pp : t.path) pts.push_back(P(pp.x, pp.y));
                dl->AddPolyline(pts.data(), (int)pts.size(), col, 0, 1.f);
            }
            if (!t.clutter && a.showIds) {
                char b[48];
                snprintf(b, sizeof(b), "#%d %s", t.id, class_ru(t));
                dl->AddText(ImVec2(q.x + h + 3, q.y - h - 2), col, b);
            }
            if (r->lockQueue.size() > 1 && t.id != r->lockId) {   // место в очереди целей
                const auto it = std::find(r->lockQueue.begin(), r->lockQueue.end(), t.id);
                if (it != r->lockQueue.end()) {
                    char b[16];
                    snprintf(b, sizeof(b), "%d", (int)(it - r->lockQueue.begin()) + 1);
                    dl->AddText(ImVec2(q.x - h - 4 - ImGui::CalcTextSize(b).x, q.y - h - 2), COL_LOCK, b);
                }
            }
            if (hovered && std::fabs(io.MousePos.x - q.x) < h + 2 && std::fabs(io.MousePos.y - q.y) < h + 2) hover = &t;
        }
    // щелчок по цели (не в разметке) — захват на неё
    if (hover && !(a.annotate && a.ann.editable) && ImGui::IsMouseReleased(ImGuiMouseButton_Left) &&
        io.MouseDragMaxDistanceSqr[0] < 16 && !hover->clutter)
        a.pipe.selectTarget(hover->id);
    if (a.annotate && a.ann.editable) {
        // разметка: опорные точки и положение объектов на этом кадре (в пикселях видео)
        const VideoInfo vi = a.pipe.info();
        const float kx = vi.w > 0 ? (float)f.w / vi.w : 1.f, ky = vi.h > 0 ? (float)f.h / vi.h : 1.f;
        for (const auto& o : a.ann.objs) {
            const bool cur = o.id == a.annCur;
            const ImU32 c = cur ? IM_COL32(0, 255, 120, 255) : IM_COL32(0, 200, 100, 170);
            if (cur && o.keys.size() > 1) {
                std::vector<ImVec2> pts;
                for (const auto& k : o.keys) pts.push_back(P(k.second.first * kx, k.second.second * ky));
                dl->AddPolyline(pts.data(), (int)pts.size(), IM_COL32(0, 255, 120, 90), 0, 1.f);
            }
            float x, y;
            if (!Annotation::pos(o, f.index, x, y)) continue;
            const ImVec2 q = P(x * kx, y * ky);
            const bool key = o.keys.count(f.index) > 0;
            dl->AddCircle(q, 16, c, 0, cur ? 2.5f : 1.5f);
            if (key) dl->AddCircleFilled(q, 4, c);
            char b[64];   // кириллица — 2 байта на букву: в 32 "#1 птица (по прямой)" обрезалась
            snprintf(b, sizeof(b), "#%d %s%s", o.id, o.bird ? "птица" : "дрон", key ? "" : " (по прямой)");
            dl->AddText(ImVec2(q.x + 18, q.y + 4), c, b);
        }
        // щелчок без перетаскивания — опорная точка выбранного объекта (прилипает к ближайшей цели)
        if (hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Left) && io.MouseDragMaxDistanceSqr[0] < 16) {
            float fx = (io.MousePos.x - o.x) / sc, fy = (io.MousePos.y - o.y) / sc;
            float best = 14.f / sc;
            const Track* snap = nullptr;
            // и к «статичной помехе»: зависший или медленный дрон трекер тоже считает помехой
            for (const auto& t : r->tracks)
                if (std::hypot(t.x - fx, t.y - fy) < best) {
                    best = std::hypot(t.x - fx, t.y - fy);
                    fx = t.x;
                    fy = t.y;
                    snap = &t;
                }
            if (!snap && !r->follow) snap_to_spot(f, fx, fy, std::min(12, std::max(2, (int)std::lround(3 / sc))));
            if (fx >= 0 && fy >= 0 && fx < f.w && fy < f.h) {
                ann_follow_stop(a, nullptr);
                if (!a.ann.find(a.annCur)) a.annCur = a.ann.add(false);
                a.ann.find(a.annCur)->keys[f.index] = {fx / kx, fy / ky};
                ann_changed(a);
                // и с кадра из истории: следующие кадры истории уже посчитаны, по ним точки встанут сразу.
                // Нет цели алгоритма под щелчком — дальше по картинке
                if (a.annAuto && !r->follow) ann_follow_start(a, *r, snap, fx / kx, fy / ky);
            }
        }
        if (hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
            if (io.KeyShift) {   // Shift + правый щелчок по кружку — удалить путь этого объекта целиком
                int id = 0;
                float bd = 20.f;
                for (const auto& ob : a.ann.objs) {
                    float x, y;
                    if (!Annotation::pos(ob, f.index, x, y)) continue;
                    const ImVec2 q = P(x * kx, y * ky);
                    const float d = std::hypot(q.x - io.MousePos.x, q.y - io.MousePos.y);
                    if (d < bd) {
                        bd = d;
                        id = ob.id;
                    }
                }
                if (id) ann_delete(a, id);
                else a.annMsg = "Shift + правый щелчок — по кружку объекта: так удаляется весь его путь";
            } else if (AnnObject* ob = a.ann.find(a.annCur)) {
                if (ob->keys.erase(f.index)) ann_changed(a);
            }
        }
    }
    dl->PopClipRect();
    // строка состояния и легенда поверх видео
    char st[512];   // с запасом: при обрезке len += snprintf ушёл бы за конец буфера
    int len = r->dup ? snprintf(st, sizeof(st), "%dx%d  кадр %lld  проход %d  повтор прошлого кадра", f.w, f.h,
                                (long long)f.index, r->loop + 1)
                     : snprintf(st, sizeof(st), "%dx%d  кадр %lld  проход %d  детекция %.2f мс  обработка %.2f мс",
                                f.w, f.h, (long long)f.index, r->loop + 1, r->dt.total_ms, r->procMs);
    if (r->motionOn && r->camValid)
        len += snprintf(st + len, sizeof(st) - len, "  камера %+.1f %+.1f", r->camDx, r->camDy);
    if (r->follow)
        len += snprintf(st + len, sizeof(st) - len, "  поворотка: окно %d,%d из %dx%d", r->fx, r->fy, r->srcW, r->srcH);
    if (a.s.pace == Pace::Realtime && std::fabs(a.s.speed - 1) > 1e-3)
        len += snprintf(st + len, sizeof(st) - len, "  x%.2f", a.s.speed);
    int hf, hl;
    if (a.viewSeq >= 0 && a.pipe.historyRange(hf, hl))
        len += snprintf(st + len, sizeof(st) - len, "  [история: %d кадр. назад]", hl - a.viewSeq);
    else if (a.pipe.paused())
        len += snprintf(st + len, sizeof(st) - len, "  [пауза]");
    dl->AddRectFilled(ImVec2(p0.x + 6, p0.y + 6), ImVec2(p0.x + 12 + ImGui::CalcTextSize(st).x, p0.y + 10 + ImGui::GetFontSize()),
                      IM_COL32(0, 0, 0, 160), 3);
    dl->AddText(ImVec2(p0.x + 9, p0.y + 8), IM_COL32(120, 255, 140, 255), st);
    struct L { ImU32 c; const char* t; } legend[] = {{COL_LOCK, "захват (пунктир — по прогнозу)"}, {COL_DRONE, "дрон"},
                                                     {COL_BIRD, "птица"}, {COL_UNKNOWN, "цель, класс неясен"},
                                                     {COL_CLUTTER, "статичная помеха (крестик — запомнена)"}, {COL_NOISE, "шум (одиночные отклики)"},
                                                     {COL_GT, "разметка: круг дрон, ромб птица"}};
    float ly = p0.y + avail.y - 8;
    for (int i = 6; i >= 0; --i) {
        if (i == 6 && r->gt.empty()) continue;
        ly -= ImGui::GetFontSize() + 4;
        dl->AddRectFilled(ImVec2(p0.x + 8, ly + 3), ImVec2(p0.x + 20, ly + 15), legend[i].c);
        dl->AddText(ImVec2(p0.x + 26, ly), IM_COL32(230, 230, 230, 255), legend[i].t);
    }
    if (hover) {
        ImGui::BeginTooltip();
        ImGui::TextColored(ImColor(class_color(*hover)).Value, "#%d %s", hover->id, class_ru(*hover));
        ImGui::Text("x %.0f  y %.0f, %s", hover->x, hover->y, hover->polarity > 0 ? "ярче фона" : "темнее фона");
        ImGui::Text("контраст %.0f, скорость %.1f пикс./кадр", hover->contrast, hover->speed);
        ImGui::Text("отклонение от плавной траектории %.2f пикс.", hover->resid);
        ImGui::Text("мигание контраста %.2f, попаданий %d", hover->contrastCv, hover->hits);
        ImGui::Text("скорость по сцене %.1f, %.1f; прошёл %.0f пикс.", hover->vsx, hover->vsy, hover->travel);
        ImGui::Text("ошибка прогноза %.1f пикс., весомость %.1f, вес для захвата %.1f", hover->sigma, hover->score,
                    hover->lockW);
        if (hover->miss) ImGui::Text("на прогнозе %d кадр.", hover->miss);
        ImGui::EndTooltip();
    }
}

// ------------------------------------------------------------ шкала видео

// Шкала под видео: перемотка в любое место файла, отметки целей, захвата и пропусков по разметке
// высота кнопок плеера и всей нижней панели (кнопки + шкала)
float timeline_buttons_h() { return ImGui::GetFrameHeight() * 1.6f; }
float timeline_height() {
    return 4 + timeline_buttons_h() + ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeight() * 1.3f + 14;
}

void draw_timeline(App& a, const Result* r, const Metrics& m) {
    ImGuiIO& io = ImGui::GetIO();
    const VideoInfo vi = a.pipe.info();
    const double fps = vi.fps > 0 ? vi.fps : 25;
    a.pipe.marks(a.marks);
    const int64_t cur = r ? r->frame->index : 0;
    int64_t n = a.pipe.length();
    const bool known = n > 0;
    n = std::max<int64_t>({n, (int64_t)a.marks.size(), cur + 1, 1});
    const float pad = 8, bh = timeline_buttons_h();
    ImGui::SetCursorPos(ImVec2(pad, ImGui::GetCursorPosY() + 4));
    const float rowY = ImGui::GetCursorPosY();
    // кнопки плеера, крупно
    const bool paused = a.pipe.paused();
    {
        FontSize f(1.2f);
        if (ImGui::Button("|<", ImVec2(bh, bh))) seek_to(a, 0, false);
        ImGui::SetItemTooltip("В начало (Home)");
        ImGui::SameLine();
        if (ImGui::Button("<", ImVec2(bh, bh))) step_back(a);
        ImGui::SetItemTooltip("Кадр назад (стрелка влево)");
        ImGui::SameLine();
        push_accent();
        if (ImGui::Button(paused ? "Пуск" : "Пауза", ImVec2(bh * 2.4f, bh))) {
            if (paused && m.status == "видео закончилось") seek_to(a, 0, false);
            toggle_pause(a);
        }
        pop_accent();
        ImGui::SetItemTooltip("Пробел");
        ImGui::SameLine();
        if (ImGui::Button(">", ImVec2(bh, bh))) step_forward(a);
        ImGui::SetItemTooltip("Кадр вперёд (стрелка вправо)");
    }
    const int64_t shown = a.scrub >= 0 ? a.scrub : cur;
    auto centered = [&] {
        ImGui::SameLine(0, 14);
        ImGui::SetCursorPosY(rowY + (bh - ImGui::GetTextLineHeight()) / 2);
    };
    {
        FontSize f(1.15f);
        centered();
        ImGui::Text("%s / %s", fmt_time(shown / fps).c_str(), fmt_time(n / fps).c_str());
    }
    // скорость справа в той же строке; на узком окне номер кадра уступает ей место
    const float sw = bh * 1.3f, sp = 4, total = 4 * sw + 3 * sp;
    const float sx = ImGui::GetWindowWidth() - pad - total;
    char frameTxt[64];
    snprintf(frameTxt, sizeof(frameTxt), "кадр %lld / %lld%s", (long long)shown, (long long)n, known ? "" : "+");
    auto room = [&](float need) { return ImGui::GetItemRectMax().x - ImGui::GetWindowPos().x + 14 + need + 20 < sx; };
    if (room(ImGui::CalcTextSize(frameTxt).x)) {
        centered();
        ImGui::TextDisabled("%s", frameTxt);
    }
    if (a.viewSeq >= 0) {
        int hf, hl;
        if (a.pipe.historyRange(hf, hl) && room(ImGui::CalcTextSize("история: 000 кадр. назад").x)) {
            centered();
            ImGui::TextColored(TXT_WARN, "история: %d кадр. назад", hl - a.viewSeq);
        }
    }
    static const double speeds[] = {0.1, 0.25, 0.5, 1.0};
    bool preset = false;
    for (double v : speeds) preset |= std::fabs(a.s.speed - v) < 1e-3;
    if (a.s.pace == Pace::Realtime && !preset && room(ImGui::CalcTextSize("x0.00").x)) {
        centered();
        ImGui::TextColored(TXT_WARN, "x%.2f", a.s.speed);
    }
    static const char* speedNames[] = {"x0.1", "x0.25", "x0.5", "x1"};
    if (room(0)) {
        ImGui::SameLine(sx);
        ImGui::SetCursorPosY(rowY);
        for (int i = 0; i < 4; ++i) {
            if (i) ImGui::SameLine(0, sp);
            const bool on = a.s.pace == Pace::Realtime && std::fabs(a.s.speed - speeds[i]) < 1e-3;
            if (on) push_accent();
            if (ImGui::Button(speedNames[i], ImVec2(sw, bh))) set_speed(a, speeds[i]);
            if (on) pop_accent();
            ImGui::SetItemTooltip("Скорость видео (клавиши + и -)");
        }
    }
    ImGui::SetCursorPosY(rowY + bh + ImGui::GetStyle().ItemSpacing.y);
    // полоса: сверху — что нашёл алгоритм, снизу — разметка (зелёный найден, красный пропущен)
    ImGui::SetCursorPosX(pad);
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const float w = std::max(10.f, ImGui::GetContentRegionAvail().x - pad), h = ImGui::GetFrameHeight() * 1.3f;
    ImGui::InvisibleButton("timeline", ImVec2(w, h));
    const bool hov = ImGui::IsItemHovered(), act = ImGui::IsItemActive();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p1(p0.x + w, p0.y + h);
    dl->AddRectFilled(p0, p1, IM_COL32(40, 42, 48, 255), 3);
    const float hTop = h * 0.62f;
    const int cols = (int)w;
    for (int x = 0; x < cols; ++x) {
        const int64_t i0 = (int64_t)((double)x / cols * n), i1 = std::max(i0 + 1, (int64_t)((double)(x + 1) / cols * n));
        uint8_t acc = 0;
        for (int64_t i = i0; i < i1 && i < (int64_t)a.marks.size(); ++i) acc |= a.marks[(size_t)i];
        if (!acc) continue;
        const ImU32 c = acc & MARK_LOCK    ? COL_LOCK
                        : acc & MARK_DRONE ? COL_DRONE
                        : acc & MARK_BIRD  ? COL_BIRD
                        : acc & MARK_TARGET ? COL_UNKNOWN
                                            : IM_COL32(85, 88, 98, 255);
        const float fx = p0.x + x;
        dl->AddRectFilled(ImVec2(fx, p0.y + 2), ImVec2(fx + 1, p0.y + hTop), c);
        if (acc & MARK_GT)
            dl->AddRectFilled(ImVec2(fx, p0.y + hTop + 2), ImVec2(fx + 1, p1.y - 2),
                              acc & MARK_MISS ? IM_COL32(255, 60, 60, 255) : COL_GT);
    }
    // опорные точки выбранного объекта разметки
    if (a.annotate)
        if (const AnnObject* ob = a.ann.find(a.annCur))
            for (const auto& k : ob->keys) {
                const float kx = p0.x + (float)((k.first + 0.5) / n * w);
                dl->AddTriangleFilled(ImVec2(kx - 4, p0.y), ImVec2(kx + 4, p0.y), ImVec2(kx, p0.y + 6), COL_GT);
            }
    // текущий кадр
    const float cx = p0.x + (float)((cur + 0.5) / n * w);
    dl->AddLine(ImVec2(cx, p0.y - 2), ImVec2(cx, p1.y + 2), IM_COL32(255, 255, 255, 255), 2);
    auto frameAt = [&](float mx) { return std::min<int64_t>(n - 1, std::max<int64_t>(0, (int64_t)((mx - p0.x) / w * n))); };
    if (hov || act) {
        const int64_t f = frameAt(io.MousePos.x);
        const float hx = p0.x + (float)((f + 0.5) / n * w);
        dl->AddLine(ImVec2(hx, p0.y), ImVec2(hx, p1.y), IM_COL32(255, 255, 255, 110), 1);
        const uint8_t mk = f < (int64_t)a.marks.size() ? a.marks[(size_t)f] : 0;
        ImGui::SetTooltip("кадр %lld, %s%s%s%s%s", (long long)f, fmt_time(f / fps).c_str(),
                          !(mk & MARK_DONE) ? ", не обработан" : "", mk & MARK_LOCK ? ", захват" : "",
                          mk & MARK_DRONE ? ", дрон" : "", mk & MARK_MISS ? ", дрон по разметке пропущен" : "");
    }
    // щелчок или перетаскивание — перемотка (при перетаскивании не чаще 10 раз в секунду)
    if (act) {
        const int64_t f = frameAt(io.MousePos.x);
        const double t = now_ms();
        if (f != a.scrub && t - a.scrubAt > 100) {
            seek_to(a, f, true);
            a.scrubAt = t;
        }
        a.scrub = f;
    } else if (a.scrub >= 0) {
        seek_to(a, a.scrub, true);
        a.scrub = -1;
    }
}

// ------------------------------------------------------------ режим замера без окна

int run_bench(int argc, char** argv) {
    plat::attach_console();
    std::string video, json, csv;
    int frames = 300, preload = 0;
    Settings s;
    s.pace = Pace::MaxSpeed;
    s.loop = true;
    s.staticFile = false;
    s.maskFile = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--frames") frames = atoi(val().c_str());
        else if (a == "--threads") s.det.threads = atoi(val().c_str());
        else if (a == "--isa") {
            std::string v = val();
            s.det.isa = v == "scalar" ? Isa::Scalar : v == "sse2" ? Isa::SSE2 : v == "avx2" ? Isa::AVX2 : v == "neon" ? Isa::NEON : Isa::Auto;
        } else if (a == "--size") sscanf(val().c_str(), "%dx%d", &s.procW, &s.procH);
        else if (a == "--preload") { preload = atoi(val().c_str()); }
        else if (a == "--learn-mask") s.learnMaskFrames = atoi(val().c_str());
        else if (a == "--scales") s.scales = atoi(val().c_str());
        else if (a == "--motion") s.motion = atoi(val().c_str());
        else if (a == "--no-scene-shift") s.sceneShift = false;
        else if (a == "--no-pan") s.panDetect = false;
        else if (a == "--pan-motion") s.panMotion = true;
        else if (a == "--no-motion-long") s.motionLong = false;
        else if (a == "--focus") s.focus = true;
        else if (a == "--static") s.trk.static_frames = atoi(val().c_str());
        else if (a == "--static-file") s.staticFile = true;
        else if (a == "--mask-file") s.maskFile = true;
        else if (a == "--slowdown") s.slowdown = atof(val().c_str());
        else if (a == "--governor") s.governor = true;
        else if (a == "--camera") s.pace = Pace::Realtime;   // кадры в темпе видео, лишние выбрасываются
        else if (a == "--profile") {
            const AlgoBlocks b = profile_blocks((Profile)std::min(3, std::max(0, atoi(val().c_str()))));
            TrackerConfig t = s.trk;
            apply_blocks(b, s.det, t);
            s.trk.static_frames = t.static_frames;
            s.scales = b.scales;
            s.motion = b.motion ? s.motion : 0;
            s.zones = b.zones ? s.zones : 0;
            s.learnMaskFrames = b.skyMask ? s.learnMaskFrames : 0;
        }
        else if (a == "--full-every") s.det.full_every = atoi(val().c_str());
        else if (a == "--zones") s.zones = atoi(val().c_str());
        else if (a == "--coarse-every") s.det.coarse_every = atoi(val().c_str());
        else if (a == "--json") json = val();
        else if (a == "--csv") csv = val();
        else if (a == "--tracks") s.trackLog = val();
        else if (a == "--dedup") s.skipDuplicates = true;
        else if (a == "--step") s.frameStep = std::max(1, atoi(val().c_str()));
        else if (a == "--phase") s.framePhase = std::max(0, atoi(val().c_str()));
        else if (a == "--cam-fps") s.targetFps = atof(val().c_str());
        else if (a == "--no-fps-scale") s.fpsScale = false;
        else if (a == "--no-anchor") s.trk.anchor_check = false;
        else if (a == "--no-size") s.trk.size_predict = false;
        else if (a == "--no-skyline") s.det.skyline = false;
        else if (a == "--no-fine-mask") s.det.fine_mask = false;
        else if (a == "--no-grow-mask") s.det.grow_mask = false;
        else if (a == "--no-mask-recheck") s.det.mask_recheck = false;
        else if (a == "--no-sky-color") s.det.sky_color = false;
        else if (a == "--no-sky-soft") s.det.sky_soft = false;
        else if (a == "--no-sky-bright") s.det.sky_bright = false;
        else if (a == "--no-adapt") s.adapt = false;
        else if (a == "--lock-cmax") s.trk.lock_cmax = (float)atof(val().c_str());
        else if (a == "--lock-slow") s.trk.lock_slow = (float)atof(val().c_str());
        else if (a == "--lock-dwell") s.trk.lock_dwell = atoi(val().c_str());
        else if (a == "--follow-k") s.trk.follow_k = (float)atof(val().c_str());
        else if (a == "--hard-switch") s.trk.hard_switch = (float)atof(val().c_str());
        else if (a == "--follow") s.follow = true;
        else if (a == "--follow-frac") s.followFrac = (float)atof(val().c_str());
        else if (a == "--follow-gain") s.followGain = (float)atof(val().c_str());
        else if (a == "--follow-any") s.followVerified = false;
        else video = a;
    }
    if (preload > 0) {
        s.source = Source::Preload;
        s.preloadFrames = preload;
    }
    Pipeline p;
    p.setSettings(s);
    std::string err;
    if (!p.open(video, err)) {
        printf("ошибка: %s\n", err.c_str());
        return 1;
    }
    printf("видео %dx%d, %.2f к/с, %s, поворот %d°\n", p.info().w, p.info().h, p.info().fps, p.info().codec.c_str(),
           p.info().rotation);
    bool counting = preload == 0;
    for (;;) {
        SDL_Delay(50);
        Metrics m = p.metrics(0);
        if (!counting && m.status.empty() && m.processed > 0) {   // предзагрузка закончилась
            p.resetStats();
            counting = true;
            continue;
        }
        if (counting && m.processed >= frames) break;
        if (m.status == "видео закончилось" || (!m.status.empty() && m.status.find("Предзагрузка") == std::string::npos && m.processed == 0)) {
            printf("%s\n", m.status.c_str());
            break;
        }
    }
    p.setPaused(true);
    const std::string rep = p.reportJson();
    printf("%s\n", rep.c_str());
    if (!json.empty())
        if (FILE* f = open_file(json, "w")) {
            fprintf(f, "%s\n", rep.c_str());
            fclose(f);
        }
    if (!csv.empty()) p.exportCsv(csv);
    p.close();
    return 0;
}

// Стресс-проверка: случайно крутит настройки, паузу, историю и память помех, как
// пользователь в интерфейсе, только много раз в секунду. Для поиска падений
// (в сборке с -fsanitize=address) и в CI.
int run_stress(int argc, char** argv) {
    plat::attach_console();
    std::string video;
    double seconds = 20;
    unsigned seed = 1;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--seconds" && i + 1 < argc) seconds = atof(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
        else video = a;
    }
    Pipeline p;
    Settings s;
    s.staticFile = false;
    s.maskFile = false;
    p.setSettings(s);
    std::string err;
    if (!p.open(video, err)) {
        printf("ошибка: %s\n", err.c_str());
        return 1;
    }
    const VideoInfo vi = p.info();
    uint64_t st = seed * 0x9E3779B97F4A7C15ull + 1;
    auto R = [&](int n) {   // xorshift: одинаково на всех платформах
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        return (int)(st % (uint64_t)n);
    };
    auto F = [&](float lo, float hi) { return lo + (hi - lo) * R(1001) / 1000.f; };
    const std::vector<Isa> isas = available_isas();
    const std::string tmp = video + ".stress.csv";
    const double end = now_ms() + seconds * 1000;
    long actions = 0, seen = 0;
    int viewSeq = -1;
    unsigned sum = 0;
    while (now_ms() < end) {
        SDL_Delay(2 + R(40));
        // как интерфейс: последний кадр, кадр из истории, метрики
        auto r = p.latest();
        int first, last;
        if (p.historyRange(first, last)) {
            if (viewSeq < 0 || R(4) == 0) viewSeq = first + R(last - first + 1);
            if (auto h = p.at(viewSeq)) r = h;
        }
        if (r) {
            ++seen;
            const Frame& f = *r->frame;
            sum += f.px[(size_t)R(f.h) * f.w + R(f.w)];
            for (const auto& t : r->tracks) sum += (unsigned)t.path.size();
        }
        p.metrics();
        bool set = true;
        int a = R(38);
        // перезапуск чтения (источник, разрешение) и сброс — реже, иначе кадры почти не идут
        if ((a >= 12 && a <= 15) || a == 30 || a == 32) a = R(6) ? 37 : a;
        switch (a) {
            case 0: s.scales = R(5); break;
            case 1: s.motion = R(3) - 1; break;
            case 2: s.zones = R(3) - 1; break;
            case 3: s.det.full_every = 1 + R(4); break;
            case 4: s.det.coarse_every = 1 + R(4); break;
            case 5: s.focus = !s.focus; break;
            case 6: s.trk.static_frames = R(2) ? 0 : 300 + R(30000); break;
            case 7: s.trk.static_hits = F(3, 40); break;
            case 8: s.det.threads = 1 + R(std::max(1, plat::logical_cpus())); break;
            case 9: s.det.isa = R(3) ? isas[(size_t)R((int)isas.size())] : Isa::Auto; break;
            case 10: s.pace = R(2) ? Pace::Realtime : Pace::MaxSpeed; break;
            case 11: s.speed = F(0.05f, 2.f); break;
            case 12:
                s.source = R(3) ? Source::Stream : Source::Preload;
                s.preloadFrames = 10 + R(40);
                break;
            case 13: {   // разрешение обработки, как в списке "разрешение"
                static const float sc[] = {1.f, 0.75f, 0.5f, 0.25f, 0.33f, 1.5f};
                const float k = sc[R(6)];
                s.procW = k == 1.f ? 0 : std::max(64, (int)(vi.w * k) & ~1);
                s.procH = k == 1.f ? 0 : std::max(64, (int)(vi.h * k) & ~1);
                break;
            }
            case 14: s.decoderThreads = R(4); break;
            case 15: s.loop = R(4) != 0; break;
            case 16:
                s.emulateMs = R(3) ? 0 : F(0, 60);
                s.slowdown = R(3) ? 1 : F(1, 20);
                s.governor = R(2);
                break;
            case 17:
                s.learnMaskFrames = R(3) ? 25 : R(100);
                s.fpsScale = R(4) != 0;
                break;
            case 18: s.det.thr = F(3, 40); break;
            case 19: s.det.k = (3 + R(13)) | 1; break;
            case 20: s.det.pool = R(2) ? 2 : 1; break;
            case 21:
                s.det.dark = R(4) != 0;
                s.det.bright = R(4) != 0;
                break;
            case 22:
                s.det.clutter_k = F(1, 8);
                s.det.max_density = F(0.002f, 0.05f);
                s.det.compact_tex = R(3) != 0;
                s.det.smooth = R(2);
                s.det.subpixel = R(2);
                break;
            case 23:
                s.det.coarse_thr_k = F(1, 6);
                s.det.focus_k = F(0.2f, 1);
                s.det.auto_thr = R(2);
                s.det.noise_k = F(4, 30);
                s.det.thr_min = F(2, 20);
                s.det.thr_max = s.det.thr_min + F(0, 60);
                break;
            case 24:
                s.trk.gate = F(5, 150);
                s.trk.q_accel = F(0.25f, 20);
                s.trk.max_miss = 1 + R(60);
                s.trk.lock_switch = F(1, 5);
                s.trk.hard_lock = R(3) != 0;
                s.trk.verify_frames = 1 + R(120);
                s.trk.hard_miss = 5 + R(296);
                s.trk.hard_full_every = 1 + R(30);
                s.trk.silhouette = R(3) != 0;
                s.trk.anchor_check = R(3) != 0;
                s.trk.size_predict = R(3) != 0;
                s.trk.quiet_frames = R(2) ? 0 : R(200);
                s.trk.edge_straight = R(2) ? 0.5f : F(0, 1);
                s.trk.silh_max = F(50, 1000);
                s.skipDuplicates = R(2);
                s.follow = R(3) == 0;
                s.followFrac = F(0.1f, 1.2f);
                s.followGain = F(0, 2);
                s.followMaxPan = F(0, 0.5f);
                s.followVerified = R(2);
                break;
            case 25:
                s.trk.confirm = 1 + R(s.trk.window);
                s.trk.cls_min_hits = 3 + R(s.trk.cls_window - 2);
                s.trk.bird_resid = F(0.5f, 10);
                s.trk.bird_speed = R(2) ? 0 : F(0, 40);
                break;
            case 26: p.setPaused(!p.paused()); set = false; break;
            case 27: p.step(); set = false; break;
            case 28: p.relearnMask(); set = false; break;
            case 29: R(2) ? p.staticSave() : p.staticClear(); set = false; break;
            case 30: p.resetStats(); set = false; break;
            case 31: p.exportCsv(tmp); p.reportJson(); set = false; break;
            case 32:   // "С начала"
                if (R(8) == 0 && !p.open(video, err)) {
                    printf("ошибка: %s\n", err.c_str());
                    return 1;
                }
                set = R(2);
                break;
            case 33:
            case 34:   // перемотка по шкале
                p.seek(R((int)std::max<int64_t>(1, std::min<int64_t>(p.length() + 20, 1 << 30))));
                set = false;
                break;
            case 35:   // маска неба: зоны, доращивание, перепроверка, полярность, облака по цвету
                s.det.skyline = R(4) != 0;
                s.det.bg_map = R(4) != 0;
                s.det.bg_near = R(3) != 0;
                s.det.fine_mask = R(3) != 0;
                s.det.fine_frames = R(3) ? 150 : 1 + R(400);
                s.det.thin_tiles = R(3);
                s.det.zone_strip = R(3) != 0;
                s.det.zone_vote = R(3);
                s.det.halo_k = R(3) ? 2.f : F(0, 4);
                s.det.grow_mask = R(3) != 0;
                s.det.grow_frames = R(3) ? 60 : 1 + R(300);
                s.det.grow_k = F(1, 8);
                s.det.grow_min = F(0, 0.5f);
                s.det.grow_cells = R(5);
                s.det.mask_recheck = R(3) != 0;
                s.det.recheck_every = R(3) ? 10 : R(40);
                s.det.recheck_frames = R(3) ? 150 : R(400);
                s.det.recheck_tex = F(0, 1);
                s.det.reopen_quiet = F(0.5f, 1.1f);
                s.det.pol_mask = R(3);
                s.det.pol_tex = F(0, 0.3f);
                s.det.sky_color = R(3) != 0;
                s.det.sky_soft = R(3) != 0;
                s.det.sky_bright = R(3) != 0;
                s.det.sky_tree = R(2) ? 0.35f : F(0, 1);
                s.det.sky_green = F(0, 0.3f);
                s.det.sky_y = F(0, 1.2f);
                s.det.sky_dark = F(0, 1.2f);
                s.det.sky_d4 = F(0, 0.3f);
                s.det.sky_s14 = F(0, 1);
                s.det.lock_through = R(3) != 0;
                s.det.lock_through_k = F(0.5f, 3);
                s.det.bg_frames = R(3) ? 250 : 1 + R(1000);
                s.det.per_tile = 1 + R(2);
                s.det.band_px = R(3) ? 0 : R(1024);
                s.det.upper_every = 1 + R(4);
                s.det.tbd = R(4) == 0;
                break;
            case 36:   // трекер: захват на летавший трек, переход жёсткого, птица, скорости в секундах
                s.trk.lock_mover = R(2);
                s.trk.lock_mover_fps = R(3) ? 20.f : F(0, 120);
                s.trk.hard_switch_drone = R(3) != 0;
                s.trk.hover_first = R(3) != 0;
                s.trk.reupdate = R(2) ? 0.f : F(0, 1);
                s.trk.mover_cam_k = F(0, 1);
                s.trk.still_speed = F(0, 3);
                s.trk.flap_max_period = F(2, 48);
                s.trk.lock_slow = R(2) ? 1.f : F(0, 3);
                s.trk.hard_switch = F(0, 4);
                s.trk.lock_dwell = R(40);
                s.trk.lock_dwell_keep = R(3) != 0;
                s.trk.lock_bird_yield = R(3) != 0;
                s.trk.follow_k = F(0, 6);
                s.trkSeconds = R(4) != 0;
                // ночной подбор 30.10: дрейф фона, ведомая цель, ворота, класс по кадру, наложение у края
                s.trk.hard_switch_follow = R(3) != 0;
                s.trk.follow_min_obs = R(3) ? 16 : R(64);
                s.trk.drift_min = R(3) ? 0.25f : F(0, 2);
                s.trk.drift_obs = R(3) ? 13 : R(80);
                s.trk.resid_frame = R(3) != 0;
                s.trk.overlay_rel = R(3) ? 0.f : F(0, 0.5f);
                s.trk.gate = R(3) ? 26.f : F(0, 150);
                s.trk.gate_speed_k = R(3) ? 0.f : F(0, 4);
                s.trk.gate_pps = R(3) ? 360.f : F(0, 2000);
                s.motionLong = R(3) != 0;
                break;
            default:   // частота камеры и шаг: ниже частоты видео кадры файла пропускаются
                s.targetFps = R(2) ? 0 : F(1, 120);
                if (R(4) == 0) s.frameStep = 1 + R(6);
                if (R(4) == 0) s.framePhase = R(8);
                break;
        }
        if (set) p.setSettings(s);
        ++actions;
    }
    const Metrics m = p.metrics(0);
    p.close();
    remove_file(tmp);
    remove_file(p.staticPath());   // "сохранить память" пишет файл и без staticFile
    printf("стресс: %.0f с, действий %ld, кадров обработано %ld, просмотрено %ld (%u) — без падений\n", seconds,
           actions, m.processed, seen, sum & 1);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") return run_bench(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "--stress") return run_stress(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "--multitest") return mt::run_cli(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "--twocam") return twocam::run_cli(argc, argv);

    SDL_SetHint(SDL_HINT_WINDOWS_DPI_AWARENESS, "permonitorv2");
    SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
    const float scale = std::max(1.f, ImGui_ImplSDL2_GetContentScaleForDisplay(0));
    SDL_Window* win = SDL_CreateWindow("Дронолок — тест детектора", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                       (int)(1500 * scale), (int)(900 * scale),
                                       SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_MAXIMIZED);
    if (!win) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        return 1;
    }
    App a;
#if defined(_WIN32)
    SDL_SysWMinfo wm;
    SDL_VERSION(&wm.version);
    if (SDL_GetWindowWMInfo(win, &wm)) a.owner = wm.info.win.window;
#endif
    a.ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!a.ren) a.ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
    if (!a.ren) {
        fprintf(stderr, "SDL_CreateRenderer: %s\n", SDL_GetError());
        return 1;
    }
    SDL_RendererInfo ri;
    if (SDL_GetRendererInfo(a.ren, &ri) == 0 && ri.max_texture_width > 0) {
        a.maxTexW = ri.max_texture_width;
        a.maxTexH = ri.max_texture_height;
    }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0;
    style.FrameRounding = 4;
    style.GrabRounding = 4;
    style.TabRounding = 4;
    style.FramePadding = ImVec2(8, 5);
    style.ItemSpacing = ImVec2(8, 6);
    style.WindowPadding = ImVec2(12, 10);
    style.GrabMinSize = 12;
    style.ScrollbarSize = 14;
    style.SeparatorTextBorderSize = 1;
    ImVec4* c = style.Colors;
    c[ImGuiCol_WindowBg] = ImVec4(0.09f, 0.10f, 0.12f, 1);
    c[ImGuiCol_ChildBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_FrameBg] = ImVec4(0.17f, 0.19f, 0.23f, 1);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.22f, 0.25f, 0.31f, 1);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.25f, 0.30f, 0.38f, 1);
    c[ImGuiCol_Button] = ImVec4(0.20f, 0.22f, 0.27f, 1);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.27f, 0.30f, 0.37f, 1);
    c[ImGuiCol_ButtonActive] = ImVec4(0.32f, 0.36f, 0.45f, 1);
    c[ImGuiCol_Tab] = ImVec4(0.15f, 0.17f, 0.21f, 1);
    c[ImGuiCol_TabHovered] = ImVec4(0.27f, 0.53f, 0.93f, 1);
    c[ImGuiCol_TabSelected] = ImVec4(0.20f, 0.45f, 0.85f, 1);
    c[ImGuiCol_TabSelectedOverline] = ImVec4(0.20f, 0.45f, 0.85f, 1);
    c[ImGuiCol_CheckMark] = ImVec4(0.45f, 0.70f, 1, 1);
    c[ImGuiCol_SliderGrab] = ImVec4(0.30f, 0.55f, 0.95f, 1);
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.45f, 0.70f, 1, 1);
    c[ImGuiCol_Header] = ImVec4(0.20f, 0.45f, 0.85f, 0.45f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.20f, 0.45f, 0.85f, 0.65f);
    c[ImGuiCol_Separator] = ImVec4(0.25f, 0.28f, 0.34f, 1);
    style.ScaleAllSizes(scale);
    style.FontScaleDpi = scale;
    for (const auto& path : plat::ui_fonts()) {
        if (FILE* f = fopen(path.c_str(), "rb")) {
            fclose(f);
            if (io.Fonts->AddFontFromFileTTF(path.c_str(), FONT_BASE)) break;
        }
    }
    ImGui_ImplSDL2_InitForSDLRenderer(win, a.ren);
    ImGui_ImplSDLRenderer2_Init(a.ren);

    // скорость ядра этого ПК для имитации устройств (~0.1 с, пока видео не открыто)
    a.pcMs = calibrate_ms_per_mp();
    for (int i = 0; i < N_DEVICES; ++i) a.devMs[i] = DEVICES[i].msPerMp;
    a.cpuOn.assign(std::min(64, plat::logical_cpus()), 1);
    const uint64_t aff = plat::process_affinity();
    for (size_t i = 0; i < a.cpuOn.size(); ++i) a.cpuOn[i] = (aff >> i) & 1;
    a.s.det.threads = 0;
    a.pipe.setSettings(a.s);
    {   // мультитест: что перебирать по умолчанию — то, что чаще всего меняли на роликах
        const auto& P = mt::params();
        a.mtSel.resize(P.size());
        for (size_t i = 0; i < P.size(); ++i) {
            snprintf(a.mtSel[i].vals, sizeof(a.mtSel[i].vals), "%s", P[i].values);
            for (const char* k : {"auto_thr", "thr", "skyline", "lock_slow"}) a.mtSel[i].on |= !strcmp(P[i].key, k);
        }
    }
    if (argc > 1) open_video(a, argv[1]);
    a.cpuWall = now_ms();
    a.cpuT = plat::process_cpu_seconds();

    bool done = false;
    while (!done) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL2_ProcessEvent(&e);
            if (e.type == SDL_QUIT) done = true;
            if (e.type == SDL_DROPFILE) {
                open_video(a, e.drop.file);
                SDL_free(e.drop.file);
            }
        }
        std::string picked;
        if (plat::video_dialog_result(picked) && !picked.empty()) open_video(a, picked);
        if (plat::folder_dialog_result(picked) && !picked.empty())
            snprintf(a.mtFolder, sizeof(a.mtFolder), "%s", picked.c_str());
        if (SDL_GetWindowFlags(win) & SDL_WINDOW_MINIMIZED) {
            SDL_Delay(20);
            continue;
        }
        // загрузка CPU процессом: раз в полсекунды
        const double nw = now_ms();
        if (nw - a.cpuWall > 500) {
            const double ct = plat::process_cpu_seconds();
            a.cpuPct = 100.0 * (ct - a.cpuT) / ((nw - a.cpuWall) / 1000.0) / plat::logical_cpus();
            a.cpuT = ct;
            a.cpuWall = nw;
        }
        // новый кадр на текстуру — не чаще, чем разрешено настройкой отображения
        std::shared_ptr<const Result> r;
        if (a.viewSeq >= 0 && !(r = a.pipe.at(a.viewSeq))) a.viewSeq = -1;   // кадр вытеснен из истории
        const bool still = r || a.pipe.paused();   // на паузе кадр показываем сразу
        if (!r) r = a.pipe.latest();
        static const double minGap[] = {0, 1000.0 / 30, 1000.0 / 10, 1e18};
        if (r && r->seq != a.texSeq && a.displayMode != 3 && (still || nw - a.texAt >= minGap[a.displayMode])) {
            upload_frame(a, *r->frame);
            if (a.showCrops) upload_crops(a, *r, top_targets(*r));
            a.texSeq = r->seq;
            a.texAt = nw;
            a.shown = r;
        }
        // своя ссылка на кадр: кнопка "Открыть видео..." посреди кадра интерфейса сбрасывает
        // a.shown, а панели ниже ещё рисуют этот кадр (раньше — обращение к удалённой памяти
        // и вылет при открытии видео)
        ann_follow_update(a);
        const std::shared_ptr<const Result> shownRef = a.shown;
        const Result* shown = shownRef.get();
        const Metrics m = a.pipe.metrics();

        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        if (a.appMode == 1) {   // эмулятор: кадр широкой — когда он новый, точной — каждый кадр интерфейса
            a.emuView = a.emu.view();
            const twocam::View& v = a.emuView;
            if (v.wide && v.wide->seq != a.emuWSeq) {
                const Frame& f = *v.wide->frame;
                upload_gray(a, a.emuWTex, a.emuWTw, a.emuWTh, f.px.data(), f.w, f.h, std::max(1, f.w / 1152));
                a.emuWSeq = v.wide->seq;
            }
            if (!v.narrow.empty()) upload_gray(a, a.emuNTex, a.emuNTw, a.emuNTh, v.narrow.data(), v.nw, v.nh, 1);
            if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Tab, false)) a.emu.nextTarget();
            if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Space, false)) a.emu.setPaused(!a.emu.paused());
        }
        if (!io.WantTextInput && a.pipe.isOpen() && a.appMode == 0) {
            if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) toggle_pause(a);
            const int64_t curIdx = shown ? shown->frame->index : 0;
            const int64_t sec = (int64_t)std::max(1.0, a.pipe.info().fps);
            if (io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) seek_to(a, curIdx - sec, true);
            else if (io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_RightArrow)) seek_to(a, curIdx + sec, false);
            else if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) step_back(a);
            else if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) step_forward(a);
            if (ImGui::IsKeyPressed(ImGuiKey_Home, false)) seek_to(a, 0, false);
            if (ImGui::IsKeyPressed(ImGuiKey_Tab, false)) a.pipe.nextTarget();
            if (a.annotate && a.ann.editable && ImGui::IsKeyPressed(ImGuiKey_Delete, false)) ann_delete(a, a.annCur);
            if (ImGui::IsKeyPressed(ImGuiKey_Equal, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd, false))
                set_speed(a, a.s.speed * 2);
            if (ImGui::IsKeyPressed(ImGuiKey_Minus, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract, false))
                set_speed(a, a.s.speed / 2);
        }
        const ImVec2 ds = io.DisplaySize;
        const float lw = 470 * scale, rw = 430 * scale;
        const ImGuiWindowFlags fixed = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                                       ImGuiWindowFlags_NoSavedSettings;
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(lw, ds.y));
        ImGui::Begin("Управление", nullptr, fixed);
        panel_controls(a);
        ImGui::End();
        ImGui::SetNextWindowPos(ImVec2(ds.x - rw, 0));
        ImGui::SetNextWindowSize(ImVec2(rw, ds.y));
        ImGui::Begin("Метрики", nullptr, fixed);
        if (a.appMode == 1) panel_twocam_metrics(a);
        else panel_metrics(a, m, shown);
        ImGui::End();
        ImGui::SetNextWindowPos(ImVec2(lw, 0));
        ImGui::SetNextWindowSize(ImVec2(std::max(50.f, ds.x - lw - rw), ds.y));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("Видео", nullptr, fixed | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        const float barH = a.pipe.isOpen() && a.appMode == 0 ? timeline_height() : 0;
        ImGui::BeginChild("view", ImVec2(0, -barH), ImGuiChildFlags_None,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        if (a.appMode == 1) draw_twocam(a);
        else draw_view(a, shown);
        ImGui::EndChild();
        if (a.pipe.isOpen() && a.appMode == 0) draw_timeline(a, shown, m);
        ImGui::End();
        ImGui::PopStyleVar();
        if (a.mtOpen && a.appMode == 0) {
            const float pad = 12 * scale;
            window_multitest(a, ImVec2(lw + pad, pad), ImVec2(std::max(300.f, ds.x - lw - 2 * pad), ds.y - 2 * pad));
        }

        ImGui::Render();
        SDL_RenderSetScale(a.ren, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
        SDL_SetRenderDrawColor(a.ren, 18, 18, 20, 255);
        SDL_RenderClear(a.ren);
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), a.ren);
        SDL_RenderPresent(a.ren);
    }
    a.emu.stop();
    a.emuView = twocam::View();
    a.pipe.close();
    a.shown.reset();
    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    if (a.tex) SDL_DestroyTexture(a.tex);
    if (a.cropTex) SDL_DestroyTexture(a.cropTex);
    if (a.emuWTex) SDL_DestroyTexture(a.emuWTex);
    if (a.emuNTex) SDL_DestroyTexture(a.emuNTex);
    SDL_DestroyRenderer(a.ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
