// Дронолок: тестовое приложение для ПК. Загружает видео, прогоняет его через нативный
// детектор и показывает, что алгоритм считает дроном, птицей, помехой и шумом, вместе
// с метриками скорости. Ресурсы можно урезать (потоки, ядра, SIMD, потолок CPU,
// разрешение, искусственное замедление), чтобы примерить алгоритм к слабому железу.
//
//   dronolok_app [видео]
//   dronolok_app --bench видео [--frames N] [--threads N] [--isa avx2] [--scales 0..4] [--motion 0|1]
//                [--focus] [--json out.json]
#include <SDL.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"
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

struct App {
    Pipeline pipe;
    Settings s;
    std::string err, note;
    // ресурсы
    std::vector<char> cpuOn;
    bool pinWorkers = false;
    int cpuCap = 100, priority = 0, resIdx = 0;
    // отображение
    bool showNoise = true, showClutter = true, showTracks = true, showPaths = true, showIds = true;
    bool showMask = false, showGt = true, showCrops = true, showPred = true, showFocus = false;
    int viewSeq = -1;      // кадр из истории на паузе (Result::seq), -1 — последний обработанный
    int displayMode = 1;   // 0 каждый кадр, 1 до 30 FPS, 2 до 10 FPS, 3 не показывать
    float zoom = 1.f, panX = 0, panY = 0;
    // текстуры
    SDL_Renderer* ren = nullptr;
    SDL_Texture* tex = nullptr;
    int texW = 0, texH = 0, texDs = 1, texSeq = -1;
    double texAt = 0;
    std::shared_ptr<const Result> shown;   // кадр, который сейчас на текстуре
    SDL_Texture* cropTex = nullptr;
    int maxTexW = 4096, maxTexH = 4096;
    // загрузка CPU
    double cpuT = 0, cpuWall = 0, cpuPct = 0;
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

// ------------------------------------------------------------ панели

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

void panel_controls(App& a) {
    bool changed = false;
    const VideoInfo vi = a.pipe.info();
    if (ImGui::CollapsingHeader("Видео", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Открыть видео...")) {
            std::string p = plat::open_video_dialog();
            if (!p.empty()) open_video(a, p);
        }
        ImGui::SameLine();
        ImGui::TextDisabled("или перетащите файл в окно");
        if (!a.err.empty()) ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "%s", a.err.c_str());
        if (a.pipe.isOpen()) {
            std::string name = a.pipe.path();
            size_t sl = name.find_last_of("/\\");
            if (sl != std::string::npos) name = name.substr(sl + 1);
            ImGui::TextWrapped("%s", name.c_str());
            ImGui::TextDisabled("%dx%d, %.2f к/с, %s (%s)%s", vi.w, vi.h, vi.fps, vi.codec.c_str(), vi.pixfmt.c_str(),
                                vi.frames ? (", ~" + std::to_string(vi.frames) + " кадров").c_str() : "");
            const bool paused = a.pipe.paused();
            if (ImGui::Button(paused ? "Продолжить" : "Пауза")) toggle_pause(a);
            ImGui::SameLine();
            if (ImGui::Button("< кадр")) step_back(a);
            ImGui::SameLine();
            if (ImGui::Button("кадр >")) step_forward(a);
            ImGui::SameLine();
            if (ImGui::Button("С начала")) open_video(a, a.pipe.path());
            // скорость
            static const double speeds[] = {0.1, 0.25, 0.5, 1.0};
            static const char* speedNames[] = {"x0.1", "x0.25", "x0.5", "x1"};
            for (int i = 0; i < 4; ++i) {
                if (i) ImGui::SameLine();
                const bool on = a.s.pace == Pace::Realtime && std::fabs(a.s.speed - speeds[i]) < 1e-3;
                if (on) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
                if (ImGui::Button(speedNames[i])) set_speed(a, speeds[i]);
                if (on) ImGui::PopStyleColor();
            }
            ImGui::SameLine();
            float sp = (float)a.s.speed;
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.6f);
            if (ImGui::SliderFloat("скорость", &sp, 0.05f, 2.f, "x%.2f", ImGuiSliderFlags_Logarithmic)) set_speed(a, sp);
            help_marker("Замедление видео: кадры подаются реже, каждый обрабатывается. Работает в режиме подачи "
                        "\"как камера\". Клавиши: пробел — пауза, стрелки влево/вправо — кадр назад/вперёд, "
                        "+/- — быстрее/медленнее.");
            // история обработанных кадров (назад на паузе)
            int first, last;
            if (paused && a.pipe.historyRange(first, last) && last > first) {
                int cur = a.viewSeq >= 0 ? a.viewSeq : last;
                ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.6f);
                char fmt[64];
                snprintf(fmt, sizeof(fmt), cur == last ? "последний" : "%d кадров назад", last - cur);
                if (ImGui::SliderInt("история", &cur, first, last, fmt)) a.viewSeq = cur >= last ? -1 : cur;
                help_marker("Последние обработанные кадры с разметкой (до 600 кадров и 1 ГБ памяти). Трекер при "
                            "этом не откатывается: вперёд с последнего кадра идёт обработка следующего.");
            }
            changed |= ImGui::Checkbox("Зациклить", &a.s.loop);
        }
    }
    if (ImGui::CollapsingHeader("Подача кадров", ImGuiTreeNodeFlags_DefaultOpen)) {
        int pace = a.s.pace == Pace::Realtime ? 0 : 1;
        changed |= ImGui::RadioButton("Как камера", &pace, 0);
        help_marker("Кадры приходят с частотой видео (или заданной). Если обработка не успевает, лишние кадры "
                    "выбрасываются, как у настоящей камеры. Счётчик пропусков — в метриках.");
        ImGui::SameLine();
        changed |= ImGui::RadioButton("Максимально быстро", &pace, 1);
        help_marker("Без пропусков и пауз: замер предельной скорости всего конвейера.");
        a.s.pace = pace == 0 ? Pace::Realtime : Pace::MaxSpeed;
        if (a.s.pace == Pace::Realtime) {
            float fps = (float)a.s.targetFps;
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.45f);
            if (ImGui::SliderFloat("к/с камеры", &fps, 0, 120, fps <= 0 ? "как в видео" : "%.0f")) {
                a.s.targetFps = fps;
                changed = true;
            }
        }
        int src = a.s.source == Source::Stream ? 0 : 1;
        bool srcCh = ImGui::RadioButton("Читать файл", &src, 0);
        ImGui::SameLine();
        srcCh |= ImGui::RadioButton("Предзагрузка в память", &src, 1);
        help_marker("Кадры заранее декодируются в память и подаются по кругу без копирования, как сырые буферы "
                    "камеры. Тогда декодирование видео не мешает замеру.");
        if (src == 1) {
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.45f);
            srcCh |= ImGui::SliderInt("кадров", &a.s.preloadFrames, 10, 1000);
            const int w = a.s.procW ? a.s.procW : vi.w, h = a.s.procH ? a.s.procH : vi.h;
            ImGui::TextDisabled("памяти: ~%.0f МБ", (double)w * h * a.s.preloadFrames / (1024.0 * 1024.0));
        }
        if (srcCh) {
            a.s.source = src == 0 ? Source::Stream : Source::Preload;
            changed = true;
        }
    }
    if (ImGui::CollapsingHeader("Ресурсы (имитация железа)", ImGuiTreeNodeFlags_DefaultOpen)) {
        const int hw = plat::logical_cpus();
        int th = a.s.det.threads > 0 ? a.s.det.threads : hw;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.45f);
        if (ImGui::SliderInt("потоков детектора", &th, 1, std::max(hw, 16))) {
            a.s.det.threads = th;
            changed = true;
        }
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.45f);
        if (ImGui::BeginCombo("SIMD", isa_ru(a.s.det.isa))) {
            std::vector<Isa> isas = available_isas();
            isas.insert(isas.begin(), Isa::Auto);
            for (Isa i : isas)
                if (ImGui::Selectable(isa_ru(i), a.s.det.isa == i)) {
                    a.s.det.isa = i;
                    changed = true;
                }
            ImGui::EndCombo();
        }
        help_marker("Набор векторных инструкций. На Raspberry Pi 3/5 есть только NEON (128 бит), по ширине это "
                    "как SSE2. AVX2 — 256 бит, есть на большинстве ПК.");
        ImGui::TextUnformatted("Ядра процесса:");
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
        if (ImGui::SmallButton("все")) { std::fill(a.cpuOn.begin(), a.cpuOn.end(), 1); cpuCh = true; }
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
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.45f);
        if (ImGui::SliderInt("потоков декодера", &dth, 0, hw, dth == 0 ? "авто" : "%d")) {
            a.s.decoderThreads = dth;
            changed = true;
        }
        ImGui::BeginDisabled(!plat::cpu_cap_supported());
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.45f);
        if (ImGui::SliderInt("потолок CPU, %", &a.cpuCap, 5, 100, a.cpuCap >= 100 ? "нет" : "%d%%"))
            plat::set_cpu_cap(a.cpuCap);
        ImGui::EndDisabled();
        help_marker("Жёсткое ограничение загрузки всем процессом, в процентах от всех ядер ПК (Windows Job Object). "
                    "Имитирует более медленный процессор.");
        static const char* prios[] = {"низкий", "ниже среднего", "обычный", "выше среднего", "высокий"};
        int pr = a.priority + 2;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.45f);
        if (ImGui::Combo("приоритет", &pr, prios, 5)) {
            a.priority = pr - 2;
            plat::set_process_priority(a.priority);
        }
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.45f);
        if (ImGui::BeginCombo("разрешение", RES_OPTS[a.resIdx].label)) {
            for (int i = 0; i < (int)(sizeof(RES_OPTS) / sizeof(RES_OPTS[0])); ++i)
                if (ImGui::Selectable(RES_OPTS[i].label, a.resIdx == i)) {
                    a.resIdx = i;
                    set_resolution(a);
                    changed = true;
                }
            ImGui::EndCombo();
        }
        help_marker("Кадр масштабируется перед детектором. 16 Мп — растянуть ролик до размера кадра "
                    "широкоугольной камеры (как --upscale в run_demo.py).");
        float emu = (float)a.s.emulateMs;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.45f);
        if (ImGui::SliderFloat("мин. мс на кадр", &emu, 0, 200, emu <= 0 ? "выкл." : "%.0f мс")) {
            a.s.emulateMs = emu;
            changed = true;
        }
        help_marker("Имитация медленного устройства: обработка кадра занимает не меньше заданного времени. "
                    "Подставьте время, которое dronolok_bench показал на Raspberry Pi, и посмотрите, сколько "
                    "кадров будет теряться в режиме \"как камера\".");
        if (ImGui::SmallButton("Пресет: весь ПК")) {
            a.s.det.threads = 0;
            a.s.det.isa = Isa::Auto;
            std::fill(a.cpuOn.begin(), a.cpuOn.end(), 1);
            a.s.emulateMs = 0;
            a.cpuCap = 100;
            plat::set_cpu_cap(100);
            apply_resources(a);
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Пресет: 4 ядра + 128-бит SIMD")) {
            a.s.det.threads = 4;
            a.s.det.isa = Isa::SSE2;
            for (size_t i = 0; i < a.cpuOn.size(); ++i) a.cpuOn[i] = i < 4;
            a.pinWorkers = true;
            apply_resources(a);
            changed = true;
        }
        help_marker("Число ядер и ширина векторов как у Raspberry Pi. Сами ядра ПК быстрее, поэтому для точной "
                    "имитации добавьте \"мин. мс на кадр\" по замеру на устройстве.");
    }
    if (ImGui::CollapsingHeader("Алгоритм")) {
        DetectorConfig& d = a.s.det;
        const float w = ImGui::GetContentRegionAvail().x * 0.4f;
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::SliderFloat("порог контраста", &d.thr, 3, 40, "%.0f");
        help_marker("Минимальная разница яркости цели и фона, уровни 0..255.");
        int kk = d.k;
        ImGui::SetNextItemWidth(w);
        if (ImGui::SliderInt("макс. размер цели k", &kk, 3, 15)) {
            d.k = kk | 1;
            changed = true;
        }
        help_marker("В пикселях после пулинга (нечётное). При пулинге 2x2 цель до ~2k пикселей кадра.");
        bool pool2 = d.pool == 2;
        if (ImGui::Checkbox("пулинг 2x2", &pool2)) {
            d.pool = pool2 ? 2 : 1;
            changed = true;
        }
        ImGui::SameLine();
        changed |= ImGui::Checkbox("тёмные", &d.dark);
        ImGui::SameLine();
        changed |= ImGui::Checkbox("яркие", &d.bright);
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::SliderFloat("выше клаттера, раз", &d.clutter_k, 1, 8, "%.1f");
        ImGui::SetNextItemWidth(w);
        float dens = d.max_density * 100;
        if (ImGui::SliderFloat("текстура тайла, %", &dens, 0.2f, 5, "%.1f")) {
            d.max_density = dens / 100;
            changed = true;
        }
        help_marker("Тайл, где откликов больше этой доли, считается текстурой (листва, рябь облаков) и пропускается.");
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::SliderInt("кадров на маску неба", &a.s.learnMaskFrames, 0, 100, a.s.learnMaskFrames ? "%d" : "выкл.");
        ImGui::SameLine();
        if (ImGui::SmallButton("заново")) a.pipe.relearnMask();
        help_marker("Тайлы, где по первым кадрам видна текстура (деревья, дома), дальше не просматриваются. Только для "
                    "неподвижной камеры: при учёте движения камеры маска не строится.");
        ImGui::SeparatorText("Крупные цели и движение камеры");
        static const char* scl[] = {"авто", "1: до ~7 пикс.", "2: до ~28 пикс.", "3: до ~56 пикс.", "4: до ~112 пикс."};
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::Combo("размер цели", &a.s.scales, scl, 5);
        help_marker("Сколько масштабов просматривать. 1 — только точки (дальний дрон), каждый следующий — кадр, "
                    "уменьшенный ещё в 2 раза, для близкой крупной цели. Каждый добавляет ~20-35% времени. "
                    "Авто: до 2 Мп — 3, больше — 1.");
        int mo = a.s.motion + 1;
        static const char* mos[] = {"авто", "не учитывать", "учитывать"};
        ImGui::SetNextItemWidth(w);
        if (ImGui::Combo("движение камеры", &mo, mos, 3)) {
            a.s.motion = mo - 1;
            changed = true;
        }
        help_marker("Съёмка с рук или с поворотного устройства: сдвиг всей картинки между кадрами оценивается "
                    "(~1-2 мс) и прогноз целей переносится вместе с ним, иначе захват срывается при каждом "
                    "движении камеры. Авто: до 2 Мп — учитывать.");
        if (a.pipe.isOpen()) {
            const Effective ef = effective(a.s, a.s.procW ? a.s.procW : vi.w, a.s.procH ? a.s.procH : vi.h);
            ImGui::TextDisabled("сейчас: масштабов %d, движение камеры %s", ef.scales, ef.motion ? "учитывается" : "нет");
        }
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::SliderFloat("порог крупных, раз", &d.coarse_thr_k, 1, 6, "%.1f");
        help_marker("Во сколько раз порог контраста выше на уменьшенных масштабах: крупная близкая цель "
                    "контрастна, а пятна облаков на них дают ложные цели.");
        ImGui::SeparatorText("Экономия");
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::SliderInt("весь кадр раз в, кадров", &d.full_every, 1, 4, d.full_every > 1 ? "%d" : "каждый");
        help_marker("В остальные кадры детектор смотрит только вокруг уже найденных целей. Ведомая цель не "
                    "теряется, новая находится с задержкой до N-1 кадров. На 16 Мп раз в 2 кадра — минус ~35% "
                    "времени; на синтетике полнота 96.7% -> 95%. Для Raspberry Pi.");
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::SliderInt("крупные масштабы раз в, кадров", &d.coarse_every, 1, 4, d.coarse_every > 1 ? "%d" : "каждый");
        help_marker("Уменьшенные копии кадра (поиск крупной цели) целиком — раз в N кадров, а между ними только "
                    "если крупная цель уже ведётся. 16 Мп, 3 масштаба: 7.4 мс -> 5.7 (раз в 2) -> 4.8 (раз в 4).");
        changed |= ImGui::Checkbox("искать возле прогноза", &a.s.focus);
        help_marker("Эксперимент: там, где трек ожидается на следующем кадре, порог ниже, чтобы не терять "
                    "бледнеющую цель. На ваших роликах пока чуть хуже, поэтому выключено.");
        if (a.s.focus) {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(w * 0.6f);
            changed |= ImGui::SliderFloat("порог там", &d.focus_k, 0.2f, 1, "x%.2f");
        }
        ImGui::SeparatorText("Трекер, захват и классы");
        TrackerConfig& t = a.s.trk;
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::SliderFloat("макс. радиус поиска, пикс.", &t.gate, 5, 150, "%.0f");
        help_marker("Где искать цель на следующем кадре: вокруг прогноза по её скорости. Окно растёт с "
                    "неуверенностью прогноза (пропуски, манёвры), но не больше этого.");
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::SliderFloat("манёвренность цели", &t.q_accel, 0.25f, 20, "%.2f", ImGuiSliderFlags_Logarithmic);
        help_marker("Ускорение в фильтре Калмана, пикс./кадр². Меньше — прогноз жёстче, захват плавно летящего дрона "
                    "устойчивее; больше — окно шире, не теряет резкие манёвры (птицы).");
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::SliderInt("держать на прогнозе, кадров", &t.max_miss, 1, 60);
        help_marker("Сколько кадров подряд цель может не находиться (ушла за провод, слилась с фоном), а трек и "
                    "захват идут по прогнозу. Новые и стоящие на месте треки держатся только 1 кадр.");
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::SliderFloat("смена захвата, раз", &t.lock_switch, 1, 5, "%.1f");
        help_marker("Захват переходит на другую цель, только если она весомее текущей во столько раз.");
        bool stOn = t.static_frames > 0;
        if (ImGui::Checkbox("память статичных помех", &stOn)) {
            t.static_frames = stOn ? 3000 : 0;
            changed = true;
        }
        help_marker("Для неподвижной камеры. Места, где цели раз за разом появляются и никуда не летят (катушки и "
                    "изоляторы ЛЭП, фонари, антенны), запоминаются, и там больше не рождаются цели. Дрон, пролетающий "
                    "мимо такого места, не теряется. Память лежит рядом с видео (.static.bin): загружается при "
                    "открытии, сохраняется при закрытии. При учёте движения камеры не ведётся.");
        if (stOn) {
            ImGui::SetNextItemWidth(w);
            changed |= ImGui::SliderFloat("запомнить после, откликов", &t.static_hits, 3, 40, "%.0f");
            help_marker("Сколько неподвижных откликов в одном месте делают его помехой. Меньше — быстрее учится, "
                        "но зависший на одном месте дрон тоже может стать \"помехой\" (если его трек много раз "
                        "терялся, а дрон так и не сдвинулся).");
            ImGui::SetNextItemWidth(w);
            changed |= ImGui::SliderInt("забывать за, кадров", &t.static_frames, 300, 30000, "%d",
                                        ImGuiSliderFlags_Logarithmic);
            if (ImGui::SmallButton("сохранить память")) a.pipe.staticSave();
            ImGui::SameLine();
            if (ImGui::SmallButton("забыть всё")) a.pipe.staticClear();
        }
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::SliderInt("подтверждение, кадров", &t.confirm, 1, t.window);
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::SliderFloat("птица: отклонение, пикс.", &t.bird_resid, 0.5f, 10, "%.1f");
        help_marker("Средний разброс точек трека вокруг плавной траектории за последние 16 попаданий. "
                    "Дрон летит плавно (~1 пикс.), ласточка дёргается (3-10 пикс.).");
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::SliderFloat("птица: мигание контраста", &t.bird_contrast_cv, 0.05f, 1, "%.2f");
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::SliderFloat("птица: скорость, пикс./кадр", &t.bird_speed, 0, 40, t.bird_speed > 0 ? "%.0f" : "не учитывать");
        ImGui::SetNextItemWidth(w);
        changed |= ImGui::SliderInt("попаданий до решения", &t.cls_min_hits, 3, t.cls_window);
    }
    if (ImGui::CollapsingHeader("Отображение")) {
        static const char* modes[] = {"каждый кадр", "до 30 к/с", "до 10 к/с", "не показывать видео"};
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.45f);
        ImGui::Combo("обновление", &a.displayMode, modes, 4);
        help_marker("Отрисовка 16 Мп кадра тоже тратит CPU. Для чистого замера скорости выберите \"не показывать\".");
        ImGui::Checkbox("цели и классы", &a.showTracks);
        ImGui::SameLine();
        ImGui::Checkbox("траектории", &a.showPaths);
        ImGui::SameLine();
        ImGui::Checkbox("номера", &a.showIds);
        ImGui::Checkbox("шум", &a.showNoise);
        ImGui::SameLine();
        ImGui::Checkbox("помехи", &a.showClutter);
        ImGui::SameLine();
        ImGui::Checkbox("маска неба", &a.showMask);
        ImGui::Checkbox("разметка (.gt.json)", &a.showGt);
        ImGui::SameLine();
        ImGui::Checkbox("вырезки целей", &a.showCrops);
        ImGui::Checkbox("прогноз движения", &a.showPred);
        help_marker("Линия — куда цель придёт за 10 кадров при нынешней скорости (у захвата и устоявшихся треков). "
                    "Пунктирная рамка — трек идёт по прогнозу; у захвата круг — где цель может быть (3 сигмы).");
        ImGui::SameLine();
        ImGui::Checkbox("зоны пониженного порога", &a.showFocus);
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

void panel_metrics(App& a, const Metrics& m, const Result* r) {
    const float fps = (float)m.procFps;
    ImGui::Text("Обработано:");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.4f, 1, 0.5f, 1), "%.1f к/с", fps);
    const double detFps = m.detect.mean > 0 ? 1000.0 / m.detect.mean : 0;
    ImGui::Text("Детектор: %.2f мс/кадр", m.detect.mean);
    ImGui::SameLine();
    ImGui::TextDisabled("(до %.0f к/с)", detFps);
    if (r) ImGui::TextDisabled("кадр %dx%d (%.1f Мп), %s, потоков %d", r->frame->w, r->frame->h,
                               r->frame->w * (double)r->frame->h / 1e6, m.backend.c_str(), m.threads);
    if (!m.status.empty()) ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1), "%s", m.status.c_str());

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
    const Settings s = a.pipe.settings();
    const double fpsSrc = s.targetFps > 0 ? s.targetFps : m.sourceFps;
    draw_chart(m, s.pace == Pace::Realtime ? (float)(1000.0 / std::max(0.1, fpsSrc)) : 0.f,
               ImVec2(ImGui::GetContentRegionAvail().x, 110));

    ImGui::SeparatorText("Конвейер");
    const double dropPct = m.processed + m.dropped ? 100.0 * m.dropped / (m.processed + m.dropped) : 0;
    ImGui::Text("кадров: %ld, пропущено: %ld (%.1f%%)", m.processed, m.dropped, dropPct);
    ImGui::Text("декодер: %.1f к/с, видео: %.1f к/с", m.decodeFps, m.sourceFps);
    ImGui::Text("очередь: %d / %d, время замера %.0f с", m.queue, m.queueCap, m.elapsedS);

    ImGui::SeparatorText("Процесс");
    ImGui::Text("CPU: %.0f%% (≈ %.1f ядра из %d)", a.cpuPct, a.cpuPct / 100.0 * plat::logical_cpus(),
                plat::logical_cpus());
    ImGui::Text("память: %.0f МБ", plat::process_memory_mb());
    ImGui::TextDisabled("%s", plat::cpu_model().c_str());

    if (r) {
        int dr = 0, bi = 0, un = 0, cl = (int)r->clutter.size();
        for (const auto& t : r->tracks) {
            if (t.clutter) { ++cl; continue; }
            dr += t.cls == TargetClass::Drone;
            bi += t.cls == TargetClass::Bird;
            un += t.cls == TargetClass::Unknown;
        }
        ImGui::SeparatorText("На кадре");
        ImGui::TextColored(ImColor(COL_DRONE).Value, "дронов %d", dr);
        ImGui::SameLine();
        ImGui::TextColored(ImColor(COL_BIRD).Value, "птиц %d", bi);
        ImGui::SameLine();
        ImGui::TextColored(ImColor(COL_UNKNOWN).Value, "неясно %d", un);
        ImGui::TextColored(ImColor(COL_CLUTTER).Value, "помех %d", cl);
        ImGui::SameLine();
        ImGui::TextColored(ImColor(COL_NOISE).Value, "шум %d", (int)r->noise.size());
        ImGui::SameLine();
        ImGui::TextDisabled("кандидатов %d", (int)r->cands.size());
        if (r->learning) ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1), "учится маска неба...");
        const Track* lk = nullptr;
        for (const auto& t : r->tracks)
            if (t.id == r->lockId) lk = &t;
        if (lk) {
            ImGui::TextColored(ImColor(COL_LOCK).Value, "захват: #%d %s", lk->id, class_ru(*lk));
            ImGui::SameLine();
            if (lk->miss) ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1), "на прогнозе %d кадр.", lk->miss);
            else ImGui::TextDisabled("ведётся");
        } else {
            ImGui::TextDisabled("захвата нет");
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(%.0f%% кадров с захватом)", m.lockPct);
        if (r->motionOn) {
            if (r->camValid) ImGui::Text("камера: сдвиг %+.1f, %+.1f пикс./кадр", r->camDx, r->camDy);
            else ImGui::TextDisabled("камера: сдвиг не определён (нет текстуры)");
        } else {
            ImGui::TextDisabled("движение камеры не учитывается");
        }
        ImGui::TextDisabled("масштабов %d", r->scales);
        if (r->staticOn) {
            ImGui::SameLine();
            ImGui::TextColored(ImColor(COL_STATIC).Value, "запомнено помех: %d", (int)r->staticCells.size());
        }
        if (!m.staticMsg.empty()) ImGui::TextDisabled("%s", m.staticMsg.c_str());
    }
    if (m.haveGt) {
        ImGui::SeparatorText("Качество по разметке");
        if (m.eval.frames == 0) {
            ImGui::TextDisabled("считается после первого прохода видео");
        } else {
            ImGui::Text("найдено дронов %.1f%%, птиц %.1f%%", 100 * m.eval.recall(0), 100 * m.eval.recall(1));
            ImGui::Text("ложных треков на кадр: %.2f", m.eval.falsePerFrame());
            ImGui::Text("класс верный: дроны %.0f%%, птицы %.0f%%", 100 * m.eval.clsAccuracy(0), 100 * m.eval.clsAccuracy(1));
        }
    }
    ImGui::Separator();
    if (ImGui::Button("Сбросить")) a.pipe.resetStats();
    ImGui::SameLine();
    if (ImGui::Button("CSV по кадрам")) {
        std::string p = a.pipe.path() + ".metrics.csv";
        a.note = a.pipe.exportCsv(p) ? "сохранено: " + p : "не удалось сохранить " + p;
    }
    ImGui::SameLine();
    if (ImGui::Button("Отчёт JSON")) {
        std::string p = a.pipe.path() + ".report.json";
        FILE* f = fopen(p.c_str(), "w");
        if (f) {
            fprintf(f, "%s\n", a.pipe.reportJson().c_str());
            fclose(f);
        }
        a.note = f ? "сохранено: " + p : "не удалось сохранить " + p;
    }
    if (!a.note.empty()) ImGui::TextWrapped("%s", a.note.c_str());

    if (a.showCrops && r && a.cropTex) {
        ImGui::SeparatorText("Цели крупно");
        const auto targets = top_targets(*r);
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
    if (a.showMask && !r->mask.empty()) {
        for (int ty = 0; ty < r->tilesY; ++ty)
            for (int tx = 0; tx < r->tilesX; ++tx)
                if (!r->mask[(size_t)ty * r->tilesX + tx])
                    dl->AddRectFilled(P((float)tx * r->tilePx, (float)ty * r->tilePx),
                                      P((float)(tx + 1) * r->tilePx, (float)(ty + 1) * r->tilePx), IM_COL32(0, 60, 120, 90));
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
    if (a.showGt)
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
            if (t.id == r->lockId) {   // прицел захвата: уголки и перекрестие
                const float g = h + 9, l = 8;
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
                char b[48];
                snprintf(b, sizeof(b), t.miss ? "ЗАХВАТ (прогноз %d)" : "ЗАХВАТ", t.miss);
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
            if (hovered && std::fabs(io.MousePos.x - q.x) < h + 2 && std::fabs(io.MousePos.y - q.y) < h + 2) hover = &t;
        }
    dl->PopClipRect();
    // строка состояния и легенда поверх видео
    char st[320];
    int len = snprintf(st, sizeof(st), "%dx%d  кадр %lld  проход %d  детекция %.2f мс  обработка %.2f мс", f.w, f.h,
                       (long long)f.index, r->loop + 1, r->dt.total_ms, r->procMs);
    if (r->motionOn && r->camValid)
        len += snprintf(st + len, sizeof(st) - len, "  камера %+.1f %+.1f", r->camDx, r->camDy);
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
        ImGui::Text("ошибка прогноза %.1f пикс., вес для захвата %.1f", hover->sigma, hover->score);
        if (hover->miss) ImGui::Text("на прогнозе %d кадр.", hover->miss);
        ImGui::EndTooltip();
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
        else if (a == "--focus") s.focus = true;
        else if (a == "--static") s.trk.static_frames = atoi(val().c_str());
        else if (a == "--full-every") s.det.full_every = atoi(val().c_str());
        else if (a == "--coarse-every") s.det.coarse_every = atoi(val().c_str());
        else if (a == "--json") json = val();
        else if (a == "--csv") csv = val();
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
    printf("видео %dx%d, %.2f к/с, %s\n", p.info().w, p.info().h, p.info().fps, p.info().codec.c_str());
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
        if (FILE* f = fopen(json.c_str(), "w")) {
            fprintf(f, "%s\n", rep.c_str());
            fclose(f);
        }
    if (!csv.empty()) p.exportCsv(csv);
    p.close();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") return run_bench(argc, argv);

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
    style.FrameRounding = 3;
    style.ScaleAllSizes(scale);
    style.FontScaleDpi = scale;
    for (const auto& path : plat::ui_fonts()) {
        if (FILE* f = fopen(path.c_str(), "rb")) {
            fclose(f);
            if (io.Fonts->AddFontFromFileTTF(path.c_str(), 16.f)) break;
        }
    }
    ImGui_ImplSDL2_InitForSDLRenderer(win, a.ren);
    ImGui_ImplSDLRenderer2_Init(a.ren);

    a.cpuOn.assign(std::min(64, plat::logical_cpus()), 1);
    const uint64_t aff = plat::process_affinity();
    for (size_t i = 0; i < a.cpuOn.size(); ++i) a.cpuOn[i] = (aff >> i) & 1;
    a.s.det.threads = 0;
    a.pipe.setSettings(a.s);
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
        const Result* shown = a.shown.get();
        const Metrics m = a.pipe.metrics();

        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        if (!io.WantTextInput && a.pipe.isOpen()) {
            if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) toggle_pause(a);
            if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) step_back(a);
            if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) step_forward(a);
            if (ImGui::IsKeyPressed(ImGuiKey_Equal, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd, false))
                set_speed(a, a.s.speed * 2);
            if (ImGui::IsKeyPressed(ImGuiKey_Minus, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract, false))
                set_speed(a, a.s.speed / 2);
        }
        const ImVec2 ds = io.DisplaySize;
        const float lw = 420 * scale, rw = 400 * scale;
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
        panel_metrics(a, m, shown);
        ImGui::End();
        ImGui::SetNextWindowPos(ImVec2(lw, 0));
        ImGui::SetNextWindowSize(ImVec2(std::max(50.f, ds.x - lw - rw), ds.y));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("Видео", nullptr, fixed | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        draw_view(a, shown);
        ImGui::End();
        ImGui::PopStyleVar();

        ImGui::Render();
        SDL_RenderSetScale(a.ren, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
        SDL_SetRenderDrawColor(a.ren, 18, 18, 20, 255);
        SDL_RenderClear(a.ren);
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), a.ren);
        SDL_RenderPresent(a.ren);
    }
    a.pipe.close();
    a.shown.reset();
    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    if (a.tex) SDL_DestroyTexture(a.tex);
    if (a.cropTex) SDL_DestroyTexture(a.cropTex);
    SDL_DestroyRenderer(a.ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
