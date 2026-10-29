#include "multitest.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <map>
#include <set>

#include "dronolok/fileutil.hpp"
#include "dronolok/tuning.hpp"
#include "gt_eval.hpp"
#include "platform.hpp"
#include "video_source.hpp"

using namespace dronolok;
namespace fs = std::filesystem;

namespace mt {

// ------------------------------------------------------------ настройки для перебора

#define D(field) [](const Setup& u) -> double { return (double)u.s.det.field; }, [](Setup& u, double v) { u.s.det.field = (decltype(u.s.det.field))v; }
#define T(field) [](const Setup& u) -> double { return (double)u.s.trk.field; }, [](Setup& u, double v) { u.s.trk.field = (decltype(u.s.trk.field))v; }
#define S(field) [](const Setup& u) -> double { return (double)u.s.field; }, [](Setup& u, double v) { u.s.field = (decltype(u.s.field))v; }

static void set_profile(Setup& u, double v) {
    const AlgoBlocks b = profile_blocks((Profile)std::min(3, std::max(0, (int)v)));
    TrackerConfig t = u.s.trk;
    apply_blocks(b, u.s.det, t);
    u.s.trk.static_frames = t.static_frames;
    u.s.scales = b.scales;
    u.s.motion = b.motion ? -1 : 0;
    u.s.zones = b.zones ? -1 : 0;
    u.s.learnMaskFrames = b.skyMask ? 25 : 0;
    u.scale = b.resolution;
}

const std::vector<Param>& params() {
    static const std::vector<Param> p = {
        // целиком профиль: блоки, как кнопки «Полный / Баланс / Эконом / Минимум»
        {"profile", "Профиль алгоритма (0 полный, 1 баланс, 2 эконом, 3 минимум)", "Главное", false, "0, 1, 2, 3",
         [](const Setup&) -> double { return -1; }, set_profile},
        {"res", "Разрешение обработки, доля", "Железо", false, "1, 0.75, 0.5",
         [](const Setup& u) -> double { return u.scale; }, [](Setup& u, double v) { u.scale = (float)v; }},
        // Алгоритм
        {"scales", "Размер цели, масштабов (0 авто)", "Алгоритм", false, "0, 1, 2, 3", S(scales)},
        {"motion", "Учитывать движение камеры (-1 авто, 0 нет, 1 да)", "Алгоритм", false, "-1, 0, 1", S(motion)},
        {"zones", "Зоны кадра (-1 авто, 0 нет, 1 да)", "Алгоритм", false, "-1, 0, 1", S(zones)},
        {"thin_tiles", "Зоны: столб не шире, тайлов — не земля (0 выкл.)", "Алгоритм", false, "0, 1, 2", D(thin_tiles)},
        {"zone_strip", "Зоны: полоса под сеткой у нижнего края кадра — земля, если там текстура", "Алгоритм", true, "0, 1", D(zone_strip)},
        {"halo_k", "Светлая кайма резкости у тёмного (столб, фонарь) — не цель: во сколько раз темнее контраста (0 выкл.)", "Алгоритм", false, "0, 1.5, 2, 3", D(halo_k)},
        {"zone_vote", "Зоны не мигают (0 нет, 1 большинство из 3 кадров, 2 земля держится кадр)", "Алгоритм", false, "0, 1, 2", D(zone_vote)},
        {"dark", "Тёмные цели", "Алгоритм", true, "1, 0", D(dark)},
        {"bright", "Яркие цели", "Алгоритм", true, "1, 0", D(bright)},
        {"trk_seconds", "Трекер в секундах", "Алгоритм", true, "1, 0", S(trkSeconds)},
        {"fps_scale", "Маска неба и поворот камеры в секундах", "Алгоритм", true, "1, 0", S(fpsScale)},
        {"follow", "Виртуальная поворотка", "Алгоритм", true, "0, 1", S(follow)},
        {"mask", "Кадров на маску неба (0 выкл.)", "Алгоритм", false, "0, 25, 50", S(learnMaskFrames)},
        {"per_tile", "Кандидатов в тайле", "Алгоритм", false, "1, 2", D(per_tile)},
        {"skyline", "Всё ниже горизонта — земля", "Алгоритм", true, "1, 0", D(skyline)},
        {"adapt", "Подстраиваться под сцену и погоду", "Алгоритм", true, "1, 0", S(adapt)},
        {"bg_map", "Фон по ячейкам у кромки леса", "Алгоритм", true, "1, 0", D(bg_map)},
        {"bg_near", "Фон по ячейкам и в небе у кромки (кончики веток)", "Алгоритм", true, "1, 0", D(bg_near)},
        {"grow_mask", "Маска дорастает, где копятся помехи", "Алгоритм", true, "1, 0", D(grow_mask)},
        {"mask_recheck", "Маска перепроверяется (облака открываются)", "Алгоритм", true, "1, 0", D(mask_recheck)},
        {"pol_mask", "Облака закрыты только для светлых целей (0 нет, 1 тёмные ищутся, 2 и наоборот)", "Алгоритм", true, "1, 0, 2", D(pol_mask)},
        {"pol_tex", "Маска по полярности: другой текстуры не чаще доли кадров", "Тонко", false, "0.02, 0.05, 0.1", D(pol_tex)},
        {"sky_color", "Облака по цвету: не зелёное, не темнее неба и не стоит на земле — не закрывать как землю", "Алгоритм", true, "1, 0", D(sky_color)},
        {"sky_green", "Облака по цвету: листва — зелёный выше синего больше доли яркости", "Тонко", false, "0.05, 0.08, 0.12", D(sky_green)},
        {"sky_y", "Облака по цвету: не темнее доли яркости неба", "Тонко", false, "0.6, 0.7, 0.8", D(sky_y)},
        {"sky_dark", "Облака по цвету: 10% самых тёмных точек не темнее доли неба", "Тонко", false, "0.5, 0.6, 0.7", D(sky_dark)},
        {"sky_soft", "Мягкое облако — облако и у самой кромки леса (линия горизонта через него не идёт)", "Алгоритм", true, "1, 0", D(sky_soft)},
        {"sky_d4", "Мягкое облако: перепад на 4 пикс. меньше доли яркости", "Тонко", false, "0.06, 0.08, 0.1", D(sky_d4)},
        {"sky_s14", "Мягкое облако: перепад на 1 пикс. меньше доли перепада на 4", "Тонко", false, "0.28, 0.32, 0.36", D(sky_s14)},
        {"sky_tree", "Мягкое облако над тёмной верхушкой (доля точек темнее sky_dark неба) поднимает линию горизонта", "Тонко", false, "0, 0.25, 0.35, 0.45", D(sky_tree)},
        {"sky_bright", "Облако по цвету открыто и для светлых целей", "Алгоритм", true, "1, 0", D(sky_bright)},
        {"lock_through", "Цель в захвате ищется и под маской (только продолжение трека)", "Алгоритм", true, "1, 0", D(lock_through)},
        {"lock_through_k", "Под маской: порог выше общего, раз", "Тонко", false, "1.2, 1.5, 2", D(lock_through_k)},
        {"reopen_quiet", "Доросшая маска открывается: тихих замеров не меньше доли (1.01 — никогда)", "Тонко", false, "0.8, 0.9, 1.01", D(reopen_quiet)},
        {"recheck_tex", "Перепроверка: сильный отклик реже доли кадров — облако", "Тонко", false, "0.1, 0.2, 0.4", D(recheck_tex)},
        {"grow_k", "Дорастание: помех выше среднего, раз", "Тонко", false, "3, 4, 6", D(grow_k)},
        {"grow_min", "Дорастание: помех на тайл за кадр, не меньше", "Тонко", false, "0.05, 0.1, 0.2", D(grow_min)},
        {"grow_frames", "Дорастание: окно, кадров при 60 к/с", "Тонко", false, "60, 120, 240", D(grow_frames)},
        {"bg_frames", "Фон ячеек у кромки: окно, кадров", "Тонко", false, "125, 250, 500", D(bg_frames)},
        {"fine_frames", "Уточнение маски у кромки: окно, кадров при 60 к/с", "Тонко", false, "75, 150, 300", D(fine_frames)},
        {"grow_cells", "Дорастание: помехи хотя бы в стольких ячейках тайла", "Тонко", false, "1, 2, 3", D(grow_cells)},
        {"tbd", "Накопление слабых целей", "Алгоритм", true, "0, 1", D(tbd)},
        {"band", "Полоса у горизонта, пикс. (0 выкл.)", "Алгоритм", false, "0, 256, 512", D(band_px)},
        {"upper_every", "Небо выше полосы раз в, кадров", "Алгоритм", false, "1, 2, 3", D(upper_every)},
        {"static", "Память статичных помех, кадров (0 выкл.)", "Алгоритм", false, "0, 3000", T(static_frames)},
        {"full_every", "Весь кадр раз в, кадров", "Алгоритм", false, "1, 2, 3", D(full_every)},
        {"coarse_every", "Крупные масштабы раз в, кадров", "Алгоритм", false, "1, 2, 4", D(coarse_every)},
        {"dedup", "Пропускать повторы кадра", "Видео", true, "0, 1", S(skipDuplicates)},
        {"step", "Камера реже видео: каждый N-й кадр", "Видео", false, "1, 4, 6", S(frameStep)},
        {"phase", "Камера реже видео: с какого кадра файла считать шаг (сдвиг)", "Видео", false, "0, 1, 2, 3", S(framePhase)},
        {"cam_fps", "Частота камеры, к/с (0 — как в видео)", "Видео", false, "0, 15, 30", S(targetFps)},
        // Тонко: порог
        {"auto_thr", "Порог по шуму", "Тонко", true, "0, 1", D(auto_thr)},
        {"noise_k", "Порог: во сколько раз выше шума", "Тонко", false, "8, 12, 16", D(noise_k)},
        {"thr", "Порог контраста", "Тонко", false, "6, 8, 10, 14", D(thr)},
        {"clutter_k", "Выше клаттера, раз", "Тонко", false, "2, 3, 4", D(clutter_k)},
        {"density", "Текстура тайла, доля", "Тонко", false, "0.006, 0.012, 0.024", D(max_density)},
        {"compact_tex", "Крупная цель — не текстура", "Тонко", true, "1, 0", D(compact_tex)},
        {"smooth", "Сглаживание 3x3 (тусклые точки)", "Тонко", true, "0, 1", D(smooth)},
        {"subpixel", "Положение точнее пикселя", "Тонко", true, "0, 1", D(subpixel)},
        {"coarse_thr_k", "Порог крупных масштабов, раз", "Тонко", false, "2, 3, 4", D(coarse_thr_k)},
        {"k", "Макс. размер цели k, пикс.", "Тонко", false, "5, 7, 9", D(k)},
        {"focus", "Искать возле прогноза", "Тонко", true, "0, 1", S(focus)},
        // Тонко: трекер и захват
        {"gate", "Макс. радиус поиска, пикс.", "Тонко", false, "20, 40, 80", T(gate)},
        {"q_accel", "Манёвренность цели", "Тонко", false, "1, 4, 8", T(q_accel)},
        {"max_miss", "Держать на прогнозе, кадров", "Тонко", false, "3, 6, 12", T(max_miss)},
        {"lock_switch", "Смена захвата, раз", "Тонко", false, "1.2, 1.5, 2.5", T(lock_switch)},
        {"confirm", "Подтверждение, кадров", "Тонко", false, "2, 3, 4", T(confirm)},
        {"quiet", "Новая точка в спокойном небе, кадров (0 выкл.)", "Тонко", false, "0, 50", T(quiet_frames)},
        {"edge_straight", "У кромки — только прямой полёт", "Тонко", false, "0, 0.5", T(edge_straight)},
        {"lock_mover", "Новый захват — только на летавший трек", "Тонко", true, "0, 1", T(lock_mover)},
        {"hard_switch_drone", "Жёсткий захват переходит только на «дрон» или ведомую камерой цель", "Тонко", true, "1, 0", T(hard_switch_drone)},
        {"hover_first", "Птица: зависание важнее взмахов", "Тонко", true, "1, 0", T(hover_first)},
        {"reupdate", "Нашлась после пропуска: скорость по прямой от последней отметки, доля", "Тонко", false, "0, 0.5, 1", T(reupdate)},
        {"lock_mover_fps", "Захват только на летавший трек при камере не быстрее, к/с", "Тонко", false, "0, 20, 30", T(lock_mover_fps)},
        {"mover_cam_k", "Летит: по сцене дальше доли пути камеры", "Тонко", false, "0, 0.25, 0.35, 0.5", T(mover_cam_k)},
        {"still_speed", "Помеха: по сцене медленнее, пикс./кадр при 25 к/с", "Тонко", false, "0.25, 0.6, 1", T(still_speed)},
        {"silhouette", "Крупная цель целиком (силуэт)", "Тонко", true, "1, 0", T(silhouette)},
        {"size_predict", "Ожидание размера", "Тонко", true, "1, 0", T(size_predict)},
        {"anchor_check", "На мачте или столбе — не дрон", "Тонко", true, "1, 0", T(anchor_check)},
        {"hard_lock", "Жёсткий захват дрона", "Тонко", true, "1, 0", T(hard_lock)},
        {"verify", "Проверка: дрон подряд, кадров", "Тонко", false, "5, 10, 25", T(verify_frames)},
        {"hard_miss", "Жёсткий: держать на прогнозе, кадров", "Тонко", false, "30, 60, 120", T(hard_miss)},
        {"hard_full_every", "Жёсткий: весь кадр раз в, кадров", "Тонко", false, "3, 6, 12", T(hard_full_every)},
        {"lock_slow", "Захват: сначала мелкий медленный дрон", "Тонко", true, "1, 0",
         [](const Setup& u) -> double { return u.s.trk.lock_slow > 0 ? 1 : 0; },
         [](Setup& u, double v) {   // как галочка в приложении: вместе с жёстким переходом
             u.s.trk.lock_slow = v > 0 ? 1.f : 0.f;
             u.s.trk.hard_switch = v > 0 ? 2.f : 0.f;
         }},
        // Тонко: птица или дрон
        {"bird_resid", "Птица: отклонение от траектории, пикс.", "Тонко", false, "1.6, 2.2, 3", T(bird_resid)},
        {"bird_cv", "Птица: мигание контраста", "Тонко", false, "0.35, 0.5, 0.65", T(bird_contrast_cv)},
        {"flap_max_period", "Птица: самый медленный взмах, кадров при 25 к/с", "Тонко", false, "16, 24", T(flap_max_period)},
        {"cls_min_hits", "Птица: попаданий до решения", "Тонко", false, "6, 8, 12", T(cls_min_hits)},
        // Главное
        {"governor", "Держать темп камеры", "Главное", true, "0, 1", S(governor)},
    };
    return p;
}

#undef D
#undef T
#undef S

const Param* find_param(const std::string& key) {
    for (const auto& p : params())
        if (key == p.key) return &p;
    return nullptr;
}

std::string fmt_value(const Param& p, double v) {
    if (p.flag) return v > 0 ? "вкл" : "выкл";
    char b[32];
    snprintf(b, sizeof(b), "%g", v);
    return b;
}

bool parse_values(const Param& p, const std::string& text, std::vector<double>& out) {
    out.clear();
    std::string tok;
    auto flush = [&]() -> bool {
        size_t a = tok.find_first_not_of(" \t"), b = tok.find_last_not_of(" \t");
        if (a == std::string::npos) {
            tok.clear();
            return true;
        }
        std::string t = tok.substr(a, b - a + 1);
        tok.clear();
        double v;
        if (p.flag && (t == "вкл" || t == "да" || t == "on" || t == "true")) v = 1;
        else if (p.flag && (t == "выкл" || t == "нет" || t == "off" || t == "false")) v = 0;
        else {
            for (char& c : t)
                if (c == ',') c = '.';
            char* e = nullptr;
            v = strtod(t.c_str(), &e);
            // nan/inf и огромные числа дальше приводятся к int настройки — неопределённое поведение
            if (!e || *e || !(std::fabs(v) <= 1e9)) return false;
            if (p.flag) v = v > 0 ? 1 : 0;
        }
        if (std::find(out.begin(), out.end(), v) == out.end()) out.push_back(v);
        return true;
    };
    // разделители: ',' (и ';', пробел); дробная часть — через точку. Если в тексте есть ';',
    // запятая считается десятичной: "0,5; 1,5"
    const bool semi = text.find(';') != std::string::npos;
    for (char c : text) {
        const bool sep = c == ';' || c == ' ' || (c == ',' && !semi);
        if (sep) {
            if (!flush()) return false;
        } else {
            tok += c;
        }
    }
    if (!flush()) return false;
    return !out.empty();
}

static std::string change_name(const Param& p, double v) {
    std::string n = p.name;
    const size_t br = n.find(" (");
    if (br != std::string::npos) n = n.substr(0, br);
    return n + ": " + fmt_value(p, v);
}

static bool same_value(const Param& p, const Setup& base, double v) {
    return std::fabs(p.get(base) - v) < 1e-6;
}

int count_variants(const Setup& base, const std::vector<Choice>& ch, Mode mode) {
    if (mode == Mode::OneAtATime) {
        int n = 1;
        for (const auto& c : ch)
            if (const Param* p = find_param(c.key))
                for (double v : c.values) n += !same_value(*p, base, v);
        return n;
    }
    long n = 1;
    for (const auto& c : ch)
        if (find_param(c.key) && !c.values.empty()) n = std::min<long>(n * (long)c.values.size(), 1 << 20);
    return (int)n;
}

std::vector<Variant> make_variants(const Setup& base, const std::vector<Choice>& ch, Mode mode, std::string& warn) {
    warn.clear();
    std::vector<Variant> out;
    Variant b;
    b.name = "база";
    b.setup = base;
    out.push_back(b);
    std::vector<std::pair<const Param*, std::vector<double>>> sel;
    for (const auto& c : ch) {
        const Param* p = find_param(c.key);
        if (!p) {
            warn += "нет такой настройки: " + c.key + "; ";
            continue;
        }
        if (!c.values.empty()) sel.push_back({p, c.values});
    }
    if (mode == Mode::OneAtATime) {
        for (const auto& [p, vals] : sel)
            for (double v : vals) {
                if (same_value(*p, base, v)) continue;   // это и есть база
                Variant x;
                x.setup = base;
                p->set(x.setup, v);
                x.changes.push_back({p->key, v});
                x.name = change_name(*p, v);
                out.push_back(x);
            }
        return out;
    }
    // все сочетания: перебор как у счётчика
    out.clear();
    std::vector<size_t> idx(sel.size(), 0);
    for (;;) {
        Variant x;
        x.setup = base;
        bool isBase = true;
        for (size_t i = 0; i < sel.size(); ++i) {
            const Param* p = sel[i].first;
            const double v = sel[i].second[idx[i]];
            p->set(x.setup, v);
            if (!same_value(*p, base, v)) {
                isBase = false;
                x.changes.push_back({p->key, v});
                x.name += (x.name.empty() ? "" : ", ") + change_name(*p, v);
            }
        }
        if (isBase) {
            x.name = "база";
            out.insert(out.begin(), x);
        } else {
            out.push_back(x);
        }
        if ((int)out.size() > GRID_MAX) {
            out.resize(GRID_MAX);
            warn += "сочетаний больше " + std::to_string(GRID_MAX) + " — взяты первые; ";
            break;
        }
        size_t i = 0;
        while (i < sel.size() && ++idx[i] == sel[i].second.size()) idx[i++] = 0;
        if (i == sel.size()) break;
    }
    if (out.empty() || out[0].name != "база") out.insert(out.begin(), b);
    return out;
}

void apply_algo(Settings& dst, const Settings& src) {
    const int threads = dst.det.threads;
    const Isa isa = dst.det.isa;
    const std::vector<int> cpus = dst.det.cpus;
    dst.det = src.det;
    dst.det.threads = threads;
    dst.det.isa = isa;
    dst.det.cpus = cpus;
    dst.trk = src.trk;
    dst.learnMaskFrames = src.learnMaskFrames;
    dst.scales = src.scales;
    dst.motion = src.motion;
    dst.focus = src.focus;
    dst.zones = src.zones;
    dst.trkSeconds = src.trkSeconds;
    dst.fpsScale = src.fpsScale;
    dst.governor = src.governor;
    dst.skipDuplicates = src.skipDuplicates;
    dst.frameStep = src.frameStep;
    dst.framePhase = src.framePhase;
    dst.follow = src.follow;
    dst.followFrac = src.followFrac;
    dst.followGain = src.followGain;
    dst.followMaxPan = src.followMaxPan;
    dst.followVerified = src.followVerified;
}

// ------------------------------------------------------------ видео в папке

bool is_video_file(const std::string& name) {
    static const char* ext[] = {".mp4", ".mkv", ".avi", ".mov", ".ts", ".y4m", ".webm", ".m4v", ".mts"};
    std::string n = name;
    for (char& c : n) c = (char)tolower((unsigned char)c);
    for (const char* e : ext) {
        const size_t l = strlen(e);
        if (n.size() > l && n.compare(n.size() - l, l, e) == 0) return true;
    }
    return false;
}

static fs::path upath(const std::string& s) {
#if defined(__cpp_char8_t)
    return fs::path(std::u8string(s.begin(), s.end()));
#else
    return fs::u8path(s);
#endif
}

static std::string ustr(const fs::path& p) {
    const auto u = p.u8string();
    return std::string(u.begin(), u.end());
}

std::vector<std::string> list_videos(const std::string& folder) {
    std::vector<std::string> out;
    std::error_code ec;
    for (fs::directory_iterator it(upath(folder), ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code e2;
        if (!it->is_regular_file(e2)) continue;
        const std::string name = ustr(it->path().filename());
        if (is_video_file(name)) out.push_back(ustr(it->path()));
    }
    std::sort(out.begin(), out.end());
    return out;
}

// ------------------------------------------------------------ один прогон

namespace {

struct Collector {
    long maxFrames = 0;
    int warmup = 25;
    float radius = 12;
    long gtN = 0;          // кадров в разметке (0 — разметки нет)
    std::atomic<long> frames{0};
    std::vector<float> det, proc;
    long tgtSum = 0, tgtFrames = 0;
    std::set<int> ids, drones, birds;
    long lockF = 0, hardF = 0, acq = 0;
    int lastLock = 0;
    // по разметке
    long evF = 0, dObj = 0, bObj = 0, dHit = 0, bHit = 0, dAsDrone = 0, dFrames = 0;
    long fp = 0, lockEv = 0, lockOnD = 0, hardOnD = 0, falseLockF = 0;
    std::map<int, bool> life;   // трек -> хоть раз был на настоящей цели
    long firstGt = -1, firstDet = -1, firstLock = -1;
    double skySum = 0, skyDarkSum = 0;
    long skyN = 0;

    static bool live(const Track& t) { return !t.clutter && t.cls != TargetClass::Clutter; }

    void add(const Result& r) {
        if (maxFrames > 0 && frames >= maxFrames) return;
        ++frames;
        if (!r.dup) {
            det.push_back((float)r.dt.total_ms);
            proc.push_back((float)r.procMs);
        }
        long n = 0;
        for (const auto& t : r.tracks) {
            if (!live(t)) continue;
            ++n;
            ids.insert(t.id);
            if (t.cls == TargetClass::Drone) drones.insert(t.id);
            if (t.cls == TargetClass::Bird) birds.insert(t.id);
        }
        tgtSum += n;
        tgtFrames += n > 0;
        if (r.lockId) {
            ++lockF;
            hardF += r.lockState == LockState::Hard;
            if (r.lockId != lastLock) ++acq;
        }
        lastLock = r.lockId;

        const int64_t idx = r.frame->index;
        if (!r.mask.empty() && idx >= warmup) {   // маска неба после обучения: сколько неба осталось в поиске
            long on = 0, dark = 0;
            const bool md = r.maskDark.size() == r.mask.size(), mb = r.maskBright.size() == r.mask.size();
            for (size_t i = 0; i < r.mask.size(); ++i) {
                // облако, где ищутся цели обеих полярностей, — небо в поиске
                on += r.mask[i] != 0 || (md && mb && r.maskDark[i] && r.maskBright[i]);
                dark += r.mask[i] != 0 || (md && r.maskDark[i]);
            }
            skySum += (double)on / r.mask.size();
            skyDarkSum += (double)dark / r.mask.size();
            ++skyN;
        }
        if (gtN <= 0 || idx < 0 || idx >= gtN) return;
        const float R = radius;
        auto near = [&](const Track& t, const GtObj& g) {
            return std::fabs(t.x - g.x * r.gsx) < R && std::fabs(t.y - g.y * r.gsy) < R;
        };
        const Track* lt = nullptr;
        for (const auto& t : r.tracks)
            if (t.id == r.lockId) lt = &t;
        bool anyDrone = false, detDrone = false, lockDrone = false;
        for (const auto& g : r.gt) {
            if (g.bird) continue;
            anyDrone = true;
            for (const auto& t : r.tracks)
                if (live(t) && near(t, g)) detDrone = true;
            if (lt && near(*lt, g)) lockDrone = true;
        }
        if (anyDrone && firstGt < 0) firstGt = idx;
        if (detDrone && firstDet < 0) firstDet = idx;
        if (lockDrone && firstLock < 0) firstLock = idx;
        if (idx < warmup) return;

        ++evF;
        std::vector<char> matched(r.tracks.size(), 0);
        for (const auto& g : r.gt) {
            const Track* hit = nullptr;
            for (size_t j = 0; j < r.tracks.size(); ++j)
                if (live(r.tracks[j]) && near(r.tracks[j], g)) {
                    matched[j] = 1;
                    if (!hit) hit = &r.tracks[j];
                }
            if (g.bird) {
                ++bObj;
                bHit += hit != nullptr;
            } else {
                ++dObj;
                if (hit) {
                    ++dHit;
                    dAsDrone += hit->cls == TargetClass::Drone;
                }
            }
        }
        for (size_t j = 0; j < r.tracks.size(); ++j) {
            if (!live(r.tracks[j])) continue;
            fp += !matched[j];
            life[r.tracks[j].id] |= matched[j] != 0;
        }
        if (anyDrone) {
            ++dFrames;
            lockOnD += lockDrone;
            hardOnD += lockDrone && r.lockState == LockState::Hard;
        }
        if (r.lockId) {
            ++lockEv;
            falseLockF += !lockDrone;
        }
    }
};

double mean_of(const std::vector<float>& v) {
    double s = 0;
    for (float x : v) s += x;
    return v.empty() ? 0 : s / v.size();
}
double p95_of(std::vector<float> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t)(0.95 * v.size()))];
}

}  // namespace

static VideoMeta probe(const std::string& path) {
    VideoMeta m;
    m.path = path;
    m.file = ustr(upath(path).filename());
    VideoSource src;
    std::string err;
    if (src.open(path, 1, err)) {
        const VideoInfo vi = src.info();
        m.w = vi.w;
        m.h = vi.h;
        m.fps = vi.fps;
        m.frames = (long)vi.frames;
    }
    GroundTruth g;
    m.gt = g.load(path + ".gt.json");
    if (m.gt) {
        m.noDrones = true;
        for (const auto& f : g.frames) m.noDrones &= f.empty();
    }
    return m;
}

static RunResult run_one(const VideoMeta& vm, const Variant& var, const Config& c, int threads,
                         const std::atomic<bool>& stop, std::atomic<long>& live) {
    RunResult rr;
    Settings s = var.setup.s;
    s.pace = c.realtime ? Pace::Realtime : Pace::MaxSpeed;
    // частота камеры — только медленнее видео (пропуск кадров); быстрее — темп подачи, здесь не нужен
    if (!(s.targetFps > 0 && vm.fps > 0 && s.targetFps < 0.99 * vm.fps)) s.targetFps = 0;
    s.speed = 1;
    s.source = Source::Stream;
    s.loop = false;
    s.staticFile = false;   // память помех от прошлых прогонов исказила бы сравнение
    s.maskFile = c.maskFile;
    s.fresh = !c.maskFile;   // чистый старт: накопленное рядом с видео не читается и не пишется
    s.trackLog.clear();
    if (!c.emulate) {
        s.slowdown = 1;
        s.emulateMs = 0;
    }
    if (threads > 0) s.det.threads = threads;
    const float sc = var.setup.scale;
    if (sc > 0 && std::fabs(sc - 1.f) > 1e-3f && vm.w > 0) {
        s.procW = std::max(64, (int)(vm.w * sc) & ~1);
        s.procH = std::max(64, (int)(vm.h * sc) & ~1);
    } else {
        s.procW = s.procH = 0;
    }

    Collector col;
    col.maxFrames = c.maxFrames;
    // номера кадров — файла, маска учится по обработанным
    const double fstep = std::max(1, s.frameStep) * (s.targetFps > 0 ? vm.fps / s.targetFps : 1.0);
    col.warmup = (int)std::lround(c.warmup * fstep) + (fstep > 1.0001 ? std::max(0, s.framePhase) : 0);
    // радиус задан в пикселях исходного кадра; при уменьшенной обработке — меньше
    col.radius = c.radius * (s.procW && vm.w ? (float)s.procW / vm.w : 1.f);
    if (vm.gt) {
        GroundTruth g;
        if (g.load(vm.path + ".gt.json")) col.gtN = (long)g.frames.size();
    }
    Pipeline p;
    p.setSettings(s);
    p.setObserver([&](const Result& r) {
        col.add(r);
        live = col.frames.load();
    });
    std::string err;
    const double t0 = now_ms();
    if (!p.open(vm.path, err)) {
        rr.error = err.empty() ? "не открылось" : err;
        return rr;
    }
    long lastN = -1;
    double lastAt = now_ms();
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (stop) {
            rr.error = "остановлено";
            break;
        }
        if (c.maxFrames > 0 && col.frames >= c.maxFrames) break;
        const Pipeline::Progress pr = p.progress();
        if (pr.processed != lastN) {
            lastN = pr.processed;
            lastAt = now_ms();
        }
        if (pr.status == "видео закончилось") {
            // дождаться кадров, что ещё в очереди и в обработке
            if ((pr.queue == 0 && pr.processed >= pr.decoded) || now_ms() - lastAt > 3000) break;
        } else if (!pr.status.empty()) {
            rr.error = pr.status;
            break;
        }
    }
    rr.wallS = (now_ms() - t0) / 1000.0;
    p.close();

    rr.frames = col.frames;
    rr.done = rr.error.empty();   // прерванный прогон в сравнение не идёт: он короче остальных
    rr.fps = rr.wallS > 0 ? rr.frames / rr.wallS : 0;
    rr.detectMs = mean_of(col.det);
    rr.detectP95 = p95_of(col.det);
    rr.procMs = mean_of(col.proc);
    rr.procP95 = p95_of(col.proc);
    const double F = std::max<long>(1, rr.frames);
    rr.targetsPerFrame = col.tgtSum / F;
    rr.targetFramesPct = 100.0 * col.tgtFrames / F;
    rr.tracks = (long)col.ids.size();
    rr.droneIds = (long)col.drones.size();
    rr.birdIds = (long)col.birds.size();
    rr.lockPct = 100.0 * col.lockF / F;
    rr.hardPct = 100.0 * col.hardF / F;
    rr.lockAcq = col.acq;
    rr.skyPct = col.skyN ? 100.0 * col.skySum / col.skyN : -1;
    rr.skyDarkPct = col.skyN ? 100.0 * col.skyDarkSum / col.skyN : -1;
    rr.gt = col.gtN > 0 && col.evF > 0;
    if (rr.gt) {
        rr.evalFrames = col.evF;
        rr.gtDroneFrames = col.dFrames;
        rr.recallDrone = col.dObj ? 100.0 * col.dHit / col.dObj : -1;
        rr.recallBird = col.bObj ? 100.0 * col.bHit / col.bObj : -1;
        rr.droneAsDrone = col.dHit ? 100.0 * col.dAsDrone / col.dHit : -1;
        rr.falsePerFrame = (double)col.fp / col.evF;
        long ft = 0;
        for (const auto& kv : col.life) ft += !kv.second;
        rr.falseTracks = ft;
        rr.lockOnDrone = col.dFrames ? 100.0 * col.lockOnD / col.dFrames : -1;
        rr.hardOnDrone = col.dFrames ? 100.0 * col.hardOnD / col.dFrames : -1;
        rr.falseLock = col.lockEv ? 100.0 * col.falseLockF / col.lockEv : 0;
        const double fps = vm.fps > 0 ? vm.fps : 25;
        if (col.firstGt >= 0 && col.firstDet >= 0) rr.firstDetectS = std::max(0.0, (col.firstDet - col.firstGt) / fps);
        if (col.firstGt >= 0 && col.firstLock >= 0) rr.firstLockS = std::max(0.0, (col.firstLock - col.firstGt) / fps);
        // оценка 0..100: обнаружение и захват на дроне, минус ложные захваты и ложные цели
        if (col.dObj == 0 && col.bObj == 0) {
            // ролик без дронов: найти нечего (полные 40), любой захват ложный
            const double lockEv = 100.0 * col.lockEv / col.evF;
            rr.score = std::max(0.0, 40 + 0.6 * (100 - lockEv) - std::min(20.0, 5 * rr.falsePerFrame));
        } else if (rr.recallDrone >= 0) {
            const double lockOn = std::max(0.0, rr.lockOnDrone);
            rr.score = 0.4 * rr.recallDrone + 0.4 * lockOn + 0.2 * (100 - rr.falseLock) -
                       std::min(20.0, 5 * rr.falsePerFrame);
            rr.score = std::max(0.0, rr.score);
        }
    }
    return rr;
}

// ------------------------------------------------------------ метрики для отчёта

const std::vector<Metric>& metrics() {
    static const std::vector<Metric> m = {
        {"score", "Оценка", "", +1, true, 1, [](const RunResult& r) { return r.score; },
         "0..100: 0.4·обнаружение дрона + 0.4·захват на дроне + 0.2·(100 − ложный захват) − 5·ложных целей на кадр "
         "(не больше 20). На ролике без дронов: 40 + 0.6·(100 − кадров с захватом) − 5·ложных целей на кадр. "
         "Только по разметке."},
        {"recall_drone", "Дрон найден", "%", +1, true, 1, [](const RunResult& r) { return r.recallDrone; },
         "Доля кадров разметки, где рядом с дроном есть цель (не помеха)."},
        {"lock_on_drone", "Захват на дроне", "%", +1, true, 1, [](const RunResult& r) { return r.lockOnDrone; },
         "Доля кадров с дроном по разметке, где захват стоит на нём."},
        {"hard_on_drone", "Жёсткий на дроне", "%", +1, true, 1, [](const RunResult& r) { return r.hardOnDrone; },
         "Доля кадров с дроном, где на нём жёсткий захват («Захвачен дрон»)."},
        {"false_lock", "Ложный захват", "%", -1, true, 1, [](const RunResult& r) { return r.falseLock; },
         "Из кадров с захватом — доля, где захват не на дроне (на птице, помехе, пустом месте)."},
        {"false_per_frame", "Ложных целей/кадр", "", -1, true, 2, [](const RunResult& r) { return r.falsePerFrame; },
         "Цели не рядом ни с одним размеченным объектом, в среднем на кадр."},
        {"false_tracks", "Ложных треков", "", -1, true, 0, [](const RunResult& r) { return (double)r.falseTracks; },
         "Сколько треков за весь прогон ни разу не были на размеченном объекте."},
        {"drone_as_drone", "Назван дроном", "%", +1, true, 1, [](const RunResult& r) { return r.droneAsDrone; },
         "Из найденных дронов — доля, где трек назван «дрон»."},
        {"recall_bird", "Птица найдена", "%", +1, true, 1, [](const RunResult& r) { return r.recallBird; },
         "Как «дрон найден», для размеченных птиц."},
        {"first_detect_s", "До обнаружения", "с", -1, true, 2, [](const RunResult& r) { return r.firstDetectS; },
         "От первого кадра с дроном в разметке до первого кадра, где он найден."},
        {"first_lock_s", "До захвата", "с", -1, true, 2, [](const RunResult& r) { return r.firstLockS; },
         "От первого кадра с дроном в разметке до первого захвата на нём."},
        {"sky_pct", "Небо в поиске", "%", 0, false, 1, [](const RunResult& r) { return r.skyPct; },
         "Доля кадра, где детектор ищет: не закрыта маской неба как лес, дома, земля, текстура. Меньше целей при "
         "меньшем «небе в поиске» — это не чище, а просто меньше смотрим. Прочерк — маски нет (камера движется)."},
        {"sky_dark_pct", "Небо в поиске тёмных", "%", 0, false, 1, [](const RunResult& r) { return r.skyDarkPct; },
         "Как «небо в поиске», но для тёмных целей: с тайлами, которые маска закрыла только из-за светлой текстуры "
         "(облака), — там тёмный дрон ищется."},
        {"lock_pct", "Кадров с захватом", "%", 0, false, 1, [](const RunResult& r) { return r.lockPct; },
         "Доля всех кадров, где есть захват (любой цели)."},
        {"hard_pct", "Кадров с жёстким", "%", 0, false, 1, [](const RunResult& r) { return r.hardPct; },
         "Доля всех кадров с жёстким захватом."},
        {"lock_acq", "Смен захвата", "", 0, false, 0, [](const RunResult& r) { return (double)r.lockAcq; },
         "Сколько раз захват брал новую цель. На ролике с одним дроном меньше — обычно лучше."},
        {"targets_per_frame", "Целей/кадр", "", 0, false, 2, [](const RunResult& r) { return r.targetsPerFrame; },
         "Цели (не помехи) в среднем на кадр. Без разметки рост — это либо новые цели, либо ложные."},
        {"tracks", "Треков", "", 0, false, 0, [](const RunResult& r) { return (double)r.tracks; },
         "Разных целей за прогон."},
        {"drone_ids", "Треков «дрон»", "", 0, false, 0, [](const RunResult& r) { return (double)r.droneIds; },
         "Сколько разных целей хоть раз названы дроном."},
        {"bird_ids", "Треков «птица»", "", 0, false, 0, [](const RunResult& r) { return (double)r.birdIds; },
         "Сколько разных целей хоть раз названы птицей."},
        {"proc_ms", "Обработка, мс", "мс", -1, false, 2, [](const RunResult& r) { return r.procMs; },
         "Среднее время обработки кадра (детектор + трекер), без декодирования."},
        {"proc_p95_ms", "Обработка p95, мс", "мс", -1, false, 2, [](const RunResult& r) { return r.procP95; },
         "95% кадров обработаны быстрее этого."},
        {"detect_ms", "Детектор, мс", "мс", -1, false, 2, [](const RunResult& r) { return r.detectMs; },
         "Среднее время детектора."},
        {"fps", "Кадров/с прогона", "", +1, false, 1, [](const RunResult& r) { return r.fps; },
         "Сколько кадров в секунду шёл прогон, вместе с декодированием видео."},
        {"frames", "Кадров", "", 0, false, 0, [](const RunResult& r) { return (double)r.frames; }, "Обработано кадров."},
    };
    return m;
}

// ------------------------------------------------------------ запуск

Runner::~Runner() {
    stop();
    if (joiner_.joinable()) joiner_.join();
    for (auto* p : live_) delete p;
}

bool Runner::start(const Config& c, std::string& err) {
    if (running_) {
        err = "мультитест уже идёт";
        return false;
    }
    if (joiner_.joinable()) joiner_.join();
    cfg_ = c;
    std::vector<std::string> files = c.videos.empty() ? list_videos(c.folder) : c.videos;
    if (files.empty()) {
        err = "в папке нет видео (mp4, mkv, avi, mov, ts, y4m, webm)";
        return false;
    }
    std::string warn;
    variants_ = make_variants(c.base, c.choices, c.mode, warn);
    videos_.clear();
    for (const auto& f : files) videos_.push_back(probe(f));
    if (cfg_.outDir.empty()) {
        const std::time_t t = std::time(nullptr);
        char b[64];
        std::strftime(b, sizeof(b), "multitest-%Y%m%d-%H%M%S", std::localtime(&t));
        const std::string base = c.folder.empty() ? ustr(upath(files[0]).parent_path()) : c.folder;
        cfg_.outDir = ustr(upath(base) / b);
    }
    std::error_code ec;
    fs::create_directories(upath(cfg_.outDir), ec);
    if (ec) {
        err = "не создать папку отчёта " + cfg_.outDir + ": " + ec.message();
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(m_);
        res_.clear();
        for (int v = 0; v < (int)videos_.size(); ++v)
            for (int k = 0; k < (int)variants_.size(); ++k) {
                RunResult r;
                r.video = v;
                r.variant = k;
                res_.push_back(r);
            }
        error_ = warn;
        html_.clear();
        finished_ = false;
    }
    const int par = std::max(1, std::min(cfg_.parallel, (int)res_.size()));
    for (auto* p : live_) delete p;
    live_.clear();
    liveJob_.assign(par, -1);
    for (int i = 0; i < par; ++i) live_.push_back(new std::atomic<long>(0));
    stop_ = false;
    next_ = 0;
    done_ = 0;
    running_ = true;
    t0_ = now_ms();
    t1_ = 0;
    th_.clear();
    for (int i = 0; i < par; ++i)
        th_.emplace_back([this, i, par] {
            // потоков детектора на прогон: ядра поровну, если не задано явно
            const int threads = cfg_.base.s.det.threads > 0 ? 0 : (par > 1 ? std::max(1, plat::logical_cpus() / par) : 0);
            for (;;) {
                const int j = next_++;
                if (j >= (int)res_.size() || stop_) break;
                RunResult r0;
                {
                    std::lock_guard<std::mutex> lk(m_);
                    r0 = res_[j];
                    liveJob_[i] = j;
                }
                *live_[i] = 0;
                RunResult r = run_one(videos_[r0.video], variants_[r0.variant], cfg_, threads, stop_, *live_[i]);
                r.video = r0.video;
                r.variant = r0.variant;
                {
                    std::lock_guard<std::mutex> lk(m_);
                    res_[j] = r;
                    liveJob_[i] = -1;
                }
                ++done_;
                writeReport();   // отчёт по сделанному — после каждого прогона
            }
        });
    joiner_ = std::thread([this] {
        for (auto& t : th_) t.join();
        th_.clear();
        {
            std::lock_guard<std::mutex> lk(m_);
            finished_ = true;
            t1_ = now_ms();
        }
        writeReport();
        running_ = false;
    });
    return true;
}

void Runner::stop() { stop_ = true; }

void Runner::writeReport() {
    std::vector<RunResult> res;
    bool fin;
    double el;
    {
        std::lock_guard<std::mutex> lk(m_);
        res = res_;
        fin = finished_;
        el = ((t1_ > 0 ? t1_ : now_ms()) - t0_) / 1000.0;
    }
    static std::mutex wm;   // два потока не пишут файлы отчёта одновременно
    std::lock_guard<std::mutex> lk(wm);
    std::string err;
    const bool ok = write_report(cfg_.outDir, cfg_, videos_, variants_, res, fin, el, err);
    std::lock_guard<std::mutex> l2(m_);
    if (ok) html_ = ustr(upath(cfg_.outDir) / "report.html");
    else error_ = err;
}

Runner::Status Runner::status() const {
    Status st;
    std::lock_guard<std::mutex> lk(m_);
    st.total = (int)res_.size();
    st.done = done_;
    st.finished = finished_;
    st.stopped = stop_;
    st.outDir = cfg_.outDir;
    st.html = html_;
    st.error = error_;
    st.elapsedS = ((t1_ > 0 ? t1_ : now_ms()) - t0_) / 1000.0;
    double part = 0;
    for (size_t i = 0; i < liveJob_.size(); ++i) {
        const int j = liveJob_[i];
        if (j < 0) continue;
        const RunResult& r = res_[j];
        const VideoMeta& vm = videos_[r.video];
        long n = cfg_.maxFrames > 0 ? cfg_.maxFrames : vm.frames;
        if (cfg_.maxFrames > 0 && vm.frames > 0) n = std::min(n, vm.frames);
        const long f = *live_[i];
        if (n > 0) part += std::min(1.0, (double)f / n);
        if (!st.current.empty()) st.current += "\n";
        st.current += vm.file + " — " + variants_[r.variant].name + " (кадр " + std::to_string(f) +
                      (n > 0 ? " из " + std::to_string(n) : "") + ")";
    }
    st.frac = st.total ? std::min(1.0, (st.done + part) / st.total) : 0;
    if (st.frac > 0.02 && !st.finished) st.etaS = st.elapsedS * (1 - st.frac) / st.frac;
    return st;
}

void Runner::snapshot(std::vector<VideoMeta>& v, std::vector<Variant>& var, std::vector<RunResult>& res) const {
    std::lock_guard<std::mutex> lk(m_);
    v = videos_;
    var = variants_;
    res = res_;
}

// ------------------------------------------------------------ без окна

static void usage() {
    printf("dronolok_app --multitest папка|видео... [--vary ключ[=знач,знач]]... [--mode one|grid]\n"
           "            [--frames N] [--warmup N] [--radius пикс] [--parallel N] [--out папка]\n"
           "            [--profile 0..3] [--set ключ=знач]... [--realtime] [--emulate] [--mask-file]\n"
           "  --vary  что перебирать (без значений — значения по умолчанию); --set — изменить базу\n"
           "  --mode  one: база + каждая настройка по отдельности (по умолч.), grid: все сочетания\n"
           "  --list  список настроек\n");
}

static bool parse_kv(const std::string& a, std::string& key, std::string& vals) {
    const size_t eq = a.find('=');
    key = a.substr(0, eq);
    vals = eq == std::string::npos ? "" : a.substr(eq + 1);
    return find_param(key) != nullptr;
}

int run_cli(int argc, char** argv) {
    plat::attach_console();
    Config c;
    std::vector<std::string> inputs;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        std::string key, vals;
        if (a == "--list") {
            for (const auto& p : params()) printf("  %-16s %-12s %s  [%s]\n", p.key, p.group, p.name, p.values);
            return 0;
        } else if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else if (a == "--vary") {
            if (!parse_kv(val(), key, vals)) {
                printf("нет такой настройки: %s (--list — список)\n", key.c_str());
                return 2;
            }
            const Param* p = find_param(key);
            Choice ch;
            ch.key = key;
            if (!parse_values(*p, vals.empty() ? p->values : vals, ch.values)) {
                printf("не разобрать значения %s: %s\n", key.c_str(), vals.c_str());
                return 2;
            }
            c.choices.push_back(ch);
        } else if (a == "--set") {
            std::vector<double> v;
            if (!parse_kv(val(), key, vals) || !parse_values(*find_param(key), vals, v)) {
                printf("--set: ожидается ключ=значение (%s)\n", key.c_str());
                return 2;
            }
            find_param(key)->set(c.base, v[0]);
        } else if (a == "--profile") {
            set_profile(c.base, atof(val().c_str()));
        } else if (a == "--mode") {
            c.mode = val() == "grid" ? Mode::Grid : Mode::OneAtATime;
        } else if (a == "--frames") c.maxFrames = atol(val().c_str());
        else if (a == "--warmup") c.warmup = atoi(val().c_str());
        else if (a == "--radius") c.radius = (float)atof(val().c_str());
        else if (a == "--parallel") c.parallel = std::max(1, atoi(val().c_str()));
        else if (a == "--threads") c.base.s.det.threads = atoi(val().c_str());
        else if (a == "--out") c.outDir = val();
        else if (a == "--realtime") c.realtime = true;
        else if (a == "--emulate") c.emulate = true;
        else if (a == "--mask-file") c.maskFile = true;
        else if (!a.empty() && a[0] == '-') {
            printf("неизвестный ключ %s\n", a.c_str());
            usage();
            return 2;
        } else inputs.push_back(a);
    }
    if (inputs.empty()) {
        usage();
        return 2;
    }
    for (const auto& in : inputs) {
        std::error_code ec;
        if (fs::is_directory(upath(in), ec)) {
            if (c.folder.empty()) c.folder = in;
            for (const auto& v : list_videos(in)) c.videos.push_back(v);
        } else {
            c.videos.push_back(in);
        }
    }
    if (c.videos.empty()) {
        printf("видео не найдены\n");
        return 1;
    }
    Runner r;
    std::string err;
    if (!r.start(c, err)) {
        printf("ошибка: %s\n", err.c_str());
        return 1;
    }
    {
        std::vector<VideoMeta> v;
        std::vector<Variant> var;
        std::vector<RunResult> res;
        r.snapshot(v, var, res);
        printf("мультитест: видео %zu, вариантов %zu, прогонов %zu -> %s\n", v.size(), var.size(), res.size(),
               r.status().outDir.c_str());
        for (const auto& x : var) printf("  вариант: %s\n", x.name.c_str());
    }
    int shown = 0;
    while (r.running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const Runner::Status st = r.status();
        if (st.done != shown) {
            shown = st.done;
            printf("  %d/%d прогонов, %.0f с\n", st.done, st.total, st.elapsedS);
            fflush(stdout);
        }
    }
    const Runner::Status st = r.status();
    std::vector<VideoMeta> v;
    std::vector<Variant> var;
    std::vector<RunResult> res;
    r.snapshot(v, var, res);
    int bad = 0;
    for (const auto& x : res)
        if (!x.done || !x.error.empty()) {
            ++bad;
            printf("  ошибка: %s / %s: %s\n", v[x.video].file.c_str(), var[x.variant].name.c_str(), x.error.c_str());
        }
    if (!st.error.empty()) printf("  %s\n", st.error.c_str());
    printf("готово за %.0f с: %s\n", st.elapsedS, st.html.c_str());
    return bad ? 1 : 0;
}

}  // namespace mt
