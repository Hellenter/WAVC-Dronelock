// Замер скорости и качества нативного детектора без GUI (для ПК и Raspberry Pi).
//
//   ffmpeg -i sky16mp.mp4 -pix_fmt gray -f yuv4mpegpipe sky16mp.y4m
//   dronolok_bench sky16mp.y4m --gt sky16mp.mp4.gt.json --loops 3 --learn-mask 25
//   dronolok_bench sky16mp.y4m --sweep              # таблица: набор инструкций x потоки
//   ffmpeg -i real.mp4 -pix_fmt gray -f yuv4mpegpipe - | dronolok_bench - --stream
//
// Кадры по умолчанию заранее загружаются в память (как сырой поток камеры),
// время чтения/декодирования в замер не входит. --stream меряет вместе с чтением.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "dronolok/detector.hpp"
#include "dronolok/scene.hpp"
#include "dronolok/tuning.hpp"
#include "gt_eval.hpp"
#include "y4m_reader.hpp"

using namespace dronolok;

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static void usage() {
    std::printf(
        "usage: dronolok_bench <video.y4m | raw | -> [options]\n"
        "       dronolok_bench --calibrate   эталонный замер ядра CPU для эмуляции устройства в приложении\n"
        "  --size WxH [--yuv420]   сырой поток вместо y4m\n"
        "  --frames N              сколько кадров загрузить (100)\n"
        "  --loops L               проходов по кадрам (3)\n"
        "  --threads N             рабочих потоков (0 = все)\n"
        "  --cpus 0,1,2            привязать потоки к ядрам\n"
        "  --isa auto|scalar|sse2|avx2|neon\n"
        "  --pool 1|2 --k 7 --thr 10 --tile 32 --job RxC\n"
        "  --polarity dark|bright|both\n"
        "  --auto-thr [--noise-k 4.5 --thr-min 5]   порог по оценке шума (CFAR)\n"
        "  --ev 3 [--ev-hit 0.3]   подтверждение трека по накопленной уверенности\n"
        "  --confirm 3 --window 5  подтверждение трека: попаданий из окна кадров\n"
        "  --learn-mask N          авто-маска неба по первым N кадрам (до замера)\n"
        "  --learn-live N          учить маску на первых N кадрах замера, как приложение при открытии\n"
        "  --mask-file f           маска неба из файла (есть — загрузить и сверить, нет — выучить и сохранить)\n"
        "  --eval-all              оценивать и первый проход (обычно — со второго)\n"
        "  --scales 1..4           доп. масштабы для крупных целей\n"
        "  --focus                 пониженный порог возле прогноза треков\n"
        "  --motion                компенсация движения камеры\n"
        "  --zones                 зоны кадра: небо / смешанные / земля\n"
        "  --full-every N          весь кадр раз в N кадров, между ними — только вокруг треков\n"
        "  --coarse-every N        грубые масштабы целиком раз в N кадров\n"
        "  --per-tile N            до стольких кандидатов в тайле (2; второй — из ячеек 16x16)\n"
        "  --no-bg-map             без фона по ячейкам у кромки леса (маска неба с тайлом запаса, как раньше)\n"
        "  --no-sky-color          маска без облаков по цвету (цвет берётся из y4m и --yuv420)\n"
        "  --bg-map [--bg-k 3 --bg-frames 250]  фон по ячейкам у кромки: порог ячейки = медиана + K*СКО\n"
        "  --tbd [--tbd-thr 7 --tbd-z0 1.5 --tbd-decay 0.7 --tbd-min 0.5]   накопление по кадрам (включает --bg-map)\n"
        "  --band PX [--band-k 1 --upper-every N]   полоса у горизонта каждый кадр, небо выше — раз в N\n"
        "  --smooth [--smooth-k 0.5] / --no-smooth   сглаживание 3x3 перед пулингом (согласованный фильтр)\n"
        "  --static N              память статичных помех, кадров (0 — выкл.; с --motion выкл.)\n"
        "  --gt file.gt.json       разметка: полнота, ложные треки, классы\n"
        "  --bird-resid 3 --bird-cv 0.3 --bird-speed 0   пороги классификатора дрон/птица\n"
        "  --flap-window 48 --flap-r2 0.45 --flap-depth 0.3 --flap-hold 50   взмахи (окно 0 — выкл.)\n"
        "  --hover 0.3 --flock 150     зависание = дрон, стая с машущей = птицы (0 — выкл.)\n"
        "  --quiet N [--quiet-max 750]  новая неподвижная точка в спокойном небе (N кадров без\n"
        "                          откликов вокруг) — зависший дрон (0 — выкл.)\n"
        "  --fps-scale             параметры трекера в секундах: пересчитать под частоту кадров ролика\n"
        "  --scene                 оценивать сцену и погоду (свет, шум, облака, ветер, осадки, окружение)\n"
        "  --adapt                 ...и подстраивать под неё параметры (включает --scene)\n"
        "  --scene-log out.csv     оценка сцены по кадрам анализа\n"
        "  --stream                не загружать заранее (замер вместе с чтением)\n"
        "  --sweep                 таблица скоростей: наборы инструкций x потоки\n"
        "  --json out.jsonl        дописать итог строкой JSON\n"
        "  --dump-tracks out.csv   признаки треков по кадрам (для настройки классификатора)\n");
}

// Цвет загруженного кадра для облаков по цвету (DetectorConfig::sky_color): по отсчёту U и V на
// блок 8x8 — столько детектор и читает (8x8 отсчётов на тайл 64x64), в памяти это 1/32 кадра
struct Color8 {
    std::vector<uint8_t> u, v;
};
static bool sample_color8(const FrameReader& rd, int W, int H, Color8& c) {
    const uint8_t *u, *v;
    int cw, ch, stride, step;
    if (!rd.color(u, v, cw, ch, stride, step)) return false;
    const int bw = W / 8, bh = H / 8;
    c.u.resize((size_t)bw * bh);
    c.v.resize(c.u.size());
    for (int by = 0; by < bh; ++by) {
        const size_t row = (size_t)std::min(ch - 1, (by * 8 + 4) / step) * stride;
        for (int bx = 0; bx < bw; ++bx) {
            const int cx = std::min(cw - 1, (bx * 8 + 4) / step);
            c.u[(size_t)by * bw + bx] = u[row + cx];
            c.v[(size_t)by * bw + bx] = v[row + cx];
        }
    }
    return true;
}

static void print_sky_color(const Detector& det) {
    size_t n = 0;
    for (auto m : det.skyColored()) n += m;
    if (n) std::printf("облака по цвету: %zu тайлов закрыты только для светлых целей\n", n);
}

static Isa parse_isa(const std::string& s) {
    if (s == "scalar") return Isa::Scalar;
    if (s == "sse2") return Isa::SSE2;
    if (s == "avx2") return Isa::AVX2;
    if (s == "neon") return Isa::NEON;
    return Isa::Auto;
}

struct Pct {
    double mean = 0, p50 = 0, p95 = 0, max = 0;
};
static Pct pct(std::vector<double> v) {
    Pct p;
    if (v.empty()) return p;
    std::sort(v.begin(), v.end());
    for (double x : v) p.mean += x;
    p.mean /= v.size();
    p.p50 = v[v.size() / 2];
    p.p95 = v[std::min(v.size() - 1, (size_t)std::ceil(0.95 * v.size()) - 1)];
    p.max = v.back();
    return p;
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    if (std::string(argv[1]) == "--calibrate") {   // число для эмуляции устройства в приложении
        const double v = calibrate_ms_per_mp();
        std::printf("калибровка: %.2f мс на 1 Мп (1 поток, 128-бит SIMD)\n"
                    "введите это число в приложении: Ресурсы > Устройство > мс/Мп на устройстве\n", v);
        return 0;
    }
    std::string input, gtPath, jsonPath, dumpPath, maskPath;
    int rawW = 0, rawH = 0, maxFrames = 100, loops = 3, learn = 0, learnLive = 0;
    bool evalAll = false, sceneOn = false, adapt = false;
    std::string sceneLog;
    bool yuv420 = false, stream = false, sweep = false, focus = false, motion = false, fpsScale = false;
    GlobalMotion gm;
    std::vector<FocusArea> focusAreas, roiAreas;
    DetectorConfig cfg;
    TrackerConfig tcfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--size") { std::string v = val(); std::sscanf(v.c_str(), "%dx%d", &rawW, &rawH); }
        else if (a == "--yuv420") yuv420 = true;
        else if (a == "--frames") maxFrames = std::atoi(val().c_str());
        else if (a == "--loops") loops = std::atoi(val().c_str());
        else if (a == "--threads") cfg.threads = std::atoi(val().c_str());
        else if (a == "--cpus") {
            std::string v = val();
            for (size_t p = 0; p < v.size();) {
                cfg.cpus.push_back(std::atoi(v.c_str() + p));
                p = v.find(',', p);
                if (p == std::string::npos) break;
                ++p;
            }
        }
        else if (a == "--isa") cfg.isa = parse_isa(val());
        else if (a == "--pool") cfg.pool = std::atoi(val().c_str());
        else if (a == "--k") cfg.k = std::atoi(val().c_str());
        else if (a == "--thr") cfg.thr = (float)std::atof(val().c_str());
        else if (a == "--auto-thr") cfg.auto_thr = true;
        else if (a == "--noise-k") cfg.noise_k = (float)std::atof(val().c_str());
        else if (a == "--thr-min") cfg.thr_min = (float)std::atof(val().c_str());
        else if (a == "--ev") tcfg.ev_confirm = (float)std::atof(val().c_str());
        else if (a == "--ev-hit") tcfg.ev_hit = (float)std::atof(val().c_str());
        else if (a == "--confirm") tcfg.confirm = std::atoi(val().c_str());
        else if (a == "--window") tcfg.window = std::atoi(val().c_str());
        else if (a == "--tile") cfg.tile = std::atoi(val().c_str());
        else if (a == "--job") { std::string v = val(); std::sscanf(v.c_str(), "%dx%d", &cfg.job_rows, &cfg.job_cols); }
        else if (a == "--polarity") { std::string v = val(); cfg.dark = v != "bright"; cfg.bright = v != "dark"; }
        else if (a == "--bird-resid") tcfg.bird_resid = (float)std::atof(val().c_str());
        else if (a == "--bird-cv") tcfg.bird_contrast_cv = (float)std::atof(val().c_str());
        else if (a == "--bird-speed") tcfg.bird_speed = (float)std::atof(val().c_str());
        else if (a == "--flap-window") tcfg.flap_window = std::atoi(val().c_str());
        else if (a == "--flap-r2") tcfg.flap_r2 = (float)std::atof(val().c_str());
        else if (a == "--flap-depth") tcfg.flap_depth = (float)std::atof(val().c_str());
        else if (a == "--flap-hold") tcfg.flap_hold = std::atoi(val().c_str());
        else if (a == "--hover") tcfg.hover_speed = (float)std::atof(val().c_str());
        else if (a == "--quiet") tcfg.quiet_frames = std::atoi(val().c_str());
        else if (a == "--edge-straight") tcfg.edge_straight = (float)std::atof(val().c_str());
        else if (a == "--quiet-max") tcfg.quiet_max = std::atoi(val().c_str());
        else if (a == "--flock") tcfg.flock_px = (float)std::atof(val().c_str());
        else if (a == "--learn-mask") learn = std::atoi(val().c_str());
        else if (a == "--learn-live") learnLive = std::atoi(val().c_str());
        else if (a == "--mask-file") maskPath = val();
        else if (a == "--eval-all") evalAll = true;
        else if (a == "--scales") cfg.scales = std::atoi(val().c_str());
        else if (a == "--focus") focus = true;
        else if (a == "--q") tcfg.q_accel = (float)std::atof(val().c_str());
        else if (a == "--gate-sigma") tcfg.gate_sigma = (float)std::atof(val().c_str());
        else if (a == "--motion") motion = true;
        else if (a == "--zones") cfg.zones = true;
        else if (a == "--mixed-k") cfg.mixed_k = (float)std::atof(val().c_str());
        else if (a == "--full-every") cfg.full_every = std::atoi(val().c_str());
        else if (a == "--coarse-every") cfg.coarse_every = std::atoi(val().c_str());
        else if (a == "--per-tile") cfg.per_tile = std::atoi(val().c_str());
        else if (a == "--bg-map") cfg.bg_map = true;
        else if (a == "--no-bg-map") cfg.bg_map = false;
        else if (a == "--no-skyline") cfg.skyline = false;
        else if (a == "--no-fine-mask") cfg.fine_mask = false;
        else if (a == "--no-sky-color") cfg.sky_color = false;
        else if (a == "--lock-slow") tcfg.lock_slow = (float)std::atof(val().c_str());
        else if (a == "--lock-dwell") tcfg.lock_dwell = std::atoi(val().c_str());
        else if (a == "--follow-k") tcfg.follow_k = (float)std::atof(val().c_str());
        else if (a == "--hard-switch") tcfg.hard_switch = (float)std::atof(val().c_str());
        else if (a == "--edge-mover-k") tcfg.edge_mover_k = (float)std::atof(val().c_str());
        else if (a == "--bg-k") cfg.bg_k = (float)std::atof(val().c_str());
        else if (a == "--bg-frames") cfg.bg_frames = std::atoi(val().c_str());
        else if (a == "--tbd") cfg.tbd = cfg.bg_map = true;
        else if (a == "--tbd-thr") cfg.tbd_thr = (float)std::atof(val().c_str());
        else if (a == "--tbd-z0") cfg.tbd_z0 = (float)std::atof(val().c_str());
        else if (a == "--tbd-decay") cfg.tbd_decay = (float)std::atof(val().c_str());
        else if (a == "--tbd-min") cfg.tbd_min = (float)std::atof(val().c_str());
        else if (a == "--band") cfg.band_px = std::atoi(val().c_str());
        else if (a == "--band-k") cfg.band_k = (float)std::atof(val().c_str());
        else if (a == "--upper-every") cfg.upper_every = std::atoi(val().c_str());
        else if (a == "--smooth") cfg.smooth = true;
        else if (a == "--no-smooth") cfg.smooth = false;
        else if (a == "--smooth-k") cfg.smooth_k = (float)std::atof(val().c_str());
        else if (a == "--no-compact") cfg.smooth_compact = false;
        else if (a == "--no-compact-tex") cfg.compact_tex = false;
        else if (a == "--compact") { cfg.compact_r0 = (float)std::atof(val().c_str()); cfg.compact_r1 = (float)std::atof(val().c_str()); }
        else if (a == "--subpixel") cfg.subpixel = true;
        else if (a == "--static") tcfg.static_frames = std::atoi(val().c_str());
        else if (a == "--static-hits") tcfg.static_hits = (float)std::atof(val().c_str());
        else if (a == "--gt") gtPath = val();
        else if (a == "--stream") stream = true;
        else if (a == "--scene") sceneOn = true;
        else if (a == "--adapt") sceneOn = adapt = true;
        else if (a == "--scene-log") { sceneLog = val(); sceneOn = true; }
        else if (a == "--fps-scale") fpsScale = true;
        else if (a == "--sweep") sweep = true;
        else if (a == "--json") jsonPath = val();
        else if (a == "--dump-tracks") dumpPath = val();
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a[0] == '-' && a != "-") { std::printf("неизвестный параметр %s\n", a.c_str()); return 1; }
        else input = a;
    }
    if (motion) tcfg.static_frames = 0;   // память привязана к кадру неподвижной камеры
    FrameReader rd;
    if (!rd.open(input, rawW, rawH, yuv420)) {
        std::printf("не удалось открыть %s\n", input.c_str());
        return 1;
    }
    const int W = rd.width(), H = rd.height();
    if (fpsScale) {   // параметры трекера — в секундах, под частоту кадров ролика
        tcfg = tracker_for_fps(tcfg, (float)rd.fps());
        std::printf("трекер под %.1f к/с\n", rd.fps());
    }
    GroundTruth gt;
    const bool haveGt = !gtPath.empty() && gt.load(gtPath);
    if (!gtPath.empty() && !haveGt) std::printf("разметка %s не прочитана\n", gtPath.c_str());
    const float gsx = haveGt && gt.w ? (float)W / gt.w : 1, gsy = haveGt && gt.h ? (float)H / gt.h : 1;

    std::vector<std::vector<uint8_t>> frames;
    std::vector<Color8> colors;   // цвет загруженных кадров (пусто — его нет)
    std::vector<uint8_t> buf;
    if (!stream) {
        const double t0 = now_ms();
        Color8 c8;
        while ((int)frames.size() < maxFrames && rd.next(buf)) {
            frames.push_back(buf);
            if (cfg.sky_color && colors.size() + 1 == frames.size() && sample_color8(rd, W, H, c8)) colors.push_back(c8);
        }
        if (colors.size() != frames.size()) colors.clear();
        std::printf("загружено %zu кадров %dx%d за %.1f c\n", frames.size(), W, H, (now_ms() - t0) / 1000);
        if (frames.empty()) return 1;
    }

    if (sweep) {
        const int hw = hardware_threads();
        std::vector<int> ths{1, 2, 4};
        if (hw > 4) ths.push_back(hw);
        std::printf("%-8s %7s %10s %8s\n", "isa", "потоки", "мс/кадр", "FPS");
        for (Isa isa : available_isas())
            for (int th : ths) {
                if (th > hw) continue;
                DetectorConfig c = cfg;
                c.isa = isa;
                c.threads = th;
                Detector det(c);
                std::vector<Candidate> cands;
                for (int i = 0; i < 3; ++i) det.detect(frames[i % frames.size()].data(), W, H, W, cands);
                std::vector<double> ms;
                for (int l = 0; l < loops; ++l)
                    for (auto& f : frames) {
                        det.detect(f.data(), W, H, W, cands);
                        ms.push_back(det.timing().total_ms);
                    }
                Pct p = pct(ms);
                std::printf("%-8s %7d %10.2f %8.1f\n", isa_name(isa), th, p.mean, 1000 / p.mean);
            }
        return 0;
    }

    Detector det(cfg);
    Tracker trk(W, H, tcfg);
    SceneAnalyzer scene;
    const DetectorConfig baseCfg = cfg;
    const TrackerConfig baseT = tcfg;
    std::string why, lastWhy;
    long adaptChanges = 0;
    std::vector<double> sceneMs;
    std::FILE* slog = sceneLog.empty() ? nullptr : std::fopen(sceneLog.c_str(), "w");
    if (slog) std::fprintf(slog, "frame,sky_lum,ground_lum,noise,clouds,wind,precip,haze,glare,sky_frac,hv,flat,line_sd,light,terrain,event,thr,profile\n");
    std::FILE* dump = dumpPath.empty() ? nullptr : std::fopen(dumpPath.c_str(), "w");
    if (dump) std::fprintf(dump, "loop,frame,id,x,y,hits,speed,resid,contrast,contrast_cv,flap_r2,flap_period,flap_depth,hover,flock,energy,level,cls,gt\n");
    bool maskLoaded = false;
    if (!maskPath.empty() && det.loadMask(maskPath)) {
        maskLoaded = true;
        learn = learnLive = 0;
        std::printf("маска неба загружена из %s, сверка 25 кадров\n", maskPath.c_str());
    }
    if (learn > 0 && !stream) {
        det.beginMaskLearning();
        std::vector<Candidate> c;
        for (int i = 0; i < std::min(learn, (int)frames.size()); ++i) {
            if (!colors.empty()) det.setColor(colors[i].u.data(), colors[i].v.data(), W / 8, H / 8, W / 8, 8);
            det.detect(frames[i].data(), W, H, W, c);
        }
        det.endMaskLearning();
        long on = 0;
        for (auto m : det.tileMask()) on += m;
        std::printf("авто-маска: исключено %.1f%% тайлов (лес/земля)\n",
                    100.0 * (1 - (double)on / std::max<size_t>(1, det.tileMask().size())));
        print_sky_color(det);
    }
    std::printf("бэкенд %s, потоков %d\n", det.backendName().c_str(), det.threads());

    std::vector<double> total, resp, sel, trackMs, readMs;
    std::vector<Candidate> cands;
    EvalStats ev, ev3;   // ev3 — первые 3 с первого прохода (с --eval-all)
    const int n = stream ? 0 : (int)frames.size();
    const long nIter = stream ? (long)maxFrames * 1000000 : (long)n * loops;
    for (long i = 0; i < nIter; ++i) {
        const uint8_t* g;
        int fi;
        if (stream) {
            const double r0 = now_ms();
            if (!rd.next(buf)) break;
            readMs.push_back(now_ms() - r0);
            g = buf.data();
            fi = (int)i;
        } else {
            fi = (int)(i % n);
            g = frames[fi].data();
            if (fi == 0) trk.resetTracks();   // видео зациклено — трек не должен "прыгать" через склейку
        }
        if (i == 0 && learnLive > 0) det.beginMaskLearning();
        if (learnLive > 0 && i == learnLive) {
            det.endMaskLearning();
            trk.resetTracks();
            print_sky_color(det);
        }
        const double t0 = now_ms();
        MotionEstimate me;
        if (motion) me = gm.estimate(g, W, H, W);
        if (focus) {
            trk.focus(me.dx, me.dy, focusAreas);
            det.setFocus(focusAreas);
        }
        if (cfg.full_every > 1 || cfg.coarse_every > 1 || cfg.upper_every > 1 || cfg.bg_map) {
            trk.roi(me.dx, me.dy, roiAreas);
            det.setRoi(roiAreas);
        }
        const double ts0 = now_ms();
        const bool analyzed = sceneOn && scene.update(g, W, H, W, (float)rd.fps(), motion);
        if (analyzed) sceneMs.push_back(now_ms() - ts0);
        if (analyzed) {
            const SceneState& ss = scene.state();
            if (adapt) {
                DetectorConfig dc;
                TrackerConfig tc;
                scene_profile(ss, baseCfg, baseT, dc, tc, &why);
                dc.threads = cfg.threads;
                const bool stale = scene_mask_stale(det.config(), dc);
                if (why != lastWhy || stale) det.reconfigure(dc);
                trk.config() = tc;
                if (ss.lightJump) det.resetBackground();
                // как приложение: маска заново по следующим кадрам (если её уже выучили)
                if ((ss.moved || stale) && (learn > 0 || learnLive > 0 || maskLoaded) && !det.learningMask()) {
                    det.beginMaskLearning();
                    learnLive = (int)i + 25;
                }
                if (why != lastWhy) {
                    ++adaptChanges;
                    std::printf("  кадр %ld: %s\n", i, why.empty() ? "база" : why.c_str());
                    lastWhy = why;
                }
            }
            if (ss.lightJump || ss.moved) std::printf("  кадр %ld: %s\n", i, ss.lastEvent.c_str());
            if (slog)
                std::fprintf(slog, "%ld,%.1f,%.1f,%.2f,%.3f,%.2f,%.2f,%.3f,%.4f,%.3f,%.3f,%.3f,%.1f,%s,%s,%s,%.1f,\"%s\"\n", i,
                             ss.skyLum, ss.groundLum, ss.noise, ss.clouds, ss.wind, ss.precip, ss.haze, ss.glare,
                             ss.skyFrac, ss.hv, ss.skylineFlat, ss.skylineSd, light_name(ss.light), terrain_name(ss.terrain),
                             ss.lightJump || ss.moved ? ss.lastEvent.c_str() : "", det.threshold(), why.c_str());
        }
        // цвет кадра: детектор читает его только в кадры обучения маски
        const uint8_t *cu, *cv;
        int cw, ch, cs, cst;
        if (stream && rd.color(cu, cv, cw, ch, cs, cst)) det.setColor(cu, cv, cw, ch, cs, cst);
        else if (!stream && !colors.empty()) det.setColor(colors[fi].u.data(), colors[fi].v.data(), W / 8, H / 8, W / 8, 8);
        det.detect(g, W, H, W, cands);
        trk.setPartial(det.timing().partial);
        const double t1 = now_ms();
        const auto& tracks = trk.update(cands, me.dx, me.dy);
        const double t2 = now_ms();
        if (i >= 3) {
            total.push_back(t2 - t0);
            resp.push_back(det.timing().response_ms);
            sel.push_back(det.timing().select_ms);
            trackMs.push_back(t2 - t1);
        }
        const bool evalFrame = haveGt && fi < (int)gt.frames.size() && (stream || evalAll || i >= n);
        if (evalFrame) {
            ev.add(gt.frames[fi], tracks, gsx, gsy);
            ev.addCands(gt.frames[fi], cands, gsx, gsy);
            if (i < (long)std::lround(3 * rd.fps())) ev3.add(gt.frames[fi], tracks, gsx, gsy);
        }
        if (dump && haveGt && fi < (int)gt.frames.size()) {
            for (const auto& t : tracks) {
                const char* kind = "none";
                for (const auto& o : gt.frames[fi])
                    if (std::fabs(t.x - o.x * gsx) < 10 && std::fabs(t.y - o.y * gsy) < 10) { kind = o.bird ? "bird" : "drone"; break; }
                std::fprintf(dump, "%ld,%d,%d,%.1f,%.1f,%d,%.2f,%.2f,%.1f,%.3f,%.2f,%.1f,%.2f,%d,%d,%.0f,%d,%s,%s\n", n ? i / n : 0,
                             fi, t.id, t.x, t.y, t.hits, t.speed, t.resid, t.contrast, t.contrastCv, t.flapR2,
                             t.flapPeriod, t.flapDepth, (int)t.hovering, (int)t.flock,
                             t.obs.empty() ? 0.f : t.obs.back().e, t.obs.empty() ? 0 : t.obs.back().level, class_name(t.cls), kind);
            }
        }
    }
    if (dump) std::fclose(dump);
    if (slog) std::fclose(slog);
    if (sceneOn) {
        const SceneState& ss = scene.state();
        std::printf("сцена: %s, %s; небо %.0f, шум %.2f, облачность %.2f, ветер у кромки %.2f пикс., осадки %.2f, "
                    "дымка %.2f, неба %.0f%% кадра; перемен %d\n",
                    light_name(ss.light), terrain_name(ss.terrain), ss.skyLum, ss.noise, ss.clouds, ss.wind, ss.precip,
                    ss.haze, 100 * ss.skyFrac, ss.events);
        std::printf("оценка сцены: %.2f мс на анализ (%zu анализов)\n", pct(sceneMs).mean, sceneMs.size());
        if (adapt) std::printf("подстройка: %s (смен профиля %ld)\n", lastWhy.empty() ? "база" : lastWhy.c_str(), adaptChanges);
    }
    if (!maskPath.empty()) {
        if (maskLoaded && det.maskReplaced()) std::printf("маска из файла не совпала с кадром — выучена заново\n");
        if (!maskLoaded || det.maskReplaced()) {
            if (det.saveMask(maskPath)) std::printf("маска неба сохранена в %s\n", maskPath.c_str());
        }
    }
    const Pct pt = pct(total), pr = pct(resp), ps = pct(sel), pk = pct(trackMs);
    std::printf("кадров %zu: detect+track %.2f мс (p50 %.2f, p95 %.2f, max %.2f) = %.1f FPS\n", total.size(), pt.mean,
                pt.p50, pt.p95, pt.max, pt.mean > 0 ? 1000 / pt.mean : 0);
    std::printf("  отклик (бэкенд) %.2f мс, выбор кандидатов %.3f мс, трекер %.3f мс\n", pr.mean, ps.mean, pk.mean);
    if (stream) std::printf("  чтение кадра %.2f мс\n", pct(readMs).mean);
    std::printf("  шум отклика %.2f, порог %.1f%s\n", det.noiseLevel(), det.threshold(), cfg.auto_thr ? " (по шуму)" : "");
    char json[1024];
    int len = std::snprintf(json, sizeof(json),
                            "{\"resolution\": \"%dx%d\", \"backend\": \"%s\", \"threads\": %d, \"pool\": %d, \"k\": %d, "
                            "\"mask\": %s, \"frames\": %zu, \"mean_ms\": %.2f, \"p95_ms\": %.2f, \"fps\": %.1f, "
                            "\"response_ms\": %.2f, \"track_ms\": %.3f",
                            W, H, det.backendName().c_str(), det.threads(), cfg.pool, cfg.k, learn ? "true" : "false",
                            total.size(), pt.mean, pt.p95, pt.mean > 0 ? 1000 / pt.mean : 0, pr.mean, pk.mean);
    if (haveGt) {
        std::printf("  полнота: дроны %.1f%%, птицы %.1f%%; ложных треков на кадр %.2f\n", 100 * ev.recall(0),
                    100 * ev.recall(1), ev.falsePerFrame());
        const double gfps = gt.fps > 0 ? gt.fps : rd.fps();
        for (const auto& g : ev.groups) {
            int fnd = 0, miss = 0;
            const double d = ev.meanDelay(g.first, fnd, miss);
            std::printf("  группа %-5s: полнота %5.1f%%, обнаружение через %.2f с (найдено %d, ни разу %d)\n", g.first.c_str(),
                        100.0 * g.second.first / std::max(1L, g.second.second), d / gfps, fnd, miss);
        }
        if (evalAll)
            std::printf("  первые 3 с: полнота дронов %.1f%%, ложных треков на кадр %.2f\n", 100 * ev3.recall(0),
                        ev3.falsePerFrame());
        std::printf("  ложные тревоги (треки, ни разу не бывшие на цели): %ld за %.0f с = %.1f в минуту, висят %.3f на кадр\n",
                    ev.falseAlarms(), ev.frames / gfps, ev.falseAlarms() * 60.0 * gfps / std::max(1L, ev.frames),
                    ev.falseAlarmPerFrame());
        std::printf("  кандидаты детектора ближе 6 пикс.: дроны %.1f%%, птицы %.1f%%\n", 100 * ev.candRecall(0),
                    100 * ev.candRecall(1));
        std::printf("  классы найденных дронов: дрон %ld, птица %ld, неясно %ld; птиц: птица %ld, дрон %ld, неясно %ld\n",
                    ev.conf[0][1], ev.conf[0][2], ev.conf[0][0], ev.conf[1][2], ev.conf[1][1], ev.conf[1][0]);
        len += std::snprintf(json + len, sizeof(json) - len,
                             ", \"recall_drone\": %.3f, \"recall_bird\": %.3f, \"false_tracks_per_frame\": %.2f, "
                             "\"drone_cls_acc\": %.3f, \"bird_cls_acc\": %.3f, \"cand_recall_drone\": %.3f",
                             ev.recall(0), ev.recall(1), ev.falsePerFrame(), ev.clsAccuracy(0), ev.clsAccuracy(1),
                             ev.candRecall(0));
    }
    std::snprintf(json + len, sizeof(json) - len, "}");
    std::printf("%s\n", json);
    if (!jsonPath.empty()) {
        if (std::FILE* f = std::fopen(jsonPath.c_str(), "a")) {
            std::fprintf(f, "%s\n", json);
            std::fclose(f);
        }
    }
    return 0;
}
