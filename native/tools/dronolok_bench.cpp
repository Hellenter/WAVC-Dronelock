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
        "  --learn-mask N          авто-маска неба по первым N кадрам\n"
        "  --scales 1..4           доп. масштабы для крупных целей\n"
        "  --focus                 пониженный порог возле прогноза треков\n"
        "  --motion                компенсация движения камеры\n"
        "  --zones                 зоны кадра: небо / смешанные / земля\n"
        "  --full-every N          весь кадр раз в N кадров, между ними — только вокруг треков\n"
        "  --coarse-every N        грубые масштабы целиком раз в N кадров\n"
        "  --static N              память статичных помех, кадров (0 — выкл.; с --motion выкл.)\n"
        "  --gt file.gt.json       разметка: полнота, ложные треки, классы\n"
        "  --bird-resid 3 --bird-cv 0.3 --bird-speed 0   пороги классификатора дрон/птица\n"
        "  --flap-window 48 --flap-r2 0.45 --flap-depth 0.3 --flap-hold 50   взмахи (окно 0 — выкл.)\n"
        "  --hover 0.3 --flock 150     зависание = дрон, стая с машущей = птицы (0 — выкл.)\n"
        "  --stream                не загружать заранее (замер вместе с чтением)\n"
        "  --sweep                 таблица скоростей: наборы инструкций x потоки\n"
        "  --json out.jsonl        дописать итог строкой JSON\n"
        "  --dump-tracks out.csv   признаки треков по кадрам (для настройки классификатора)\n");
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
    std::string input, gtPath, jsonPath, dumpPath;
    int rawW = 0, rawH = 0, maxFrames = 100, loops = 3, learn = 0;
    bool yuv420 = false, stream = false, sweep = false, focus = false, motion = false;
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
        else if (a == "--flock") tcfg.flock_px = (float)std::atof(val().c_str());
        else if (a == "--learn-mask") learn = std::atoi(val().c_str());
        else if (a == "--scales") cfg.scales = std::atoi(val().c_str());
        else if (a == "--focus") focus = true;
        else if (a == "--q") tcfg.q_accel = (float)std::atof(val().c_str());
        else if (a == "--gate-sigma") tcfg.gate_sigma = (float)std::atof(val().c_str());
        else if (a == "--motion") motion = true;
        else if (a == "--zones") cfg.zones = true;
        else if (a == "--mixed-k") cfg.mixed_k = (float)std::atof(val().c_str());
        else if (a == "--full-every") cfg.full_every = std::atoi(val().c_str());
        else if (a == "--coarse-every") cfg.coarse_every = std::atoi(val().c_str());
        else if (a == "--static") tcfg.static_frames = std::atoi(val().c_str());
        else if (a == "--static-hits") tcfg.static_hits = (float)std::atof(val().c_str());
        else if (a == "--gt") gtPath = val();
        else if (a == "--stream") stream = true;
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
    GroundTruth gt;
    const bool haveGt = !gtPath.empty() && gt.load(gtPath);
    if (!gtPath.empty() && !haveGt) std::printf("разметка %s не прочитана\n", gtPath.c_str());
    const float gsx = haveGt && gt.w ? (float)W / gt.w : 1, gsy = haveGt && gt.h ? (float)H / gt.h : 1;

    std::vector<std::vector<uint8_t>> frames;
    std::vector<uint8_t> buf;
    if (!stream) {
        const double t0 = now_ms();
        while ((int)frames.size() < maxFrames && rd.next(buf)) frames.push_back(buf);
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
    std::FILE* dump = dumpPath.empty() ? nullptr : std::fopen(dumpPath.c_str(), "w");
    if (dump) std::fprintf(dump, "loop,frame,id,x,y,hits,speed,resid,contrast,contrast_cv,flap_r2,flap_period,flap_depth,hover,flock,energy,level,cls,gt\n");
    if (learn > 0 && !stream) {
        det.beginMaskLearning();
        std::vector<Candidate> c;
        for (int i = 0; i < std::min(learn, (int)frames.size()); ++i) det.detect(frames[i].data(), W, H, W, c);
        det.endMaskLearning();
        long on = 0;
        for (auto m : det.tileMask()) on += m;
        std::printf("авто-маска: исключено %.1f%% тайлов (лес/земля)\n",
                    100.0 * (1 - (double)on / std::max<size_t>(1, det.tileMask().size())));
    }
    std::printf("бэкенд %s, потоков %d\n", det.backendName().c_str(), det.threads());

    std::vector<double> total, resp, sel, trackMs, readMs;
    std::vector<Candidate> cands;
    EvalStats ev;
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
        const double t0 = now_ms();
        MotionEstimate me;
        if (motion) me = gm.estimate(g, W, H, W);
        if (focus) {
            trk.focus(me.dx, me.dy, focusAreas);
            det.setFocus(focusAreas);
        }
        if (cfg.full_every > 1 || cfg.coarse_every > 1) {
            trk.roi(me.dx, me.dy, roiAreas);
            det.setRoi(roiAreas);
        }
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
        const bool evalFrame = haveGt && fi < (int)gt.frames.size() && (stream || i >= n);
        if (evalFrame) ev.add(gt.frames[fi], tracks, gsx, gsy);
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
    const Pct pt = pct(total), pr = pct(resp), ps = pct(sel), pk = pct(trackMs);
    std::printf("кадров %zu: detect+track %.2f мс (p50 %.2f, p95 %.2f, max %.2f) = %.1f FPS\n", total.size(), pt.mean,
                pt.p50, pt.p95, pt.max, pt.mean > 0 ? 1000 / pt.mean : 0);
    std::printf("  отклик (бэкенд) %.2f мс, выбор кандидатов %.3f мс, трекер %.3f мс\n", pr.mean, ps.mean, pk.mean);
    if (stream) std::printf("  чтение кадра %.2f мс\n", pct(readMs).mean);
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
        std::printf("  классы найденных дронов: дрон %ld, птица %ld, неясно %ld; птиц: птица %ld, дрон %ld, неясно %ld\n",
                    ev.conf[0][1], ev.conf[0][2], ev.conf[0][0], ev.conf[1][2], ev.conf[1][1], ev.conf[1][0]);
        len += std::snprintf(json + len, sizeof(json) - len,
                             ", \"recall_drone\": %.3f, \"recall_bird\": %.3f, \"false_tracks_per_frame\": %.2f, "
                             "\"drone_cls_acc\": %.3f, \"bird_cls_acc\": %.3f",
                             ev.recall(0), ev.recall(1), ev.falsePerFrame(), ev.clsAccuracy(0), ev.clsAccuracy(1));
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
