#include "pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include "platform.hpp"

using namespace dronolok;

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Ждать до момента t точно: sleep грубый (особенно в Windows), последнюю миллисекунду — активно
static void wait_until(double t, const std::atomic<bool>& stop) {
    for (double d; !stop && (d = t - now_ms()) > 1.5;)
        std::this_thread::sleep_for(std::chrono::microseconds((long)((d - 1.0) * 1000)));
    while (!stop && now_ms() < t) std::this_thread::yield();
}

struct BufPool {
    std::mutex m;
    std::vector<std::vector<uint8_t>> free;
};

Pipeline::Pipeline() : pool_(std::make_shared<BufPool>()) {}
Pipeline::~Pipeline() { close(); }

FramePtr Pipeline::allocFrame(int w, int h) {
    std::vector<uint8_t> buf;
    {
        std::lock_guard<std::mutex> lk(pool_->m);
        if (!pool_->free.empty()) {
            buf = std::move(pool_->free.back());
            pool_->free.pop_back();
        }
    }
    buf.resize((size_t)w * h);
    std::weak_ptr<BufPool> wp = pool_;
    Frame* f = new Frame;
    f->px = std::move(buf);
    f->w = w;
    f->h = h;
    return FramePtr(f, [wp](Frame* fr) {
        if (auto p = wp.lock()) {
            std::lock_guard<std::mutex> lk(p->m);
            if (p->free.size() < 8) p->free.push_back(std::move(fr->px));
        }
        delete fr;
    });
}

bool Pipeline::open(const std::string& path, std::string& err) {
    close();
    VideoSource probe;
    if (!probe.open(path, 1, err)) return false;
    info_ = probe.info();
    probe.close();
    path_ = path;
    haveGt_ = gt_.load(path + ".gt.json");
    open_ = true;
    relearn_ = true;
    resetStats();
    startThreads();
    return true;
}

void Pipeline::close() {
    stopThreads();
    open_ = false;
    std::lock_guard<std::mutex> lk(rm_);
    latest_.reset();
}

VideoInfo Pipeline::info() const { return info_; }

void Pipeline::startThreads() {
    stop_ = false;
    {
        std::lock_guard<std::mutex> lk(qm_);
        q_.clear();
    }
    dec_ = std::thread([this] { decodeLoop(); });
    proc_ = std::thread([this] { processLoop(); });
}

void Pipeline::stopThreads() {
    stop_ = true;
    qcv_.notify_all();
    qspace_.notify_all();
    if (dec_.joinable()) dec_.join();
    if (proc_.joinable()) proc_.join();
    std::lock_guard<std::mutex> lk(qm_);
    q_.clear();
}

Settings Pipeline::settings() const {
    std::lock_guard<std::mutex> lk(sm_);
    return settings_;
}

void Pipeline::setSettings(const Settings& s) {
    bool restart;
    {
        std::lock_guard<std::mutex> lk(sm_);
        const Settings& o = settings_;
        restart = open_ && (o.source != s.source || o.preloadFrames != s.preloadFrames || o.procW != s.procW ||
                            o.procH != s.procH || o.decoderThreads != s.decoderThreads || o.loop != s.loop);
        settings_ = s;
        ++settingsVer_;
    }
    if (restart) {   // источник кадров поменялся — перезапуск чтения с начала файла
        stopThreads();
        relearn_ = true;
        startThreads();
    }
}

void Pipeline::setPaused(bool p) {
    paused_ = p;
    qcv_.notify_all();
}

void Pipeline::step() {
    ++steps_;
    qcv_.notify_all();
}

void Pipeline::relearnMask() { relearn_ = true; }

void Pipeline::resetStats() {
    std::lock_guard<std::mutex> lk(rm_);
    log_.clear();
    dropped_ = decoded_ = 0;
    decodeTimes_.clear();
    eval_ = EvalStats();
    t0_ = now_ms();
}

void Pipeline::push(Packet p, bool realtime) {
    std::unique_lock<std::mutex> lk(qm_);
    if (realtime) {
        // камера не ждёт: если обработка не успевает, старый кадр выбрасывается
        if (q_.size() >= qcap_) {
            q_.pop_front();
            std::lock_guard<std::mutex> lr(rm_);
            ++dropped_;
        }
    } else {
        qspace_.wait(lk, [&] { return q_.size() < qcap_ || stop_; });
        if (stop_) return;
    }
    q_.push_back(std::move(p));
    lk.unlock();
    qcv_.notify_one();
}

void Pipeline::decodeLoop() {
    Settings s = settings();
    VideoSource src;
    std::string err;
    if (!src.open(path_, s.decoderThreads, err)) {
        std::lock_guard<std::mutex> lk(rm_);
        status_ = err;
        return;
    }
    const int W = s.procW > 0 ? s.procW : info_.w, H = s.procH > 0 ? s.procH : info_.h;
    int64_t idx = 0;
    int loop = 0;
    auto decodeOne = [&](FramePtr& f) -> bool {
        f = allocFrame(W, H);
        const double t = now_ms();
        if (!src.read(f->px.data(), W, H, W)) {
            if (!s.loop || !src.rewind() || !src.read(f->px.data(), W, H, W)) return false;
            idx = 0;
            ++loop;
        }
        f->decodeMs = now_ms() - t;
        f->index = idx++;
        std::lock_guard<std::mutex> lk(rm_);
        ++decoded_;
        decodeTimes_.push_back(now_ms());
        while (decodeTimes_.size() > 200) decodeTimes_.pop_front();
        return true;
    };

    std::vector<FramePtr> pre;
    if (s.source == Source::Preload) {
        const bool loopSave = s.loop;
        s.loop = false;
        for (int i = 0; i < s.preloadFrames && !stop_; ++i) {
            FramePtr f;
            if (!decodeOne(f)) break;
            pre.push_back(f);
            char b[96];
            snprintf(b, sizeof(b), "Предзагрузка в память: %d / %d", i + 1, s.preloadFrames);
            std::lock_guard<std::mutex> lk(rm_);
            status_ = b;
        }
        s.loop = loopSave;
        std::lock_guard<std::mutex> lk(rm_);
        status_.clear();
        if (pre.empty()) status_ = "не удалось прочитать ни одного кадра";
    }

    size_t pi = 0;
    int ploop = 0;
    double next = now_ms();
    bool wasPaused = false;
    while (!stop_) {
        // на паузе в очереди держим один кадр — для покадрового режима
        if (paused_) {
            bool have;
            {
                std::lock_guard<std::mutex> lk(qm_);
                have = !q_.empty();
            }
            if (have) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                wasPaused = true;
                continue;
            }
        }
        if (wasPaused) {
            next = now_ms();
            wasPaused = false;
        }
        const Settings cur = settings();
        Packet p;
        if (s.source == Source::Preload) {
            if (pre.empty()) break;
            if (pi == pre.size()) {
                if (!s.loop) break;
                pi = 0;
                ++ploop;
            }
            p.frame = pre[pi];
            p.loop = ploop;
            p.loopStart = pi == 0;
            ++pi;
        } else {
            FramePtr f;
            if (!decodeOne(f)) break;
            p.frame = f;
            p.loop = loop;
            p.loopStart = f->index == 0;
        }
        const bool realtime = cur.pace == Pace::Realtime && !paused_;
        if (realtime) {
            const double fps = cur.targetFps > 0 ? cur.targetFps : info_.fps;
            const double period = 1000.0 / std::max(0.1, fps);
            wait_until(next, stop_);
            next += period;
            if (now_ms() - next > period) next = now_ms();   // декодер не успевает — без рывков вдогонку
        }
        p.readyAt = now_ms();
        push(std::move(p), realtime);
    }
    if (!stop_) {
        std::lock_guard<std::mutex> lk(rm_);
        if (status_.empty()) status_ = "видео закончилось";
    }
}

void Pipeline::processLoop() {
    Settings s;
    int ver;
    {
        std::lock_guard<std::mutex> lk(sm_);
        s = settings_;
        ver = settingsVer_;
    }
    Detector det(s.det);
    std::unique_ptr<Tracker> trk;
    int trkW = 0, trkH = 0, trkCell = 0;
    int learnLeft = 0;
    {
        std::lock_guard<std::mutex> lk(rm_);
        backend_ = det.backendName();
        threads_ = det.threads();
    }
    std::vector<Candidate> cands;
    int seq = 0;
    while (!stop_) {
        Packet p;
        {
            std::unique_lock<std::mutex> lk(qm_);
            qcv_.wait_for(lk, std::chrono::milliseconds(50),
                          [&] { return stop_ || (!q_.empty() && (!paused_ || steps_ > 0)); });
            if (stop_) break;
            if (q_.empty() || (paused_ && steps_ <= 0)) continue;
            p = std::move(q_.front());
            q_.pop_front();
            if (paused_) --steps_;
        }
        qspace_.notify_one();
        {
            std::lock_guard<std::mutex> lk(sm_);
            if (settingsVer_ != ver) {
                s = settings_;
                ver = settingsVer_;
                det.reconfigure(s.det);
                if (trk) trk->config() = s.trk;
                std::lock_guard<std::mutex> lr(rm_);
                backend_ = det.backendName();
                threads_ = det.threads();
            }
        }
        const Frame& f = *p.frame;
        if (!trk || trkW != f.w || trkH != f.h || trkCell != s.trk.clutter_cell) {
            trk.reset(new Tracker(f.w, f.h, s.trk));
            trkW = f.w;
            trkH = f.h;
            trkCell = s.trk.clutter_cell;
        }
        if (p.loopStart) trk->resetTracks();   // склейка зацикленного видео — треки не должны "прыгать"
        if (relearn_.exchange(false)) {
            if (s.learnMaskFrames > 0) {
                det.beginMaskLearning();
                learnLeft = s.learnMaskFrames;
            } else {
                det.clearMask();
                learnLeft = 0;
            }
        }
        const double t0 = now_ms();
        det.detect(f.px.data(), f.w, f.h, f.w, cands);
        const double t1 = now_ms();
        const auto& tracks = trk->update(cands);
        const double t2 = now_ms();
        const bool learning = det.learningMask();
        if (learnLeft > 0 && --learnLeft == 0) det.endMaskLearning();
        if (s.emulateMs > 0) wait_until(t0 + s.emulateMs, stop_);
        const double t3 = now_ms();

        auto r = std::make_shared<Result>();
        r->frame = p.frame;
        r->loop = p.loop;
        r->cands = cands;
        r->clutter = trk->clutterDets();
        r->noise = trk->noiseDets();
        r->tracks = tracks;
        r->dt = det.timing();
        r->trackMs = t2 - t1;
        r->procMs = t3 - t0;
        r->latencyMs = t3 - p.readyAt;
        r->mask = det.tileMask();
        r->tilesX = det.tilesX();
        r->tilesY = det.tilesY();
        r->tilePx = det.tileSizePx();
        r->learning = learning;
        r->seq = ++seq;
        const bool gtFrame = haveGt_ && f.index < (int64_t)gt_.frames.size();
        if (gtFrame) {
            r->gt = gt_.frames[f.index];
            r->gsx = gt_.w ? (float)f.w / gt_.w : 1.f;
            r->gsy = gt_.h ? (float)f.h / gt_.h : 1.f;
        }
        FrameRec rec;
        rec.decodeMs = (float)f.decodeMs;
        rec.detectMs = (float)r->dt.total_ms;
        rec.responseMs = (float)r->dt.response_ms;
        rec.selectMs = (float)r->dt.select_ms;
        rec.trackMs = (float)r->trackMs;
        rec.procMs = (float)r->procMs;
        rec.latencyMs = (float)r->latencyMs;
        rec.index = f.index;
        rec.cands = (int)cands.size();
        rec.drones = rec.birds = rec.unknown = 0;
        for (const auto& t : tracks) {
            if (t.clutter) continue;
            rec.drones += t.cls == TargetClass::Drone;
            rec.birds += t.cls == TargetClass::Bird;
            rec.unknown += t.cls == TargetClass::Unknown;
        }
        rec.clutter = (int)r->clutter.size();
        rec.noise = (int)r->noise.size();
        std::lock_guard<std::mutex> lk(rm_);
        rec.t = t3 - t0_;
        if (log_.size() < 500000) log_.push_back(rec);
        // качество считаем после первого прохода (как run_demo.py) или после обучения маски
        const bool evalOn = s.loop ? p.loop >= 1 : f.index >= s.learnMaskFrames;
        if (gtFrame && evalOn) eval_.add(r->gt, tracks, r->gsx, r->gsy);
        latest_ = r;
    }
}

std::shared_ptr<const Result> Pipeline::latest() const {
    std::lock_guard<std::mutex> lk(rm_);
    return latest_;
}

static Stat stat_of(std::vector<float> v) {
    Stat s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (float x : v) sum += x;
    s.mean = sum / v.size();
    s.p50 = v[v.size() / 2];
    s.p95 = v[std::min(v.size() - 1, (size_t)(0.95 * v.size()))];
    s.max = v.back();
    return s;
}

Metrics Pipeline::metrics(int histLen) const {
    Metrics m;
    {
        std::lock_guard<std::mutex> lk(qm_);
        m.queue = (int)q_.size();
        m.queueCap = (int)qcap_;
    }
    std::lock_guard<std::mutex> lk(rm_);
    m.processed = (long)log_.size();
    m.dropped = dropped_;
    m.decoded = decoded_;
    m.backend = backend_;
    m.threads = threads_;
    m.status = status_;
    m.eval = eval_;
    m.haveGt = haveGt_;
    m.sourceFps = info_.fps;
    const double now = now_ms();
    m.elapsedS = (now - t0_) / 1000.0;
    // статистика по последним 300 кадрам
    const size_t n = log_.size(), from = n > 300 ? n - 300 : 0;
    std::vector<float> dec, det, resp, sel, trk, proc, lat;
    for (size_t i = from; i < n; ++i) {
        const FrameRec& r = log_[i];
        dec.push_back(r.decodeMs);
        det.push_back(r.detectMs);
        resp.push_back(r.responseMs);
        sel.push_back(r.selectMs);
        trk.push_back(r.trackMs);
        proc.push_back(r.procMs);
        lat.push_back(r.latencyMs);
    }
    m.decode = stat_of(dec);
    m.detect = stat_of(det);
    m.response = stat_of(resp);
    m.select = stat_of(sel);
    m.track = stat_of(trk);
    m.proc = stat_of(proc);
    m.latency = stat_of(lat);
    // FPS: кадров за последнюю секунду (или за всё время, если прошло меньше)
    const double tnow = now - t0_;
    int cnt = 0;
    for (size_t i = n; i-- > 0 && log_[i].t > tnow - 1000;) ++cnt;
    m.procFps = cnt / std::min(1.0, std::max(1e-3, tnow / 1000.0));
    cnt = 0;
    for (auto it = decodeTimes_.rbegin(); it != decodeTimes_.rend() && *it > now - 1000; ++it) ++cnt;
    m.decodeFps = cnt / std::min(1.0, std::max(1e-3, tnow / 1000.0));
    const size_t hf = n > (size_t)histLen ? n - histLen : 0;
    for (size_t i = hf; i < n; ++i) {
        m.histDecode.push_back(log_[i].decodeMs);
        m.histDetect.push_back(log_[i].detectMs);
        m.histTrack.push_back(log_[i].trackMs);
        m.histProc.push_back(log_[i].procMs);
    }
    return m;
}

bool Pipeline::exportCsv(const std::string& path) const {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;
    fprintf(f, "t_ms,frame,decode_ms,detect_ms,response_ms,select_ms,track_ms,proc_ms,latency_ms,"
               "candidates,drones,birds,unknown,clutter,noise\n");
    std::lock_guard<std::mutex> lk(rm_);
    for (const auto& r : log_)
        fprintf(f, "%.2f,%lld,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%d\n", r.t, (long long)r.index,
                r.decodeMs, r.detectMs, r.responseMs, r.selectMs, r.trackMs, r.procMs, r.latencyMs, r.cands, r.drones,
                r.birds, r.unknown, r.clutter, r.noise);
    fclose(f);
    return true;
}

std::string Pipeline::reportJson() const {
    const Metrics m = metrics(0);
    const Settings s = settings();
    auto r = latest();
    char buf[2048];
    int len = snprintf(
        buf, sizeof(buf),
        "{\"video\": \"%dx%d\", \"processing\": \"%dx%d\", \"cpu\": \"%s\", \"backend\": \"%s\", \"threads\": %d, "
        "\"frames\": %ld, \"dropped\": %ld, \"fps\": %.1f, \"decode_ms\": %.2f, \"detect_ms\": %.2f, "
        "\"detect_p95_ms\": %.2f, \"response_ms\": %.2f, \"track_ms\": %.3f, \"proc_ms\": %.2f, "
        "\"detect_fps_potential\": %.1f, \"latency_ms\": %.1f, \"emulate_ms\": %.1f",
        info_.w, info_.h, r ? r->frame->w : 0, r ? r->frame->h : 0, plat::cpu_model().c_str(), m.backend.c_str(),
        m.threads, m.processed, m.dropped, m.processed / std::max(1e-3, m.elapsedS), m.decode.mean, m.detect.mean,
        m.detect.p95, m.response.mean, m.track.mean, m.proc.mean, m.detect.mean > 0 ? 1000 / m.detect.mean : 0,
        m.latency.mean, s.emulateMs);
    if (m.haveGt)
        len += snprintf(buf + len, sizeof(buf) - len,
                        ", \"recall_drone\": %.3f, \"recall_bird\": %.3f, \"false_tracks_per_frame\": %.2f, "
                        "\"drone_cls_acc\": %.3f, \"bird_cls_acc\": %.3f",
                        m.eval.recall(0), m.eval.recall(1), m.eval.falsePerFrame(), m.eval.clsAccuracy(0),
                        m.eval.clsAccuracy(1));
    snprintf(buf + len, sizeof(buf) - len, "}");
    return buf;
}
