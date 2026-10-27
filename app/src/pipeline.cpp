#include "pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>

#include "dronolok/fileutil.hpp"
#include "dronolok/tuning.hpp"
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

Effective effective(const Settings& s, int w, int h) {
    const bool small = (double)w * h <= 2.2e6;
    Effective e;
    e.scales = s.scales > 0 ? std::min(4, s.scales) : small ? 3 : 1;
    e.motion = s.motion >= 0 ? s.motion > 0 : small;
    // при неподвижной камере землю убирает маска неба, зоны нужны при съёмке с рук
    e.zones = s.zones >= 0 ? s.zones > 0 : e.motion;
    return e;
}

// история для покадрового просмотра назад
static const size_t HIST_MAX = 600;
static const size_t HIST_BYTES = sizeof(void*) > 4 ? (size_t)1 << 30 : (size_t)256 << 20;

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
    {
        auto g = std::make_shared<GroundTruth>();
        std::lock_guard<std::mutex> lk(rm_);
        haveGt_ = g->load(gtPath());
        gt_ = haveGt_ ? g : nullptr;
    }
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
    hist_.clear();
    histBytes_ = 0;
    marks_.clear();
}

VideoInfo Pipeline::info() const { return info_; }

void Pipeline::startThreads() {
    stop_ = false;
    {
        std::lock_guard<std::mutex> lk(qm_);
        q_.clear();
    }
    {   // номера кадров seq начнутся заново
        std::lock_guard<std::mutex> lk(rm_);
        hist_.clear();
        histBytes_ = 0;
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

void Pipeline::setGroundTruth(const GroundTruth& g) {
    auto p = std::make_shared<GroundTruth>(g);
    std::lock_guard<std::mutex> lk(rm_);
    haveGt_ = !g.frames.empty();
    gt_ = haveGt_ ? p : nullptr;
    eval_ = EvalStats();   // качество считается заново по новой разметке
    for (auto& m : marks_) m &= (uint8_t)~(MARK_GT | MARK_MISS);
}

void Pipeline::seek(int64_t frame) {
    seekReq_ = std::max<int64_t>(0, frame);
    qspace_.notify_all();
}

int64_t Pipeline::length() const {
    const int64_t n = preN_;
    return n > 0 ? n : info_.frames;
}

void Pipeline::marks(std::vector<uint8_t>& out) const {
    std::lock_guard<std::mutex> lk(rm_);
    out = marks_;
}

int Pipeline::findInHistory(int64_t frame) const {
    std::lock_guard<std::mutex> lk(rm_);
    for (auto it = hist_.rbegin(); it != hist_.rend(); ++it)
        if ((*it)->frame->index == frame) return (*it)->seq;
    return -1;
}

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
    int loop = 0;
    bool wrapped = false, first = true;
    preN_ = 0;
    auto decodeOne = [&](FramePtr& f) -> bool {
        f = allocFrame(W, H);
        const double t = now_ms();
        wrapped = false;
        if (!src.read(f->px.data(), W, H, W)) {
            if (!s.loop || !src.rewind() || !src.read(f->px.data(), W, H, W)) return false;
            ++loop;
            wrapped = true;
        }
        f->decodeMs = now_ms() - t;
        f->index = std::max<int64_t>(0, src.index());
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
        // номера кадров в памяти — подряд с 0, шкала видео — по ним
        for (size_t i = 0; i < pre.size(); ++i) pre[i]->index = (int64_t)i;
        preN_ = (int64_t)pre.size();
        std::lock_guard<std::mutex> lk(rm_);
        status_.clear();
        if (pre.empty()) status_ = "не удалось прочитать ни одного кадра";
    }

    size_t pi = 0;
    int ploop = 0;
    double next = now_ms();
    bool wasPaused = false, seeked = false;
    while (!stop_) {
        const int64_t sk = seekReq_.exchange(-1);
        if (sk >= 0) {   // перемотка: кадры, что уже в очереди, больше не нужны
            {
                std::lock_guard<std::mutex> lk(qm_);
                q_.clear();
            }
            qspace_.notify_all();
            if (s.source == Source::Preload) {
                if (!pre.empty()) pi = (size_t)std::min<int64_t>(sk, (int64_t)pre.size() - 1);
            } else if (!src.seek(sk)) {
                std::lock_guard<std::mutex> lk(rm_);
                status_ = "не удалось перемотать";
            }
            seeked = true;
            next = now_ms();
            {
                std::lock_guard<std::mutex> lk(rm_);
                if (status_ == "видео закончилось") status_.clear();
            }
        }
        // на паузе в очереди держим один кадр — для покадрового режима
        if (paused_ && !seeked) {
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
                if (!s.loop) {
                    {
                        std::lock_guard<std::mutex> lk(rm_);
                        status_ = "видео закончилось";
                    }
                    while (!stop_ && seekReq_ < 0) std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    continue;
                }
                pi = 0;
                ++ploop;
            }
            p.frame = pre[pi];
            p.loop = ploop;
            p.loopStart = pi == 0;
            ++pi;
        } else {
            FramePtr f;
            if (!decodeOne(f)) {
                // конец файла без повтора: ждём перемотку
                {
                    std::lock_guard<std::mutex> lk(rm_);
                    status_ = "видео закончилось";
                }
                while (!stop_ && seekReq_ < 0) std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            p.frame = f;
            p.loop = loop;
            p.loopStart = (first || wrapped) && !seeked;
            first = false;
        }
        p.seekStart = seeked;
        p.force = seeked && paused_;
        seeked = false;
        const bool realtime = cur.pace == Pace::Realtime && !paused_;
        if (realtime) {
            const double fps = cur.targetFps > 0 ? cur.targetFps : info_.fps;
            // замедление: кадры "приходят с камеры" реже, обработка та же
            const double period = 1000.0 / std::max(0.1, fps) / std::min(4.0, std::max(0.01, cur.speed));
            wait_until(next, stop_);
            next += period;
            if (now_ms() - next > period) next = now_ms();   // декодер не успевает — без рывков вдогонку
        }
        p.readyAt = now_ms();
        const bool drop = realtime && !p.force;
        push(std::move(p), drop);
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
    DetectorConfig dc = s.det;   // то, что сейчас стоит в детекторе (с учётом "авто")
    bool dirty = true;
    GlobalMotion gm;
    LoadGovernor gov;
    std::vector<FocusArea> focus, roi;
    std::unique_ptr<Tracker> trk;
    int trkW = 0, trkH = 0, trkCell = 0;
    int learnLeft = 0;
    bool maskOn = false, motionWas = false;
    {
        std::lock_guard<std::mutex> lk(rm_);
        backend_ = det.backendName();
        threads_ = det.threads();
    }
    std::vector<Candidate> cands;
    std::vector<TrackPoint> statCells;
    // при закрытии и пересоздании трекера память помех сохраняется рядом с видео
    auto saveStatic = [&] {
        if (!trk || trk->config().static_frames <= 0 || !s.staticFile) return;
        trk->staticCells(statCells);
        if (!statCells.empty()) trk->saveStatic(staticPath());
    };
    int seq = 0;
    while (!stop_) {
        Packet p;
        {
            std::unique_lock<std::mutex> lk(qm_);
            qcv_.wait_for(lk, std::chrono::milliseconds(50),
                          [&] { return stop_ || (!q_.empty() && (!paused_ || steps_ > 0 || q_.front().force)); });
            if (stop_) break;
            if (q_.empty() || (paused_ && steps_ <= 0 && !q_.front().force)) continue;
            p = std::move(q_.front());
            q_.pop_front();
            if (paused_ && !p.force && steps_ > 0) --steps_;
        }
        qspace_.notify_one();
        {
            std::lock_guard<std::mutex> lk(sm_);
            if (settingsVer_ != ver) {
                s = settings_;
                ver = settingsVer_;
                dirty = true;
                if (trk) trk->config() = s.trk;
            }
        }
        const Frame& f = *p.frame;
        const Effective ef = effective(s, f.w, f.h);
        DetectorConfig want = s.det;
        want.scales = ef.scales;
        want.zones = ef.zones;
        if (s.governor) LoadGovernor::apply(gov.level(), want);
        else gov.reset();
        if (dirty || want.scales != dc.scales || want.zones != dc.zones || want.full_every != dc.full_every ||
            want.coarse_every != dc.coarse_every) {
            dc = want;
            det.reconfigure(dc);
            dirty = false;
            std::lock_guard<std::mutex> lr(rm_);
            backend_ = det.backendName();
            threads_ = det.threads();
        }
        if (!trk || trkW != f.w || trkH != f.h || trkCell != s.trk.clutter_cell) {
            saveStatic();
            trk.reset(new Tracker(f.w, f.h, s.trk));
            trkW = f.w;
            trkH = f.h;
            trkCell = s.trk.clutter_cell;
            if (s.staticFile && trk->loadStatic(staticPath())) {
                trk->staticCells(statCells);
                std::lock_guard<std::mutex> lr(rm_);
                staticMsg_ = "память помех загружена: " + std::to_string(statCells.size()) + " ячеек";
            }
        }
        // память помех — в координатах кадра, при движущейся камере не ведётся
        trk->config().static_frames = ef.motion ? 0 : s.trk.static_frames;
        if (const int cmd = staticCmd_.exchange(0)) {
            if (cmd == 2) {
                trk->clearStatic();
                remove_file(staticPath());
            }
            trk->staticCells(statCells);
            const bool ok = cmd == 2 || trk->saveStatic(staticPath());
            std::lock_guard<std::mutex> lr(rm_);
            staticMsg_ = cmd == 2 ? "память помех очищена"
                         : ok     ? "сохранено " + std::to_string(statCells.size()) + " ячеек: " + staticPath()
                                  : "не удалось сохранить " + staticPath();
        }
        if (p.loopStart || p.seekStart) {   // склейка зацикленного видео или перемотка — треки не должны "прыгать"
            trk->resetTracks();
            gm.reset();
        }
        // Маска неба привязана к кадру и годится только для неподвижной камеры: при съёмке с рук
        // земля через секунду окажется там, где было небо, и наоборот (дрон пропадёт в "маске").
        if (ef.motion != motionWas) {
            if (!ef.motion) relearn_ = true;
            motionWas = ef.motion;
        }
        if (ef.motion) {
            if (maskOn) {
                det.clearMask();
                maskOn = false;
                learnLeft = 0;
            }
            relearn_ = false;
        } else if (relearn_.exchange(false)) {
            if (s.learnMaskFrames > 0) {
                det.beginMaskLearning();
                learnLeft = s.learnMaskFrames;
                maskOn = true;
            } else {
                det.clearMask();
                learnLeft = 0;
                maskOn = false;
            }
        }
        const double t0 = now_ms();
        // сдвиг картинки от прошлого кадра: прогноз треков переносится вместе с камерой
        MotionEstimate me;
        if (ef.motion) me = gm.estimate(f.px.data(), f.w, f.h, f.w);
        else gm.reset();
        const double tm = now_ms();
        focus.clear();
        if (s.focus) trk->focus(me.dx, me.dy, focus);
        det.setFocus(focus);
        // экономия: между полными просмотрами кадра детектор смотрит только вокруг треков
        roi.clear();
        if (dc.full_every > 1 || (dc.coarse_every > 1 && dc.scales > 1)) trk->roi(me.dx, me.dy, roi);
        det.setRoi(roi);
        det.detect(f.px.data(), f.w, f.h, f.w, cands);
        trk->setPartial(det.timing().partial);
        const double t1 = now_ms();
        const auto& tracks = trk->update(cands, me.dx, me.dy);
        const double t2 = now_ms();
        const bool learning = det.learningMask();
        if (learnLeft > 0 && --learnLeft == 0) det.endMaskLearning();
        // эмуляция устройства: ядро медленнее в slowdown раз, плюс нижняя граница времени кадра
        if (s.slowdown > 1) wait_until(t0 + (t2 - t0) * std::min(100.0, s.slowdown), stop_);
        if (s.emulateMs > 0) wait_until(t0 + s.emulateMs, stop_);
        const double t3 = now_ms();
        // авто-упрощение по времени кадра против периода камеры (без замедления просмотра)
        if (s.governor) gov.update(t3 - t0, 1000.0 / std::max(0.1, s.targetFps > 0 ? s.targetFps : info_.fps));

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
        r->zones = det.zones();
        r->learning = learning;
        r->seq = ++seq;
        r->lockId = trk->lockedId();
        r->camDx = me.dx;
        r->camDy = me.dy;
        r->camQ = me.quality;
        r->camValid = me.valid;
        r->motionOn = ef.motion;
        r->motionMs = ef.motion ? tm - t0 : 0;
        r->scales = dc.scales;
        r->govLevel = s.governor ? gov.level() : -1;
        r->fullEvery = dc.full_every;
        r->coarseEvery = dc.scales > 1 ? dc.coarse_every : 0;
        r->focus = focus;
        r->staticOn = trk->config().static_frames > 0;
        if (r->staticOn && seq % 15 == 0) trk->staticCells(statCells);
        if (r->staticOn) r->staticCells = statCells;
        r->staticCellPx = trk->staticCellPx();
        std::shared_ptr<const GroundTruth> gt;
        {
            std::lock_guard<std::mutex> lk(rm_);
            gt = gt_;
        }
        const bool gtFrame = gt && f.index < (int64_t)gt->frames.size();
        if (gtFrame) {
            r->gt = gt->frames[f.index];
            r->gsx = gt->w ? (float)f.w / gt->w : 1.f;
            r->gsy = gt->h ? (float)f.h / gt->h : 1.f;
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
        rec.motionMs = (float)r->motionMs;
        rec.lockId = r->lockId;
        rec.camDx = r->camDx;
        rec.camDy = r->camDy;
        rec.govLevel = r->govLevel;
        uint8_t mk = MARK_DONE;
        if (rec.drones) mk |= MARK_DRONE;
        if (rec.birds) mk |= MARK_BIRD;
        if (rec.unknown) mk |= MARK_TARGET;
        if (r->lockId) mk |= MARK_LOCK;
        for (const auto& g : r->gt) {
            if (g.bird) continue;
            mk |= MARK_GT;
            bool found = false;
            for (const auto& t : tracks)
                found |= !t.clutter && t.cls != TargetClass::Clutter && std::fabs(t.x - g.x * r->gsx) < 10 &&
                         std::fabs(t.y - g.y * r->gsy) < 10;
            if (!found) mk |= MARK_MISS;
        }
        std::lock_guard<std::mutex> lk(rm_);
        if (f.index >= 0 && f.index < (int64_t)4 << 20) {
            if ((size_t)f.index >= marks_.size()) marks_.resize((size_t)f.index + 1, 0);
            marks_[(size_t)f.index] = mk;
        }
        rec.t = t3 - t0_;
        if (log_.size() < 500000) log_.push_back(rec);
        // качество считаем после первого прохода (как run_demo.py) или после обучения маски
        // при съёмке с рук маска неба не учится — качество считаем сразу (своя разметка роликов)
        const bool evalOn = ef.motion || (s.loop ? p.loop >= 1 : f.index >= s.learnMaskFrames);
        if (gtFrame && evalOn) eval_.add(r->gt, tracks, r->gsx, r->gsy);
        latest_ = r;
        hist_.push_back(r);
        histBytes_ += f.px.size();
        while (hist_.size() > 1 && (hist_.size() > HIST_MAX || histBytes_ > HIST_BYTES)) {
            histBytes_ -= hist_.front()->frame->px.size();
            hist_.pop_front();
        }
    }
    saveStatic();
}

std::shared_ptr<const Result> Pipeline::latest() const {
    std::lock_guard<std::mutex> lk(rm_);
    return latest_;
}

std::shared_ptr<const Result> Pipeline::at(int seq) const {
    std::lock_guard<std::mutex> lk(rm_);
    if (hist_.empty()) return nullptr;
    const long i = (long)seq - hist_.front()->seq;
    return i >= 0 && i < (long)hist_.size() ? hist_[(size_t)i] : nullptr;
}

bool Pipeline::historyRange(int& first, int& last) const {
    std::lock_guard<std::mutex> lk(rm_);
    if (hist_.empty()) return false;
    first = hist_.front()->seq;
    last = hist_.back()->seq;
    return true;
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
    m.staticMsg = staticMsg_;
    m.eval = eval_;
    m.haveGt = haveGt_;
    m.sourceFps = info_.fps;
    const double now = now_ms();
    m.elapsedS = (now - t0_) / 1000.0;
    // статистика по последним 300 кадрам
    const size_t n = log_.size(), from = n > 300 ? n - 300 : 0;
    std::vector<float> dec, det, resp, sel, trk, proc, lat, mot;
    int locked = 0;
    for (size_t i = from; i < n; ++i) {
        const FrameRec& r = log_[i];
        dec.push_back(r.decodeMs);
        det.push_back(r.detectMs);
        resp.push_back(r.responseMs);
        sel.push_back(r.selectMs);
        trk.push_back(r.trackMs);
        proc.push_back(r.procMs);
        lat.push_back(r.latencyMs);
        mot.push_back(r.motionMs);
        locked += r.lockId != 0;
    }
    m.lockPct = n > from ? 100.0 * locked / (n - from) : 0;
    m.motion = stat_of(mot);
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
    FILE* f = open_file(path, "w");
    if (!f) return false;
    fprintf(f, "t_ms,frame,decode_ms,motion_ms,detect_ms,response_ms,select_ms,track_ms,proc_ms,latency_ms,"
               "candidates,drones,birds,unknown,clutter,noise,lock_id,cam_dx,cam_dy,gov_level\n");
    std::lock_guard<std::mutex> lk(rm_);
    for (const auto& r : log_)
        fprintf(f, "%.2f,%lld,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%d,%d,%.2f,%.2f,%d\n", r.t,
                (long long)r.index, r.decodeMs, r.motionMs, r.detectMs, r.responseMs, r.selectMs, r.trackMs, r.procMs,
                r.latencyMs, r.cands, r.drones, r.birds, r.unknown, r.clutter, r.noise, r.lockId, r.camDx, r.camDy,
                r.govLevel);
    fclose(f);
    return true;
}

// 8-битный BMP с серой палитрой: открывается любым просмотрщиком, кодировщик не нужен
static bool write_bmp(const std::string& path, const uint8_t* px, int w, int h) {
    FILE* f = open_file(path, "wb");
    if (!f) return false;
    const int stride = (w + 3) & ~3;
    const uint32_t off = 14 + 40 + 1024, size = off + (uint32_t)stride * h;
    uint8_t hdr[54] = {'B', 'M'};
    auto put32 = [&](int at, uint32_t v) { for (int i = 0; i < 4; ++i) hdr[at + i] = (uint8_t)(v >> (8 * i)); };
    put32(2, size);
    put32(10, off);
    put32(14, 40);
    put32(18, (uint32_t)w);
    put32(22, (uint32_t)h);
    hdr[26] = 1;
    hdr[28] = 8;
    put32(34, (uint32_t)stride * h);
    put32(46, 256);
    fwrite(hdr, 1, 54, f);
    for (int i = 0; i < 256; ++i) {
        const uint8_t c[4] = {(uint8_t)i, (uint8_t)i, (uint8_t)i, 0};
        fwrite(c, 1, 4, f);
    }
    std::vector<uint8_t> row(stride, 0);
    for (int y = h - 1; y >= 0; --y) {   // BMP хранит строки снизу вверх
        std::copy(px + (size_t)y * w, px + (size_t)y * w + w, row.begin());
        fwrite(row.data(), 1, stride, f);
    }
    return fclose(f) == 0;
}

int Pipeline::exportCrops(const std::string& dir, std::string& msg) const {
    std::vector<std::shared_ptr<const Result>> hist;
    {
        std::lock_guard<std::mutex> lk(rm_);
        hist.assign(hist_.begin(), hist_.end());
    }
    if (hist.empty()) {
        msg = "история пуста: сначала прогоните видео";
        return 0;
    }
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::u8path(dir), ec);
    FILE* idx = open_file(dir + "/crops.csv", "w");
    if (!idx) {
        msg = "не удалось создать " + dir;
        return 0;
    }
    // признаки трека — те же, что видит классификатор; truth — по разметке (если есть)
    fprintf(idx, "file,frame,track,x,y,class,truth,contrast,speed,resid,contrast_cv,hits,miss,locked\n");
    const int S = 48;
    std::vector<uint8_t> crop((size_t)S * S);
    int n = 0;
    for (const auto& r : hist) {
        const Frame& f = *r->frame;
        if (f.w < S || f.h < S) continue;
        for (const auto& t : r->tracks) {
            if (t.clutter || t.miss > 0 || !t.confirmed) continue;
            const char* truth = "";
            if (!r->gt.empty()) {
                truth = "ложная";
                for (const auto& g : r->gt)
                    if (std::fabs(t.x - g.x * r->gsx) < 10 && std::fabs(t.y - g.y * r->gsy) < 10) truth = g.bird ? "птица" : "дрон";
            }
            const char* cls = t.cls == TargetClass::Drone ? "дрон" : t.cls == TargetClass::Bird ? "птица" : "неясно";
            const int x0 = std::min(std::max(0, (int)t.x - S / 2), f.w - S), y0 = std::min(std::max(0, (int)t.y - S / 2), f.h - S);
            for (int y = 0; y < S; ++y)
                std::copy(f.px.data() + (size_t)(y0 + y) * f.w + x0, f.px.data() + (size_t)(y0 + y) * f.w + x0 + S,
                          crop.begin() + (size_t)y * S);
            char name[96];
            snprintf(name, sizeof(name), "f%06lld_t%d_%s.bmp", (long long)f.index, t.id, *truth ? truth : cls);
            if (!write_bmp(dir + "/" + name, crop.data(), S, S)) continue;
            fprintf(idx, "%s,%lld,%d,%.1f,%.1f,%s,%s,%.1f,%.2f,%.2f,%.3f,%d,%d,%d\n", name, (long long)f.index, t.id,
                    t.x, t.y, cls, truth, t.contrast, t.speed, t.resid, t.contrastCv, t.hits, t.miss, t.id == r->lockId);
            ++n;
        }
    }
    fclose(idx);
    msg = "вырезок " + std::to_string(n) + " из " + std::to_string(hist.size()) + " кадров: " + dir;
    return n;
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
        "\"detect_fps_potential\": %.1f, \"latency_ms\": %.1f, \"emulate_ms\": %.1f, \"scales\": %d, "
        "\"motion\": %s, \"motion_ms\": %.2f, \"lock_pct\": %.1f, \"slowdown\": %.1f, \"gov_level\": %d",
        info_.w, info_.h, r ? r->frame->w : 0, r ? r->frame->h : 0, plat::cpu_model().c_str(), m.backend.c_str(),
        m.threads, m.processed, m.dropped, m.processed / std::max(1e-3, m.elapsedS), m.decode.mean, m.detect.mean,
        m.detect.p95, m.response.mean, m.track.mean, m.proc.mean, m.detect.mean > 0 ? 1000 / m.detect.mean : 0,
        m.latency.mean, s.emulateMs, r ? r->scales : 0, r && r->motionOn ? "true" : "false", m.motion.mean, m.lockPct, s.slowdown,
        r ? r->govLevel : -1);
    if (m.haveGt)
        len += snprintf(buf + len, sizeof(buf) - len,
                        ", \"recall_drone\": %.3f, \"recall_bird\": %.3f, \"false_tracks_per_frame\": %.2f, "
                        "\"drone_cls_acc\": %.3f, \"bird_cls_acc\": %.3f",
                        m.eval.recall(0), m.eval.recall(1), m.eval.falsePerFrame(), m.eval.clsAccuracy(0),
                        m.eval.clsAccuracy(1));
    snprintf(buf + len, sizeof(buf) - len, "}");
    return buf;
}
