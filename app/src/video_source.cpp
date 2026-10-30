#include "video_source.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "dronolok/fileutil.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/display.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/time.h>
#include <libswscale/swscale.h>
}

void expand_tv_range(const uint8_t* src, uint8_t* dst, int n) {
    static uint8_t lut[256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; ++i) {
            int v = ((i - 16) * 255 + 109) / 219;
            lut[i] = (uint8_t)(i <= 16 ? 0 : v > 255 ? 255 : v);
        }
        init = true;
    }
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        dst[i] = lut[src[i]];
        dst[i + 1] = lut[src[i + 1]];
        dst[i + 2] = lut[src[i + 2]];
        dst[i + 3] = lut[src[i + 3]];
    }
    for (; i < n; ++i) dst[i] = lut[src[i]];
}

// Отображение кадра по матрице поворота — как автоповорот в ffmpeg (fftools, insert_autorotate):
// 90/180/270° и отражения; прочие углы не поворачиваются
static void plane_orient(const uint8_t* s, int sw, int sh, size_t ss, uint8_t* d, size_t ds, bool transpose, bool fx,
                         bool fy) {
    if (!transpose) {
        for (int y = 0; y < sh; ++y) {
            const uint8_t* r = s + (size_t)(fy ? sh - 1 - y : y) * ss;
            uint8_t* o = d + (size_t)y * ds;
            if (!fx) memcpy(o, r, sw);
            else
                for (int x = 0; x < sw; ++x) o[x] = r[sw - 1 - x];
        }
        return;
    }
    // выход sh x sw: out(x, y) = in(y, x) с отражениями; блоками 32x32 — чтение по столбцу не вымывает кэш
    const int B = 32;
    for (int y0 = 0; y0 < sw; y0 += B)
        for (int x0 = 0; x0 < sh; x0 += B) {
            const int y1 = std::min(sw, y0 + B), x1 = std::min(sh, x0 + B);
            for (int y = y0; y < y1; ++y) {
                const int sx = fx ? sw - 1 - y : y;
                uint8_t* o = d + (size_t)y * ds;
                for (int x = x0; x < x1; ++x) o[x] = s[(size_t)(fy ? sh - 1 - x : x) * ss + sx];
            }
        }
}

// «Голый» поток кадров без контейнера (rpicam-vid пишет .h264 и .mjpeg так): времени кадров в нём нет,
// FFmpeg нумерует их сам по частоте из параметров потока или по 25 к/с
static bool is_raw_stream(const std::string& f) {
    static const char* raw[] = {"h264",     "hevc",      "vvc",       "mjpeg",      "mjpeg_2000", "jpeg_pipe",
                                "image2pipe", "m4v",      "mpegvideo", "h263",       "rawvideo"};
    for (const char* r : raw)
        if (f == r) return true;
    return false;
}

std::string fps_note(const VideoInfo& vi) {
    char b[512];
    switch (vi.fpsFrom) {
        case FpsFrom::Assumed:
            snprintf(b, sizeof(b),
                     "в файле (%s) нет времени кадров: частота %.4g к/с взята из потока или по умолчанию — если камера "
                     "снимала с другой частотой, укажите её (Видео → частота кадров файла) или положите рядом файл меток "
                     "rpicam-vid --save-pts",
                     vi.format.c_str(), vi.fps);
            return b;
        case FpsFrom::Timecodes: {
            std::string f = vi.timecodes;
            const size_t sl = f.find_last_of("/\\");
            if (sl != std::string::npos) f = f.substr(sl + 1);
            snprintf(b, sizeof(b), "частота по меткам времени %s, пропущено кадров: %lld", f.c_str(), (long long)vi.dropped);
            return b;
        }
        case FpsFrom::User:
            snprintf(b, sizeof(b), "частота задана вручную (в файле %.4g к/с)", vi.fileFps);
            return b;
        default:
            return "";
    }
}

VideoSource::~VideoSource() { close(); }

// Метки времени кадров рядом с «голым» потоком — rpicam-vid --save-pts <файл>, mkvmerge timecode v2:
// первая строка "# timecode format v2", дальше по строке на кадр, миллисекунды. Частота — по
// обычному интервалу, кадр, которого камера не успела снять, — пропуск номера
void VideoSource::loadTimecodes(const std::string& path) {
    std::string stem = path;
    const size_t dot = stem.find_last_of('.'), sl = stem.find_last_of("/\\");
    if (dot != std::string::npos && (sl == std::string::npos || dot > sl)) stem.resize(dot);
    const std::string cand[] = {path + ".pts",     stem + ".pts",        stem + ".txt",           stem + "_pts.txt",
                                stem + ".timestamps", stem + "_timestamps.txt", path + ".txt"};
    for (const auto& c : cand) {
        FILE* f = dronolok::open_file(c, "rb");
        if (!f) continue;
        std::vector<double> t;
        char line[256];
        bool v2 = false;
        for (int n = 0; fgets(line, sizeof(line), f); ++n) {
            if (n == 0) {
                v2 = strstr(line, "timecode format v2") != nullptr;
                if (!v2) break;
                continue;
            }
            if (line[0] == '#') continue;
            char* e = nullptr;
            const double ms = strtod(line, &e);
            if (e == line || !std::isfinite(ms)) continue;
            t.push_back(ms / 1000.0);
            if (t.size() >= 100000000) break;
        }
        fclose(f);
        if (!v2 || t.size() < 3) continue;
        std::vector<double> d;
        for (size_t i = 1; i < t.size(); ++i)
            if (t[i] > t[i - 1]) d.push_back(t[i] - t[i - 1]);
        if (d.size() < 2) continue;
        std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
        const double dt = d[d.size() / 2];
        if (!(dt > 1e-4 && dt < 100)) continue;
        tcIdx_.resize(t.size());
        int64_t drop = 0;
        for (size_t i = 0; i < t.size(); ++i) {
            int64_t k = std::llround((t[i] - t[0]) / dt);
            if (i && k <= tcIdx_[i - 1]) k = tcIdx_[i - 1] + 1;   // метки не по порядку — подряд
            if (i) drop += k - tcIdx_[i - 1] - 1;
            tcIdx_[i] = k;
        }
        info_.fps = 1.0 / dt;
        info_.fpsFrom = FpsFrom::Timecodes;
        info_.timecodes = c;
        info_.dropped = drop;
        return;
    }
}

void VideoSource::close() {
    if (sws_) sws_freeContext(sws_);
    if (frame_) av_frame_free(&frame_);
    if (pkt_) av_packet_free(&pkt_);
    if (ctx_) avcodec_free_context(&ctx_);
    if (fmt_) avformat_close_input(&fmt_);
    sws_ = nullptr;
    stream_ = -1;
    eof_ = false;
    info_ = VideoInfo();
    orient_ = Orient();
    tcIdx_.clear();
    idxFps_ = 25;
    seq_ = false;
    keys_.clear();
}

bool VideoSource::open(const std::string& path, int threads, std::string& err) {
    close();
    char buf[256];
    int r = avformat_open_input(&fmt_, path.c_str(), nullptr, nullptr);
    if (r < 0) {
        av_strerror(r, buf, sizeof(buf));
        err = std::string("не удалось открыть файл: ") + buf;
        fmt_ = nullptr;
        return false;
    }
    if (avformat_find_stream_info(fmt_, nullptr) < 0) {
        err = "не удалось прочитать параметры потоков";
        close();
        return false;
    }
    const AVCodec* codec = nullptr;
    stream_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (stream_ < 0 || !codec) {
        err = "в файле нет видео, которое умеет декодировать эта сборка";
        close();
        return false;
    }
    AVStream* st = fmt_->streams[stream_];
    ctx_ = avcodec_alloc_context3(codec);
    if (!ctx_ || avcodec_parameters_to_context(ctx_, st->codecpar) < 0) {
        err = "не удалось подготовить декодер (испорченные параметры потока или нет памяти)";
        close();
        return false;
    }
    ctx_->thread_count = threads;   // 0 = FFmpeg выберет сам
    ctx_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    if ((r = avcodec_open2(ctx_, codec, nullptr)) < 0) {
        av_strerror(r, buf, sizeof(buf));
        err = std::string("не удалось открыть декодер: ") + buf;
        close();
        return false;
    }
    frame_ = av_frame_alloc();
    pkt_ = av_packet_alloc();
    if (!frame_ || !pkt_) {
        err = "не хватило памяти под кадр";
        close();
        return false;
    }
    if (ctx_->width <= 0 || ctx_->height <= 0) {
        err = "в потоке не указан размер кадра";
        close();
        return false;
    }
    info_.w = ctx_->width;
    info_.h = ctx_->height;
    // матрица отображения (телефоны пишут поворот в метаданные, а кадр — как лежал датчик)
    const int32_t* dm = nullptr;
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(60, 29, 100)
    if (const AVPacketSideData* sd = av_packet_side_data_get(st->codecpar->coded_side_data,
                                                             st->codecpar->nb_coded_side_data,
                                                             AV_PKT_DATA_DISPLAYMATRIX))
        if (sd->size >= 9 * sizeof(int32_t)) dm = (const int32_t*)sd->data;
#else
    size_t dmSize = 0;
    if (const uint8_t* sd = av_stream_get_side_data(st, AV_PKT_DATA_DISPLAYMATRIX, &dmSize))
        if (dmSize >= 9 * sizeof(int32_t)) dm = (const int32_t*)sd;
#endif
    if (dm) {
        double theta = -std::round(av_display_rotation_get(dm));   // по часовой
        if (std::isnan(theta)) theta = 0;
        theta -= 360 * std::floor(theta / 360 + 0.9 / 360);
        Orient& o = orient_;
        if (std::fabs(theta - 90) < 1) {
            o.transpose = true;
            o.fy = dm[3] <= 0;   // иначе (dm[3] > 0) — с отражением, только транспонирование
            info_.rotation = 90;
        } else if (std::fabs(theta - 180) < 1) {
            o.fx = dm[0] < 0;
            o.fy = dm[4] < 0;
            info_.rotation = 180;
        } else if (std::fabs(theta - 270) < 1) {
            o.transpose = true;
            o.fx = true;
            o.fy = dm[3] < 0;
            info_.rotation = 270;
        } else if (std::fabs(theta) < 1) {
            o.fy = dm[4] < 0;
        }
        if (o.transpose) std::swap(info_.w, info_.h);
    }
    AVRational fr = av_guess_frame_rate(fmt_, st, nullptr);
    info_.fps = (fr.num > 0 && fr.den > 0) ? av_q2d(fr) : 25.0;
    idxFps_ = info_.fps;
    tb_ = av_q2d(st->time_base);
    start_ = st->start_time != AV_NOPTS_VALUE ? st->start_time : 0;
    index_ = -1;
    skipTo_ = 0;
    tcIdx_.clear();
    info_.format = fmt_->iformat && fmt_->iformat->name ? fmt_->iformat->name : "";
    if (is_raw_stream(info_.format)) {
        info_.fpsFrom = FpsFrom::Assumed;
        seq_ = true;
        scanRaw();
        loadTimecodes(path);
    }
    if (!tcIdx_.empty()) info_.frames = tcIdx_.back() + 1 + std::max<int64_t>(0, rawFrames_ - (int64_t)tcIdx_.size());
    else if (seq_ && rawFrames_ > 0) info_.frames = rawFrames_;
    else if (st->nb_frames > 0) info_.frames = st->nb_frames;
    else if (fmt_->duration > 0) info_.frames = (int64_t)(fmt_->duration / 1e6 * idxFps_);
    info_.codec = codec->name;
    const char* pf = av_get_pix_fmt_name(ctx_->pix_fmt);
    info_.pixfmt = pf ? pf : "?";
    return true;
}

bool VideoSource::decodeNext() {
    for (;;) {
        int r = avcodec_receive_frame(ctx_, frame_);
        if (r == 0) return true;
        if (r == AVERROR_EOF) return false;
        if (r != AVERROR(EAGAIN)) return false;
        if (eof_) return false;
        r = av_read_frame(fmt_, pkt_);
        if (r < 0) {   // конец файла: досливаем буферы декодера
            eof_ = true;
            avcodec_send_packet(ctx_, nullptr);
            continue;
        }
        if (pkt_->stream_index == stream_) {
            if (seq_) notePacket();
            avcodec_send_packet(ctx_, pkt_);
        }
        av_packet_unref(pkt_);
    }
}

void VideoSource::convert(uint8_t* dst, int outW, int outH, int dstStride) {
    const AVPixelFormat f = (AVPixelFormat)frame_->format;
    const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(f);
    const bool full = frame_->color_range == AVCOL_RANGE_JPEG || f == AV_PIX_FMT_YUVJ420P ||
                      f == AV_PIX_FMT_YUVJ422P || f == AV_PIX_FMT_YUVJ444P || f == AV_PIX_FMT_GRAY8;
    // 8-битный YUV/серый того же размера: яркость берётся прямо из плоскости Y
    const bool plainY = d && !(d->flags & AV_PIX_FMT_FLAG_RGB) && !(d->flags & AV_PIX_FMT_FLAG_PAL) &&
                        d->comp[0].depth == 8 && d->comp[0].plane == 0 && d->comp[0].step == 1;
    if (plainY && outW == frame_->width && outH == frame_->height) {
        for (int y = 0; y < outH; ++y) {
            const uint8_t* s = frame_->data[0] + (size_t)y * frame_->linesize[0];
            uint8_t* o = dst + (size_t)y * dstStride;
            if (full) memcpy(o, s, outW);
            else expand_tv_range(s, o, outW);
        }
        return;
    }
    const int flags = (outW < frame_->width) ? SWS_AREA : SWS_BILINEAR;
    sws_ = sws_getCachedContext(sws_, frame_->width, frame_->height, f, outW, outH, AV_PIX_FMT_GRAY8, flags,
                                nullptr, nullptr, nullptr);
    if (!sws_) return;
    const int* coef = sws_getCoefficients(SWS_CS_DEFAULT);
    sws_setColorspaceDetails(sws_, coef, full ? 1 : 0, coef, 1, 0, 1 << 16, 1 << 16);
    uint8_t* planes[4] = {dst, nullptr, nullptr, nullptr};
    int strides[4] = {dstStride, 0, 0, 0};
    sws_scale(sws_, frame_->data, frame_->linesize, 0, frame_->height, planes, strides);
}

// Цвет: отсчёт U и V в точке (off, off) каждого блока step x step выходного кадра (уменьшение и
// увеличение — пересчётом координат). Плоскости 8-16 бит, планарные и NV12/P010; RGB и палитра — без
// цвета. Смещения в строке и перевод диапазона — таблицами: 2560x1440 с шагом 4 (640x360 отсчётов)
// ~0.7 мс, половина — чтение строк цвета из памяти (с шагом 8 — 0.25 мс)
void VideoSource::sampleColor(ColorPlanes& c, int outW, int outH) {
    c.w = c.h = 0;
    const AVPixFmtDescriptor* d = av_pix_fmt_desc_get((AVPixelFormat)frame_->format);
    if (!d || d->nb_components < 3 || (d->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_PAL | AV_PIX_FMT_FLAG_BE)) ||
        (d->flags & AV_PIX_FMT_FLAG_HWACCEL) || c.step <= 0 || outW < c.step || outH < c.step)
        return;
    const AVComponentDescriptor& cu = d->comp[1];
    const AVComponentDescriptor& cv = d->comp[2];
    if (cu.depth < 8 || cu.depth > 16 || cv.depth != cu.depth || !frame_->data[cu.plane] || !frame_->data[cv.plane]) return;
    const AVPixelFormat f = (AVPixelFormat)frame_->format;
    const bool full = frame_->color_range == AVCOL_RANGE_JPEG || f == AV_PIX_FMT_YUVJ420P || f == AV_PIX_FMT_YUVJ422P ||
                      f == AV_PIX_FMT_YUVJ444P;
    c.w = outW / c.step;
    c.h = outH / c.step;
    c.u.resize((size_t)c.w * c.h);
    c.v.resize((size_t)c.w * c.h);
    const double sx = (double)frame_->width / outW, sy = (double)frame_->height / outH;
    const int cwMax = -((-frame_->width) >> d->log2_chroma_w) - 1;
    const int chMax = -((-frame_->height) >> d->log2_chroma_h) - 1;
    colOffU_.resize(c.w);
    colOffV_.resize(c.w);
    for (int bx = 0; bx < c.w; ++bx) {   // байт отсчёта в строке (шаг и смещение — из описания формата)
        const int x = std::min(frame_->width - 1, (int)((bx * c.step + c.off) * sx));
        const int cx = std::min(cwMax, x >> d->log2_chroma_w);
        colOffU_[bx] = cx * cu.step + cu.offset;
        colOffV_[bx] = cx * cv.step + cv.offset;
    }
    uint8_t lut[256];   // ТВ-диапазон цвета 16..240 -> полный 0..255 (как яркость в convert)
    for (int i = 0; i < 256; ++i) {
        const int val = full ? i : 128 + ((i - 128) * 255 + (i >= 128 ? 112 : -112)) / 224;
        lut[i] = (uint8_t)std::min(255, std::max(0, val));
    }
    const bool wide = cu.depth > 8;   // отсчёт в 16 битах: старшие 8 бит значения
    const int su = cu.shift + cu.depth - 8, sv = cv.shift + cv.depth - 8;
    for (int by = 0; by < c.h; ++by) {
        const int y = std::min(frame_->height - 1, (int)((by * c.step + c.off) * sy));
        const int cy = std::min(chMax, y >> d->log2_chroma_h);
        const uint8_t* ru = frame_->data[cu.plane] + (size_t)cy * frame_->linesize[cu.plane];
        const uint8_t* rv = frame_->data[cv.plane] + (size_t)cy * frame_->linesize[cv.plane];
        uint8_t* ou = &c.u[(size_t)by * c.w];
        uint8_t* ov = &c.v[(size_t)by * c.w];
        if (wide) {
            for (int bx = 0; bx < c.w; ++bx) {
                const uint8_t* qu = ru + colOffU_[bx];
                const uint8_t* qv = rv + colOffV_[bx];
                ou[bx] = lut[((qu[0] | (qu[1] << 8)) >> su) & 255];
                ov[bx] = lut[((qv[0] | (qv[1] << 8)) >> sv) & 255];
            }
        } else {
            for (int bx = 0; bx < c.w; ++bx) {
                ou[bx] = lut[ru[colOffU_[bx]]];
                ov[bx] = lut[rv[colOffV_[bx]]];
            }
        }
    }
}

bool VideoSource::read(uint8_t* dst, int outW, int outH, int dstStride, ColorPlanes* color) {
    if (!fmt_) return false;
    for (;;) {
        if (!decodeNext()) return false;
        // номер кадра — по времени показа: после перемотки счёт кадров с начала неизвестен
        const int64_t pts = frame_->best_effort_timestamp;
        if (seq_) {   // «голый» поток: кадры по счёту от ключевого, с которого читали
            const int64_t k = ++seqN_;
            if (tcIdx_.empty()) index_ = k;
            else if (k < (int64_t)tcIdx_.size()) index_ = tcIdx_[(size_t)k];   // по меткам времени камеры
            else index_ = tcIdx_.back() + (k - ((int64_t)tcIdx_.size() - 1));
        } else {
            index_ = pts != AV_NOPTS_VALUE && tb_ > 0 ? (int64_t)std::llround((pts - start_) * tb_ * idxFps_) : index_ + 1;
        }
        if (index_ >= skipTo_) break;
        av_frame_unref(frame_);   // до нужного кадра после перемотки — без преобразования
    }
    skipTo_ = 0;
    if (orient_.identity()) {
        convert(dst, outW, outH, dstStride);
        if (color) sampleColor(*color, outW, outH);
    } else {   // декодированный кадр в размере до поворота, затем поворот в dst
        const Orient& o = orient_;
        const int rw = o.transpose ? outH : outW, rh = o.transpose ? outW : outH;
        raw_.resize((size_t)rw * rh);
        convert(raw_.data(), rw, rh, rw);
        plane_orient(raw_.data(), rw, rh, rw, dst, dstStride, o.transpose, o.fx, o.fy);
        if (color) {
            sampleColor(*color, rw, rh);
            if (color->w > 0) {
                const int cw = color->w, ch = color->h;
                rawU_.swap(color->u);
                rawV_.swap(color->v);
                color->u.resize(rawU_.size());
                color->v.resize(rawV_.size());
                if (o.transpose) std::swap(color->w, color->h);
                plane_orient(rawU_.data(), cw, ch, cw, color->u.data(), color->w, o.transpose, o.fx, o.fy);
                plane_orient(rawV_.data(), cw, ch, cw, color->v.data(), color->w, o.transpose, o.fx, o.fy);
            }
        }
    }
    av_frame_unref(frame_);
    return true;
}

bool VideoSource::seek(int64_t frame) {
    if (!fmt_) return false;
    frame = std::max<int64_t>(0, frame);
    if (seq_) return seekSeq(frame);
    const int64_t ts = start_ + (tb_ > 0 ? (int64_t)(frame / idxFps_ / tb_) : 0);
    if (av_seek_frame(fmt_, stream_, ts, AVSEEK_FLAG_BACKWARD) < 0 && !rewind()) return false;
    avcodec_flush_buffers(ctx_);
    eof_ = false;
    index_ = -1;
    skipTo_ = frame;
    return true;
}

// ---- «голый» поток: меток времени у пакетов нет, перемотка — по байтам до ключевого кадра.
// Номер ключевого кадра = число пакетов до него (у IDR кадры до него в порядке показа идут и в
// порядке декодирования раньше), позиции ключевых — из просмотра файла при открытии и по ходу чтения

// Ключевой кадр, с которого номер по счёту пакетов верен: IDR у H.264/H.265 (у I-кадра «открытой»
// группы кадры, показанные до него, в порядке декодирования идут после и при перемотке выпадают —
// счёт сбился бы), любой кадр у MJPEG; у прочих — если в потоке нет B-кадров
static bool seek_key(const AVPacket* p, AVCodecID id, bool reorder) {
    if (!(p->flags & AV_PKT_FLAG_KEY) || p->pos < 0) return false;
    if (id != AV_CODEC_ID_H264 && id != AV_CODEC_ID_HEVC) return id == AV_CODEC_ID_MJPEG || !reorder;
    for (int i = 0; i + 3 < p->size; ++i) {
        if (p->data[i] || p->data[i + 1] || p->data[i + 2] != 1) continue;
        const uint8_t b = p->data[i + 3];
        if (id == AV_CODEC_ID_H264 ? (b & 0x1f) == 5 : ((b >> 1) & 0x3f) == 19 || ((b >> 1) & 0x3f) == 20) return true;
        i += 2;
    }
    return false;
}

void VideoSource::notePacket() {
    if (seek_key(pkt_, ctx_->codec_id, reorder_)) keys_.emplace(pktN_, pkt_->pos);
    ++pktN_;
}

// Просмотр пакетов без декодирования: число кадров и позиции ключевых. Не дольше ~2 с — у очень
// длинного файла остальное доберётся по ходу чтения
void VideoSource::scanRaw() {
    keys_.clear();
    rawFrames_ = 0;
    reorder_ = fmt_->streams[stream_]->codecpar->video_delay > 0 || ctx_->has_b_frames > 0;
    const double t0 = (double)av_gettime_relative() / 1000.0;
    AVPacket* p = av_packet_alloc();
    if (!p) return;
    int64_t n = 0;
    bool full = true;
    for (int r; (r = av_read_frame(fmt_, p)) >= 0;) {
        if (p->stream_index == stream_) {
            if (seek_key(p, ctx_->codec_id, reorder_)) keys_.emplace(n, p->pos);
            ++n;
        }
        av_packet_unref(p);
        if ((n & 63) == 0 && (double)av_gettime_relative() / 1000.0 - t0 > 2000) {
            full = false;
            break;
        }
    }
    av_packet_free(&p);
    if (full) rawFrames_ = n;
    rewindSeq();
}

bool VideoSource::byteSeek(int64_t pos) {
    if (av_seek_frame(fmt_, -1, pos, AVSEEK_FLAG_BYTE) < 0 &&
        avformat_seek_file(fmt_, -1, pos, pos, pos, AVSEEK_FLAG_BYTE) < 0)
        return false;
    avcodec_flush_buffers(ctx_);
    eof_ = false;
    return true;
}

bool VideoSource::rewindSeq() {
    const int64_t pos = keys_.empty() ? 0 : keys_.begin()->second;
    if (!byteSeek(keys_.empty() || keys_.begin()->first != 0 ? 0 : pos)) return false;
    pktN_ = 0;
    seqN_ = -1;
    index_ = -1;
    skipTo_ = 0;
    return true;
}

bool VideoSource::seekSeq(int64_t frame) {
    int64_t k = frame;   // кадр потока, с которого начинается нужный номер
    if (!tcIdx_.empty())
        k = std::min<int64_t>(std::lower_bound(tcIdx_.begin(), tcIdx_.end(), frame) - tcIdx_.begin(),
                              (int64_t)tcIdx_.size() - 1);
    auto it = keys_.upper_bound(k);
    const bool ahead = seqN_ + 1 <= k && (it == keys_.begin() || std::prev(it)->first <= seqN_ + 1);
    if (!ahead) {   // назад или вперёд за ключевой кадр: с ближайшего ключевого до него
        if (it == keys_.begin()) {
            if (!rewindSeq()) return false;
        } else {
            --it;
            if (!byteSeek(it->second)) return false;
            pktN_ = it->first;
            seqN_ = it->first - 1;
        }
    }
    index_ = -1;
    skipTo_ = frame;
    return true;
}

bool VideoSource::rewind() {
    if (!fmt_) return false;
    if (seq_) return rewindSeq();
    if (av_seek_frame(fmt_, stream_, 0, AVSEEK_FLAG_BACKWARD) < 0 &&
        avformat_seek_file(fmt_, -1, INT64_MIN, 0, INT64_MAX, 0) < 0)
        return false;
    avcodec_flush_buffers(ctx_);
    eof_ = false;
    index_ = -1;
    skipTo_ = 0;
    return true;
}
