#include "video_source.hpp"

#include <cstring>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
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

VideoSource::~VideoSource() { close(); }

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
    avcodec_parameters_to_context(ctx_, st->codecpar);
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
    info_.w = ctx_->width;
    info_.h = ctx_->height;
    AVRational fr = av_guess_frame_rate(fmt_, st, nullptr);
    info_.fps = (fr.num > 0 && fr.den > 0) ? av_q2d(fr) : 25.0;
    if (st->nb_frames > 0) info_.frames = st->nb_frames;
    else if (fmt_->duration > 0) info_.frames = (int64_t)(fmt_->duration / 1e6 * info_.fps);
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
        if (pkt_->stream_index == stream_) avcodec_send_packet(ctx_, pkt_);
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

bool VideoSource::read(uint8_t* dst, int outW, int outH, int dstStride) {
    if (!fmt_ || !decodeNext()) return false;
    convert(dst, outW, outH, dstStride);
    av_frame_unref(frame_);
    return true;
}

bool VideoSource::rewind() {
    if (!fmt_) return false;
    if (av_seek_frame(fmt_, stream_, 0, AVSEEK_FLAG_BACKWARD) < 0 &&
        avformat_seek_file(fmt_, -1, INT64_MIN, 0, INT64_MAX, 0) < 0)
        return false;
    avcodec_flush_buffers(ctx_);
    eof_ = false;
    return true;
}
