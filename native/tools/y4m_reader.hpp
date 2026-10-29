// Чтение кадров без FFmpeg: YUV4MPEG2 (.y4m, любой C420*/C422/C444/Cmono — берётся Y)
// или сырой поток gray/yuv420p заданного размера. Источник — файл или stdin ("-"), поэтому
// на Raspberry Pi можно подать поток прямо из ffmpeg или rpicam-vid:
//   ffmpeg -i in.mp4 -pix_fmt gray -f yuv4mpegpipe - | dronolok_bench -
//   rpicam-vid -t 0 --codec yuv420 --width 4608 --height 2592 -o - | dronolok_bench - --size 4608x2592 --yuv420
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

namespace dronolok {

class FrameReader {
public:
    // размер кадра из заголовка ограничен: испорченный файл не выделяет гигабайты на кадр
    static constexpr int MAX_SIDE = 1 << 15;
    static constexpr size_t MAX_LINE = 4096;   // строка заголовка и "FRAME ..."
    static bool sizeOk(int w, int h) { return w > 0 && h > 0 && w <= MAX_SIDE && h <= MAX_SIDE; }

    // rawW/rawH > 0 — сырой поток; yuv420 — после Y идут ещё U и V (w*h/2 байт)
    bool open(const std::string& path, int rawW = 0, int rawH = 0, bool yuv420 = false) {
        if (path == "-") {
#if defined(_WIN32)
            _setmode(_fileno(stdin), _O_BINARY);
#endif
            f_ = stdin;
        } else {
            f_ = std::fopen(path.c_str(), "rb");
        }
        if (!f_) return false;
        if (rawW > 0) {
            if (!sizeOk(rawW, rawH)) return false;
            w_ = rawW;
            h_ = rawH;
            skip_ = yuv420 ? (size_t)w_ * h_ / 2 : 0;
            y4m_ = false;
            return true;
        }
        std::string hdr;
        for (int c; hdr.size() < MAX_LINE && (c = std::fgetc(f_)) != EOF && c != '\n';) hdr += (char)c;
        if (hdr.compare(0, 10, "YUV4MPEG2 ") != 0) return false;
        std::string cs = "420";
        size_t p = 0;
        while ((p = hdr.find(' ', p)) != std::string::npos) {
            ++p;
            const char t = hdr[p];
            const std::string val = hdr.substr(p + 1, hdr.find(' ', p) - p - 1);
            auto side = [&] { return (int)std::min(std::max(std::strtol(val.c_str(), nullptr, 10), 0L), (long)MAX_SIDE + 1); };
            if (t == 'W') w_ = side();
            if (t == 'H') h_ = side();
            if (t == 'F') {
                int a = 0, b = 1;
                if (std::sscanf(val.c_str(), "%d:%d", &a, &b) == 2 && b) fps_ = (double)a / b;
            }
            if (t == 'C') cs = val;
        }
        if (!sizeOk(w_, h_)) return false;
        const size_t n = (size_t)w_ * h_;
        if (cs.compare(0, 4, "mono") == 0) skip_ = 0;
        else if (cs.compare(0, 3, "444") == 0) skip_ = 2 * n;
        else if (cs.compare(0, 3, "422") == 0) skip_ = n;
        else skip_ = 2 * (((size_t)(w_ + 1) / 2) * ((h_ + 1) / 2));
        y4m_ = true;
        return true;
    }
    ~FrameReader() {
        if (f_ && f_ != stdin) std::fclose(f_);
    }
    int width() const { return w_; }
    int height() const { return h_; }
    double fps() const { return fps_; }

    // Читает Y-плоскость следующего кадра (w*h байт). false — конец потока.
    bool next(std::vector<uint8_t>& y) {
        if (y4m_) {
            std::string line;
            for (int c; line.size() < MAX_LINE && (c = std::fgetc(f_)) != EOF && c != '\n';) line += (char)c;
            if (line.compare(0, 5, "FRAME") != 0) return false;
        }
        y.resize((size_t)w_ * h_);
        if (std::fread(y.data(), 1, y.size(), f_) != y.size()) return false;
        if (skip_) {
            skipBuf_.resize(skip_);
            if (std::fread(skipBuf_.data(), 1, skip_, f_) != skip_) return false;
        }
        return true;
    }

private:
    std::FILE* f_ = nullptr;
    int w_ = 0, h_ = 0;
    double fps_ = 25;
    size_t skip_ = 0;
    bool y4m_ = false;
    std::vector<uint8_t> skipBuf_;
};

}  // namespace dronolok
