// Чтение видео через FFmpeg (libavformat + libavcodec + swscale) в серые кадры u8.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

// Цвет кадра для детектора (облака по цвету, DetectorConfig::sky_color; цвет целей, leaf_color): U и V
// (Cb, Cr, полный диапазон, 128 — серый), по отсчёту на блок step x step пикс. выходного кадра — в его
// точке (off, off) от угла. Шаг 4 с точкой в углу: каждый второй отсчёт — точно в центрах блоков 8x8
// (по ним учится маска неба), а все вместе — цвет целей в 5-20 пикс. (лист у камеры).
// w = 0 — цвета нет (серое видео, RGB, палитра)
struct ColorPlanes {
    std::vector<uint8_t> u, v;
    int w = 0, h = 0, step = 4, off = 0;
};

struct VideoInfo {
    int w = 0, h = 0;
    double fps = 25;
    int64_t frames = 0;      // оценка по длительности, может быть 0
    std::string codec;
    std::string pixfmt;
};

class VideoSource {
public:
    ~VideoSource();
    // threads: потоков декодера FFmpeg (0 = авто)
    bool open(const std::string& path, int threads, std::string& err);
    void close();
    bool isOpen() const { return fmt_ != nullptr; }
    const VideoInfo& info() const { return info_; }
    // Следующий кадр в gray outW x outH (полный диапазон 0..255, как cv2 BGR2GRAY) и, если
    // color не пусто, его цвет (выборка из плоскостей U и V декодера, без пересчёта кадра).
    // false — конец файла или ошибка.
    bool read(uint8_t* dst, int outW, int outH, int dstStride, ColorPlanes* color = nullptr);
    bool rewind();
    // Перейти к кадру frame (от 0): к ближайшему ключевому кадру до него, затем
    // декодировать вперёд; следующий read() вернёт frame (или ближайший после него).
    bool seek(int64_t frame);
    int64_t index() const { return index_; }   // номер кадра, который вернул последний read()

private:
    bool decodeNext();
    void convert(uint8_t* dst, int outW, int outH, int dstStride);
    void sampleColor(ColorPlanes& c, int outW, int outH);

    AVFormatContext* fmt_ = nullptr;
    AVCodecContext* ctx_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVPacket* pkt_ = nullptr;
    SwsContext* sws_ = nullptr;
    int stream_ = -1;
    bool eof_ = false;
    double tb_ = 0;                 // единица времени потока, с
    int64_t start_ = 0;             // pts первого кадра
    int64_t index_ = -1, skipTo_ = 0;
    VideoInfo info_;
    std::vector<int> colOffU_, colOffV_;   // sampleColor: байт отсчёта цвета в строке
};

// Расширение диапазона яркости Y из ТВ-диапазона [16, 235] в полный [0, 255]
void expand_tv_range(const uint8_t* src, uint8_t* dst, int n);
