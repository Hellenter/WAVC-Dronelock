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

// Цвет кадра для детектора (облака по цвету, DetectorConfig::sky_color): U и V (Cb, Cr, полный
// диапазон, 128 — серый), по отсчёту на блок step x step пикс. выходного кадра — в центре блока.
// w = 0 — цвета нет (серое видео, RGB, палитра)
struct ColorPlanes {
    std::vector<uint8_t> u, v;
    int w = 0, h = 0, step = 8;
};

struct VideoInfo {
    int w = 0, h = 0;
    double fps = 25;
    int64_t frames = 0;      // оценка по длительности, может быть 0
    std::string codec;
    std::string pixfmt;
    int rotation = 0;        // поворот из метаданных (матрица отображения), градусов по часовой: 0/90/180/270
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

    // Как показать кадр по матрице отображения (телефон снимает «вверх ногами» или стоя):
    // выходной (x, y) берётся из декодированного (y, x) при transpose, затем отражается по fx/fy
    struct Orient {
        bool transpose = false, fx = false, fy = false;
        bool identity() const { return !transpose && !fx && !fy; }
    };

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
    Orient orient_;
    std::vector<uint8_t> raw_;             // кадр до поворота
    std::vector<uint8_t> rawU_, rawV_;     // цвет до поворота
    std::vector<int> colOffU_, colOffV_;   // sampleColor: байт отсчёта цвета в строке
};

// Расширение диапазона яркости Y из ТВ-диапазона [16, 235] в полный [0, 255]
void expand_tv_range(const uint8_t* src, uint8_t* dst, int n);
