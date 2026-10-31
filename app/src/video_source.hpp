// Чтение видео через FFmpeg (libavformat + libavcodec + swscale) в серые кадры u8.
#pragma once
#include <cstdint>
#include <map>
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

// откуда взята частота кадров видео
enum class FpsFrom {
    File,        // из контейнера (mp4, mkv, avi, ...)
    Assumed,     // «голый» поток (.h264, .mjpeg с rpicam-vid): времени кадров в файле нет, частота —
                 // из параметров потока или 25 по умолчанию FFmpeg, может быть неверной
    Timecodes,   // из файла меток времени рядом с видео (rpicam-vid --save-pts, mkvmerge timecode v2)
    User,        // задана вручную (Settings::videoFps)
};

struct VideoInfo {
    int w = 0, h = 0;
    double fps = 25;
    int64_t frames = 0;      // оценка по длительности, может быть 0
    std::string codec;
    std::string pixfmt;
    int rotation = 0;        // поворот из метаданных (матрица отображения), градусов по часовой: 0/90/180/270
    FpsFrom fpsFrom = FpsFrom::File;
    std::string format;      // контейнер (демуксер FFmpeg): mov,mp4,..., h264, mjpeg
    std::string timecodes;   // файл меток времени, если частота и пропуски взяты из него
    int64_t dropped = 0;     // по меткам времени: пропущено кадров (интервал больше полутора)
    double fileFps = 0;      // при частоте, заданной вручную, — какая была в файле
};

// Откуда частота кадров, для человека: "" — из контейнера, иначе пояснение (предупреждение для «голого»
// потока, где частота могла быть взята наугад)
std::string fps_note(const VideoInfo& vi);

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
    void loadTimecodes(const std::string& path);
    void scanRaw();
    void notePacket();
    bool byteSeek(int64_t pos);
    bool rewindSeq();
    bool seekSeq(int64_t frame);
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
    int badFrames_ = 0;             // кадров подряд, которые декодер не смог собрать
    std::vector<uint8_t> lutY_;     // яркость 9-16 бит -> 8 бит (HDR/10-бит с телефона)
    int lutKey_ = -1;
    // частота, по которой метки времени переводятся в номера кадров: у контейнера — его частота, у
    // «голого» потока — шаг меток, которые проставил FFmpeg (не обязательно настоящая частота камеры)
    double idxFps_ = 25;
    // «голый» поток (seq_): номер кадра — по счёту от ключевого кадра, с которого читали; перемотка — по
    // байтовым позициям ключевых кадров (номер кадра потока -> позиция в файле)
    bool seq_ = false;
    bool reorder_ = false;   // в потоке есть B-кадры (порядок показа не совпадает с порядком декодирования)
    int64_t seqN_ = -1, pktN_ = 0, rawFrames_ = 0;
    std::map<int64_t, int64_t> keys_;
    // по файлу меток времени: номер кадра (по настоящему времени) k-го кадра потока; пропуск кадра
    // камерой — скачок номера, как у записи с потерями
    std::vector<int64_t> tcIdx_;
    VideoInfo info_;
    Orient orient_;
    std::vector<uint8_t> raw_;             // кадр до поворота
    std::vector<uint8_t> rawU_, rawV_;     // цвет до поворота
    std::vector<int> colOffU_, colOffV_;   // sampleColor: байт отсчёта цвета в строке
};

// Расширение диапазона яркости Y из ТВ-диапазона [16, 235] в полный [0, 255]
void expand_tv_range(const uint8_t* src, uint8_t* dst, int n);
