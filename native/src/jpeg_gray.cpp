// Серый baseline JPEG (ITU T.81, стандартные таблицы Хаффмана из приложения K). Без зависимостей:
// на Raspberry Pi ядру не нужен libjpeg, а вырезки журнала маленькие (160x160) — простого DCT хватает.
#include <cmath>
#include <algorithm>
#include <cstring>

#include "dronolok/fieldlog.hpp"

namespace dronolok {
namespace {

const uint8_t ZIGZAG[64] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
                            12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
                            35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
                            58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};
// таблица квантования яркости (K.1), по строкам
const uint8_t STD_Q[64] = {16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
                           14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
                           18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
                           49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};
const uint8_t DC_BITS[16] = {0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
const uint8_t DC_VALS[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const uint8_t AC_BITS[16] = {0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d};
const uint8_t AC_VALS[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71,
    0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72,
    0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83,
    0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3,
    0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};

struct Huff {
    uint16_t code[256];
    uint8_t size[256];
    Huff(const uint8_t* bits, const uint8_t* vals) {
        std::memset(size, 0, sizeof(size));
        std::memset(code, 0, sizeof(code));
        unsigned c = 0, k = 0;
        for (int len = 1; len <= 16; ++len) {
            for (int i = 0; i < bits[len - 1]; ++i, ++k) {
                code[vals[k]] = (uint16_t)c++;
                size[vals[k]] = (uint8_t)len;
            }
            c <<= 1;
        }
    }
};

struct Tables {
    Huff dc{DC_BITS, DC_VALS}, ac{AC_BITS, AC_VALS};
    float cosT[8][8];   // C(u)/2 * cos((2x+1)u pi/16)
    Tables() {
        for (int u = 0; u < 8; ++u)
            for (int x = 0; x < 8; ++x)
                cosT[u][x] = (float)((u ? 0.5 : 0.5 / std::sqrt(2.0)) * std::cos((2 * x + 1) * u * 3.14159265358979323846 / 16));
    }
};
const Tables& tables() {
    static const Tables t;
    return t;
}

struct Bits {
    std::vector<uint8_t>& out;
    uint32_t acc = 0;
    int n = 0;
    explicit Bits(std::vector<uint8_t>& o) : out(o) {}
    void put(uint32_t v, int len) {
        acc = (acc << len) | (v & ((1u << len) - 1));
        n += len;
        while (n >= 8) {
            const uint8_t b = (uint8_t)(acc >> (n - 8));
            out.push_back(b);
            if (b == 0xFF) out.push_back(0);   // байт 0xFF в данных — с нулём после
            n -= 8;
        }
        acc &= (1u << n) - 1;
    }
    void flush() {
        if (n) put(0x7F, 8 - n);   // добить единицами
    }
};

void u16(std::vector<uint8_t>& o, int v) {
    o.push_back((uint8_t)(v >> 8));
    o.push_back((uint8_t)v);
}

int category(int v) {
    v = v < 0 ? -v : v;
    int c = 0;
    while (v) ++c, v >>= 1;
    return c;
}

}  // namespace

std::vector<uint8_t> encode_jpeg_gray(const uint8_t* px, int w, int h, int stride, int quality) {
    std::vector<uint8_t> o;
    if (!px || w <= 0 || h <= 0 || w > 65535 || h > 65535) return o;
    const Tables& T = tables();
    quality = quality < 1 ? 1 : quality > 100 ? 100 : quality;
    const int sc = quality < 50 ? 5000 / quality : 200 - 2 * quality;
    uint8_t q[64];
    float qinv[64];
    for (int i = 0; i < 64; ++i) {
        const int v = (STD_Q[i] * sc + 50) / 100;
        q[i] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : v);
        qinv[i] = 1.f / q[i];
    }
    o.reserve((size_t)w * h / 4 + 1024);
    const uint8_t head[] = {0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 'J', 'F', 'I', 'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0};
    o.insert(o.end(), head, head + sizeof(head));
    o.push_back(0xFF), o.push_back(0xDB), u16(o, 67), o.push_back(0);
    for (int i = 0; i < 64; ++i) o.push_back(q[ZIGZAG[i]]);
    o.push_back(0xFF), o.push_back(0xC0), u16(o, 11), o.push_back(8), u16(o, h), u16(o, w);
    o.push_back(1), o.push_back(1), o.push_back(0x11), o.push_back(0);
    auto dht = [&](int cls, const uint8_t* bits, const uint8_t* vals) {
        int n = 0;
        for (int i = 0; i < 16; ++i) n += bits[i];
        o.push_back(0xFF), o.push_back(0xC4), u16(o, 3 + 16 + n), o.push_back((uint8_t)(cls << 4));
        o.insert(o.end(), bits, bits + 16);
        o.insert(o.end(), vals, vals + n);
    };
    dht(0, DC_BITS, DC_VALS);
    dht(1, AC_BITS, AC_VALS);
    const uint8_t sos[] = {0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00};
    o.insert(o.end(), sos, sos + sizeof(sos));

    Bits b(o);
    int prevDc = 0;
    float blk[64], tmp[64];
    for (int by = 0; by < h; by += 8) {
        for (int bx = 0; bx < w; bx += 8) {
            for (int y = 0; y < 8; ++y) {   // за краем — повтор крайнего пикселя
                const uint8_t* row = px + (size_t)std::min(by + y, h - 1) * stride;
                for (int x = 0; x < 8; ++x) blk[y * 8 + x] = (float)row[std::min(bx + x, w - 1)] - 128.f;
            }
            for (int y = 0; y < 8; ++y)   // по строкам
                for (int u = 0; u < 8; ++u) {
                    float s = 0;
                    for (int x = 0; x < 8; ++x) s += T.cosT[u][x] * blk[y * 8 + x];
                    tmp[y * 8 + u] = s;
                }
            int coef[64];
            for (int u = 0; u < 8; ++u)   // по столбцам
                for (int v = 0; v < 8; ++v) {
                    float s = 0;
                    for (int y = 0; y < 8; ++y) s += T.cosT[v][y] * tmp[y * 8 + u];
                    coef[v * 8 + u] = (int)std::lround(s * qinv[v * 8 + u]);
                }
            const int dc = coef[0], diff = dc - prevDc;
            prevDc = dc;
            const int dcat = category(diff);
            b.put(T.dc.code[dcat], T.dc.size[dcat]);
            if (dcat) b.put((uint32_t)(diff < 0 ? diff - 1 : diff), dcat);
            int run = 0;
            for (int i = 1; i < 64; ++i) {
                const int v = coef[ZIGZAG[i]];
                if (!v) {
                    ++run;
                    continue;
                }
                while (run > 15) {
                    b.put(T.ac.code[0xF0], T.ac.size[0xF0]);
                    run -= 16;
                }
                const int vc = v > 1023 ? 1023 : v < -1023 ? -1023 : v;   // baseline: AC не больше 10 бит
                const int cat = category(vc);
                const int sym = (run << 4) | cat;
                b.put(T.ac.code[sym], T.ac.size[sym]);
                b.put((uint32_t)(vc < 0 ? vc - 1 : vc), cat);
                run = 0;
            }
            if (run) b.put(T.ac.code[0], T.ac.size[0]);
        }
    }
    b.flush();
    o.push_back(0xFF), o.push_back(0xD9);
    return o;
}

}  // namespace dronolok
