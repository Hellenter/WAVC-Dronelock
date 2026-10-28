// Таблица вычислительных ядер одного набора инструкций (scalar / SSE2 / AVX2 / NEON).
// Все ядра работают над строками u8 и не знают ни про кадр, ни про потоки.
#pragma once
#include <cstdint>

namespace dronolok {

struct Kernels {
    const char* name;
    int width;   // байт в векторе
    // 2x2 пулинг n выходных пикселей из строк r0, r1 (по 2n байт):
    // dark = min 2x2, bright = 255 - max 2x2 (яркая цель становится тёмной).
    void (*pool2)(const uint8_t* r0, const uint8_t* r1, uint8_t* dark, uint8_t* bright, int n);
    // pool = 1: dark = src, bright = 255 - src
    void (*pool1)(const uint8_t* r0, uint8_t* dark, uint8_t* bright, int n);
    // Дилатация линейным ядром k сразу по 4 направлениям (—, |, \, /).
    // rows[i] — строка P[y + i - k/2], уже сдвинутая на столбец, с которого начинается выход;
    // выход направления d пишется в out + d * ostride. Читает rows[i][-k/2 .. n + k/2).
    void (*dilate4)(const uint8_t* const* rows, int k, uint8_t* out, int ostride, int n);
    // Эрозия по 4 направлениям, минимум по направлениям и вычитание P:
    // out[x] = min_d min_i D_d(y + o_i * dy_d, x + o_i * dx_d) - p[x] (с насыщением).
    // rows[i] — блок из 4 строк D (шаг dstride) для строки y + i - k/2.
    void (*erode4_sub)(const uint8_t* const* rows, int k, int dstride, const uint8_t* p, uint8_t* out, int n);
    // Статистика по тайлам tile x tile для rows строк полосы, начиная со строки r0 тайла:
    // сумма, максимум, число пикселей > hot_thr и позиция первого максимума (y*tile + x).
    // Накопительная: перед первой частью тайла все счётчики должны быть нулевыми.
    void (*tile_stats)(const uint8_t* band, int stride, int tile, int rows, int r0, int ntiles, uint8_t hot_thr,
                       uint32_t* sum, uint8_t* maxv, uint16_t* hot, uint16_t* argmax);
    // Сглаживание 3x3 [1 2 1]x[1 2 1]/16 с округлением: out[x] по строкам a (выше), b, c (ниже).
    // Читает столбцы -1..n каждой строки.
    void (*smooth3)(const uint8_t* a, const uint8_t* b, const uint8_t* c, uint8_t* out, int n);
};

const Kernels* kernels_scalar();
const Kernels* kernels_sse2();   // nullptr, если не собрано под эту платформу
const Kernels* kernels_avx2();
const Kernels* kernels_neon();

// лучший доступный набор (или запрошенный, если он поддержан)
enum class Isa : int;
const Kernels* select_kernels(Isa isa);

}  // namespace dronolok
