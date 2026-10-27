// Тонкая обёртка над векторными инструкциями. Один и тот же код ядер
// (kernels_impl.hpp) компилируется под AVX2, SSE2, NEON и скалярный вариант:
// единица перед включением выбирает ровно один из макросов DL_SIMD_*.
//
// Всё объявлено static inline: каждый .cpp с ядрами получает свою копию,
// и AVX2-код не может "утечь" в SSE2-путь через общий символ при линковке.
#pragma once
#include <cstdint>
#include <cstring>

#if defined(DL_SIMD_AVX2)
#include <immintrin.h>
enum { W = 32 };
struct V { __m256i v; };
struct S { __m256i v; };   // аккумулятор суммы (4 x u64)
static inline V ld(const uint8_t* p) { return {_mm256_loadu_si256((const __m256i*)p)}; }
static inline void st(uint8_t* p, V a) { _mm256_storeu_si256((__m256i*)p, a.v); }
static inline V vdup(uint8_t x) { return {_mm256_set1_epi8((char)x)}; }
static inline V vmax(V a, V b) { return {_mm256_max_epu8(a.v, b.v)}; }
static inline V vmin(V a, V b) { return {_mm256_min_epu8(a.v, b.v)}; }
static inline V vsubs(V a, V b) { return {_mm256_subs_epu8(a.v, b.v)}; }
static inline V vadd(V a, V b) { return {_mm256_add_epi8(a.v, b.v)}; }
static inline S szero() { return {_mm256_setzero_si256()}; }
static inline S sacc(S s, V a) { return {_mm256_add_epi64(s.v, _mm256_sad_epu8(a.v, _mm256_setzero_si256()))}; }
static inline uint32_t shsum(S s) {
    alignas(32) uint64_t t[4];
    _mm256_store_si256((__m256i*)t, s.v);
    return (uint32_t)(t[0] + t[1] + t[2] + t[3]);
}
static inline uint8_t vhmax(V a) {
    __m128i m = _mm_max_epu8(_mm256_castsi256_si128(a.v), _mm256_extracti128_si256(a.v, 1));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 8));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 4));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 2));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 1));
    return (uint8_t)_mm_cvtsi128_si32(m);
}
// индекс первого лейна, равного x, или -1
static inline int vfirst_eq(V a, uint8_t x) {
    unsigned m = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(a.v, _mm256_set1_epi8((char)x)));
    return m ? __builtin_ctz(m) : -1;
}
// 2x2 пулинг: W выходов из 2W входов двух строк. dark = min, bright = 255 - max.
static inline void pool2_vec(const uint8_t* r0, const uint8_t* r1, uint8_t* dk, uint8_t* br) {
    __m256i a0 = _mm256_loadu_si256((const __m256i*)r0), a1 = _mm256_loadu_si256((const __m256i*)(r0 + 32));
    __m256i b0 = _mm256_loadu_si256((const __m256i*)r1), b1 = _mm256_loadu_si256((const __m256i*)(r1 + 32));
    const __m256i lo = _mm256_set1_epi16(0x00FF);
    __m256i n0 = _mm256_min_epu8(a0, b0), n1 = _mm256_min_epu8(a1, b1);
    __m256i x0 = _mm256_max_epu8(a0, b0), x1 = _mm256_max_epu8(a1, b1);
    n0 = _mm256_and_si256(_mm256_min_epu8(n0, _mm256_srli_epi16(n0, 8)), lo);
    n1 = _mm256_and_si256(_mm256_min_epu8(n1, _mm256_srli_epi16(n1, 8)), lo);
    x0 = _mm256_and_si256(_mm256_max_epu8(x0, _mm256_srli_epi16(x0, 8)), lo);
    x1 = _mm256_and_si256(_mm256_max_epu8(x1, _mm256_srli_epi16(x1, 8)), lo);
    // packus перемешивает 128-битные половины, permute возвращает порядок
    __m256i n = _mm256_permute4x64_epi64(_mm256_packus_epi16(n0, n1), 0xD8);
    __m256i x = _mm256_permute4x64_epi64(_mm256_packus_epi16(x0, x1), 0xD8);
    _mm256_storeu_si256((__m256i*)dk, n);
    _mm256_storeu_si256((__m256i*)br, _mm256_xor_si256(x, _mm256_set1_epi8(-1)));
}

#elif defined(DL_SIMD_SSE2)
#include <emmintrin.h>
enum { W = 16 };
struct V { __m128i v; };
struct S { __m128i v; };
static inline V ld(const uint8_t* p) { return {_mm_loadu_si128((const __m128i*)p)}; }
static inline void st(uint8_t* p, V a) { _mm_storeu_si128((__m128i*)p, a.v); }
static inline V vdup(uint8_t x) { return {_mm_set1_epi8((char)x)}; }
static inline V vmax(V a, V b) { return {_mm_max_epu8(a.v, b.v)}; }
static inline V vmin(V a, V b) { return {_mm_min_epu8(a.v, b.v)}; }
static inline V vsubs(V a, V b) { return {_mm_subs_epu8(a.v, b.v)}; }
static inline V vadd(V a, V b) { return {_mm_add_epi8(a.v, b.v)}; }
static inline S szero() { return {_mm_setzero_si128()}; }
static inline S sacc(S s, V a) { return {_mm_add_epi64(s.v, _mm_sad_epu8(a.v, _mm_setzero_si128()))}; }
static inline uint32_t shsum(S s) {
    alignas(16) uint64_t t[2];
    _mm_store_si128((__m128i*)t, s.v);
    return (uint32_t)(t[0] + t[1]);
}
static inline uint8_t vhmax(V a) {
    __m128i m = _mm_max_epu8(a.v, _mm_srli_si128(a.v, 8));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 4));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 2));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 1));
    return (uint8_t)_mm_cvtsi128_si32(m);
}
static inline int vfirst_eq(V a, uint8_t x) {
    unsigned m = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(a.v, _mm_set1_epi8((char)x)));
    return m ? __builtin_ctz(m) : -1;
}
static inline void pool2_vec(const uint8_t* r0, const uint8_t* r1, uint8_t* dk, uint8_t* br) {
    __m128i a0 = _mm_loadu_si128((const __m128i*)r0), a1 = _mm_loadu_si128((const __m128i*)(r0 + 16));
    __m128i b0 = _mm_loadu_si128((const __m128i*)r1), b1 = _mm_loadu_si128((const __m128i*)(r1 + 16));
    const __m128i lo = _mm_set1_epi16(0x00FF);
    __m128i n0 = _mm_min_epu8(a0, b0), n1 = _mm_min_epu8(a1, b1);
    __m128i x0 = _mm_max_epu8(a0, b0), x1 = _mm_max_epu8(a1, b1);
    n0 = _mm_and_si128(_mm_min_epu8(n0, _mm_srli_epi16(n0, 8)), lo);
    n1 = _mm_and_si128(_mm_min_epu8(n1, _mm_srli_epi16(n1, 8)), lo);
    x0 = _mm_and_si128(_mm_max_epu8(x0, _mm_srli_epi16(x0, 8)), lo);
    x1 = _mm_and_si128(_mm_max_epu8(x1, _mm_srli_epi16(x1, 8)), lo);
    _mm_storeu_si128((__m128i*)dk, _mm_packus_epi16(n0, n1));
    _mm_storeu_si128((__m128i*)br, _mm_xor_si128(_mm_packus_epi16(x0, x1), _mm_set1_epi8(-1)));
}

#elif defined(DL_SIMD_NEON)
#include <arm_neon.h>
enum { W = 16 };
struct V { uint8x16_t v; };
struct S { uint32x4_t v; };
static inline V ld(const uint8_t* p) { return {vld1q_u8(p)}; }
static inline void st(uint8_t* p, V a) { vst1q_u8(p, a.v); }
static inline V vdup(uint8_t x) { return {vdupq_n_u8(x)}; }
static inline V vmax(V a, V b) { return {vmaxq_u8(a.v, b.v)}; }
static inline V vmin(V a, V b) { return {vminq_u8(a.v, b.v)}; }
static inline V vsubs(V a, V b) { return {vqsubq_u8(a.v, b.v)}; }
static inline V vadd(V a, V b) { return {vaddq_u8(a.v, b.v)}; }
static inline S szero() { return {vdupq_n_u32(0)}; }
static inline S sacc(S s, V a) { return {vpadalq_u16(s.v, vpaddlq_u8(a.v))}; }
static inline uint32_t shsum(S s) {
#if defined(__aarch64__)
    return vaddvq_u32(s.v);
#else
    uint64x2_t t = vpaddlq_u32(s.v);
    return (uint32_t)(vgetq_lane_u64(t, 0) + vgetq_lane_u64(t, 1));
#endif
}
static inline uint8_t vhmax(V a) {
#if defined(__aarch64__)
    return vmaxvq_u8(a.v);
#else
    uint8x8_t m = vpmax_u8(vget_low_u8(a.v), vget_high_u8(a.v));
    m = vpmax_u8(m, m);
    m = vpmax_u8(m, m);
    m = vpmax_u8(m, m);
    return vget_lane_u8(m, 0);
#endif
}
static inline int vfirst_eq(V a, uint8_t x) {
    uint8x16_t m = vceqq_u8(a.v, vdupq_n_u8(x));
    // сужение: по 4 бита на лейн в 64-битной маске
    uint64_t bits = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(m), 4)), 0);
    return bits ? __builtin_ctzll(bits) >> 2 : -1;
}
static inline void pool2_vec(const uint8_t* r0, const uint8_t* r1, uint8_t* dk, uint8_t* br) {
    // vld2 сам раскладывает чётные/нечётные пиксели строки
    uint8x16x2_t a = vld2q_u8(r0), b = vld2q_u8(r1);
    uint8x16_t n = vminq_u8(vminq_u8(a.val[0], a.val[1]), vminq_u8(b.val[0], b.val[1]));
    uint8x16_t x = vmaxq_u8(vmaxq_u8(a.val[0], a.val[1]), vmaxq_u8(b.val[0], b.val[1]));
    vst1q_u8(dk, n);
    vst1q_u8(br, vmvnq_u8(x));
}

#elif defined(DL_SIMD_SCALAR)
enum { W = 1 };
struct V { uint8_t v; };
struct S { uint32_t v; };
static inline V ld(const uint8_t* p) { return {*p}; }
static inline void st(uint8_t* p, V a) { *p = a.v; }
static inline V vdup(uint8_t x) { return {x}; }
static inline V vmax(V a, V b) { return {a.v > b.v ? a.v : b.v}; }
static inline V vmin(V a, V b) { return {a.v < b.v ? a.v : b.v}; }
static inline V vsubs(V a, V b) { return {(uint8_t)(a.v > b.v ? a.v - b.v : 0)}; }
static inline V vadd(V a, V b) { return {(uint8_t)(a.v + b.v)}; }
static inline S szero() { return {0}; }
static inline S sacc(S s, V a) { return {s.v + a.v}; }
static inline uint32_t shsum(S s) { return s.v; }
static inline uint8_t vhmax(V a) { return a.v; }
static inline int vfirst_eq(V a, uint8_t x) { return a.v == x ? 0 : -1; }
static inline void pool2_vec(const uint8_t* r0, const uint8_t* r1, uint8_t* dk, uint8_t* br) {
    uint8_t a = r0[0] < r0[1] ? r0[0] : r0[1], b = r1[0] < r1[1] ? r1[0] : r1[1];
    uint8_t c = r0[0] > r0[1] ? r0[0] : r0[1], d = r1[0] > r1[1] ? r1[0] : r1[1];
    *dk = a < b ? a : b;
    *br = (uint8_t)(255 - (c > d ? c : d));
}

#else
#error "define one of DL_SIMD_AVX2 / DL_SIMD_SSE2 / DL_SIMD_NEON / DL_SIMD_SCALAR"
#endif
