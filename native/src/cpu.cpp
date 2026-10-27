// Определение возможностей процессора и выбор ядер.
#include <thread>

#include "dronolok/detector.hpp"
#include "kernels.hpp"

#if defined(DL_HAVE_X86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace dronolok {

#if defined(DL_HAVE_X86)
static bool cpu_has_avx2() {
    unsigned a, b, c, d;
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    c = (unsigned)r[2];
#else
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid(1, a, b, c, d);
#endif
    const bool osxsave = c & (1u << 27), avx = c & (1u << 28);
    if (!osxsave || !avx) return false;
    // ОС должна сохранять YMM-регистры (XCR0 биты 1 и 2)
#if defined(_MSC_VER)
    unsigned long long xcr0 = _xgetbv(0);
#else
    unsigned xlo, xhi;
    __asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
    unsigned long long xcr0 = ((unsigned long long)xhi << 32) | xlo;
#endif
    if ((xcr0 & 6) != 6) return false;
#if defined(_MSC_VER)
    __cpuidex(r, 7, 0);
    b = (unsigned)r[1];
#else
    __cpuid_count(7, 0, a, b, c, d);
#endif
    return b & (1u << 5);
}
#endif

const char* isa_name(Isa isa) {
    switch (isa) {
        case Isa::Auto: return "auto";
        case Isa::Scalar: return "scalar";
        case Isa::SSE2: return "sse2";
        case Isa::AVX2: return "avx2";
        case Isa::NEON: return "neon";
    }
    return "?";
}

std::vector<Isa> available_isas() {
    std::vector<Isa> v{Isa::Scalar};
#if defined(DL_HAVE_X86)
    if (kernels_sse2()) v.push_back(Isa::SSE2);
    if (kernels_avx2() && cpu_has_avx2()) v.push_back(Isa::AVX2);
#endif
#if defined(DL_HAVE_NEON)
    if (kernels_neon()) v.push_back(Isa::NEON);
#endif
    return v;
}

int hardware_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n ? (int)n : 1;
}

const Kernels* select_kernels(Isa isa) {
    auto av = available_isas();
    if (isa == Isa::Auto) isa = av.back();
    bool ok = false;
    for (Isa a : av) ok |= a == isa;
    if (!ok) isa = av.back();
    switch (isa) {
        case Isa::SSE2: return kernels_sse2();
        case Isa::AVX2: return kernels_avx2();
        case Isa::NEON: return kernels_neon();
        default: return kernels_scalar();
    }
}

}  // namespace dronolok
