// Ядра для набора инструкций SSE2. Сам код в kernels_impl.hpp.
#include "kernels.hpp"

#if defined(DL_HAVE_X86)
#define DL_SIMD_SSE2
// системные заголовки до анонимного namespace: внутри simd.hpp они уже пустые
#include <cstdint>
#include <cstring>
#include <emmintrin.h>
namespace {
#include "simd.hpp"
#include "kernels_impl.hpp"
}  // namespace

const dronolok::Kernels* dronolok::kernels_sse2() { return make_table("sse2"); }
#else
const dronolok::Kernels* dronolok::kernels_sse2() { return nullptr; }
#endif
