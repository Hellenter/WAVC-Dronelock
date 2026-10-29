// Ядра для набора инструкций NEON. Сам код в kernels_impl.hpp.
#include "kernels.hpp"

#if defined(DL_HAVE_NEON)
#define DL_SIMD_NEON
// системные заголовки до анонимного namespace: внутри simd.hpp они уже пустые
#include <cstdint>
#include <cstring>
#include <arm_neon.h>
namespace {
#include "simd.hpp"
#include "kernels_impl.hpp"
}  // namespace

const dronolok::Kernels* dronolok::kernels_neon() { return make_table("neon"); }
#else
const dronolok::Kernels* dronolok::kernels_neon() { return nullptr; }
#endif
