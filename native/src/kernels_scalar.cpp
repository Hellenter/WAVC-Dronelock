// Ядра для набора инструкций SCALAR. Сам код в kernels_impl.hpp.
#include "kernels.hpp"

#if 1
#define DL_SIMD_SCALAR
// системные заголовки до анонимного namespace: внутри simd.hpp они уже пустые
#include <cstdint>
#include <cstring>
namespace {
#include "simd.hpp"
#include "kernels_impl.hpp"
}  // namespace

const dronolok::Kernels* dronolok::kernels_scalar() { return make_table("scalar"); }
#else
const dronolok::Kernels* dronolok::kernels_scalar() { return nullptr; }
#endif
