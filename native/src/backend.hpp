// Бэкенд считает "отклик точечной цели" и статистику по тайлам — это 95% работы
// детектора. Выбор кандидатов, маска и трекинг от бэкенда не зависят.
//
// Сейчас есть CPU-бэкенд (скаляр / SSE2 / AVX2 / NEON). GPU-бэкенд (Vulkan compute
// на Pi 5, OpenCL на Mali/RK3588, CUDA на Jetson) реализует тот же интерфейс:
// на вход кадр u8, на выход TileStats для двух полярностей.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "dronolok/detector.hpp"

namespace dronolok {

struct FrameView {
    const uint8_t* data;
    int w, h, stride;
};

struct ResponseParams {
    int k, pool, tile;
    uint8_t hot_thr;          // пиксель "горячий", если отклик > hot_thr
    bool dark, bright;
    int job_rows, job_cols;   // размер задания, тайлы
};

// Статистика одной полярности по сетке тайлов tilesY x tilesX
struct TileStats {
    std::vector<uint32_t> sum;     // сумма отклика
    std::vector<uint8_t> maxv;     // максимум отклика
    std::vector<uint16_t> hot;     // число пикселей > hot_thr
    std::vector<uint16_t> argmax;  // первый максимум: y * tile + x
    void resize(size_t n) { sum.assign(n, 0); maxv.assign(n, 0); hot.assign(n, 0); argmax.assign(n, 0); }
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string name() const = 0;
    virtual int threads() const = 0;
    // mask (tilesY x tilesX, может быть nullptr): тайлы с 0 можно не считать,
    // их статистика тогда нулевая. jobs/skipped — для метрик.
    virtual void compute(const FrameView& f, const ResponseParams& p, const uint8_t* mask,
                         TileStats& dark, TileStats& bright, int& jobs, int& skipped) = 0;
};

std::unique_ptr<Backend> make_cpu_backend(Isa isa, int threads, const std::vector<int>& cpus);

}  // namespace dronolok
