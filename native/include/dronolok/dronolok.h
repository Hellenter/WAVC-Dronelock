/* Дронолок: C-API нативного детектора (для Python/ctypes, Rust и других языков). */
#ifndef DRONOLOK_H
#define DRONOLOK_H
#include <stdint.h>

#if defined(_WIN32) && defined(DRONOLOK_SHARED)
#define DL_API __declspec(dllexport)
#elif defined(__GNUC__)
#define DL_API __attribute__((visibility("default")))
#else
#define DL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* isa: 0 auto, 1 scalar, 2 sse2, 3 avx2, 4 neon */
typedef struct {
    int k, pool;
    float thr, clutter_k;
    int tile;
    float max_density;
    int dark, bright, max_cands;
    int threads, isa, job_rows, job_cols;
    int ncpus;
    const int* cpus; /* привязка потоков к ядрам, ncpus штук */
} dl_config;

typedef struct {
    float x, y, contrast, snr;
    int polarity; /* -1 тёмная, +1 яркая */
} dl_candidate;

/* cls: 0 unknown, 1 drone, 2 bird, 3 clutter */
typedef struct {
    int id;
    float x, y, vx, vy;
    int polarity;
    float contrast;
    int clutter, cls;
    float bird_score, speed, resid, contrast_cv;
    int hits, age;
} dl_track;

typedef struct {
    double total_ms, response_ms, select_ms;
    int jobs, jobs_skipped;
} dl_timing;

typedef struct dl_detector dl_detector;
typedef struct dl_tracker dl_tracker;

DL_API void dl_config_default(dl_config* c);
DL_API dl_detector* dl_detector_create(const dl_config* c);
DL_API void dl_detector_destroy(dl_detector* d);
/* Возвращает число кандидатов (в out записывается не больше cap). */
DL_API int dl_detect(dl_detector* d, const uint8_t* gray, int w, int h, int stride, dl_candidate* out, int cap);
DL_API void dl_detector_timing(dl_detector* d, dl_timing* t);
DL_API const char* dl_detector_backend(dl_detector* d);
DL_API int dl_detector_threads(dl_detector* d);
DL_API void dl_mask_begin(dl_detector* d);
DL_API void dl_mask_end(dl_detector* d, float frac);
/* Маска тайлов (1 = искать). Возвращает tw*th или 0, если маски нет. */
DL_API int dl_mask_get(dl_detector* d, uint8_t* out, int cap, int* tw, int* th);

DL_API int dl_hardware_threads(void);
DL_API int dl_available_isas(int* out, int cap);

DL_API dl_tracker* dl_tracker_create(int w, int h);
DL_API void dl_tracker_destroy(dl_tracker* t);
/* Возвращает число подтверждённых треков. */
DL_API int dl_tracker_update(dl_tracker* t, const dl_candidate* dets, int n, dl_track* out, int cap);
DL_API int dl_tracker_clutter(dl_tracker* t, dl_candidate* out, int cap);
DL_API void dl_tracker_reset(dl_tracker* t);

#ifdef __cplusplus
}
#endif
#endif
