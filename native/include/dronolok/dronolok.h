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
    int scales;      /* 1..3: доп. масштабы x2, x4 для крупных целей */
    float focus_k;   /* порог возле прогноза трека = thr * focus_k */
} dl_config;

typedef struct {
    float x, y, contrast, snr;
    int polarity; /* -1 тёмная, +1 яркая */
    int level;    /* масштаб, на котором найдена */
    float energy; /* суммарный контраст пятна: по нему трекер видит взмахи крыльев */
    float margin; /* отклик / порог детектора; 0 — неизвестно */
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
    int miss;     /* > 0: трек держится на прогнозе столько кадров */
    float sigma;  /* СКО ошибки прогноза позиции, пикс. */
    float score;  /* весомость цели */
    int locked;   /* 1 — захваченная цель */
    float flap_r2, flap_period; /* взмахи: доля объяснённой дисперсии, период в кадрах */
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
/* Сдвиг картинки к этому кадру (камера движется): применить в следующем update. */
/* Сдвиг картинки относительно прошлого кадра. Возвращает 1, если оценка надёжна. */
typedef struct dl_motion dl_motion;
DL_API dl_motion* dl_motion_create(void);
DL_API void dl_motion_destroy(dl_motion* m);
DL_API int dl_motion_estimate(dl_motion* m, const uint8_t* gray, int w, int h, int stride, float* dx, float* dy,
                              float* quality);
DL_API void dl_tracker_set_motion(dl_tracker* t, float dx, float dy);
/* Области прогноза подтверждённых треков (x, y, r тройками) для dl_detector_set_focus. */
DL_API int dl_tracker_focus(dl_tracker* t, float* xyr, int cap);
DL_API void dl_detector_set_focus(dl_detector* d, const float* xyr, int n);
DL_API int dl_tracker_clutter(dl_tracker* t, dl_candidate* out, int cap);
DL_API void dl_tracker_reset(dl_tracker* t);
/* Очередь целей (несколько дронов в кадре): id подтверждённых треков по весомости для захвата,
   самый весомый первым; возвращает длину очереди. Захваченная цель — dl_tracker_locked.
   dl_tracker_select — захват на трек id (ручной выбор держится, пока цель жива; 1 — выбран),
   dl_tracker_next — на следующий в очереди после захваченного (по кругу), возвращает его id. */
DL_API int dl_tracker_queue(dl_tracker* t, int* ids, int cap);
DL_API int dl_tracker_locked(dl_tracker* t);
DL_API int dl_tracker_select(dl_tracker* t, int id);
DL_API int dl_tracker_next(dl_tracker* t);
/* Долгая память статичных помех (неподвижная камера): frames = 0 выключает,
   hits — сколько неподвижных откликов делают место помехой. Файл привязан к размеру кадра. */
DL_API void dl_tracker_set_static(dl_tracker* t, int frames, float hits);
DL_API int dl_tracker_static_save(dl_tracker* t, const char* path);
DL_API int dl_tracker_static_load(dl_tracker* t, const char* path);
DL_API void dl_tracker_static_clear(dl_tracker* t);

#ifdef __cplusplus
}
#endif
#endif
