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
/* Частота кадров камеры. Умолчания трекера подобраны для 25 к/с: на 10-15 к/с цель за кадр
   смещается дальше, а выдержки в кадрах длятся дольше. Вызвать один раз после create
   (как Pipeline в приложении, tracker_for_fps); без вызова — 25 к/с. */
DL_API void dl_tracker_set_fps(dl_tracker* t, float fps);
/* Возвращает число подтверждённых треков. */
DL_API int dl_tracker_update(dl_tracker* t, const dl_candidate* dets, int n, dl_track* out, int cap);
/* Сдвиг картинки к этому кадру (камера движется): применить в следующем update. */
/* Сдвиг картинки относительно прошлого кадра. Возвращает 1, если оценка надёжна. */
typedef struct dl_motion dl_motion;
DL_API dl_motion* dl_motion_create(void);
DL_API void dl_motion_destroy(dl_motion* m);
DL_API int dl_motion_estimate(dl_motion* m, const uint8_t* gray, int w, int h, int stride, float* dx, float* dy,
                              float* quality);
/* Частота кадров камеры (0 — не задана): проверка "сдвинулась только часть кадра — камера стоит"
   включается по скорости поворота, как при 60 к/с. Без вызова на 10-15 к/с плавный поворот, при
   котором часть кадра неподвижна, мог оцениваться нулём. */
DL_API void dl_motion_set_fps(dl_motion* m, float fps);
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

/* ---- Геометрия камер и передача цели (dronolok/geometry.hpp) ----
   Мир: азимут от севера по часовой, угол места от горизонта, градусы. Положение камеры — куда смотрит
   оптическая ось и поворот кадра вокруг неё. lens: 0 обычный объектив, 1 рыбий глаз (r = f·θ). */
typedef struct {
    int w, h;
    float hfov_deg;
    int lens;
    float cx, cy; /* центр кадра; < 0 — середина */
    float k1;     /* радиальная дисторсия, 0 — без неё */
} dl_camera;

typedef struct {
    float az, el, roll;
} dl_pose;

/* Сообщение о цели для другой камеры или поворотки. t — время кадра по общим часам, с. */
typedef struct {
    int id;
    double t;
    float az, el;     /* градусы */
    float vaz, vel;   /* град/с */
    float sigma;      /* СКО направления, градусы */
    float sigma_v;    /* СКО скорости, град/с */
    int cls;          /* как у dl_track */
    float confidence; /* 0..1 */
    int hard;         /* 1 — жёсткий захват */
    float size_deg;   /* видимый размер, 0 — не измерен */
} dl_target;

typedef struct {
    float u, v, radius;
    int visible;
} dl_window;

DL_API int dl_project(const dl_camera* c, const dl_pose* p, float az, float el, float* u, float* v);
DL_API void dl_unproject(const dl_camera* c, const dl_pose* p, float u, float v, float* az, float* el);
DL_API float dl_pixel_deg(const dl_camera* c, const dl_pose* p, float u, float v);
/* Цель из трека: направление, угловая скорость по vx, vy (пикс./кадр) и fps, неопределённость по sigma.
   hard — состояние захвата трекера (dl_tracker_hard). */
DL_API void dl_target_from_track(const dl_camera* c, const dl_pose* p, double t, float fps, const dl_track* tr,
                                 int hard, dl_target* out);
/* Прогноз направления на момент t (az, el, sigma в градусах). */
DL_API void dl_target_predict(const dl_target* tg, double t, float* az, float* el, float* sigma);
/* Окно поиска цели в кадре другой камеры на момент t: k_sigma — сколько СКО брать. */
DL_API dl_window dl_target_window(const dl_target* tg, double t, const dl_camera* c, const dl_pose* p, float k_sigma);
/* Куда повернуть ось, чтобы цель была в центре через lead секунд после t. */
DL_API dl_pose dl_target_aim(const dl_target* tg, double t, double lead);
/* Сдвиг картинки при повороте камеры from -> to в точке (u, v) — для dl_tracker_set_motion по энкодерам. */
DL_API void dl_pose_shift(const dl_camera* c, const dl_pose* from, const dl_pose* to, float u, float v, float* dx,
                          float* dy);
/* Состояние захвата: 0 нет, 1 захват, 2 жёсткий (подтверждён дрон). */
DL_API int dl_tracker_hard(dl_tracker* t);

/* ---- Наведение поворотки точной камеры (dronolok/gimbal.hpp) ----
   На Pi: фильтр цели (положение, скорость, ускорение по осям поворотки) по отметкам точной — со временем
   КАДРА (середина выдержки), а не обработки: задержка камеры учитывается сама — и по сообщениям широкой,
   пока точная цель не видит. Уставка двигателю — где будет цель, когда команда дойдёт (lead_cmd), с её
   скоростью и ускорением. Контроллер двигателя (или поток на Pi, ~1 кГц): dl_gimbal_drive — плавный профиль
   с пределами скорости, ускорения и рывка. Без датчиков угла: где была поворотка в момент кадра — по тому,
   что сообщал контроллер (dl_gimbal_report_pose -> dl_gimbal_pose_at), уход нуля — по широкой. */
typedef struct {
    float max_rate, max_accel, max_jerk; /* град/с, град/с², град/с³ */
    float kp, kv;                        /* 1/с: скорость к ошибке, ускорение к ошибке скорости (kv = 4 kp) */
    float brake_share;                   /* доля ускорения на торможение к цели */
    float lead_cmd;                      /* задержка команды до двигателя, с */
    float jerk_noise, acc_tau;           /* манёвр цели: СКО рывка (град/с³) и сколько длится (с) */
    float sigma_pose;                    /* ошибка положения поворотки на момент кадра, град */
    float gate;                          /* отметка дальше стольких СКО — выброс */
    int gate_reset;                      /* столько выбросов подряд — принять (цель сманеврировала) */
    float bias_tau, bias_max;            /* уход нуля по широкой: за сколько секунд, не больше скольких градусов */
} dl_gimbal_config;

typedef struct {
    double t;        /* на этот момент (now + lead_cmd) */
    float az, el;    /* град, в координатах привода */
    float vaz, vel;  /* град/с */
    float aaz, ael;  /* град/с² */
    int hold;        /* 1 — цели нет: остановиться плавно */
} dl_setpoint;

typedef struct dl_gimbal dl_gimbal;
DL_API void dl_gimbal_defaults(dl_gimbal_config* c);
DL_API dl_gimbal* dl_gimbal_create(const dl_gimbal_config* c); /* NULL — по умолчанию */
DL_API void dl_gimbal_destroy(dl_gimbal* g);
/* Цель по сообщению (обычно широкой, при назначении); NULL — цели нет. */
DL_API void dl_gimbal_target(dl_gimbal* g, const dl_target* tg);
/* Отметка цели точной: направление на момент кадра t_frame, СКО (град). 1 — принята, 0 — выброс. */
DL_API int dl_gimbal_measure(dl_gimbal* g, double t_frame, float az, float el, float sigma);
/* Оценка цели на момент t как сообщение (окно поиска, сверка с широкой). 0 — цели нет. */
DL_API int dl_gimbal_estimate(const dl_gimbal* g, double t, dl_target* out);
/* Уставка двигателю: цель через lead_cmd после now, минус уход нуля. */
DL_API dl_setpoint dl_gimbal_setpoint(const dl_gimbal* g, double now);
/* Положение поворотки со временем, как его сообщил контроллер (по профилю, шагам или датчику) — в историю;
   dl_gimbal_pose_at — где была поворотка в момент t (с поправкой ухода нуля). */
DL_API void dl_gimbal_report_pose(dl_gimbal* g, double t, float az, float el);
DL_API dl_pose dl_gimbal_pose_at(const dl_gimbal* g, double t);
/* Без датчиков угла: широкая видит ту же цель — её сообщение и направление на цель по точной (с поправкой)
   на время кадра t_frame; dt — сколько прошло с прошлой сверки. dl_gimbal_bias — нынешняя поправка. */
DL_API void dl_gimbal_wide_check(dl_gimbal* g, const dl_target* wide, double t_frame, float az, float el, double dt);
DL_API void dl_gimbal_bias(const dl_gimbal* g, float* az, float* el);
/* Профиль (контроллер двигателя): поставить в (az, el) и шаг h к уставке sp, которая дошла в момент t_recv
   (между уставками продолжается по её скорости). pos, vel — по 2 значения: азимут, угол места. */
DL_API void dl_gimbal_drive_reset(dl_gimbal* g, float az, float el);
DL_API void dl_gimbal_drive(dl_gimbal* g, const dl_setpoint* sp, double t_recv, double t, double h, float* pos, float* vel);

#ifdef __cplusplus
}
#endif
#endif
