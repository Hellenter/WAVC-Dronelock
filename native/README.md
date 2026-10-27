# native/ — нативное ядро детектора (C++17, SIMD)

Тот же алгоритм, что в [`detector/`](../detector/), переписанный на C++ без OpenCV. Результат совпадает с
Python-версией бит в бит: те же кандидаты и те же треки (проверка: `python/compare_with_python.py`).

## Что внутри

| Часть | Файлы | Что делает |
|---|---|---|
| Детектор | `src/detector.cpp`, `src/backend_cpu.cpp` | пулинг 2x2, морфологическое закрытие линиями в 4 направлениях, статистика тайлов 32x32, выбор кандидатов, авто-маска неба |
| Ядра SIMD | `src/kernels_*.cpp`, `src/simd.hpp`, `src/kernels_impl.hpp` | один шаблонный код под AVX2 (256 бит), SSE2 и NEON (128 бит) и скалярный эталон; набор выбирается при запуске по CPU |
| Потоки | `src/thread_pool.cpp` | кадр режется на блоки тайлов, блоки раздаются потокам; можно закрепить потоки за ядрами |
| Трекер | `src/tracker.cpp` | порт трекера из Python + классификатор по движению: дрон / птица / статичная помеха; одиночные отклики — шум |
| API | `include/dronolok/detector.hpp`, `include/dronolok/dronolok.h` | C++ и C (для Python/ctypes и других языков) |
| Инструменты | `tools/dronolok_bench.cpp`, `tests/test_kernels.cpp` | замеры, качество по разметке, сверка SIMD со скалярным кодом |

Почему так быстро: кадр обрабатывается полосами по нескольку строк, и все промежуточные данные (пулинг,
4 направления дилатации) живут в кольцевых буферах, которые помещаются в кэш L1/L2. В OpenCV каждый шаг
проходит по всему кадру 16 Мп и упирается в память. Плюс полностью замаскированные тайлы (лес, земля)
не считаются вовсе.

## Скорость, кадр 4608x3456 (16 Мп), мс на кадр

ПК: Xeon 2.8 ГГц, 4 vCPU (виртуалка). Python — `detector/` с OpenCV.

| Вариант | 1 поток | 4 потока |
|---|---:|---:|
| Python + OpenCV | 55–70 | 18–22 |
| нативное, скалярный код | 300 | 80 |
| нативное, SSE2 (128 бит, как NEON) | 15–20 | 4.5–6 |
| нативное, AVX2 (256 бит) | 10 | 3.5–4 |

Для Raspberry Pi ориентир — строка SSE2: NEON той же ширины. Ядра Cortex-A76 (Pi 5) примерно в 2–3 раза
медленнее ядра этого ПК, Cortex-A53 (Pi 3) ещё в 3–4 раза медленнее. Точные цифры даст `dronolok_bench`
на самом устройстве.

## Сборка

Нужны CMake 3.16+ и компилятор C++17 (GCC, Clang или MSVC).

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build -j
native/build/test_kernels           # все SIMD-варианты совпадают со скалярным кодом
```

### Raspberry Pi

На самой Pi (Raspberry Pi OS, 64 или 32 бит) — те же команды. NEON включается автоматически.

Кросс-компиляция с ПК на Linux:

```sh
sudo apt install g++-aarch64-linux-gnu      # Pi 3/4/5 с 64-битной ОС
cmake -S native -B build-pi -DCMAKE_TOOLCHAIN_FILE=$PWD/native/cmake/aarch64-linux-gnu.cmake
cmake --build build-pi -j
# для 32-битной ОС: g++-arm-linux-gnueabihf и native/cmake/arm-linux-gnueabihf.cmake
```

Готовые сборки для Pi и Windows собирает GitHub Actions (`.github/workflows/build.yml`), они лежат в
артефактах запуска.

### Windows

Из Linux: `-DCMAKE_TOOLCHAIN_FILE=$PWD/native/cmake/mingw-w64-x86_64.cmake` (пакет `mingw-w64`).
На Windows: Visual Studio 2022, `cmake -S native -B build` и `cmake --build build --config Release`.

## Замеры: `dronolok_bench`

```sh
# видео -> серые кадры y4m -> замер (кадры загружаются в память заранее, чтение не мешает)
ffmpeg -i video.mp4 -frames:v 100 -pix_fmt gray -f yuv4mpegpipe sky.y4m
dronolok_bench sky.y4m --sweep                         # таблица: набор инструкций x потоки
dronolok_bench sky.y4m --threads 2 --isa sse2 --cpus 2,3
dronolok_bench sky.y4m --gt video.mp4.gt.json --learn-mask 25   # полнота, ложные треки, классы
# с камеры или из файла потоком, вместе с декодированием:
ffmpeg -i rtsp://... -pix_fmt gray -f yuv4mpegpipe - | dronolok_bench - --stream
```

## Из Python

```python
from dronolok_native import NativePointDetector, NativeTracker   # native/python/
det = NativePointDetector(threads=4)      # тот же интерфейс, что у detector.PointDetector
cands = det.detect(gray_frame)
```

`detector/run_demo.py --native` прогоняет демо на нативном ядре.

## GPU и NPU

Вычисления спрятаны за интерфейсом `Backend` (`src/backend.hpp`): детектор отдаёт кадр и получает
статистику тайлов, всё остальное (выбор кандидатов, трекер) общее. Сейчас есть CPU-бэкенд. Следующие:

- **Vulkan compute** для GPU Raspberry Pi 5 (VideoCore VII, драйвер v3dv) и ПК. Морфология и статистика
  тайлов хорошо ложатся на шейдеры; главный вопрос на Pi — копирование кадра в GPU, нужен импорт
  DMA-буфера камеры без копии.
- **NPU** (Hailo-8L в AI Kit для Pi 5 и подобные) рассчитан на нейросети, а не на морфологию. Его место —
  второй шаг: классификатор дрон/птица по вырезкам 32x32 вокруг треков, которые уже нашёл этот детектор.
