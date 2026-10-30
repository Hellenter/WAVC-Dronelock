# Виртуальная сцена (Godot 4.3)

Лес с опушкой, деревня, мачты, холмы на горизонте и небо с облаками; в центре — виртуальная установка
Дронолока: широкая камера (по умолчанию 4608×3456, 120°, 14 к/с) и точная (640×480, 20°, 60 к/с) на
поворотке с пределом 30°/с и ускорением 150°/с², как во вкладке «Две камеры». Дроны (квадрокоптер,
самолёт) и птицы (взмахи, планирование) летят по траекториям, а разметка пишется сама:
`<имя>_wide.mp4.gt.json` и `<имя>_fine.mp4.gt.json` в формате приложения (по кадрам `kind`, `x`, `y`,
плюс `id`, `dist` — дальность, м, `px` — размер цели в пикселях). Цель за деревом или домом в разметку не
попадает. Время модельное: ролик одинаковый при каждом запуске с тем же `seed`.

## Запуск у себя (Windows)

1. Скачать Godot 4.3 (обычный, не .NET): https://godotengine.org/download/archive/4.3-stable/ — один .exe.
2. `pip install imageio-ffmpeg opencv-python` (ffmpeg для сборки mp4, если его нет в PATH).
3. Рендер:

       python sim\render.py --godot C:\Godot\Godot_v4.3-stable_win64.exe --scene sim\scenes\forest_village.json --out out\fv

   Быстрый просмотр: `--scale 0.5 --seconds 3`. В `out\fv` будут `forest_village_wide.mp4`,
   `forest_village_fine.mp4`, их `.gt.json` и `meta.json` (позы камер по кадрам, траектории целей).
   Ролики открываются в Dronolok.exe и в мультитесте как обычные записи с разметкой.
4. Покрутить камеру в окне (без записи роликов):

       python sim\render.py --godot C:\Godot\Godot_v4.3-stable_win64.exe --scene sim\scenes\city_sectors.json --view

   Стрелки или WASD — поворот (Shift быстрее), колесо или +/- — зум, 1/2/3 — сектора (лес, частный сектор,
   многоэтажки), Tab — следующий сектор, M — обзор по кругу, Пробел — пауза, R — время сначала, F12 — снимок
   в `out\view`, Esc — выход. Справа внизу — точная камера на поворотке. Цели в этом режиме летают по всему кругу.
   Ролик с камерой на секторе: `--sector highrise` (или `--az 300`), обзор по кругу: `--pan 12`.
   Полоса 360° с подписями секторов: `--panorama --out out\pano` → `out\pano\pano.jpg`.
5. Посмотреть и поправить сцену глазами: открыть `sim/godot/project.godot` в Godot. Сцена строится
   скриптом `sim.gd` при запуске, поэтому расстановку меняют в `.json` сцены.

## Свои ассеты

Готовые модели `.glb` (бесплатные CC0: Kenney Nature Kit / City Kit, Quaternius) кладутся в
`sim/godot/assets/trees/` и `sim/godot/assets/houses/` — тогда вместо простых деревьев и домов берутся они
(высота приводится к 8–22 м и 6–10 м). Другая папка: `"world": {"assets": "C:/путь"}`.
Небо из панорамы HDRI (`.hdr`/`.exr`, Poly Haven): файл в `sim/godot/assets/sky/` и `"world": {"sky_hdri":
"имя.hdr", "sky_rotation": 0, "sky_energy": 1.0}`; облака тогда не двигаются (панорама неподвижна).

## Сцена (.json)

Координаты как в `emulator/geom.py`: x — восток, y — север, z — вверх, метры; азимут от севера по часовой.
Установка в (0, 0), камера на высоте `rig.height`.

- `wide`: `w`, `h`, `hfov`, `fps`, `az`, `el`, `roll`, `msaa` (0/2/4/8 — сглаживание, от него зависит,
  как выглядит цель меньше пикселя), `pan_dps` и `sweep_deg` — поворот или качание ±sweep.
- `fine`: `enabled`, `w`, `h`, `hfov`, `fps`, `max_dps`, `accel`, `latency` (задержка передачи цели),
  `gain`, `target` — `"auto"` (первый видимый дрон широкой камеры) или `id` цели.
- `world`: `trees`, `trees_from`/`trees_to`, `forest_threshold` (больше — реже лес), `tree_lines` (опушки:
  `dist`, `depth`, `az_from`, `az_to`, `count`), `villages` (`az`, `dist`, `radius`, `houses`), `poles`
  (`pos`, `height`, `radius`), `hills`, `cloud_cover`, `cloud_soft`, `cloud_wind`, `sun_az`, `sun_el`, `sun`,
  `ambient`, `fog`, `wind` (качание крон), `shadows`.
  `sectors` — панорама по секторам, каждый `{"name", "type", "az_from", "az_to"}`: `forest` (лес как выше),
  `lowrise` (частный сектор: `from`, `to` — м от установки, `houses`), `highrise` (многоэтажки: `from`, `to`,
  `buildings`, `floors` [мин, макс], `towers` — доля башен, остальное панельные «пластины», `masts` — доля
  домов с мачтой на крыше). `trees` в секторе — доля леса, что остаётся (по умолчанию 0.12 и 0.05 в застройке).
- `targets`: `kind` (`drone`/`bird`), `model` (`quad`, `delta` — летающее крыло-треугольник, `wing`/`plane` — самолёт, птица — по умолчанию), `size` (м: размах
  дрона или крыльев птицы), `speed`, `start`/`end` (с), `path`:
  `line` (`from`, `to`), `waypoints` (`points`, `mode`: `once`/`loop`/`pingpong`), `orbit` (`center`,
  `radius`, `dir`, `phase`), `hover` (`pos`, `jitter`), `wander` (`center`, `radius`). Птицы: `flap_hz`,
  `flap_deg`, `flap_period` и `flap_on` (взмахи пачками, между ними планирование).

- `random_targets`: `drones`, `birds`, `range` [м, м], `types` (`quad`/`delta`/`plane` и доли), `seed` — случайные
  цели разных типов: квадрокоптер 0.3–0.8 м, крыло 1.8–3.5 м и 25–45 м/с, самолёт 1.5–4 м и 15–30 м/с, птицы 0.3–1.2 м.

Готовые сцены: `city_sectors` (лес 0–120°, частный сектор 120–240°, многоэтажки 240–360°; камера по умолчанию
на лес, `--sector lowrise`/`highrise` — на другие), `drone_types` (8 дронов трёх типов и 4 птицы на 150–1500 м), `forest_village` (4 дрона, 2 птицы), `forest_polyhaven` (то же с деревьями Poly Haven из
`assets/polyhaven`), `pan_sweep` (камера качается ±25° на 10°/с),
`overcast_birds` (пасмурно, только птицы — проверка ложных тревог), `quick` (3 с, полразрешения).

## CI

`.github/workflows/sim.yml` рендерит `quick` и `city_sectors` по трём секторам (плюс панораму 360°) при каждом изменении `sim/` (программный
рендер, видеокарта не нужна), прогоняет по роликам мультитест и выкладывает ролики с разметкой в артефакт
`sim-videos`. Другие сцены и длина — через «Run workflow» (`сцена:сектор` — камера на сектор).

Модели Poly Haven приходят как `.blend`; в `.glb` для сцены их перегоняет `sim/blend2glb.py` (Blender 4,
прореживание до `--tris`, текстуры до `--tex`, простые материалы с прозрачностью листвы):

    blender -b pine_sapling_small_1k.blend --python sim/blend2glb.py -- sim/godot/assets/polyhaven/trees --tris 8000 --tex 256
