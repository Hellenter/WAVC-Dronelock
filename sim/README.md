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
4. Посмотреть и поправить сцену глазами: открыть `sim/godot/project.godot` в Godot. Сцена строится
   скриптом `sim.gd` при запуске, поэтому расстановку меняют в `.json` сцены.

## Свои ассеты

Готовые модели `.glb` (бесплатные CC0: Kenney Nature Kit / City Kit, Quaternius) кладутся в
`sim/godot/assets/trees/` и `sim/godot/assets/houses/` — тогда вместо простых деревьев и домов берутся они
(высота приводится к 8–22 м и 6–10 м). Другая папка: `"world": {"assets": "C:/путь"}`.

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
- `targets`: `kind` (`drone`/`bird`), `model` (`quad`, `wing`, птица — по умолчанию), `size` (м: размах
  дрона или крыльев птицы), `speed`, `start`/`end` (с), `path`:
  `line` (`from`, `to`), `waypoints` (`points`, `mode`: `once`/`loop`/`pingpong`), `orbit` (`center`,
  `radius`, `dir`, `phase`), `hover` (`pos`, `jitter`), `wander` (`center`, `radius`). Птицы: `flap_hz`,
  `flap_deg`, `flap_period` и `flap_on` (взмахи пачками, между ними планирование).

Готовые сцены: `forest_village` (4 дрона, 2 птицы), `pan_sweep` (камера качается ±25° на 10°/с),
`overcast_birds` (пасмурно, только птицы — проверка ложных тревог), `quick` (3 с, полразрешения).

## CI

`.github/workflows/sim.yml` рендерит `quick` и `forest_village` при каждом изменении `sim/` (программный
рендер, видеокарта не нужна), прогоняет по роликам мультитест и выкладывает ролики с разметкой в артефакт
`sim-videos`. Другие сцены и длина — через «Run workflow».
