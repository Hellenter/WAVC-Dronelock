#!/usr/bin/env python3
"""Рендер сцены Godot в ролики для конвейера Дронолока.

    python sim/render.py --godot <путь к Godot 4.3> --scene sim/scenes/forest_village.json --out out/fv
    python sim/render.py --encode-only --out out/fv        # только собрать mp4 из готовых PNG
    python sim/render.py --scene sim/scenes/city_sectors.json --sector highrise --out out/hr   # камера на сектор
    python sim/render.py --scene sim/scenes/city_sectors.json --view      # окно: крутить камеру, 1/2/3 — сектора
    python sim/render.py --scene sim/scenes/city_sectors.json --panorama --out out/pano   # полоса 360°
    python sim/render.py --scene sim/scenes/city_sectors.json --weather rain --time 19:30 --out out/r   # погода, время
    python sim/render.py --scene sim/scenes/city_sectors.json --gallery "clear@13;cumulus@19;fog@8" --out out/g

Godot пишет кадры PNG (out/wide, out/fine), разметку <имя>_wide.mp4.gt.json и <имя>_fine.mp4.gt.json,
meta.json (позы камер, траектории). Затем ffmpeg собирает <имя>_wide.mp4 и <имя>_fine.mp4 (H.264).
ffmpeg ищется в PATH, иначе берётся из пакета imageio-ffmpeg (pip install imageio-ffmpeg).
"""
import argparse, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEATHERS = ["clear", "cumulus", "cloudy", "overcast", "haze", "mist", "fog", "rain", "storm", "snow"]


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("нет ffmpeg: поставьте его в PATH или `pip install imageio-ffmpeg`")


def encode(out, crf, noise, keep):
    meta = json.load(open(os.path.join(out, "meta.json"), encoding="utf-8"))
    ff = find_ffmpeg()
    for cam in ("wide", "fine"):
        if cam == "fine" and not meta["fine"]["enabled"]:
            continue
        src = os.path.join(out, cam, "f_%06d.png")
        dst = os.path.join(out, f"{meta['name']}_{cam}.mp4")
        vf = ["format=yuv420p"]
        if noise > 0:                      # шум матрицы, как у настоящей камеры
            vf.insert(0, f"noise=alls={noise}:allf=t")
        cmd = [ff, "-y", "-loglevel", "error", "-framerate", str(meta[cam]["fps"]), "-i", src,
               "-vf", ",".join(vf), "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), dst]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        if not keep:
            shutil.rmtree(os.path.join(out, cam))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--godot", default=os.environ.get("GODOT", "godot"))
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes", "forest_village.json"))
    ap.add_argument("--out", default="out/view")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--scale", type=float, help="множитель разрешения камер (0.5 — быстрый просмотр)")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--noise", type=int, default=2, help="шум ffmpeg noise alls (0 — без шума)")
    ap.add_argument("--sector", help="навести широкую камеру на середину сектора world.sectors (имя или тип)")
    ap.add_argument("--az", type=float, help="азимут широкой камеры, °")
    ap.add_argument("--el", type=float, help="угол места широкой камеры, °")
    ap.add_argument("--pan", type=float, help="поворот широкой камеры, °/с (обзор по кругу)")
    ap.add_argument("--sweep", type=float, help="качание ±° вместо обзора по кругу (вместе с --pan)")
    ap.add_argument("--view", action="store_true", help="окно просмотра вместо рендера роликов")
    ap.add_argument("--panorama", action="store_true", help="12 снимков по кругу и склейка pano.jpg")
    ap.add_argument("--weather", help="погода поверх сцены: " + ", ".join(WEATHERS))
    ap.add_argument("--time", help="время суток, ч (местное солнечное): 19.5 или 19:30")
    ap.add_argument("--date", help="дата ММ-ДД (высота солнца), по умолчанию 09-15")
    ap.add_argument("--lat", type=float, help="широта, ° (по умолчанию 55.75 — Москва)")
    ap.add_argument("--time-rate", type=float, help="во сколько раз быстрее идёт время суток (60 — час за минуту)")
    ap.add_argument("--clouds", type=float, help="облачность 0..1")
    ap.add_argument("--cirrus", type=float, help="перистые облака 0..1")
    ap.add_argument("--visibility", type=float, help="видимость, м (дымка, туман)")
    ap.add_argument("--rain", type=float, help="дождь 0..1")
    ap.add_argument("--snow", type=float, help="снег 0..1")
    ap.add_argument("--exposure", help="экспозиция камеры: auto (по умолчанию) или множитель")
    ap.add_argument("--gallery", help='кадры по погоде: "погода@часы[@азимут];..." -> gallery.jpg')
    ap.add_argument("--keep-png", action="store_true")
    ap.add_argument("--encode-only", action="store_true")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if not a.encode_only:
        os.makedirs(out, exist_ok=True)
        cmd = [a.godot, "--path", os.path.join(HERE, "godot"), "--rendering-driver", "opengl3",
               "--audio-driver", "Dummy", "--", "--config", os.path.abspath(a.scene), "--out", out]
        if a.seconds:
            cmd += ["--seconds", str(a.seconds)]
        if a.scale:
            cmd += ["--scale", str(a.scale)]
        for k in ("sector", "az", "el", "pan", "sweep", "weather", "time", "date", "lat", "time_rate", "clouds",
                  "cirrus", "visibility", "rain", "snow", "exposure"):
            if getattr(a, k) is not None:
                cmd += ["--" + k.replace("_", "-"), str(getattr(a, k))]
        if a.gallery:
            cmd += ["--gallery", a.gallery]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            subprocess.run([sys.executable, os.path.join(HERE, "gallery.py"), out], check=True)
            return
        if a.view:
            cmd += ["--view", "1"]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            return
        if a.panorama:
            cmd += ["--panorama", "1"]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            subprocess.run([sys.executable, os.path.join(HERE, "panorama.py"), out], check=True)
            return
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        if not os.path.exists(os.path.join(out, "meta.json")):
            sys.exit("Godot не записал meta.json — смотрите ошибки выше")
    encode(out, a.crf, a.noise, a.keep_png)


if __name__ == "__main__":
    main()
