#!/usr/bin/env python3
"""Рендер сцены Godot в ролики для конвейера Дронолока.

    python sim/render.py --godot <путь к Godot 4.3> --scene sim/scenes/forest_village.json --out out/fv
    python sim/render.py --encode-only --out out/fv        # только собрать mp4 из готовых PNG

Godot пишет кадры PNG (out/wide, out/fine), разметку <имя>_wide.mp4.gt.json и <имя>_fine.mp4.gt.json,
meta.json (позы камер, траектории). Затем ffmpeg собирает <имя>_wide.mp4 и <имя>_fine.mp4 (H.264).
ffmpeg ищется в PATH, иначе берётся из пакета imageio-ffmpeg (pip install imageio-ffmpeg).
"""
import argparse, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


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
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--scale", type=float, help="множитель разрешения камер (0.5 — быстрый просмотр)")
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--noise", type=int, default=3, help="шум ffmpeg noise alls (0 — без шума)")
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
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        if not os.path.exists(os.path.join(out, "meta.json")):
            sys.exit("Godot не записал meta.json — смотрите ошибки выше")
    encode(out, a.crf, a.noise, a.keep_png)


if __name__ == "__main__":
    main()
