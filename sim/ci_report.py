#!/usr/bin/env python3
"""Сводка для журнала CI: по разметке каждой сцены (сколько кадров с целями, размеры в пикселях) и
--multitest results.json — числа прогонов мультитеста."""
import glob, json, os, sys


def gt_summary(out):
    for meta_path in sorted(glob.glob(os.path.join(out, "*", "meta.json"))):
        d = os.path.dirname(meta_path)
        meta = json.load(open(meta_path, encoding="utf-8"))
        for cam in ("wide", "fine"):
            p = os.path.join(d, f"{meta['name']}_{cam}.mp4.gt.json")
            if not os.path.exists(p):
                continue
            g = json.load(open(p))
            fr = g["frames"]
            per = {}
            for rec in fr:
                for o in rec:
                    s = per.setdefault((o["kind"], o["id"]), [0, 1e9, 0, 1e9, 0])
                    s[0] += 1
                    s[1], s[2] = min(s[1], o["px"]), max(s[2], o["px"])
                    s[3], s[4] = min(s[3], o["dist"]), max(s[4], o["dist"])
            print(f"{meta['name']} {cam} {g['w']}x{g['h']} {g['fps']} к/с, кадров {len(fr)}, "
                  f"с целями {sum(1 for r in fr if r)}")
            for (k, i), s in sorted(per.items(), key=lambda x: x[0][1]):
                print(f"   {k:5s} id={i}: в кадре {100 * s[0] / max(len(fr), 1):5.1f}%  "
                      f"{s[1]:.1f}–{s[2]:.1f} пикс.  {s[3]:.0f}–{s[4]:.0f} м")
        if meta["fine"]["enabled"]:
            poses = meta["fine"]["poses"]
            led = sum(1 for p in poses if p[5] is not None)
            print(f"   точная: цель назначена в {100 * led / max(len(poses), 1):.0f}% кадров")


def mt_summary(path):
    d = json.load(open(path))
    for r in d.get("runs", []):
        flat = {k: v for k, v in r.items() if isinstance(v, (int, float, str)) or v is None}
        print(json.dumps(flat, ensure_ascii=False))
        for v in r.get("videos", []) if isinstance(r.get("videos"), list) else []:
            print("   ", json.dumps({k: x for k, x in v.items() if isinstance(x, (int, float, str)) or x is None},
                                    ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1] == "--multitest":
        mt_summary(sys.argv[2])
    else:
        gt_summary(sys.argv[1])
