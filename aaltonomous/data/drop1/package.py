"""S4 / Drop 1: bundle a capture session (+ stats, phone photos) into drop1_<date>.tar with a README data card.

    python -m aaltonomous.data.drop1.package data/drop1/<session> --phone-dir ~/phone_drop1 --out data/
    # too big? keep fewer frames per step / drop wrist cameras:
    python -m aaltonomous.data.drop1.package data/drop1/<session> --max-frames 20 --no-wrist

Runs stats.py first if <session>/stats/stats.json is missing. Refuses to write more than --max-gb (default 2)
and prints what to cut. Layout inside the tar: drop1_<date>/{README.md, MANIFEST.sha256, calibration.yaml,
scene.yaml, checklist.yaml, session.json, steps/<step>/..., stats/..., phone/...}. PNGs are already compressed,
so the tar is uncompressed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import time
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PHONE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".dng", ".exr", ".hdr", ".tif", ".tiff", ".mp4", ".mov", ".txt", ".md"}


def collect(session: Path, phone_dir=None, max_frames=None, no_wrist=False):
    """[(path_in_tar, src_path)] for everything that goes in."""
    items = []
    for n in ("calibration.yaml", "scene.yaml", "checklist.yaml", "session.json"):
        if (session / n).exists():
            items.append((n, session / n))
    for p in sorted((session / "stats").glob("*")) if (session / "stats").exists() else []:
        items.append((f"stats/{p.name}", p))
    for sd in sorted(p for p in session.iterdir() if (p / "meta.json").exists()):
        for f in ("meta.json", "frames.npz"):
            items.append((f"steps/{sd.name}/{f}", sd / f))
        for cam in sorted(d for d in sd.iterdir() if d.is_dir()):
            if no_wrist and "wrist" in cam.name:
                continue
            fs = sorted(cam.glob("*.png"))
            for f in fs[:max_frames] if max_frames else fs:
                items.append((f"steps/{sd.name}/{cam.name}/{f.name}", f))
    if phone_dir:
        for f in sorted(Path(phone_dir).expanduser().rglob("*")):
            if f.is_file() and f.suffix.lower() in PHONE_EXT:
                items.append((f"phone/{f.relative_to(Path(phone_dir).expanduser())}", f))
    return items


def fmt(x, unit=""):
    return "n/a" if x is None else f"{x}{unit}"


def readme(session_json, stats, items, name, max_frames, no_wrist):
    src = session_json.get("source", {})
    backend = src.get("backend", "?")
    s = stats.get("summary", {}) if stats else {}
    steps = stats.get("steps", {}) if stats else {}
    n_phone = sum(1 for a, _ in items if a.startswith("phone/"))
    L = [f"# {name}: Aaltonomous Humanoid, Drop 1 (venue conditions)", ""]
    if backend != "real":
        L += [f"> **TEST DATA from backend `{backend}`, not the real robot.** Do not distribute.", ""]
    L += ["Captured at the venue on the real setup (robot behind the counter, mat, bins, our printed objects, event lights).",
          "Use it to tune perception and your sim domain randomization to the real cameras: depth noise and holes on",
          "our objects, lighting / white balance, and how the empty scene looks.", "",
          f"- session `{session_json.get('session')}`, start {session_json.get('start')}, git `{session_json.get('git')}`",
          f"- source: {backend}" + (f", DDS domain {src.get('dds_domain')}" if src.get("dds_domain") is not None else ""),
          f"- frames per step: all captured" + (f", capped at {max_frames}" if max_frames else "")
          + ("; wrist cameras left out" if no_wrist else ""),
          f"- phone photos / panorama: {n_phone} file(s) in `phone/`", "",
          "## Steps", "", "| step | kind | lights | frames | depth holes | depth std (mm, median) | head RGB noise |",
          "|---|---|---|---|---|---|---|"]
    for k, v in steps.items():
        d = v["head_depth"]
        L.append(f"| {k} | {v['kind']} | {v.get('lights')} | {v['n_frames']} | {fmt(d['hole_frac_mean'])} | "
                 f"{fmt(d['std_mm_median'])} | {fmt(v['cams']['head_rgb']['noise_std_median'])} |")
    hf = s.get("object_hole_frac_mean", {})
    if hf:
        L += ["", "## Depth holes per object type (head camera, fraction of mask pixels with depth 0)", "",
              "| object | hole fraction |", "|---|---|"] + [f"| {k} | {v} |" for k, v in hf.items()]
    L += ["", f"Calibration check: median offset colour-mask centre vs projected object centre = "
          f"{fmt(s.get('calib_offset_px_median'), ' px')} (masks: {s.get('mask_sources')}).", "",
          "## Files", "",
          "- `steps/<step>/head_rgb|left_wrist_rgb|right_wrist_rgb/NNNNNN.png`: RGB8, 640x480, lossless.",
          "- `steps/<step>/head_depth/NNNNNN.png`: uint16 depth in mm aligned to head colour, 0 = invalid (hole).",
          "- `steps/<step>/frames.npz`: per frame `t` (s since step start), `wall` (unix time), `head_frame_id`,",
          "  joint states `arm_q`, `arm_dq`, `hand_q` (joint_spec order), `body_q` (29), stream ages, image hashes.",
          "- `steps/<step>/meta.json`: step definition, objects on the mat (mat coords, m; `scene.yaml` frames),",
          "  lights, time stamps, quick check.",
          "- `calibration.yaml`: camera intrinsics / extrinsics used (same file as in Drop 0).",
          "- `stats/stats.json` + PNGs: noise / hole / brightness / white-balance statistics (`stats.py`).",
          "- `phone/`: phone photos and panorama of the counter area and the light sources.", "",
          "```python",
          "import cv2, numpy as np",
          "d = cv2.imread('steps/empty_on/head_depth/000000.png', cv2.IMREAD_UNCHANGED)  # uint16 mm",
          "rgb = cv2.imread('steps/empty_on/head_rgb/000000.png')[..., ::-1]",
          "```", "",
          "Lights: `venue_on` = event lighting, `venue_off` = ceiling lights off / our lamps on, `own_off` = our lamps off.",
          "Known limits: one venue session; static scenes only (no motion blur); wrist cameras mostly see the counter edge",
          "unless an arm-pose step was captured.", ""]
    return "\n".join(L)


def package(session, out=".", phone_dir=None, max_gb=2.0, max_frames=None, no_wrist=False, day=None, log=print):
    session = Path(session)
    if not (session / "stats/stats.json").exists():
        from aaltonomous.data.drop1.stats import analyse
        analyse(session, log=log)
    stats = json.loads((session / "stats/stats.json").read_text())
    sj = json.loads((session / "session.json").read_text())
    items = collect(session, phone_dir, max_frames, no_wrist)
    size = sum(p.stat().st_size for _, p in items)
    by = {}
    for a, p in items:
        k = "phone" if a.startswith("phone/") else (a.split("/")[2] if a.startswith("steps/") and a.count("/") >= 3 else "other")
        by[k] = by.get(k, 0) + p.stat().st_size
    log(f"[package] {len(items)} files, {size / 1e9:.2f} GB: " + ", ".join(f"{k} {v / 1e6:.0f} MB" for k, v in sorted(by.items())))
    if size > max_gb * 1e9:
        raise SystemExit(f"[package] {size / 1e9:.2f} GB > --max-gb {max_gb}: use --max-frames 20 and/or --no-wrist "
                         "(or raise --max-gb)")
    name = f"drop1_{day or date.today().isoformat()}"
    path = Path(out) / f"{name}.tar"
    path.parent.mkdir(parents=True, exist_ok=True)
    rd = readme(sj, stats, items, name, max_frames, no_wrist).encode()
    man = []
    t0 = time.time()

    def addbytes(tf, arc, data):
        ti = tarfile.TarInfo(f"{name}/{arc}")
        ti.size, ti.mtime = len(data), int(time.time())
        tf.addfile(ti, io.BytesIO(data))

    tmp = path.with_suffix(".tar.partial")
    with tarfile.open(tmp, "w") as tf:
        addbytes(tf, "README.md", rd)
        for arc, p in items:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
            man.append(f"{h.hexdigest()}  {arc}")
            tf.add(p, f"{name}/{arc}")
        addbytes(tf, "MANIFEST.sha256", ("\n".join(man) + "\n").encode())
    tmp.rename(path)
    log(f"[package] wrote {path} ({path.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f} s)")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--out", default=".")
    ap.add_argument("--phone-dir", default=None, help="folder of phone photos / panorama to include")
    ap.add_argument("--max-gb", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=None, help="keep at most N frames per camera per step")
    ap.add_argument("--no-wrist", action="store_true")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD for the file name (default today)")
    a = ap.parse_args(argv)
    package(a.session, a.out, a.phone_dir, a.max_gb, a.max_frames, a.no_wrist, a.date)


if __name__ == "__main__":
    main()
