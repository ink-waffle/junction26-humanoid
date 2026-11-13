"""S4 / Drop 1: noise and lighting statistics of a capture session (capture.py output) -> stats/stats.json + PNGs.

    python -m aaltonomous.data.drop1.stats data/drop1/<session>          # writes <session>/stats/

Per step and camera:
  rgb      temporal noise (per-pixel std over the frames, per channel; median / p95 over pixels), brightness
           (mean luma, per-channel means), gray-world white balance (R/G, B/G), clipped fractions.
  depth    (head) invalid fraction (0 = hole), per-pixel temporal std (mm) where valid, std vs depth (binned) and a
           fitted RealSense-style model std = k * z^2 (z in m).
  objects  (objects/layout steps, head) per object: mask = layout projected with the session's calibration.yaml
           (object box -> pixel ROI) refined by an HSV colour threshold on the median frame (fallback: the
           projected box), then hole fraction, depth std, measured top height above the counter, and the pixel
           offset between colour centroid and projection (= calibration check).
Plots: depth std / hole maps of the empty scenes, depth noise vs range, hole fraction per object type,
RGB noise and brightness per camera and lighting, mask overlays per object step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CAMS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
# OpenCV HSV (H 0-180). Matte PLA under unknown light: wide hue bands, saturation floor.
HSV = {"red": [((0, 90, 50), (10, 255, 255)), ((165, 90, 50), (180, 255, 255))],
       "yellow": [((15, 90, 80), (40, 255, 255))],
       "blue": [((95, 90, 30), (135, 255, 255))]}


def load_frames(step_dir: Path, key):
    import cv2
    fs = sorted((step_dir / key).glob("*.png"))
    if key == "head_depth":
        return np.stack([cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in fs])
    return np.stack([cv2.imread(str(f), cv2.IMREAD_COLOR)[..., ::-1] for f in fs])


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def rgb_stats(F):
    """F: (N,H,W,3) uint8."""
    f = F.astype(np.float32)
    std = f.std(0, ddof=1) if len(F) > 1 else np.zeros(f.shape[1:])
    mean = f.mean(0)
    luma = mean @ np.array([0.299, 0.587, 0.114])
    ch = mean.reshape(-1, 3).mean(0)
    s = std.mean(-1)
    return {"n_frames": len(F), "noise_std_median": r(np.median(s)), "noise_std_p95": r(np.percentile(s, 95)),
            "noise_std_per_channel": [r(np.median(std[..., c])) for c in range(3)],
            "luma_mean": r(luma.mean(), 1), "luma_p05": r(np.percentile(luma, 5), 1), "luma_p95": r(np.percentile(luma, 95), 1),
            "channel_mean": [r(c, 1) for c in ch], "wb_r_over_g": r(ch[0] / max(ch[1], 1e-6)),
            "wb_b_over_g": r(ch[2] / max(ch[1], 1e-6)),
            "clipped_high_frac": r((F.max(-1) >= 254).mean(), 4), "clipped_low_frac": r((F.max(-1) <= 5).mean(), 4),
            "distinct_frames": int(len({F[i].tobytes().__hash__() for i in range(len(F))}))}, s


def depth_stats(D, min_valid=0.5):
    """D: (N,H,W) uint16 mm. Returns summary + per-pixel std map (mm, NaN where too few valid) + hole map."""
    v = D > 0
    cnt = v.sum(0)
    hole = 1.0 - cnt / len(D)
    d = D.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = (d * v).sum(0) / cnt
        var = ((d - mu) ** 2 * v).sum(0) / np.maximum(cnt - 1, 1)
    ok = cnt >= max(2, min_valid * len(D))
    std = np.where(ok, np.sqrt(var), np.nan)
    mu = np.where(ok, mu, np.nan)
    out = {"n_frames": len(D), "hole_frac_mean": r(hole.mean(), 4), "hole_frac_always": r((cnt == 0).mean(), 4),
           "hole_frac_sometimes": r(((cnt > 0) & (cnt < len(D))).mean(), 4),
           "std_mm_median": r(np.nanmedian(std)) if ok.any() else None,
           "std_mm_p95": r(np.nanpercentile(std, 95)) if ok.any() else None,
           "range_mm_median": r(np.nanmedian(mu), 1) if ok.any() else None}
    bins = []
    if ok.any() and len(D) > 1:
        z, s = mu[ok], std[ok]
        for lo in range(0, 3000, 100):
            m = (z >= lo) & (z < lo + 100)
            if m.sum() >= 200:
                bins.append({"z_mm": lo + 50, "std_mm_median": r(np.median(s[m])), "n_px": int(m.sum())})
        zm = z / 1000.0
        k = float((s * zm ** 2).sum() / max((zm ** 4).sum(), 1e-12))   # least squares s = k z^2
        out["model_std_mm_eq_k_z2"] = {"k_mm_per_m2": r(k), "note": "std(mm) ~= k * z(m)^2, fitted on valid pixels"}
    out["std_vs_depth"] = bins
    return out, std, hole, mu


def object_masks(cal, sc, objects, med_rgb, pad=12):
    """Per object: (mask, info). Projected box ROI (calibration) refined by colour threshold."""
    import cv2
    H, W = med_rgb.shape[:2]
    hsv = cv2.cvtColor(med_rgb, cv2.COLOR_RGB2HSV)
    res = []
    for o in objects:
        h = sc.object_height(o.get("shape"))
        a = sc.object_dims(o.get("shape", "cube"))["circumradius"]
        cs = [[o["x"] + dx, o["y"] + dy, z] for dx in (-a, a) for dy in (-a, a) for z in (0.0, h)]
        uvz = cal.project("head", sc.mat_to_pelvis(np.array(cs)))
        ctr = cal.project("head", sc.mat_to_pelvis(np.array([o["x"], o["y"], h / 2])))
        info = {k: o.get(k) for k in ("color", "shape", "x", "y")}
        if (uvz[:, 2] <= 0).any():
            res.append((None, dict(info, error="behind camera")))
            continue
        hull = cv2.convexHull(uvz[:, :2].astype(np.float32)).astype(np.int32)
        box = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(box, hull, 1)
        roi = cv2.dilate(box, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)).astype(bool)
        col = np.zeros((H, W), bool)
        for lo, hi in HSV.get(o["color"], []):
            col |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)).astype(bool)
        m = col & roi
        if m.sum() >= 0.3 * box.sum() and m.sum() >= 20:
            # largest connected component of the colour mask (neighbouring same-colour objects stay out)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
            m = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))) if n > 1 else m
            m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) if m.sum() > 100 else m
            src = "colour"
        else:
            m, src = box.astype(bool), "projection"
        info.update(mask_source=src, mask_px=int(m.sum()), projected_px=int(box.sum()),
                    proj_center_uv=[r(ctr[0], 1), r(ctr[1], 1)])
        if src == "colour" and m.any():
            vv, uu = np.nonzero(m)
            info["colour_center_uv"] = [r(uu.mean(), 1), r(vv.mean(), 1)]
            info["calib_offset_px"] = r(np.hypot(uu.mean() - ctr[0], vv.mean() - ctr[1]), 1)
        res.append((m if m.any() else None, info))
    return res


def object_stats(cal, sc, D, std, mu, mask):
    v = D[:, mask] > 0
    out = {"hole_frac": r(1 - v.mean(), 4), "hole_frac_always": r((v.sum(0) == 0).mean(), 4),
           "depth_std_mm_median": r(np.nanmedian(std[mask])) if np.isfinite(std[mask]).any() else None}
    vv, uu = np.nonzero(mask)
    z = mu[mask]
    ok = np.isfinite(z)
    if ok.sum() >= 5 and cal.cams["head"].T_pelvis_optical is not None:
        p = cal.pixel_to_pelvis("head", uu[ok], vv[ok], z[ok])
        out["top_height_m_measured"] = r(np.percentile(p[:, 2], 90) - sc.counter_z)
    return out


def analyse(session: Path, plots=True, log=print):
    from aaltonomous.interface.calibration import Calibration
    from aaltonomous.scene.scene import Scene
    session = Path(session)
    cal = Calibration.load(session / "calibration.yaml" if (session / "calibration.yaml").exists() else None)
    sc = Scene.load(session / "scene.yaml" if (session / "scene.yaml").exists() else None)
    sess = json.loads((session / "session.json").read_text())
    od = session / "stats"
    od.mkdir(exist_ok=True)
    out = {"session": sess.get("session"), "source": sess.get("source"), "steps": {}, "objects": [],
           "calibration_extrinsics_default": True}
    maps, overlays = {}, {}
    dirs = [p for p in session.iterdir() if (p / "meta.json").exists()]
    for sd, meta in sorted(((d, json.loads((d / "meta.json").read_text())) for d in dirs), key=lambda x: x[1]["start"]):
        st = {"kind": meta["kind"], "lights": meta.get("lights"), "n_frames": meta["n_frames"], "cams": {}}
        for k in CAMS:
            F = load_frames(sd, k)
            st["cams"][k], _ = rgb_stats(F)
            if k == "head_rgb":
                med = np.median(F, 0).astype(np.uint8)
        D = load_frames(sd, "head_depth")
        st["head_depth"], std, hole, mu = depth_stats(D)
        if meta["kind"] == "empty":
            maps[sd.name] = (std, hole, mu)
        if meta.get("objects"):
            ov = []
            for m, info in object_masks(cal, sc, meta["objects"], med):
                if m is not None:
                    info.update(object_stats(cal, sc, D, std, mu, m))
                    info["expected_height_m"] = r(sc.object_height(info.get("shape")))
                info.update(step=sd.name, lights=meta.get("lights"))
                out["objects"].append(info)
                ov.append((m, info))
            overlays[sd.name] = (med, ov)
        out["steps"][sd.name] = st
        log(f"  {sd.name:20s} depth holes {st['head_depth']['hole_frac_mean']}, std {st['head_depth']['std_mm_median']} mm,"
            f" head rgb noise {st['cams']['head_rgb']['noise_std_median']}")
    out["summary"] = summarize(out)
    (od / "stats.json").write_text(json.dumps(out, indent=1))
    if plots:
        make_plots(od, out, maps, overlays)
    log(f"[stats] -> {od / 'stats.json'}")
    return out


def summarize(out):
    s = {}
    emp = {k: v for k, v in out["steps"].items() if v["kind"] == "empty"}
    if emp:
        s["empty_depth"] = {k: {x: v["head_depth"][x] for x in ("hole_frac_mean", "std_mm_median", "range_mm_median")}
                            for k, v in emp.items()}
    by = {}
    for o in out["objects"]:
        if o.get("hole_frac") is not None:
            by.setdefault(f"{o['color']} {o['shape']}", []).append(o["hole_frac"])
    s["object_hole_frac_mean"] = {k: r(np.mean(v), 4) for k, v in sorted(by.items())}
    offs = [o["calib_offset_px"] for o in out["objects"] if o.get("calib_offset_px") is not None]
    s["calib_offset_px_median"] = r(np.median(offs), 1) if offs else None
    s["mask_sources"] = {k: sum(o.get("mask_source") == k for o in out["objects"]) for k in ("colour", "projection")}
    lights = {}
    for v in out["steps"].values():
        for c, cs in v["cams"].items():
            lights.setdefault(v["lights"], {}).setdefault(c, []).append(cs)
    s["rgb_by_lights"] = {L: {c: {x: r(np.mean([q[x] for q in qs])) for x in
                                  ("noise_std_median", "luma_mean", "wb_r_over_g", "wb_b_over_g")}
                              for c, qs in cs.items()} for L, cs in lights.items()}
    return s


def make_plots(od, out, maps, overlays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, (std, hole, mu) in maps.items():
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        vmax = np.nanpercentile(std, 98) if np.isfinite(std).any() else 1
        im = ax[0].imshow(std, cmap="magma", vmin=0, vmax=max(vmax, 1e-3))
        ax[0].set_title(f"{name}: depth temporal std (mm)")
        fig.colorbar(im, ax=ax[0], fraction=0.035)
        im = ax[1].imshow(hole, cmap="Greys", vmin=0, vmax=1)
        ax[1].set_title("hole fraction per pixel (0 = always valid)")
        fig.colorbar(im, ax=ax[1], fraction=0.035)
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(od / f"depth_maps_{name}.png", dpi=90)
        plt.close(fig)
    curves = {k: v["head_depth"]["std_vs_depth"] for k, v in out["steps"].items()
              if v["kind"] == "empty" and v["head_depth"]["std_vs_depth"]}
    if curves:
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, b in curves.items():
            ax.plot([x["z_mm"] for x in b], [x["std_mm_median"] for x in b], marker="o", ms=4, lw=2, label=k)
        ax.set_xlabel("depth (mm)")
        ax.set_ylabel("median temporal std (mm)")
        ax.set_title("Head depth noise vs range (empty scene)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(od / "depth_noise_vs_range.png", dpi=90)
        plt.close(fig)
    hf = out["summary"].get("object_hole_frac_mean", {})
    if hf:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ks = list(hf)
        ax.barh(ks, [hf[k] * 100 for k in ks], color="#3b6fb6")
        for i, k in enumerate(ks):
            ax.text(hf[k] * 100, i, f" {hf[k] * 100:.1f}%", va="center", fontsize=8)
        ax.set_xlabel("depth hole fraction in object mask (%)")
        ax.set_title("Depth holes per object type (head camera)")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(od / "object_holes.png", dpi=90)
        plt.close(fig)
    rb = out["summary"].get("rgb_by_lights", {})
    if rb:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        Ls = list(rb)
        w = 0.8 / len(CAMS)
        for j, c in enumerate(CAMS):
            xs = np.arange(len(Ls)) + j * w
            ax[0].bar(xs, [rb[L].get(c, {}).get("noise_std_median") or 0 for L in Ls], w, label=c,
                      color=["#3b6fb6", "#d08a2e", "#4a9a6a"][j])
            ax[1].bar(xs, [rb[L].get(c, {}).get("luma_mean") or 0 for L in Ls], w, label=c,
                      color=["#3b6fb6", "#d08a2e", "#4a9a6a"][j])
        for a, t in zip(ax, ("RGB temporal noise (std, 0-255)", "mean brightness (luma, 0-255)")):
            a.set_xticks(np.arange(len(Ls)) + w, Ls)
            a.set_title(t)
            a.grid(axis="y", alpha=0.3)
        ax[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(od / "rgb_by_lights.png", dpi=90)
        plt.close(fig)
    import cv2
    for name, (med, ov) in overlays.items():
        img = med.copy()
        for m, info in ov:
            if m is None:
                continue
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            col = (0, 255, 0) if info.get("mask_source") == "colour" else (255, 0, 255)
            cv2.drawContours(img, cnts, -1, col, 1)
            u, v = info["proj_center_uv"]
            if u is not None:
                cv2.drawMarker(img, (int(u), int(v)), (255, 255, 255), cv2.MARKER_CROSS, 8, 1)
                cv2.putText(img, f"{info.get('hole_frac', 0) * 100:.0f}%", (int(u) + 6, int(v) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imwrite(str(od / f"masks_{name}.png"), img[..., ::-1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    analyse(Path(a.session), plots=not a.no_plots)


if __name__ == "__main__":
    main()
