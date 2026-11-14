"""Colours of drop 2 in the head camera, first frame of every episode: foam objects, box tape, table, hand, forearm.

    python -m halal.eval.real.drop2_colours [--sheets DIR] [--sweep]   -> hue / HSV tables, what the detector's colour stage makes of them
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from halal import config
from halal.eval.real import colours, dataset, drop2_frames
from halal.perception.color import ColorClassifier, chromas, hue_angles, turned


@dataclass(frozen=True)
class Seed:
    """Hue range (deg) that seeds one foam colour and the medians a blob needs to be that foam."""
    hue: tuple[float, float]
    min_chroma: float
    min_value: float


@dataclass(frozen=True)
class Blob:
    """One seeded blob: its outline filled, and whether it is one compact convex object rather than two touching ones."""
    mask: np.ndarray
    single: bool
    x: float


SEEDS = {"purple": Seed((208, 300), 0.40, 90), "orange": Seed((17, 34), 0.50, 90), "yellow": Seed((36, 110), 0.20, 140)}
TAPE_HUE, FOREARM_HUE = (345, 34), (170, 212)
BLOB_AREA = (900, 14000)
BOX_AREA = 20000
PATCH = 40
FAMILY = {"purple": "blue", "orange": "red", "yellow": "yellow"}
SWEEP_DEG = (-15, -10, -5, 0, 5, 10, 15)
OUTLINE_BGR = {"purple": (255, 0, 255), "orange": (0, 255, 0), "yellow": (255, 255, 255)}
# Shapes of the single orange / yellow blobs of every first frame, left to right: c cube, p prism, y cylinder, - unclear.
SHAPES: dict[int, tuple[str, str]] = {
    0: ("pc", "y"), 1: ("pc", "cy"), 2: ("pc", "c"), 3: ("pc", "c"), 4: ("pc", "c"), 5: ("pc", "c"), 6: ("cp", "c"), 7: ("pc", "c"),
    8: ("pc", "c"), 9: ("cp", "c"), 10: ("pc", "yc"), 11: ("cp", "yc"), 12: ("c", "c"), 13: ("c", "yc"), 14: ("c", "yc"),
    15: ("-c", "yc"), 16: ("cp", "-"), 17: ("cp", "-"), 18: ("cp", "-"), 19: ("p", ""), 20: ("p", "-"), 21: ("cp", "yc"),
    22: ("c", "yc"), 23: ("cp", "yc"), 24: ("c-", "c"), 25: ("cp", "c"), 26: ("cp", "c"), 27: ("c-", "c"), 28: ("c-", "c"),
    29: ("cp", "c"), 30: ("cp", "c"), 31: ("cp", "yc"), 32: ("cp", "yc"), 33: ("cp", "yc"), 34: ("cp", "yc"), 35: ("cp", ""),
    36: ("", "-"), 37: ("pc", "yc"), 38: ("pc", "yc"), 39: ("pc", "yc"), 40: ("pc", "yc"), 41: ("pc", "yc"), 42: ("pc", ""),
    43: ("pc", ""), 44: ("p", "y"), 45: ("p", "y"), 46: ("p", "y"), 47: ("p", "y"),
}


def in_hue(hue: np.ndarray, span: tuple[float, float]) -> np.ndarray:
    """True where a hue (deg) lies in the range running from span[0] up to span[1]."""
    return (hue - span[0]) % 360.0 <= (span[1] - span[0]) % 360.0


def seed_mask(rgb: np.ndarray, span: tuple[float, float], min_chroma: float, min_value: float) -> np.ndarray:
    """Pixels of an RGB image in a hue range that are saturated and bright enough."""
    return in_hue(hue_angles(rgb), span) & (chromas(rgb) >= min_chroma) & (rgb.max(axis=-1) >= min_value)


def grow(mask: np.ndarray, px: int) -> np.ndarray:
    """Mask dilated (px > 0) or eroded (px < 0) by a square of that half side."""
    kernel = np.ones((2 * abs(px) + 1,) * 2, np.uint8)
    return (cv2.dilate if px > 0 else cv2.erode)(mask.astype(np.uint8), kernel) > 0


def foam_blobs(rgb: np.ndarray, seed: Seed, on_table: np.ndarray) -> list[Blob]:
    """Blobs of one foam colour lying on the table in front of the boxes, left to right: object-sized and vivid."""
    opened = cv2.morphologyEx(seed_mask(rgb, seed.hue, 0.15, 90).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labelled, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    found = []
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if not BLOB_AREA[0] <= area <= BLOB_AREA[1]:
            continue
        contours, _ = cv2.findContours((labelled == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        outline = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(outline)
        mask = cv2.drawContours(np.zeros(opened.shape, np.uint8), [outline], -1, 1, cv2.FILLED) > 0
        v, u = np.argwhere(mask).mean(axis=0).astype(int)
        vivid = np.median(chromas(rgb[mask])) >= seed.min_chroma and np.median(rgb[mask].max(axis=-1)) >= seed.min_value
        if not vivid or not on_table[v, u]:
            continue
        _, _, wide, tall = cv2.boundingRect(hull)
        single = area >= 0.8 * cv2.contourArea(hull) and max(wide, tall) <= 2.2 * min(wide, tall)
        found.append(Blob(mask, bool(single), float(u)))
    return sorted(found, key=lambda blob: blob.x)


def below_boxes(boxes: np.ndarray) -> np.ndarray:
    """Pixels below the lowest box pixel of their image column (of the median column where a column has none)."""
    rows = np.arange(boxes.shape[0])[:, None]
    lowest = np.where(boxes.any(axis=0), boxes.shape[0] - 1 - np.argmax(boxes[::-1], axis=0), -1)
    lowest[lowest < 0] = int(np.median(lowest[lowest >= 0])) if (lowest >= 0).any() else 0
    return rows > lowest[None, :]


def segment(rgb: np.ndarray) -> tuple[dict[str, list[Blob]], dict[str, np.ndarray]]:
    """Foam blobs per colour, and masks of the tape, the bare table, the dark hand and the blue forearm cover."""
    warm = colours.large_components(seed_mask(rgb, TAPE_HUE, 0.30, 50).astype(np.uint8), 2000)
    on_table = below_boxes(colours.large_components((warm & ~seed_mask(rgb, SEEDS["orange"].hue, 0.15, 90)).astype(np.uint8), BOX_AREA))
    blobs = {name: foam_blobs(rgb, seed, on_table) for name, seed in SEEDS.items()}
    foam = grow(np.any([blob.mask for found in blobs.values() for blob in found] or [np.zeros(rgb.shape[:2], bool)], axis=0), 4)
    tape = warm & ~foam
    value = rgb.max(axis=-1)
    table = (chromas(rgb) < 0.15) & (value >= 110) & on_table & ~foam
    forearm = colours.large_components((seed_mask(rgb, FOREARM_HUE, 0.30, 60) & ~foam).astype(np.uint8), 300)
    return blobs, {"tape": tape, "table": table, "hand (dark)": (value < 60) & ~foam, "forearm cover": forearm}


def shape_of(episode: int, colour: str, index: int) -> str:
    """Hand label of the index-th single blob of a colour in an episode's first frame; purple is always the cube."""
    if colour == "purple":
        return "cube"
    letters = SHAPES.get(episode, ("", ""))[colour == "yellow"]
    return {"c": "cube", "p": "prism", "y": "cylinder"}.get(letters[index] if index < len(letters) else "-", "unclear")


def lit_shaded(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixels (N, 3) of one object split at the Otsu threshold of their brightness: lit faces, shaded faces."""
    value = rgb.max(axis=-1).astype(np.uint8)
    threshold, _ = cv2.threshold(value[:, None], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return rgb[value > threshold], rgb[value <= threshold]


def unwrap(hue: np.ndarray) -> np.ndarray:
    """Hues (deg) moved to -90..270 so that reds on either side of 0 stay together."""
    return np.where(hue > 270.0, hue - 360.0, hue)


def describe(rgb: np.ndarray) -> str:
    """One table row for pixels (N, 3): hue in degrees and OpenCV units, saturation, value."""
    hue = np.percentile(unwrap(hue_angles(rgb)), [1, 5, 50, 95, 99])
    hsv = cv2.cvtColor(rgb.astype(np.uint8)[:, None], cv2.COLOR_RGB2HSV)[:, 0].astype(float)
    hsv[:, 0] = np.where(hsv[:, 0] > 135, hsv[:, 0] - 180, hsv[:, 0])
    cv_h, sat, val = np.percentile(hsv, [5, 50, 95], axis=0).T
    return ("n={:8d}  hue deg p1/p5/p50/p95/p99 {:4.0f} {:4.0f} {:4.0f} {:4.0f} {:4.0f}   cvH {:3.0f} {:3.0f} {:3.0f}   "
            "S {:3.0f} {:3.0f} {:3.0f}   V {:3.0f} {:3.0f} {:3.0f}").format(len(rgb), *hue, *cv_h, *sat, *val)


def tape_patches(rgb: np.ndarray, tape: np.ndarray) -> list[np.ndarray]:
    """Pixels of every object-sized image cell that is nearly all tape: what an object-sized piece of box looks like."""
    cells = [(slice(v, v + PATCH), slice(u, u + PATCH)) for v in range(0, tape.shape[0], PATCH) for u in range(0, tape.shape[1], PATCH)]
    return [rgb[cell][tape[cell]] for cell in cells if tape[cell].mean() > 0.9]


def print_pixels(pixels: dict[str, list[np.ndarray]]) -> None:
    print("== pixels per class, pooled over the first frames (hue: detector's angle; cvH S V: OpenCV, p5 / p50 / p95)")
    for name, parts in pixels.items():
        if parts:
            print(f"  {name:26s} {describe(np.concatenate(parts))}")


def print_instances(named: dict[str, list[tuple[np.ndarray, str, float]]]) -> None:
    print("== per object (or tape patch): the colour the detector names (most saturated share, table-balanced)")
    for name, rows in named.items():
        hue = unwrap(hue_angles(np.array([colour for colour, _, _ in rows])))
        chroma = chromas(np.array([colour for colour, _, _ in rows]))
        families = sorted({family for _, family, _ in rows})
        tally = ", ".join(f"{f} {sum(r[1] == f for r in rows)} (conf min {min(r[2] for r in rows if r[1] == f):.2f})" for f in families)
        low, mid, high = np.percentile(hue, [5, 50, 95])
        print(f"  {name:16s} n={len(rows):4d}  hue min/p5/p50/p95/max {hue.min():5.1f} {low:5.1f} {mid:5.1f} {high:5.1f} {hue.max():5.1f}   "
              f"chroma min/p50 {chroma.min():.2f} {np.median(chroma):.2f}   -> {tally}")


def print_take(take: dict[str, dict[str, float]], frames: int) -> None:
    print("== mean pixels per frame that each colour family takes from each source (pixel classes of the detector)")
    sources = list(next(iter(take.values())))
    print("  " + " " * 8 + "".join(f"{s[:13]:>15s}" for s in sources))
    for family, row in take.items():
        print(f"  {family:8s}" + "".join(f"{row[s] / frames:15.0f}" for s in sources))


def look(rgb: np.ndarray, blobs: dict[str, list[Blob]], table: np.ndarray, classifier: ColorClassifier) -> list[tuple[str, np.ndarray]]:
    """Foam colour and sampled colour of every single blob of one frame: what one look of the detector would have to name."""
    gains = classifier.white_balance(rgb[table])
    return [(name, classifier.sample(rgb[grow(blob.mask, -2)], gains)) for name, group in blobs.items() for blob in group if blob.single]


def sweep(frames: dict[int, np.ndarray], classifier: ColorClassifier) -> None:
    """Print how many objects keep their family when the whole image's hue is turned: fixed boundary against the look's own."""
    print("== whole-image hue turn: objects named right, fixed red / yellow boundary | boundary from the look's hue gap (moved in n looks)")
    segments = {episode: segment(rgb) for episode, rgb in frames.items()}
    for degrees in SWEEP_DEG:
        right = {scheme: {name: 0 for name in FAMILY} for scheme in ("fixed", "look")}
        total, moved = {name: 0 for name in FAMILY}, 0
        for episode, rgb in frames.items():
            blobs, masks = segments[episode]
            seen = look(turned(rgb, degrees), blobs, masks["table"], classifier)
            split = classifier.split([colour for _, colour in seen])
            moved += split is not None and split != classifier.boundary()
            for name, colour in seen:
                total[name] += 1
                right["fixed"][name] += classifier.name(colour, classifier.boundary())[0] == FAMILY[name]
                right["look"][name] += classifier.name(colour, split)[0] == FAMILY[name]
        cells = lambda scheme: "  ".join(f"{name} {right[scheme][name]:2d}/{total[name]}" for name in FAMILY)
        print(f"  {degrees:+4d} deg   {cells('fixed')}   |   {cells('look')}   ({moved})")


def save_sheet(frames: dict[int, np.ndarray], found: dict[int, dict[str, list[Blob]]], out: Path, per_row: int = 4) -> None:
    """Contact sheets of the first frames with every blob outlined and numbered, to check the segmentation and label shapes."""
    tiles = []
    for episode, rgb in frames.items():
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for name, blobs in found[episode].items():
            singles = iter(range(len(blobs)))
            for blob in blobs:
                outline, _ = cv2.findContours(blob.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.polylines(bgr, outline, True, OUTLINE_BGR[name], 1)
                v, u = np.argwhere(blob.mask).mean(axis=0).astype(int)
                cv2.putText(bgr, f"{name[0]}{next(singles)}" if blob.single else "m", (u - 16, v + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(bgr, f"ep{episode:02d}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        tiles.append(cv2.resize(bgr, (320, 240)))
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % (3 * per_row))
    out.mkdir(parents=True, exist_ok=True)
    for sheet in range(len(tiles) // (3 * per_row)):
        part = tiles[sheet * 3 * per_row:(sheet + 1) * 3 * per_row]
        cv2.imwrite(str(out / f"drop2_blobs_{sheet}.png"), np.vstack([np.hstack(part[r * per_row:(r + 1) * per_row]) for r in range(3)]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=drop2_frames.ROOT)
    ap.add_argument("--sheets", type=Path, help="write contact sheets of the segmentation here")
    ap.add_argument("--sweep", action="store_true", help="also turn the hue of the whole image and count what keeps its family")
    args = ap.parse_args(argv)
    classifier = ColorClassifier(config.load()["perception"]["topface"]["color"])
    families = list(classifier.cfg["basins"])
    frames = colours.first_frames(dataset.load(args.root))
    pixels: dict[str, list[np.ndarray]] = {}
    named: dict[str, list[tuple[np.ndarray, str, float]]] = {}
    take: dict[str, dict[str, float]] = {family: {} for family in families}
    found = {}
    for episode, rgb in frames.items():
        blobs, masks = segment(rgb)
        found[episode] = blobs
        gains = classifier.white_balance(rgb[masks["table"]])
        sources = {**{f"{name} foam": np.any([b.mask for b in group] or [np.zeros(rgb.shape[:2], bool)], axis=0) for name, group in blobs.items()}, **masks}
        sources["other"] = ~np.any(list(sources.values()), axis=0)
        classes = classifier.pixel_classes(rgb.reshape(-1, 3), gains).reshape(rgb.shape[:2])
        for index, family in enumerate(families):
            for source, mask in sources.items():
                take[family][source] = take[family].get(source, 0.0) + float((mask & (classes == index)).sum())
        for name, mask in masks.items():
            pixels.setdefault(name, []).append(rgb[mask])
        split = classifier.split([colour for _, colour in look(rgb, blobs, masks["table"], classifier)])
        for patch in tape_patches(rgb, masks["tape"]):
            named.setdefault("tape patch", []).append((colour := classifier.sample(patch, gains), *classifier.name(colour, split)))
        for name, group in blobs.items():
            for index, blob in enumerate(b for b in group if b.single):
                body = rgb[grow(blob.mask, -2)]
                label = f"{name} {shape_of(episode, name, index)}"
                lit, shaded = lit_shaded(body)
                for key, part in ((label, body), (f"{name} lit faces", lit), (f"{name} shaded faces", shaded)):
                    pixels.setdefault(key, []).append(part)
                named.setdefault(label, []).append((colour := classifier.sample(body, gains), *classifier.name(colour, split)))
    print(f"{len(frames)} first frames of {args.root.name}; basins {dict(classifier.cfg['basins'])}")
    print_pixels(dict(sorted(pixels.items())))
    print_instances(dict(sorted(named.items())))
    print_take(take, len(frames))
    if args.sweep:
        sweep(frames, classifier)
    if args.sheets:
        save_sheet(frames, found, args.sheets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
