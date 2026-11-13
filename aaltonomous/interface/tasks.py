"""Task definitions shared by sim, real and eval: instructions and random cube layouts (spec §3).

The real robot uses printed layout cards made with sample_layout(), the same code the sim uses.
Held-out (finals) instruction templates are deliberately NOT in this file.
"""
from __future__ import annotations

import numpy as np

COLORS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")  # extruded square, circle, equilateral triangle (spec §2)
BINS = ("left", "center", "right")
BIN_COLOR = {"left": "red", "center": "yellow", "right": "blue"}  # (default) physical bin colours
FIXED_INSTRUCTION = "Sort each cube into the bin of the same color."
INSERT_INSTRUCTION = "Put each object through the hole of the same shape in the lid."

# Training templates for sort_language. {c0..c2} colours, {b0..b2} bins (positions or colours).
TEMPLATES = (
    "Put the {c0} cubes in the {b0} bin, the {c1} cubes in the {b1} bin and the {c2} cubes in the {b2} bin.",
    "{C0} goes to the {b0} bin, {c1} to the {b1} bin, {c2} to the {b2} bin.",
    "Sort the cubes: {c0} into {b0}, {c1} into {b1}, {c2} into {b2}.",
    "Place every {c0} cube in the {b0} bin. {C1} cubes belong in the {b1} bin, and {c2} ones in the {b2} bin.",
    "The {b0} bin is for {c0} cubes, the {b1} bin for {c1}, the {b2} bin for {c2}.",
    "Move {c0} cubes to the {b0} bin, {c1} cubes to the {b1} bin, and {c2} cubes to the {b2} bin.",
    "Drop the {c0} ones into the {b0} bin, the {c1} ones into the {b1} bin, the {c2} ones into the {b2} bin.",
    "Use the {b0} bin for {c0}, the {b1} bin for {c1} and the {b2} bin for {c2} cubes.",
)


def bin_phrase(b, rng):
    """A bin named by position ('left') or by its colour ('red'), so instructions can contradict colours."""
    return b if rng.random() < 0.6 else BIN_COLOR[b]


def make_instruction(task, rng):
    """Returns (instruction, mapping colour -> bin position)."""
    if task == "sort_fixed":
        return FIXED_INSTRUCTION, {c: b for b, c in BIN_COLOR.items()}
    if task == "insert_shapes":  # rung 3: no bin mapping, the lid holes define the target
        return INSERT_INSTRUCTION, {}
    if task != "sort_language":
        raise ValueError(f"unknown task {task!r}")
    colors = list(rng.permutation(COLORS))
    bins = list(rng.permutation(BINS))
    words = {f"c{i}": c for i, c in enumerate(colors)}
    words.update({f"C{i}": c.capitalize() for i, c in enumerate(colors)})
    words.update({f"b{i}": bin_phrase(b, rng) for i, b in enumerate(bins)})
    text = TEMPLATES[rng.integers(len(TEMPLATES))].format(**words)
    return text, dict(zip(colors, bins))


def sample_layout(rng, n_min=3, n_max=6, region=((0.20, 0.60), (0.10, 0.22)), min_spacing=0.08,
                  yaw_range=(-np.pi / 4, np.pi / 4), tries=1000, shapes=SHAPES):
    """Cubes at random positions/yaw in `region` (m, counter-mat coordinates), 1-2 per colour.

    Defaults mirror aaltonomous/scene/scene.yaml; pass Scene.layout_kwargs() to follow that file exactly.

    Returns a list of dicts {color, shape, x, y, yaw} (shape drawn per object after the
    positions, so positions for a seed are unchanged; at most 2 of one colour so at most 2 of one type). Raises if the spacing can't be met."""
    n = int(rng.integers(n_min, n_max + 1))
    per = {c: 1 for c in COLORS}
    for c in rng.permutation(COLORS)[: max(0, n - 3)]:
        per[c] += 1
    colors = [c for c in COLORS for _ in range(per[c])][:n]
    (x0, x1), (y0, y1) = region
    for _ in range(50):  # greedy placement can paint itself into a corner: restart the whole layout
        cubes = []
        for color in rng.permutation(colors):
            for _ in range(tries // 10):
                x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
                if all(np.hypot(x - c["x"], y - c["y"]) >= min_spacing for c in cubes):
                    cubes.append({"color": str(color), "x": round(x, 3), "y": round(y, 3),
                                  "yaw": round(float(rng.uniform(*yaw_range)), 3)})
                    break
            else:
                break
        if len(cubes) == len(colors):
            for c in cubes:
                c["shape"] = str(shapes[int(rng.integers(len(shapes)))])
            return cubes
    raise RuntimeError("could not place cubes with the requested spacing")


def layout_card(cubes, instruction):
    """Plain-text layout card for staff resetting the real robot."""
    lines = [f"INSTRUCTION: {instruction}", "colour  shape     x(cm)  y(cm)  yaw(deg)"]
    lines += [f"{c['color']:<7} {c.get('shape', 'cube'):<9} {c['x'] * 100:5.1f}  {c['y'] * 100:5.1f}  {np.degrees(c['yaw']):6.0f}" for c in cubes]
    return "\n".join(lines)
