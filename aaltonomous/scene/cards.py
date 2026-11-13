"""Layout cards + staff scoring sheets for real-robot episodes (spec §3: same sampler as sim).

    python -m aaltonomous.scene.cards --task sort_language --n 10 --seed 1000 --out cards_lang.md
    python -m aaltonomous.scene.cards --score sort_language correct correct wrong off   # -> points

Each card (shapes: ■ cube, ● cylinder, ▲ triangle; --figs DIR also draws each layout true-to-scale, top view): seed, instruction, cube positions on the printed mat grid (cm, mat frame, see scene.yaml),
the expected bin per colour, and a checkbox row per cube for the judge. Seeds make cards reproducible,
so the same episode can be rebuilt in sim from the card's layout (the card seed is not an AaltoEnv seed). Finals use a seed range never published to teams.
"""
from __future__ import annotations

import argparse

import numpy as np

from aaltonomous.interface.tasks import make_instruction, sample_layout

from .scene import INSERT_OUTCOMES, OUTCOMES, Scene, resolve_bin, shape_polygon

GLYPH = {"cube": "■ cube", "cylinder": "● cylinder", "triangle": "▲ triangle"}


def make_card(sc, task, seed):
    rng = np.random.default_rng(seed)
    instruction, mapping = make_instruction(task, rng)
    cubes = sample_layout(rng, **sc.layout_kwargs())
    return {"seed": seed, "task": task, "instruction": instruction, "mapping": mapping, "cubes": cubes}


def target_text(sc, card, c):
    """Where this object should end up: bin (by colour/shape mapping) or the matching lid hole (rung 3)."""
    if card["task"] == "insert_shapes":
        return f"{c['shape']} hole, {sc.lid_bin} bin lid"
    b = resolve_bin(card["mapping"], c["color"], c["shape"])
    return f"{b} ({sc.cfg['bins']['color'][b]})"


def card_figure(sc, card, path):
    """Top view of the mat with every object drawn at its true size, shape and yaw (robot at the bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    W, D = sc.cfg["counter"]["mat_size"]
    fig, ax = plt.subplots(figsize=(8, 8 * D / W + 0.4))
    ax.set_xlim(W * 100, 0)                                     # x grows to the robot's left = image left
    ax.set_ylim(0, D * 100)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, int(W * 100) + 1, 10))
    ax.set_yticks(range(0, int(D * 100) + 1, 10))
    ax.grid(color="0.8", lw=0.5)
    hx, hy = np.asarray(sc.cfg["bins"]["outer_size"]) * 50
    for n, (bx, by) in sc.cfg["bins"]["centers"].items():
        ax.add_patch(Rectangle(((bx * 100) + hx, by * 100 - hy), -2 * hx, 2 * hy, fill=False, lw=2,
                               ec=np.array(sc.colors[sc.cfg["bins"]["color"][n]]) / 255))
        ax.text(bx * 100, by * 100, n.upper() + (" + LID" if card["task"] == "insert_shapes" and n == sc.lid_bin else ""),
                ha="center", va="center", fontsize=9, color="0.4")
    for i, c in enumerate(card["cubes"], 1):
        a = sc.object_dims(c["shape"])["apothem"]
        rot = c["yaw"]
        R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        poly = (shape_polygon(c["shape"], a) @ R.T + [c["x"], c["y"]]) * 100
        ax.add_patch(Polygon(poly, fc=np.array(sc.colors[c["color"]]) / 255, ec="k", lw=1))
        ax.text(c["x"] * 100, c["y"] * 100 - 4.5, str(i), ha="center", fontsize=9)
    ax.set_title(f"{card['task']} seed {card['seed']}: {card['instruction']}", fontsize=8)
    ax.set_xlabel("mat x (cm), towards robot's left (left of image)   -   robot at the bottom")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def card_markdown(sc, card):
    pts = sc.cfg["scoring"][card["task"]]
    ins = card["task"] == "insert_shapes"
    cols = ("inserted", "forced", "dropped", "wrong hole", "counter") if ins else ("correct", "wrong", "on counter", "off")
    L = [f"## {card['task']} — seed {card['seed']}", "",
         f"**Instruction (read/send verbatim):** {card['instruction']}", "",
         "| # | colour | shape | x (cm) | y (cm) | yaw (deg) | target | " + " | ".join(cols) + " |",
         "|---|---|---|---|---|---|---|" + "---|" * len(cols)]
    for i, c in enumerate(card["cubes"], 1):
        L.append(f"| {i} | {c['color']} | {GLYPH[c['shape']]} | {c['x'] * 100:.1f} | {c['y'] * 100:.1f} | "
                 f"{np.degrees(c['yaw']):.0f} | {target_text(sc, card, c)} | " + " | ".join("☐" for _ in cols) + " |")
    pt = ", ".join(f"{k.replace('_', ' ')} {v:+d}" for k, v in pts.items())
    L += ["", f"Points: {pt}. Safety abort: keep points so far.  Score: ______  Judge: ______", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="sort_fixed", choices=("sort_fixed", "sort_language", "insert_shapes"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000, help="first seed; cards use seed, seed+1, ...")
    ap.add_argument("--out", help="markdown file (default: stdout)")
    ap.add_argument("--figs", metavar="DIR", help="also write card_<task>_<seed>.png top views here")
    ap.add_argument("--score", nargs="+", metavar="TASK OUTCOME", help=f"score: TASK then outcomes {OUTCOMES}")
    a = ap.parse_args(argv)
    sc = Scene.load()
    if a.score:
        print(sc.score(a.score[0], a.score[1:]))
        return
    md = [f"# Layout cards ({a.task}, seeds {a.seed}–{a.seed + a.n - 1})", "",
          "Coordinates are object centres on the printed mat grid. Origin = mat corner at the robot's right, "
          "nearest edge; x towards the robot's left, y away from the robot.", "",
          "**Rule:** " + sc.rule_text(), ""]
    cards = [make_card(sc, a.task, s) for s in range(a.seed, a.seed + a.n)]
    md += [card_markdown(sc, c) for c in cards]
    if a.figs:
        import os
        os.makedirs(a.figs, exist_ok=True)
        for c in cards:
            card_figure(sc, c, os.path.join(a.figs, f"card_{a.task}_{c['seed']}.png"))
    text = "\n".join(md)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
