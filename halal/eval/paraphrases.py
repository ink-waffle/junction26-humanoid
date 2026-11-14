"""Held-out paraphrase benchmark for sort_language instructions: loader, renderer and validator."""
from __future__ import annotations

import itertools
import re
import string
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

import yaml

if TYPE_CHECKING:
    import numpy

COLORS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")
BINS = ("left", "center", "right")
SPLITS = ("dev", "test")
CATEGORIES = ("color_full", "color_respectively", "bin_synonym", "color_rest", "color_leave", "negation", "shape",
              "color_shape", "noisy")
BIJECTIVE = ("color_full", "color_respectively", "bin_synonym", "noisy")  # three colours onto three bins
SIM_CATEGORIES = ("color_full", "bin_synonym", "color_respectively")  # the realistic finals distribution
COLOR_SLOTS = ("c0", "c1", "c2")
BIN_SLOTS = ("b0", "b1", "b2")
PERMUTATIONS = tuple(itertools.product(itertools.permutations(COLORS), itertools.permutations(BINS)))
PATH = Path(__file__).with_name("paraphrases.yaml")


@dataclass(frozen=True)
class Paraphrase:
    """One template: text with colour and bin slots, its ground-truth mapping and its own surface words."""
    id: str
    split: str
    category: str
    text: str
    mapping: dict[str, str]
    words: dict[str, str]


@lru_cache(maxsize=None)
def _entries() -> tuple[Paraphrase, ...]:
    """Every entry of paraphrases.yaml, parsed once."""
    raw = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    return tuple(Paraphrase(e["id"], e["split"], e["category"], e["text"], dict(e["mapping"]),
                            dict(e.get("words") or {})) for e in raw)


def load(split: str | None = None, categories: Iterable[str] | None = None) -> list[Paraphrase]:
    """Entries of one split (None: both), restricted to the given categories (None: all)."""
    wanted = None if categories is None else set(categories)
    return [p for p in _entries() if split in (None, p.split) and (wanted is None or p.category in wanted)]


def _slot_words(colors: Sequence[str], bins: Sequence[str]) -> dict[str, str]:
    """Slot name -> canonical word for one colour and bin permutation."""
    return {**dict(zip(COLOR_SLOTS, colors)), **dict(zip(BIN_SLOTS, bins))}


def _fill(p: Paraphrase, slots: dict[str, str]) -> dict[str, str]:
    """Format arguments: every slot as this entry verbalises it, plus the capitalised twins."""
    low = {k: p.words.get(w, w) for k, w in slots.items()}
    return {**low, **{k.upper(): s[:1].upper() + s[1:] for k, s in low.items()}}


def _key(key: str, slots: dict[str, str]) -> str:
    """Mapping key with its colour slot (if any) replaced by the colour."""
    head, sep, shape = key.partition("/")
    return slots.get(head, head) + sep + shape


def render(p: Paraphrase, colors: Sequence[str], bins: Sequence[str]) -> tuple[str, dict[str, str]]:
    """Instruction text and canonical mapping of an entry for one colour and bin permutation."""
    slots = _slot_words(colors, bins)
    mapping = {_key(k, slots): slots.get(v, v) for k, v in p.mapping.items()}
    return p.text.format(**_fill(p, slots)), mapping


def _renders(p: Paraphrase) -> Iterator[tuple[str, dict[str, str]]]:
    """Distinct renderings of one entry (entries with fewer slots repeat across permutations)."""
    seen: set[str] = set()
    for colors, bins in PERMUTATIONS:
        text, mapping = render(p, colors, bins)
        if text not in seen:
            seen.add(text)
            yield text, mapping


def cases(split: str | None = None,
          categories: Iterable[str] | None = None) -> Iterator[tuple[Paraphrase, str, dict[str, str]]]:
    """Every entry rendered for every colour and bin permutation, without repeated texts."""
    for p in load(split, categories):
        for text, mapping in _renders(p):
            yield p, text, mapping


def sample(rng: numpy.random.Generator, split: str = "test",
           categories: Iterable[str] = SIM_CATEGORIES) -> tuple[str, dict[str, str]]:
    """One random rendered instruction and its mapping, for sim evaluation."""
    pool = load(split, categories)
    p = pool[int(rng.integers(len(pool)))]
    return render(p, [str(c) for c in rng.permutation(COLORS)], [str(b) for b in rng.permutation(BINS)])


def resolve(mapping: dict[str, str], color: str, shape: str) -> str | None:
    """Target bin of one object with the precedence of scene.resolve_bin; None means leave it."""
    return next((mapping[k] for k in (f"{color}/{shape}", color, shape, "default") if k in mapping), None)


def assignment(mapping: dict[str, str]) -> dict[tuple[str, str], str | None]:
    """Target bin of each of the nine object types: compare parsers on this, not on the raw dict."""
    return {(c, s): resolve(mapping, c, s) for c in COLORS for s in SHAPES}


def _text_slots(p: Paraphrase) -> set[str]:
    """Slot names as written in the text (c0, B1, ...)."""
    return {name for _, name, _, _ in string.Formatter().parse(p.text) if name is not None}


def _mapping_slots(p: Paraphrase) -> set[str]:
    """Colour and bin slots the mapping refers to."""
    parts = [part for k, v in p.mapping.items() for part in (str(k).partition("/")[0], v)]
    return {part for part in parts if part in COLOR_SLOTS + BIN_SLOTS}


def _valid_key(key: object) -> bool:
    """True for a colour slot, a shape, a colour-slot/shape pair or 'default'."""
    if not isinstance(key, str):
        return False
    head, sep, shape = key.partition("/")
    return (head in COLOR_SLOTS and shape in SHAPES) if sep else key in COLOR_SLOTS + SHAPES + ("default",)


def _check_fields(p: Paraphrase) -> list[str]:
    """Problems with id, split, category and words."""
    errs = [] if re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", str(p.id)) else ["id is not kebab-case"]
    errs += [] if p.split in SPLITS else [f"unknown split {p.split!r}"]
    errs += [] if p.category in CATEGORIES else [f"unknown category {p.category!r}"]
    errs += [f"words key {k!r} is not a canonical colour or bin" for k in p.words if k not in COLORS + BINS]
    errs += [f"words[{k!r}] is not a non-empty string" for k, v in p.words.items() if not (isinstance(v, str) and v)]
    return errs


def _check_mapping(p: Paraphrase) -> list[str]:
    """Problems with mapping keys and values."""
    errs = [] if p.mapping else ["empty mapping"]
    errs += [f"bad mapping key {k!r}" for k in p.mapping if not _valid_key(k)]
    errs += [f"mapping value {v!r} is not a bin slot or bin" for v in p.mapping.values() if v not in BIN_SLOTS + BINS]
    full = set(p.mapping) == set(COLOR_SLOTS) and len(set(p.mapping.values())) == 3
    if p.category in BIJECTIVE and not full:
        errs.append("this category needs c0, c1 and c2 on three different bins")
    return errs


def _check_slots(p: Paraphrase) -> list[str]:
    """Unknown text slots, and mapping slots the text neither names nor pins by elimination."""
    used = _text_slots(p)
    errs = [f"unknown slot {{{s}}}" for s in sorted(used) if s.lower() not in COLOR_SLOTS + BIN_SLOTS]
    named = {s.lower() for s in used}
    for kind in "cb":
        missing = sorted(s for s in _mapping_slots(p) - named if s[0] == kind)
        if missing and sum(s[0] == kind for s in named) != 2:  # the third slot is implied when the text names two
            errs.append(f"mapping uses {missing} but the text does not determine them")
    return errs


def _norm(text: str) -> str:
    """Text reduced to lower-case words, so punctuation-only edits of a template still match."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _training() -> set[str]:
    """Normalised renderings of the organisers' training templates."""
    from aaltonomous.interface.tasks import TEMPLATES
    templates = [Paraphrase("train", "dev", "color_full", t, {}, {}) for t in TEMPLATES]
    return {_norm(render(p, colors, bins)[0]) for p in templates for colors, bins in PERMUTATIONS}


def _check_render(p: Paraphrase, training: set[str]) -> list[str]:
    """Problems that only show after substitution."""
    by_text: dict[str, dict[str, str]] = {}
    errs: set[str] = set()
    for colors, bins in PERMUTATIONS:
        text, mapping = render(p, colors, bins)
        if by_text.setdefault(text, mapping) != mapping:
            errs.add("one rendered text has two different mappings")
        if _norm(text) in training:
            errs.add("renders to a training template")
    return sorted(errs)


def _problems(p: Paraphrase, training: set[str]) -> list[str]:
    """All problems of one entry; rendering is only tried on a statically sound entry."""
    try:
        return _check_fields(p) + _check_mapping(p) + _check_slots(p) or _check_render(p, training)
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as err:
        return [f"text does not render: {err!r}"]


def _shared(entries: Sequence[Paraphrase]) -> list[str]:
    """Problems spanning entries: repeated ids, repeated instructions, near-duplicates across the splits."""
    errs = {f"{i}: duplicate id" for i, n in Counter(p.id for p in entries).items() if n > 1}
    exact: dict[str, Paraphrase] = {}
    loose: dict[str, Paraphrase] = {}
    for p in entries:
        for text, _ in _renders(p):
            same, near = exact.setdefault(text, p), loose.setdefault(_norm(text), p)
            if same is not p:
                errs.add(f"{p.id}: renders the same instruction as {same.id}")
            if near.split != p.split:
                errs.add(f"{p.id}: differs from {near.id} only in punctuation, but they are in different splits")
    return sorted(errs)


def _report(entries: Sequence[Paraphrase]) -> None:
    """Print entry and rendered-case counts per category and split."""
    n = Counter((p.category, p.split) for p in entries)
    r = Counter((p.category, p.split) for p in entries for _ in _renders(p))
    rows = [(c, [t[c, s] for t in (n, r) for s in SPLITS]) for c in CATEGORIES]
    rows.append(("TOTAL", [sum(v[i] for _, v in rows) for i in range(4)]))
    print(f"{'category':<20}{'dev':>5}{'test':>6}{'all':>6}{'dev cases':>11}{'test cases':>12}{'all cases':>11}")
    for name, (dev, test, dev_cases, test_cases) in rows:
        print(f"{name:<20}{dev:>5}{test:>6}{dev + test:>6}{dev_cases:>11}{test_cases:>12}{dev_cases + test_cases:>11}")


def validate() -> int:
    """Check the whole benchmark, print the counts, and return the number of problems."""
    entries = load()
    training = _training()
    errs = [f"{p.id}: {msg}" for p in entries for msg in _problems(p, training)]
    if not errs:  # the cross-entry checks and the counts need every entry to render
        errs = _shared(entries) + [f"{c}: no {s} entries" for c in CATEGORIES for s in SPLITS if not load(s, [c])]
        _report(entries)
    print("\n".join(["OK"] if not errs else [f"ERROR {e}" for e in errs] + [f"{len(errs)} problem(s)"]))
    return len(errs)


if __name__ == "__main__":
    sys.exit(1 if validate() else 0)
