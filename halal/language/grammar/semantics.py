"""Turn a reading (units + unpaired mentions) into the object-key -> bin mapping and list what stayed doubtful."""
from __future__ import annotations

from dataclasses import dataclass, field

from halal.language.grammar import tuning
from halal.language.grammar.association import REMAINING, Reading, Unit, is_quantifier
from halal.language.grammar.lexicon import BINS, COLOURS, DEFAULT_KEY, SHAPES
from halal.language.grammar.mentions import BinMention, Mention, Negation, ObjectMention, Scope
from halal.language.grammar.tokens import Issue

Table = dict[tuple[str, str], "str | None"]


@dataclass(frozen=True)
class Interpretation:
    """Meaning of one reading: the mapping, its total cost and the doubts attached to it."""
    mapping: dict[str, str]
    cost: float
    issues: tuple[Issue, ...]
    informative: bool            # the reading says something about at least one object group


@dataclass
class _Ledger:
    """Everything the units state: bins per key, keys to leave alone, bins a key must avoid."""
    placed: dict[str, list[tuple[str, float]]] = field(default_factory=dict)     # key -> [(bin, unit cost)]
    excluded: set[str] = field(default_factory=set)
    forbidden: dict[str, set[str]] = field(default_factory=dict)
    generic_right: bool = False      # the instruction opens with only "everything -> right bin"
    issues: list[Issue] = field(default_factory=list)
    penalty: float = 0.0

    def place(self, key: str, bin_name: str | None, cost: float) -> None:
        if bin_name is None:
            self.excluded.add(key)
        else:
            self.placed.setdefault(key, []).append((bin_name, cost))

    @property
    def stated(self) -> set[str]:
        return set(self.placed) | self.excluded | set(self.forbidden)


def interpret(reading: Reading, mentions: list[Mention]) -> Interpretation:
    """Mapping stated by `reading`; cost adds what only shows once everything is put together."""
    ledger = _Ledger()
    for unit in reading.units:
        _record_unit(ledger, unit, mentions)
    ledger.generic_right = _first_unit_is_generic_right(reading, mentions)
    for index in reading.unpaired:
        _record_unpaired(ledger, mentions[index])
    _resolve_remaining_bin(ledger)
    resolved = _settle(ledger)
    _apply_forbidden(ledger, resolved)
    _charge_unpaired(ledger, resolved, [mentions[index] for index in reading.unpaired])
    mapping = _faithful_mapping(resolved, ledger.excluded)
    ledger.issues += _overlaps(mapping)
    return Interpretation(mapping, reading.cost + ledger.penalty, tuple(dict.fromkeys(ledger.issues)),
                          bool(ledger.stated))


# ---------------------------------------------------------------- statements
def _record_unit(ledger: _Ledger, unit: Unit, mentions: list[Mention]) -> None:
    objects = [mentions[index] for index in unit.objects]
    targets = unit.bins if len(unit.bins) > 1 else (unit.bins or (None,)) * len(objects)
    covers_all = any(is_quantifier(mention) for mention in objects)
    for mention, bin_name in zip(objects, targets):
        if not mention.known:
            continue
        for key, target in _statements(mention, bin_name, covers_all):
            if target is not None or unit.bins:
                ledger.place(key, target, unit.cost)
        if unit.forbidden:
            ledger.forbidden.setdefault(mention.key, set()).update(unit.forbidden)


def _statements(mention: ObjectMention, bin_name: str | None, covers_all: bool) -> list[tuple[str, str | None]]:
    """(key, bin) pairs a mention states when sent to `bin_name`; bin None means 'not these'."""
    if mention.negated is Negation.WHOLE:
        return [(mention.key, None if covers_all else bin_name)]
    if mention.negated is Negation.COLOUR:
        return [(mention.shape or DEFAULT_KEY, bin_name), (mention.key, None)]
    if mention.negated is Negation.SHAPE:
        return [(mention.colour or DEFAULT_KEY, bin_name), (mention.key, None)]
    return [(mention.key, bin_name)]


def _record_unpaired(ledger: _Ledger, mention: Mention) -> None:
    """An unpaired 'leave red' or 'except red' still says: do not move red on the strength of a broader rule."""
    if isinstance(mention, ObjectMention) and mention.known and (mention.leave or mention.negated is not Negation.NONE):
        ledger.excluded.add(mention.key)
        if mention.negated is not Negation.NONE:
            ledger.generic_right = False


def _first_unit_is_generic_right(reading: Reading, mentions: list[Mention]) -> bool:
    """'Put each object in the right bin: ...' opens the instruction; 'right' there means 'correct'."""
    if len(reading.units) < 2 or reading.units[0].bins != ("right",):
        return False
    objects = [mentions[index] for index in reading.units[0].objects]
    return all(isinstance(m, ObjectMention) and m.scope is Scope.ALL and is_quantifier(m) for m in objects)


def _resolve_remaining_bin(ledger: _Ledger) -> None:
    """'the remaining bin' is the one bin no other statement uses; otherwise those groups stay unassigned."""
    used = {bin_name for bins in ledger.placed.values() for bin_name, _ in bins} - {REMAINING}
    free = [bin_name for bin_name in BINS if bin_name not in used]
    for key, bins in list(ledger.placed.items()):
        if all(bin_name != REMAINING for bin_name, _ in bins):
            continue
        if len(free) == 1:
            ledger.placed[key] = [(free[0] if bin_name == REMAINING else bin_name, cost) for bin_name, cost in bins]
        else:
            del ledger.placed[key]
            ledger.excluded.add(key)
            ledger.issues.append(Issue(f"cannot tell which bin is 'the remaining one' for '{key}'",
                                       tuning.F_UNDETERMINED))


def _settle(ledger: _Ledger) -> dict[str, str]:
    """One bin per key: an opening 'everything in the right (= correct) bin' yields, conflicts keep the cleanest."""
    resolved: dict[str, str] = {}
    for key, bins in ledger.placed.items():
        choices = list(dict.fromkeys(bin_name for bin_name, _ in bins))
        if key == DEFAULT_KEY and ledger.generic_right:
            choices = _without_correct_bin(ledger, bins)
        if len(choices) > 1:
            best = min(bins, key=lambda entry: entry[1])[0]
            ledger.issues.append(Issue(f"conflicting bins for '{key}': {' vs '.join(choices)}; kept {best}",
                                       tuning.F_CONFLICT))
            ledger.penalty += tuning.CONFLICT
            choices = [best]
        if choices:
            resolved[key] = choices[0]
    return resolved


def _without_correct_bin(ledger: _Ledger, bins: list[tuple[str, float]]) -> list[str]:
    """Default bins once 'the right bin' is read as 'the correct bin': dropped only when every colour has a bin."""
    if all(colour in ledger.placed for colour in COLOURS):
        ledger.issues.append(Issue("read 'the right bin' for everything as 'the correct bin'", 1.0))
        return list(dict.fromkeys(bin_name for bin_name, _ in bins[1:]))
    ledger.issues.append(Issue("'the right bin' for everything may mean 'the correct bin'", tuning.F_CORRECT_BIN))
    return list(dict.fromkeys(bin_name for bin_name, _ in bins))


def _apply_forbidden(ledger: _Ledger, resolved: dict[str, str]) -> None:
    """'not left, not center' leaves right; one excluded bin leaves the group unassigned."""
    for key, banned in ledger.forbidden.items():
        allowed = [bin_name for bin_name in BINS if bin_name not in banned]
        if key in resolved:
            if resolved[key] in banned:
                ledger.issues.append(Issue(f"'{key}' is sent to a bin it is told to avoid", tuning.F_CONFLICT))
        elif len(allowed) == 1:
            resolved[key] = allowed[0]
            ledger.issues.append(Issue(f"'{key}' -> {allowed[0]} by elimination", tuning.F_ELIMINATION))
        elif key != DEFAULT_KEY:
            ledger.excluded.add(key)
            ledger.issues.append(Issue(f"'{key}' must avoid {' and '.join(sorted(banned))}; its bin is not stated",
                                       tuning.F_UNDETERMINED))


def _charge_unpaired(ledger: _Ledger, resolved: dict[str, str], unpaired: list[Mention]) -> None:
    """An unpaired mention is harmless when its colour or bin is settled elsewhere, a failure otherwise."""
    for mention in unpaired:
        if mention.soft:
            continue
        if isinstance(mention, BinMention):
            if mention.position not in resolved.values():
                ledger.penalty += tuning.UNPAIRED_BIN - tuning.REDUNDANT
                ledger.issues.append(Issue(f"no objects found for the {mention.position or 'referenced'} bin",
                                           tuning.F_UNPAIRED_BIN))
        elif mention.key not in ledger.stated:
            ledger.penalty += tuning.UNPAIRED_OBJECT - tuning.REDUNDANT
            ledger.issues.append(Issue(f"no bin found for '{mention.key}'", tuning.F_UNPAIRED_OBJECT))


# ---------------------------------------------------------------- mapping
def _lookup(keys: dict[str, str | None], colour: str, shape: str) -> str | None:
    """Most specific key wins: colour/shape, colour, shape, default."""
    for key in (f"{colour}/{shape}", colour, shape, DEFAULT_KEY):
        if key in keys:
            return keys[key]
    return None


def _faithful_mapping(resolved: dict[str, str], excluded: set[str]) -> dict[str, str]:
    """The stated keys when they resolve as intended, else explicit keys that keep excluded groups unmoved."""
    stated: dict[str, str | None] = {**{key: None for key in excluded}, **resolved}
    table = {(colour, shape): _lookup(stated, colour, shape) for colour in COLOURS for shape in SHAPES}
    if all(_lookup(resolved, colour, shape) == target for (colour, shape), target in table.items()):
        return dict(resolved)
    return _compile(table)


def _compile(table: Table) -> dict[str, str]:
    """Smallest explicit mapping for a table: colour keys, then shape keys, then colour/shape keys."""
    mapping: dict[str, str] = {}
    for colour in COLOURS:
        bins = {table[colour, shape] for shape in SHAPES}
        if len(bins) == 1 and None not in bins:
            mapping[colour] = bins.pop()
    open_colours = [colour for colour in COLOURS if colour not in mapping]
    for shape in SHAPES:
        bins = {table[colour, shape] for colour in open_colours}
        if len(bins) == 1 and None not in bins:
            mapping[shape] = bins.pop()
    for colour in open_colours:
        for shape in SHAPES:
            if shape not in mapping and table[colour, shape] is not None:
                mapping[f"{colour}/{shape}"] = table[colour, shape]
    return mapping


def _overlaps(mapping: dict[str, str]) -> list[Issue]:
    """A colour rule and a shape rule that disagree about an object type neither rule singles out."""
    return [Issue(f"'{colour}' -> {mapping[colour]} and '{shape}' -> {mapping[shape]} disagree on {colour} {shape}s; "
                  f"colour wins", tuning.F_OVERLAP)
            for colour in COLOURS for shape in SHAPES
            if colour in mapping and shape in mapping and mapping[colour] != mapping[shape]
            and f"{colour}/{shape}" not in mapping]
