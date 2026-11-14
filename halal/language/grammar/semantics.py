"""Turn a reading into rules over the nine object types, settle their precedence and emit the mapping keys."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from halal.language.grammar import tuning
from halal.language.grammar.association import BIN_OF, REMAINING, Reading, Unit, is_quantifier
from halal.language.grammar.lexicon import BINS, COLOURS, DEFAULT_KEY, SHAPES
from halal.language.grammar.mentions import BinMention, Mention, Negation, ObjectMention, Scope
from halal.language.grammar.tokens import Issue

Atom = tuple[str, str]                      # (colour, shape): one of the nine object types
Table = dict[Atom, "str | None"]
ATOMS: tuple[Atom, ...] = tuple((colour, shape) for colour in COLOURS for shape in SHAPES)


@dataclass(frozen=True)
class Interpretation:
    """Meaning of one reading: the mapping, its total cost and the doubts attached to it."""
    mapping: dict[str, str]
    cost: float
    issues: tuple[Issue, ...]
    informative: bool            # the reading says something about at least one object group


@dataclass(frozen=True)
class Rule:
    """One statement: these object types go to this bin; bin None keeps them on the table."""
    atoms: frozenset[Atom]
    bin: str | None
    key: str | None = None       # mapping key that states the rule as written, when there is one
    weight: int = 0              # strength when a colour rule and a shape rule claim the same object type
    cost: float = 0.0
    corrects: bool = False       # stated as a correction: replaces earlier rules for the same object types


@dataclass
class _Ledger:
    """Rules and constraints collected from a reading, with the doubts and costs found on the way."""
    rules: list[Rule] = field(default_factory=list)
    bans: list[tuple[frozenset[Atom], frozenset[str]]] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    penalty: float = 0.0


def interpret(reading: Reading, mentions: list[Mention]) -> Interpretation:
    """Mapping stated by `reading`; its cost adds what only shows once everything is put together."""
    ledger = _Ledger()
    _collect(ledger, _with_remaining_bin(_with_companion_bins(reading, mentions), ledger), mentions)
    _drop_correct_bin_opening(ledger, reading, mentions)
    _apply_bans(ledger)
    table = {atom: _winner(ledger, atom) for atom in ATOMS}
    ruled = {atom for atom in ATOMS if any(atom in rule.atoms for rule in ledger.rules)}
    _check_bans(ledger, table)
    _charge_unpaired(ledger, ruled, [mentions[index] for index in reading.unpaired])
    issues = tuple(dict.fromkeys(ledger.issues))
    return Interpretation(_mapping(ledger.rules, table), reading.cost + ledger.penalty, issues,
                          bool(ledger.rules or ledger.bans))


# ---------------------------------------------------------------- rules from the reading
def _with_companion_bins(reading: Reading, mentions: list[Mention]) -> Reading:
    """Replace 'the bin of yellow' by the bin another unit gives yellow; without one, that unit names no bin."""
    units = list(reading.units)
    if not any(bin_name.startswith(BIN_OF) for unit in units for bin_name in unit.bins):
        return reading
    for _ in units:
        stated = {mentions[index].key: bin_name for unit in units
                  for index, bin_name in zip(unit.objects, unit.bins * (len(unit.objects) if len(unit.bins) == 1 else 1))
                  if not bin_name.startswith(BIN_OF)}
        units = [replace(unit, bins=tuple(stated.get(b[len(BIN_OF):], b) if b.startswith(BIN_OF) else b
                                          for b in unit.bins)) for unit in units]
    return replace(reading, units=tuple(replace(unit, bins=tuple(b for b in unit.bins if not b.startswith(BIN_OF)))
                                        for unit in units))


def _with_remaining_bin(reading: Reading, ledger: _Ledger) -> Reading:
    """Replace 'the remaining bin' by the one bin no other unit uses; without one, those units name no bin."""
    used = {bin_name for unit in reading.units for bin_name in unit.bins} - {REMAINING}
    free = [bin_name for bin_name in BINS if bin_name not in used]
    if len(free) != 1 and any(REMAINING in unit.bins for unit in reading.units):
        ledger.issues.append(Issue("cannot tell which bin 'the remaining one' is", tuning.F_UNDETERMINED))
    units = tuple(replace(unit, bins=tuple(free[0] if b == REMAINING else b for b in unit.bins))
                  if len(free) == 1 else replace(unit, bins=tuple(b for b in unit.bins if b != REMAINING))
                  for unit in reading.units)
    return replace(reading, units=units)


def _collect(ledger: _Ledger, reading: Reading, mentions: list[Mention]) -> None:
    """Walk units and unpaired mentions in text order; exceptions cut into the nearest group of rules."""
    steps = [(unit.objects[0], unit) for unit in reading.units] + [(index, None) for index in reading.unpaired]
    group: list[int] = []            # indices into ledger.rules of the latest unit or 'leave' list
    pending: frozenset[Atom] = frozenset()
    residual = False                 # an unpaired 'the other shapes ...' makes every later rule a fallback
    topics: dict[int, frozenset[Atom]] = {}      # 'only the cubes: ...' limits the rules it introduces
    opened: set[int] = set()
    for index, unit in sorted(steps, key=lambda step: step[0]):
        mention = mentions[index]
        if unit is not None:
            opened.add(mention.topic)
            group = _add_rules(ledger, _within(topics.get(mention.topic), _unit_rules(ledger, unit, mentions, residual)))
            pending = _cut(ledger, group, pending | _excepted(unit, mentions))
        elif not isinstance(mention, ObjectMention) or not mention.known:
            continue
        elif mention.negated is Negation.WHOLE:
            pending = _cut(ledger, group, _selector(mention)) if group else pending | _selector(mention)
        elif mention.leave:
            group = _add_rules(ledger, _within(topics.get(mention.topic), [_rule(mention, None, 0.0, residual)]))
        elif mention.scope is Scope.REST:
            residual = True
        elif mention.topic not in opened and (not mention.soft or mention.negated is not Negation.NONE):
            topics[mention.topic] = topics.get(mention.topic, frozenset()) | _selector(mention)


def _unit_rules(ledger: _Ledger, unit: Unit, mentions: list[Mention], residual: bool) -> list[Rule]:
    """Rules a unit states; bins it must avoid are recorded as bans; a unit without a bin keeps its objects put."""
    objects = [mentions[index] for index in unit.objects]
    known = [mention for mention in objects if isinstance(mention, ObjectMention) and mention.known]
    targets = unit.bins if len(unit.bins) > 1 else (unit.bins or (None,)) * len(objects)
    excepted = _excepted(unit, mentions)
    if unit.forbidden:
        ledger.bans += [(_selector(mention), frozenset(unit.forbidden)) for mention in known]
    if not unit.bins and unit.forbidden:
        return []
    return [replace(_rule(mention, bin_name, unit.cost, residual), corrects=unit.corrects)
            for mention, bin_name in zip(objects, targets)
            if mention in known and not (excepted and mention.negated is Negation.WHOLE)]


def _excepted(unit: Unit, mentions: list[Mention]) -> frozenset[Atom]:
    """Object types a unit takes out of its own 'everything': 'everything except the red ones goes left'."""
    objects = [mentions[index] for index in unit.objects]
    if not any(is_quantifier(mention) for mention in objects):
        return frozenset()
    return frozenset().union(*(_selector(m) for m in objects if m.known and m.negated is Negation.WHOLE))


def _within(topic: frozenset[Atom] | None, rules: list[Rule]) -> list[Rule]:
    """Limit rules to the object types their introduction names, when they overlap it at all."""
    return [replace(rule, atoms=rule.atoms & topic) if topic and rule.atoms & topic else rule for rule in rules]


def _add_rules(ledger: _Ledger, rules: list[Rule]) -> list[int]:
    first = len(ledger.rules)
    ledger.rules += rules
    return list(range(first, len(ledger.rules)))


def _cut(ledger: _Ledger, group: list[int], excepted: frozenset[Atom]) -> frozenset[Atom]:
    """Take `excepted` object types out of the broader rules of a group; nothing stays pending."""
    for index in group:
        rule = ledger.rules[index]
        if excepted < rule.atoms:
            ledger.rules[index] = replace(rule, atoms=rule.atoms - excepted)
    return frozenset()


def _selector(mention: ObjectMention) -> frozenset[Atom]:
    """Object types a mention denotes; a negated colour or shape selects its complement."""
    colours = {mention.colour} if mention.colour else set(COLOURS)
    shapes = {mention.shape} if mention.shape else set(SHAPES)
    if mention.negated is Negation.COLOUR:
        colours = set(COLOURS) - colours
    if mention.negated is Negation.SHAPE:
        shapes = set(SHAPES) - shapes
    return frozenset((colour, shape) for colour in colours for shape in shapes)


def _rule(mention: ObjectMention, bin_name: str | None, cost: float, residual: bool) -> Rule:
    """The rule a mention states; its written key is the group the negated part is taken from."""
    key = {Negation.COLOUR: mention.shape or DEFAULT_KEY, Negation.SHAPE: mention.colour or DEFAULT_KEY}.get(
        mention.negated, mention.key)
    weight = (tuning.W_UNIVERSAL * mention.universal + tuning.W_QUANTIFIED * (mention.scope is Scope.ALL)
              - tuning.W_RESIDUAL * (residual or mention.scope is Scope.REST))
    return Rule(_selector(mention), bin_name, key if bin_name else None, weight, cost)


def _drop_correct_bin_opening(ledger: _Ledger, reading: Reading, mentions: list[Mention]) -> None:
    """'Put each object in the right bin: ...' opens the instruction; 'right' there means 'correct'."""
    if len(reading.units) < 2 or reading.units[0].bins != ("right",):
        return
    opening = [mentions[index] for index in reading.units[0].objects]
    if not all(isinstance(m, ObjectMention) and m.scope is Scope.ALL and is_quantifier(m) for m in opening):
        return
    generic = next((rule for rule in ledger.rules if rule.key == DEFAULT_KEY and rule.bin == "right"), None)
    if generic is None or generic.atoms != frozenset(ATOMS):
        return
    others = [rule for rule in ledger.rules if rule is not generic and rule.bin]
    if all(any(atom in rule.atoms for rule in others) for atom in ATOMS):
        ledger.rules.remove(generic)
        ledger.issues.append(Issue("read 'the right bin' for everything as 'the correct bin'", 1.0))
    else:
        ledger.issues.append(Issue("'the right bin' for everything may mean 'the correct bin'", tuning.F_CORRECT_BIN))


def _apply_bans(ledger: _Ledger) -> None:
    """'not left, not center' leaves right; a single excluded bin leaves the group where it is."""
    for atoms, banned in _merged_bans(ledger.bans).items():
        allowed = [bin_name for bin_name in BINS if bin_name not in banned]
        if any(rule.atoms == atoms and rule.bin for rule in ledger.rules) or atoms == frozenset(ATOMS):
            continue
        name = _name(atoms)
        if len(allowed) == 1:
            ledger.rules.append(Rule(atoms, allowed[0], _key(atoms)))
            ledger.issues.append(Issue(f"'{name}' -> {allowed[0]} by elimination", tuning.F_ELIMINATION))
        else:
            ledger.rules.append(Rule(atoms, None))
            ledger.issues.append(Issue(f"'{name}' must avoid {' and '.join(sorted(banned))}; its bin is not stated",
                                       tuning.F_UNDETERMINED))


def _merged_bans(bans: list[tuple[frozenset[Atom], frozenset[str]]]) -> dict[frozenset[Atom], frozenset[str]]:
    merged: dict[frozenset[Atom], frozenset[str]] = {}
    for atoms, banned in bans:
        merged[atoms] = merged.get(atoms, frozenset()) | banned
    return merged


def _check_bans(ledger: _Ledger, table: Table) -> None:
    for atoms, banned in _merged_bans(ledger.bans).items():
        if any(table[atom] in banned for atom in atoms):
            ledger.issues.append(Issue(f"'{_name(atoms)}' is sent to a bin it is told to avoid", tuning.F_CONFLICT))


def _key(atoms: frozenset[Atom]) -> str | None:
    """Mapping key that selects exactly these object types, if one exists."""
    colours, shapes = {colour for colour, _ in atoms}, {shape for _, shape in atoms}
    if len(atoms) == len(ATOMS):
        return DEFAULT_KEY
    if len(atoms) != len(colours) * len(shapes) or (len(colours) > 1 and len(shapes) > 1):
        return None
    if len(colours) == 1 and len(shapes) == 1:
        return f"{min(colours)}/{min(shapes)}"
    whole_row = len(shapes) == len(SHAPES) or len(colours) == len(COLOURS)
    return (min(colours) if len(colours) == 1 else min(shapes)) if whole_row else None


def _name(atoms: frozenset[Atom]) -> str:
    return _key(atoms) or " + ".join(sorted(f"{colour}/{shape}" for colour, shape in atoms))


# ---------------------------------------------------------------- precedence
def _winner(ledger: _Ledger, atom: Atom) -> str | None:
    """Bin for one object type: the most specific rule wins; cues settle colour rule against shape rule."""
    applicable = [rule for rule in ledger.rules if atom in rule.atoms]
    if not applicable:
        return None
    size = min(len(rule.atoms) for rule in applicable)
    groups: dict[frozenset[Atom], list[Rule]] = {}
    for rule in applicable:
        if len(rule.atoms) == size:
            groups.setdefault(rule.atoms, []).append(rule)
    return _settle(ledger, _strongest(ledger, list(groups.values())))


def _strongest(ledger: _Ledger, groups: list[list[Rule]]) -> list[Rule]:
    """Among equally specific rule groups: higher cue weight wins, then the colour rule (the mapping's own order)."""
    ranked = sorted(groups, key=lambda group: (max(rule.weight for rule in group), _by_colour(group[0])), reverse=True)
    if len(ranked) > 1 and {rule.bin for rule in ranked[0]} != {rule.bin for rule in ranked[1]}:
        lead = max(rule.weight for rule in ranked[0]) - max(rule.weight for rule in ranked[1])
        first, second = (_name(group[0].atoms) for group in ranked[:2])
        if lead < tuning.W_DECISIVE:
            ledger.issues.append(Issue(f"rules for '{first}' and '{second}' overlap; '{first}' wins",
                                       tuning.F_OVERLAP_CUED if lead else tuning.F_OVERLAP))
    return ranked[0]


def _by_colour(rule: Rule) -> bool:
    return len({colour for colour, _ in rule.atoms}) == 1


def _settle(ledger: _Ledger, rules: list[Rule]) -> str | None:
    """One bin for rules with the same selector: a placement beats 'leave'; two placements are a conflict."""
    bins = list(dict.fromkeys(rule.bin for rule in rules if rule.bin))
    corrections = [rule.bin for rule in rules if rule.corrects and rule.bin]
    if corrections:
        if len(bins) > 1:
            ledger.issues.append(Issue(f"'{_name(rules[0].atoms)}' -> {corrections[-1]} after a self-correction",
                                       tuning.F_CORRECTED))
        return corrections[-1]
    if len(bins) > 1:
        best = min((rule for rule in rules if rule.bin), key=lambda rule: rule.cost).bin
        issue = Issue(f"conflicting bins for '{_name(rules[0].atoms)}': {' vs '.join(bins)}; kept {best}",
                      tuning.F_CONFLICT)
        if issue not in ledger.issues:
            ledger.issues.append(issue)
            ledger.penalty += tuning.CONFLICT
        return best
    return bins[0] if bins else None


def _charge_unpaired(ledger: _Ledger, ruled: set[Atom], unpaired: list[Mention]) -> None:
    """An unpaired mention is harmless when settled elsewhere, a loose end when its sentence names no bin."""
    used = {rule.bin for rule in ledger.rules}
    for mention in unpaired:
        if mention.soft or (isinstance(mention, BinMention) and mention.position in used):
            continue
        if isinstance(mention, ObjectMention) and _selector(mention) <= ruled:
            continue
        is_bin = isinstance(mention, BinMention)
        name = f"the {mention.position or 'referenced'} bin" if is_bin else f"'{mention.key}'"
        if mention.descriptive:
            ledger.penalty += tuning.DESCRIPTIVE - tuning.REDUNDANT
            ledger.issues.append(Issue(f"{name} is mentioned without an assignment", tuning.F_DESCRIPTIVE))
        elif is_bin:
            ledger.penalty += tuning.UNPAIRED_BIN - tuning.REDUNDANT
            ledger.issues.append(Issue(f"no objects found for {name}", tuning.F_UNPAIRED_BIN))
        else:
            ledger.penalty += tuning.UNPAIRED_OBJECT - tuning.REDUNDANT
            ledger.issues.append(Issue(f"no bin found for {name}", tuning.F_UNPAIRED_OBJECT))


# ---------------------------------------------------------------- mapping keys
def _lookup(keys: dict[str, str], atom: Atom) -> str | None:
    """Most specific key wins: colour/shape, colour, shape, default."""
    colour, shape = atom
    for key in (f"{colour}/{shape}", colour, shape, DEFAULT_KEY):
        if key in keys:
            return keys[key]
    return None


def _reproduces(keys: dict[str, str], table: Table) -> bool:
    return all(_lookup(keys, atom) == target for atom, target in table.items())


def _mapping(rules: list[Rule], table: Table) -> dict[str, str]:
    """Keys as written when they give the table; else written keys patched per object type, or rebuilt, if smaller."""
    written: dict[str, str] = {}
    for rule in rules:
        if rule.key and rule.bin:
            written.setdefault(rule.key, rule.bin)
    if _reproduces(written, table):
        return written
    patched = {**written, **{f"{c}/{s}": target for (c, s), target in table.items()
                             if target and _lookup(written, (c, s)) != target}}
    rebuilt = _rebuilt(table, written.get(DEFAULT_KEY))
    return patched if _reproduces(patched, table) and len(patched) <= len(rebuilt) else rebuilt


def _rebuilt(table: Table, default: str | None) -> dict[str, str]:
    """Smallest explicit mapping for a table: default (if every type has a bin), colours, shapes, single types."""
    mapping = {DEFAULT_KEY: default} if default and None not in table.values() else {}
    for colour in COLOURS:
        bins = {table[colour, shape] for shape in SHAPES}
        if len(bins) == 1 and None not in bins:
            mapping[colour] = bins.pop()
    open_colours = [colour for colour in COLOURS if colour not in mapping]
    for shape in SHAPES:
        bins = {table[colour, shape] for colour in open_colours}
        if len(bins) == 1 and None not in bins:
            mapping[shape] = bins.pop()
    for atom, target in table.items():
        if target and _lookup(mapping, atom) != target:
            mapping["/".join(atom)] = target
    return {key: target for key, target in mapping.items() if key == DEFAULT_KEY or target != default
            or not _reproduces({k: v for k, v in mapping.items() if k != key}, table)}
