"""Pair object mentions with bin mentions: beam search over segmentations of the mention sequence into units."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator

from halal.language.grammar import tuning
from halal.language.grammar.lexicon import DEFAULT_KEY, Tag
from halal.language.grammar.mentions import BinMention, Gap, Mention, Negation, ObjectMention, Scope, gap_between
from halal.language.grammar.tokens import Issue, Token

REMAINING = "?"      # placeholder for "the remaining bin", resolved once every other bin is known
BIN_OF = "="         # prefix of a placeholder for "the bin of <object key>": "blue goes with yellow"
OPPOSITE = {"left": "right", "right": "left"}


@dataclass(frozen=True)
class Unit:
    """One clause's assignment: every object to one bin, or parallel lists paired in order."""
    objects: tuple[int, ...]
    bins: tuple[str, ...]
    forbidden: tuple[str, ...] = ()
    cost: float = 0.0
    issues: tuple[Issue, ...] = ()
    corrects: bool = False       # stated after "sorry" / "actually": overrides earlier rules for the same objects


@dataclass(frozen=True)
class Reading:
    """A segmentation of the mention sequence into units and unpaired mentions."""
    units: tuple[Unit, ...] = ()
    unpaired: tuple[int, ...] = ()
    cost: float = 0.0            # what the units cost; loose ends are charged once the whole meaning is known
    bound: float = 0.0           # cost plus the search's estimate for the loose ends, used to rank partial readings


@dataclass(frozen=True)
class _Move:
    """One way to consume mentions from a position: the units it adds and their cost."""
    end: int
    units: tuple[Unit, ...]
    cost: float


@dataclass
class _Sequence:
    """The mention sequence with what the search looks up repeatedly."""
    mentions: list[Mention]
    gaps: list[Gap]                  # gaps[i] stands between mentions i and i + 1
    cued: list[bool]                 # mention i shares a sentence with "respectively"
    turns: list[bool] = field(init=False)        # turns[i]: mentions i and i + 1 are of different kinds
    plain_pairs: list[bool] = field(init=False)  # plain_pairs[i]: mentions i, i + 1 form a plain "red in the left bin"
    moves: dict[tuple[int, str | None], list[_Move]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kinds = [isinstance(mention, ObjectMention) for mention in self.mentions]
        self.turns = [a != b for a, b in zip(kinds, kinds[1:])]
        self.plain_pairs = [_opens_plain_pair(self, index) for index in range(len(self.mentions))]


def candidate_readings(mentions: list[Mention], tokens: list[Token]) -> list[Reading]:
    """Complete readings worth interpreting, from a search that forgives loose ends and one that does not."""
    sequence = _Sequence(mentions, [gap_between(tokens, a, b) for a, b in zip(mentions, mentions[1:])],
                         _cued(mentions, tokens))
    found = _search(sequence, forgiving=True) + _search(sequence, forgiving=False)
    return list({(reading.units, reading.unpaired): reading for reading in found}.values())


def _search(sequence: _Sequence, forgiving: bool) -> list[Reading]:
    """Beam search over segmentations; `forgiving` assumes unpaired mentions will turn out settled elsewhere."""
    count = len(sequence.mentions)
    width = max(2, tuning.BEAM_WIDTH * tuning.BEAM_FULL_UNTIL // max(count, tuning.BEAM_FULL_UNTIL))
    frontier: list[list[Reading]] = [[] for _ in range(count + 1)]
    frontier[0].append(Reading())
    for start in range(count):
        skip = _skip_charges(sequence, start, forgiving)
        for partial in sorted(frontier[start], key=lambda reading: reading.bound)[:width]:
            frontier[start + 1].append(Reading(partial.units, partial.unpaired + (start,), partial.cost + skip[0],
                                               partial.bound + skip[1]))
            for move in _moves(sequence, start, _previous_bin(partial)):
                frontier[move.end].append(Reading(partial.units + move.units, partial.unpaired,
                                                  partial.cost + move.cost, partial.bound + move.cost))
            frontier[start + 1] += _amended(partial, sequence.mentions[start], start)
    return sorted(frontier[-1], key=lambda reading: reading.bound)[:tuning.MAX_CANDIDATES]


def _skip_charges(sequence: _Sequence, index: int, forgiving: bool) -> tuple[float, float]:
    """(cost, bound) added by leaving a mention unpaired: skipping a plain pair always costs, the rest is estimated."""
    mention = sequence.mentions[index]
    if mention.soft:
        return 0.0, 0.0
    cost = tuning.SKIPPED_PAIR - tuning.REDUNDANT if sequence.plain_pairs[index] else 0.0
    worst = tuning.UNPAIRED_OBJECT if isinstance(mention, ObjectMention) else tuning.UNPAIRED_BIN
    return cost, cost + (tuning.REDUNDANT if forgiving else worst)


def _cued(mentions: list[Mention], tokens: list[Token]) -> list[bool]:
    """Per mention: does its sentence contain 'respectively' / 'in that order' / 'from left to right'?"""
    sentence, cued_sentences, sentence_of = 0, set(), []
    for token in tokens:
        sentence += token.tag is Tag.BREAK
        sentence_of.append(sentence)
        if token.tag is Tag.RESPECTIVELY:
            cued_sentences.add(sentence)
    return [sentence_of[mention.start] in cued_sentences for mention in mentions]


def _moves(sequence: _Sequence, start: int, previous_bin: str | None) -> list[_Move]:
    """Ways to pair the mention at `start`: attach it to the last bin, send it with a companion, or open a unit."""
    key = (start, previous_bin)
    if key not in sequence.moves:
        singles = [_attached(sequence.mentions[start], start, previous_bin), _companion(sequence, start)]
        moves = [_Move(start + 1, (unit,), unit.cost) for unit in singles if unit]
        sequence.moves[key] = moves + _unit_moves(sequence, start, previous_bin)
    return sequence.moves[key]


def _unit_moves(sequence: _Sequence, start: int, previous_bin: str | None) -> list[_Move]:
    """Units (plain or nested) that start at `start`, for every span with a single change of mention kind."""
    moves: list[_Move] = []
    boundary = _boundary_cost(sequence, start)
    for end in range(start + 2, min(len(sequence.mentions), start + tuning.MAX_UNIT_SPAN) + 1):
        changes = sum(sequence.turns[start:end - 1])
        if changes != 1:
            if changes:
                break
            continue
        unit = _unit(sequence, start, end, previous_bin)
        nested = _nested_units(sequence, start, end, previous_bin)
        for units in ([(unit,)] if unit else []) + nested:
            moves.append(_Move(end, units, sum(u.cost for u in units) + boundary))
    return moves


def _attached(mention: Mention, index: int, previous_bin: str | None) -> Unit | None:
    """'blue too', 'same for blue': the mention joins the bin of the unit before it."""
    if not isinstance(mention, ObjectMention) or not mention.add or not previous_bin:
        return None
    note = Issue(f"'{mention.key}' joins the previous group's bin", tuning.F_ATTACHED)
    return Unit((index,), (previous_bin,), cost=tuning.ATTACHED, issues=(note,))


def _opens_plain_pair(sequence: _Sequence, start: int) -> bool:
    """Mentions start, start + 1 are a self-contained 'red in the left bin', not items of longer lists."""
    mentions, gaps, turns = sequence.mentions, sequence.gaps, sequence.turns
    if start + 1 >= len(mentions) or not turns[start] or mentions[start].soft or mentions[start + 1].soft:
        return False
    listed_before = start > 0 and not turns[start - 1] and gaps[start - 1].list_like
    listed_after = start + 2 < len(mentions) and not turns[start + 1] and gaps[start + 1].list_like
    plain = _transition_charge(gaps[start], isinstance(mentions[start], BinMention))[0] == 0
    return plain and not listed_before and not listed_after


def _nested_units(sequence: _Sequence, start: int, end: int, previous_bin: str | None) -> list[tuple[Unit, Unit]]:
    """'what is left [after red goes left] goes right': inner objects take the first bin, outer ones the second."""
    turn = start + sequence.turns[start:end - 1].index(True)
    breaks = [i for i in range(start, turn) if not sequence.gaps[i].list_like]
    objects_first = isinstance(sequence.mentions[start], ObjectMention)
    if not objects_first or end - turn != 3 or len(breaks) != 1 or sequence.gaps[turn + 1].list_like:
        return []
    inner = _unit(sequence, breaks[0] + 1, turn + 2, previous_bin)
    last = sequence.mentions[turn + 2]
    outer_bin = _position(last, previous_bin)
    if inner is None or len(inner.bins) != 1 or outer_bin in (None, *inner.bins) or last.negated:
        return []
    note = Issue("one clause is nested inside another", tuning.F_NESTED)
    return [(inner, Unit(tuple(range(start, breaks[0] + 1)), (outer_bin,), cost=tuning.NESTED, issues=(note,)))]


def _companion(sequence: _Sequence, start: int) -> Unit | None:
    """'blue goes with yellow': the mention at `start` takes the bin of the object mention right after it."""
    pair = sequence.mentions[start:start + 2]
    if len(pair) < 2 or not all(isinstance(mention, ObjectMention) and mention.known for mention in pair):
        return None
    gap = sequence.gaps[start]
    if not gap.has(Tag.COMPANION) or gap.has(Tag.BREAK, Tag.COMMA, Tag.CLAUSE):
        return None
    note = Issue(f"'{pair[0].key}' goes wherever '{pair[1].key}' goes", tuning.F_COMPANION)
    return Unit((start,), (BIN_OF + pair[1].key,), cost=tuning.COMPANION, issues=(note,))


def _amended(partial: Reading, mention: Mention, index: int) -> list[Reading]:
    """'..., sorry, I meant the right bin' / '... I meant blue': rebind the bin or the objects of the last unit."""
    last = partial.units[-1] if partial.units else None
    if not mention.corrects or last is None or len(last.bins) != 1 or last.bins[0].startswith((REMAINING, BIN_OF)):
        return []
    if isinstance(mention, BinMention) and not mention.position:
        return []
    note = Issue("a self-correction replaced part of the previous assignment", tuning.F_CORRECTED)
    change = {"objects": (index,)} if isinstance(mention, ObjectMention) else {"bins": (mention.position,)}
    unit = replace(last, cost=last.cost + tuning.AMENDED, issues=last.issues + (note,), corrects=True, **change)
    return [Reading(partial.units[:-1] + (unit,), partial.unpaired, partial.cost + tuning.AMENDED,
                    partial.bound + tuning.AMENDED)]


def _previous_bin(partial: Reading) -> str | None:
    """The single stated bin of the last unit, for 'the same bin' and 'blue too'."""
    bins = partial.units[-1].bins if partial.units else ()
    return bins[0] if len(bins) == 1 and bins[0] != REMAINING and not bins[0].startswith(BIN_OF) else None


def _boundary_cost(sequence: _Sequence, start: int) -> float:
    """Clauses are normally set apart by punctuation or 'and'; a verb or preposition should not be left over."""
    if start == 0 or sequence.gaps[start - 1].separated:
        return 0.0
    dangling_link = sequence.gaps[start - 1].has(Tag.LINK, Tag.ARROW)
    return tuning.LINK_BETWEEN_CLAUSES if dangling_link else tuning.UNSEPARATED


def _unit(sequence: _Sequence, start: int, end: int, previous_bin: str | None) -> Unit | None:
    """The unit over mentions [start, end), a span with one change of kind; None when its bins make no sense."""
    members = sequence.mentions[start:end]
    turn = start + sequence.turns[start:end - 1].index(True)
    objects = tuple(i for i in range(start, end) if isinstance(sequence.mentions[i], ObjectMention))
    bins = [mention for mention in members if isinstance(mention, BinMention)]
    positions = [_position(mention, previous_bin) for mention in bins]
    allowed = _unique(p for p, mention in zip(positions, bins) if not mention.negated)
    forbidden = _unique(p for p, mention in zip(positions, bins) if mention.negated)
    broken = any(sequence.gaps[i].has(Tag.BREAK) for i in range(start, end - 1) if i != turn)
    if None in positions or broken or (len(allowed) > 1 and forbidden):
        return None
    if len(allowed) > 1 and len(allowed) != len(objects):
        note = Issue(f"several bins named for one group ({' / '.join(allowed)}): not placed", tuning.F_UNDETERMINED)
        return Unit(objects, (), cost=tuning.UNDECIDED, issues=(note,))
    cost, issues = _unit_cost(sequence, start, end, turn, len(allowed) > 1)
    return Unit(objects, allowed, forbidden, cost, issues, corrects=any(mention.corrects for mention in members))


def _position(mention: BinMention, previous_bin: str | None) -> str | None:
    if mention.anaphor == "same":
        return previous_bin
    if mention.anaphor == "opposite":
        return OPPOSITE.get(previous_bin or "")
    return REMAINING if mention.anaphor == "remaining" else mention.position


def _unique(items: Iterator[str | None]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _unit_cost(sequence: _Sequence, start: int, end: int, turn: int,
               distributes: bool) -> tuple[float, tuple[Issue, ...]]:
    """Cost of reading mentions [start, end) as one unit, with the doubts that cost stands for."""
    members = sequence.mentions[start:end]
    charges = [_run_charge(sequence, i) for i in range(start, end - 1) if i != turn]
    charges.append(_transition_charge(sequence.gaps[turn], isinstance(members[0], BinMention)))
    if distributes and not any(sequence.cued[start:end]):
        charges.append((tuning.DISTRIBUTION, Issue("lists paired in order without 'respectively'",
                                                   tuning.F_DISTRIBUTION)))
    charges.append(_object_charge([m for m in members if isinstance(m, ObjectMention)]))
    cost = sum(charge for charge, _ in charges)
    return cost, tuple(dict.fromkeys(issue for _, issue in charges if issue))


def _run_charge(sequence: _Sequence, index: int) -> tuple[float, Issue | None]:
    """A list should hold only commas and conjunctions; a contrasted bin ('left, not right') is exempt."""
    gap = sequence.gaps[index]
    first, second = sequence.mentions[index:index + 2]
    restated = isinstance(first, BinMention) and (first.negated or second.negated or first.position == second.position)
    if gap.list_like or restated:
        return 0.0, None
    if gap.directional:
        return tuning.RUN_DIRECTED, Issue("a list item has a destination of its own", tuning.F_RUN_DIRECTED)
    if gap.has(Tag.ARROW, Tag.CLAUSE):
        return tuning.RUN_BROKEN, Issue("a list runs across an arrow or 'but'", tuning.F_RUN_BROKEN)
    if gap.tags.isdisjoint({Tag.LINK, Tag.LEAVE, Tag.NOT, Tag.VOID}):
        return tuning.RUN_UNKNOWN, Issue("a list contains unknown words", tuning.F_RUN_UNKNOWN)
    return tuning.RUN_MIXED, Issue("a list is interrupted by other words", tuning.F_RUN_MIXED)


def _transition_charge(gap: Gap, bin_first: bool) -> tuple[float, Issue | None]:
    """Objects and their bin are normally joined by a verb, preposition or arrow, not split by punctuation."""
    linked = gap.has(Tag.LINK, Tag.ARROW, Tag.NOT, Tag.LEAVE, Tag.COMPANION)
    if gap.has(Tag.BREAK):
        return tuning.ACROSS_BREAK, Issue("objects and bin stand in different sentences", tuning.F_ACROSS_BREAK)
    if gap.has(Tag.CLAUSE):
        return tuning.ACROSS_CLAUSE, Issue("objects and bin stand on two sides of 'but'", tuning.F_ACROSS_CLAUSE)
    if gap.has(Tag.AND) and not linked:
        return tuning.ACROSS_AND, Issue("only 'and' joins objects and bin", tuning.F_ACROSS_AND)
    if gap.has(Tag.COMMA) and not linked:
        return tuning.ACROSS_COMMA, Issue("only a comma joins objects and bin", tuning.F_ACROSS_COMMA)
    if bin_first and gap.directional and not gap.has(Tag.ARROW):
        return tuning.BIN_FIRST_DIRECTIONAL, Issue("bin named before a preposition that points to it",
                                                   tuning.F_BIN_FIRST_DIRECTIONAL)
    if gap.has(Tag.OTHER):
        return 0.0, Issue("unknown wording between objects and bin", tuning.UNKNOWN_LINK)
    return (tuning.LINKED_AFTER_COMMA if gap.separated else 0.0), None


def _object_charge(objects: list[ObjectMention]) -> tuple[float, Issue | None]:
    """Doubtful object lists: a placement after 'leave', 'everything' beside specifics, a lone exception."""
    cost = tuning.DEPOSIT_AFTER_LEAVE if any(m.leave for m in objects) else 0.0
    quantifiers = [m for m in objects if is_quantifier(m)]
    if quantifiers and any(m.negated is Negation.NONE for m in objects if m not in quantifiers):
        cost += tuning.QUANTIFIER_IN_LIST
    if not quantifiers and any(m.negated is Negation.WHOLE for m in objects):
        return cost + tuning.LONE_EXCEPTION, Issue("an excepted group is given its own bin", tuning.F_LONE_EXCEPTION)
    return cost, None


def is_quantifier(mention: ObjectMention) -> bool:
    """'everything', 'the rest': a mention that stands for all objects not named more specifically."""
    return mention.scope is not Scope.PLAIN and mention.key == DEFAULT_KEY and mention.negated is Negation.NONE
