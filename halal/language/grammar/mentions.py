"""Chunk tagged tokens into object and bin mentions, then apply pronouns, negation, exceptions and 'leave'."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Iterable, Union

from halal.language.grammar import tuning
from halal.language.grammar.lexicon import (DEFAULT_KEY, NOUN_FILLERS, PRONOUN_QUANTIFIERS, RELATIVE_PRONOUNS, Tag)
from halal.language.grammar.tokens import Issue, Token

_NOUN_TAGS = frozenset({Tag.ALL, Tag.REST, Tag.COLOUR, Tag.SHAPE, Tag.OBJECT})
_CONTENT_TAGS = frozenset({Tag.LINK, Tag.ARROW, Tag.OTHER, Tag.LEAVE, Tag.NOT, Tag.VOID, Tag.CLAUSE, Tag.BREAK})
_SEPARATOR_TAGS = frozenset({Tag.BREAK, Tag.COMMA, Tag.AND, Tag.CLAUSE})
_SCOPE_BREAKERS = frozenset({Tag.BREAK, Tag.COMMA, Tag.CLAUSE, Tag.ARROW})
_NOT_TRANSPARENT = frozenset(Tag) - {Tag.FILL, Tag.OBJECT}


class Scope(Enum):
    """How an object mention picks its objects."""
    PLAIN = auto()
    ALL = auto()         # "everything", "every object"
    REST = auto()        # "everything else", "the other red ones"


class Negation(Enum):
    """Which part of an object mention is negated."""
    NONE = auto()
    COLOUR = auto()      # "cubes that are not red"
    SHAPE = auto()       # "red objects that are not cubes"
    WHOLE = auto()       # "except the red cubes"


@dataclass
class ObjectMention:
    """A group of objects named by colour, shape, both, or only a quantifier."""
    start: int
    end: int
    colour: str | None = None
    shape: str | None = None
    scope: Scope = Scope.PLAIN
    negated: Negation = Negation.NONE
    known: bool = True              # False for a colour the scene does not have
    headless: bool = False          # bare colour adjective that may share a later noun: "red and blue cubes"
    pronoun: str = ""               # the pronoun text while it still awaits its antecedent
    leave: bool = False             # the instruction says not to move these
    add: bool = False               # an additive cue stands next to it: "blue too", "same for blue"
    universal: bool = False         # "regardless of colour", "whatever its shape" stands next to it
    corrects: bool = False          # follows "sorry, I meant" / "actually": replaces what was said before
    sentence: int = 0
    topic: int = 0                  # first of the sentences chained to its own by introducing colons
    descriptive: bool = False       # its sentence names no bin: "you will see red, yellow and blue objects"
    issues: tuple[Issue, ...] = ()

    @property
    def key(self) -> str:
        """Mapping key: 'red/cube', 'red', 'cube' or 'default'."""
        return "/".join(part for part in (self.colour, self.shape) if part) or DEFAULT_KEY

    @property
    def soft(self) -> bool:
        """May stay without a bin without that being a parse failure."""
        quantifier_only = self.scope is Scope.ALL and self.key == DEFAULT_KEY
        return self.leave or not self.known or quantifier_only or self.negated is not Negation.NONE


@dataclass
class BinMention:
    """A bin named by position, or by reference ('the same bin', 'the remaining bin')."""
    start: int
    end: int
    position: str | None = None
    anaphor: str = ""               # "same" | "opposite" | "remaining" when the position is not stated
    negated: bool = False           # "not in the left bin", "the center bin stays empty"
    optional: bool = False          # named in passing: "from left to right", "skip the center bin"
    corrects: bool = False          # follows "sorry, I meant" / "actually": replaces what was said before
    sentence: int = 0
    topic: int = 0
    descriptive: bool = False       # its sentence names no objects: "there are three bins: left, center, right"
    issues: tuple[Issue, ...] = ()

    @property
    def soft(self) -> bool:
        """May stay without objects without that being a parse failure."""
        return self.negated or self.optional


Mention = Union[ObjectMention, BinMention]


@dataclass(frozen=True)
class Gap:
    """What stands between two neighbouring mentions."""
    tags: frozenset[Tag]
    directional: bool = False

    def has(self, *tags: Tag) -> bool:
        return not self.tags.isdisjoint(tags)

    @property
    def list_like(self) -> bool:
        """Only commas, conjunctions and transparent words: the two mentions can be items of one list."""
        return self.tags.isdisjoint(_CONTENT_TAGS)

    @property
    def separated(self) -> bool:
        return not self.tags.isdisjoint(_SEPARATOR_TAGS)


def gap_between(tokens: list[Token], left: Mention, right: Mention) -> Gap:
    between = tokens[left.end:right.start]
    return Gap(frozenset(token.tag for token in between), any(token.value == "directional" for token in between))


def find_mentions(tokens: list[Token]) -> list[Mention]:
    """Object and bin mentions in text order, with pronouns resolved and operator scopes applied."""
    mentions = _chunk(tokens)
    _retag_punctuation(mentions, tokens)
    _share_heads(mentions, tokens)
    mentions = _resolve_pronouns(mentions, tokens)
    for position, token in enumerate(tokens):
        if token.tag in _OPERATORS:
            _OPERATORS[token.tag](mentions, tokens, position)
    _mark_cues(mentions, tokens)
    _mark_sentences(mentions, tokens)
    _read_cubes_as_objects(mentions)
    return mentions


# ---------------------------------------------------------------- chunking
def _chunk(tokens: list[Token]) -> list[Mention]:
    mentions: list[Mention] = []
    position = 0
    while position < len(tokens):
        end, found = _mentions_at(tokens, position)
        mentions.extend(found)
        position = max(end, position + 1)
    return mentions


def _mentions_at(tokens: list[Token], position: int) -> tuple[int, list[Mention]]:
    """(end index, mentions) of the phrase starting at `position`; no mentions when none starts there."""
    token, end = tokens[position], _phrase_end(tokens, position)
    if token.tag is Tag.BIN:
        end = _bin_noun_end(tokens, position + 1)
        return end, [BinMention(position, end, token.value, optional=token.soft, issues=_issues([token]))]
    if token.tag is Tag.PRONOUN:
        return position + 1, [ObjectMention(position, position + 1, pronoun=token.text)]
    if token.tag is Tag.SAME and _bin_noun_end(tokens, position + 1) > position + 1:
        return position + 1, [BinMention(position, position + 1, anaphor=token.value)]
    return end, _noun_phrase(tokens, position, end) if end > position else []


def _bin_noun_end(tokens: list[Token], position: int) -> int:
    """Index after the nouns that follow a bin word: 'left bin', 'left hand side', 'middle one'."""
    while position < len(tokens) and (tokens[position].tag is Tag.CONTAINER or tokens[position].text == "one"):
        position += 1
    return position


def _retag_punctuation(mentions: list[Mention], tokens: list[Token]) -> None:
    """':' and '?' after a mention pair it with what follows, elsewhere they end an introduction; 'red/left' pairs."""
    owner = {index: mention for mention in mentions for index in range(mention.start, mention.end)}
    for index, token in enumerate(tokens):
        if token.text in (":", "?"):
            before = index - 1
            while before >= 0 and tokens[before].tag is Tag.COMMA:
                before -= 1
            token.tag = Tag.ARROW if before in owner else Tag.BREAK
        elif token.text == "/":
            after = index + 1
            while after < len(tokens) and tokens[after].tag is Tag.FILL:
                after += 1
            if index - 1 in owner and after in owner and not _same_kind(owner[index - 1], owner[after]):
                token.tag = Tag.ARROW


def _issues(tokens: list[Token]) -> tuple[Issue, ...]:
    return tuple(token.issue for token in tokens if token.issue)


def _next_tag(tokens: list[Token], position: int) -> Tag | None:
    """Tag of the first non-filler token at or after `position`."""
    while position < len(tokens) and tokens[position].tag is Tag.FILL:
        position += 1
    return tokens[position].tag if position < len(tokens) else None


def _negates_property(token: Token) -> bool:
    return token.tag is Tag.NOT and token.value == "constituent"


def _phrase_end(tokens: list[Token], start: int) -> int:
    """End of the noun phrase that starts at `start`; `start` itself when none starts there."""
    if tokens[start].tag not in _NOUN_TAGS and not _negates_property(tokens[start]):
        return start
    end = position = start
    seen: set[Tag] = set()
    while position < len(tokens):
        token = tokens[position]
        if token.tag in _NOUN_TAGS:
            seen.add(token.tag)
            position = end = position + 1
        elif _negates_property(token) or (token.tag is Tag.FILL and token.text in NOUN_FILLERS):
            position += 1
        elif _bridges(tokens, position, seen):
            position += 1 + (token.text in RELATIVE_PRONOUNS)
        else:
            break
    return end


def _bridges(tokens: list[Token], position: int, seen: set[Tag]) -> bool:
    """'(that) are' here links the phrase to a colour or shape it still lacks: 'objects that are red'."""
    copula = position + (tokens[position].text in RELATIVE_PRONOUNS)
    if copula >= len(tokens) or tokens[copula].value != "copula":
        return False
    target = copula + 1
    while target < len(tokens) and (_negates_property(tokens[target]) or tokens[target].text in NOUN_FILLERS):
        target += 1
    return target < len(tokens) and tokens[target].tag in (Tag.COLOUR, Tag.SHAPE) and tokens[target].tag not in seen


def _noun_phrase(tokens: list[Token], start: int, end: int) -> list[Mention]:
    """Mentions of one noun phrase: one per colour and shape it names, or a quantifier, or a bin reference."""
    span = tokens[start:end]
    colours = [token for token in span if token.tag is Tag.COLOUR]
    shapes = [token for token in span if token.tag is Tag.SHAPE]
    scope = _scope(span)
    if not colours and not shapes:
        return _quantifier_phrase(tokens, start, end, scope)
    headless = not shapes and all(token.tag is not Tag.OBJECT for token in span)
    return [ObjectMention(start, end, colour.value or None if colour else None, shape.value if shape else None,
                          scope, _negated_property(span), known=colour is None or bool(colour.value),
                          headless=headless, issues=_issues(span))
            for colour in colours or [None] for shape in shapes or [None]]


def _scope(span: list[Token]) -> Scope:
    tags = {token.tag for token in span}
    return Scope.REST if Tag.REST in tags else Scope.ALL if Tag.ALL in tags else Scope.PLAIN


def _negated_property(span: list[Token]) -> Negation:
    """Colour or shape, whichever first follows a 'not' / 'non' inside the phrase."""
    negating = False
    for token in span:
        negating = negating or _negates_property(token)
        if negating and token.tag in (Tag.COLOUR, Tag.SHAPE):
            return Negation.COLOUR if token.tag is Tag.COLOUR else Negation.SHAPE
    return Negation.NONE


def _quantifier_phrase(tokens: list[Token], start: int, end: int, scope: Scope) -> list[Mention]:
    """'everything else' and 'all objects' are object groups; 'the other bin' refers to a bin."""
    if scope is Scope.REST and _names_bin(tokens, start, end):
        return [BinMention(start, end, anaphor="remaining")]
    span = tokens[start:end]
    pronominal = any(token.tag is Tag.OBJECT or token.text in PRONOUN_QUANTIFIERS for token in span)
    counts_bins = _next_tag(tokens, end) in (Tag.CONTAINER, Tag.PRONOUN)
    stands_alone = scope is Scope.ALL and pronominal and not counts_bins and not _floats(tokens, start, end)
    return [ObjectMention(start, end, scope=scope)] if scope is Scope.REST or stands_alone else []


def _floats(tokens: list[Token], start: int, end: int) -> bool:
    """A lone 'all' after its noun or an auxiliary repeats the subject: 'the red ones should all go left'."""
    if end - start > 1 or start == 0:
        return False
    before = tokens[start - 1]
    return before.value in ("modal", "copula") or before.tag in (Tag.OBJECT, Tag.COLOUR, Tag.SHAPE, Tag.PRONOUN)


def _names_bin(tokens: list[Token], start: int, end: int) -> bool:
    """'the remaining bin', 'the bin that is left', 'into the other one'."""
    if _next_tag(tokens, end) is Tag.CONTAINER:
        return True
    before = start - 1
    while before >= 0 and (tokens[before].text in RELATIVE_PRONOUNS or tokens[before].value == "copula"):
        before -= 1
    participle = tokens[start].text == "left" or before < start - 1
    if participle and before >= 0 and tokens[before].tag is Tag.CONTAINER:
        return True
    article = start - 1 - (start > 0 and tokens[start - 1].text == "the")
    return tokens[end - 1].text == "one" and article >= 0 and tokens[article].value == "directional"


def _share_heads(mentions: list[Mention], tokens: list[Token]) -> None:
    """'red and blue cubes' are both cubes; 'red cubes and cylinders' are both red."""
    pairs = [(a, b) for a, b in zip(mentions, mentions[1:])
             if isinstance(a, ObjectMention) and isinstance(b, ObjectMention)]
    for a, b in reversed(pairs):
        joined = gap_between(tokens, a, b).tags <= {Tag.AND, Tag.COMMA, Tag.FILL}
        if joined and a.headless and b.colour and b.shape and not a.shape:
            a.shape = b.shape
    for a, b in pairs:
        joined = bool(gap_between(tokens, a, b).tags) and gap_between(tokens, a, b).tags <= {Tag.AND, Tag.COMMA}
        if joined and a.colour and a.shape and b.shape and not b.colour and b.scope is Scope.PLAIN:
            b.colour = a.colour
            b.issues += (Issue(f"read '{b.shape}' listed after '{a.colour} {a.shape}' as {a.colour} too",
                               tuning.SHARED_COLOUR),)


# ---------------------------------------------------------------- pronouns
def _resolve_pronouns(mentions: list[Mention], tokens: list[Token]) -> list[Mention]:
    """Replace each pronoun by copies of the object mentions it refers to."""
    resolved: list[Mention] = []
    for mention in mentions:
        if isinstance(mention, ObjectMention) and mention.pronoun:
            resolved.extend(_antecedents(resolved, mention, tokens))
        else:
            resolved.append(mention)
    return resolved


def _antecedents(prior: list[Mention], pronoun: ObjectMention, tokens: list[Token]) -> list[ObjectMention]:
    """Copies, at the pronoun's place, of the nearest preceding object list ('it' after a bin refers to the bin)."""
    if not prior or (pronoun.pronoun in ("it", "which") and isinstance(prior[-1], BinMention)):
        return []
    last = max((i for i, mention in enumerate(prior) if isinstance(mention, ObjectMention)), default=-1)
    if last < 0:
        return []
    run = _run_back(prior, tokens, last)
    copies = {prior[i].key: replace(prior[i], start=pronoun.start, end=pronoun.end, issues=()) for i in run}
    return list(copies.values())


# ---------------------------------------------------------------- operator scopes
def _same_kind(a: Mention, b: Mention) -> bool:
    return isinstance(a, ObjectMention) == isinstance(b, ObjectMention)


def _list_run(mentions: list[Mention], tokens: list[Token], first: int) -> list[int]:
    """Indices of the list of same-kind mentions starting at `first`."""
    run = [first]
    while run[-1] + 1 < len(mentions) and _continues_list(mentions, tokens, run[-1]):
        run.append(run[-1] + 1)
    return _without_comma_tail(mentions, tokens, run)


def _continues_list(mentions: list[Mention], tokens: list[Token], index: int) -> bool:
    a, b = mentions[index], mentions[index + 1]
    return _same_kind(a, b) and gap_between(tokens, a, b).list_like


def _without_comma_tail(mentions: list[Mention], tokens: list[Token], run: list[int]) -> list[int]:
    """Cut 'red, yellow goes ...' after 'red': a list that goes on ends in 'and'; a bare comma starts a new clause."""
    gaps = [gap_between(tokens, mentions[i], mentions[i + 1]) for i in run[:-1]]
    after = run[-1] + 1
    goes_on = after < len(mentions) and not gap_between(tokens, mentions[run[-1]], mentions[after]).has(Tag.BREAK)
    if not gaps or gaps[-1].has(Tag.AND) or not goes_on:
        return run
    bare_commas = [i for i, gap in enumerate(gaps) if gap.has(Tag.COMMA) and not gap.has(Tag.AND)]
    return run[:bare_commas[0] + 1] if bare_commas else run


def _run_back(mentions: list[Mention], tokens: list[Token], last: int) -> list[int]:
    """Indices of the list of same-kind mentions ending at `last`."""
    first = last
    while first > 0 and _continues_list(mentions, tokens, first - 1):
        first -= 1
    return list(range(first, last + 1))


def _first_after(mentions: list[Mention], position: int) -> int:
    """Index of the first mention starting after token `position` (len(mentions) when none)."""
    return next((i for i, mention in enumerate(mentions) if mention.start > position), len(mentions))


def _last_before(mentions: list[Mention], position: int) -> int:
    """Index of the last mention ending at or before token `position` (-1 when none)."""
    return max((i for i, mention in enumerate(mentions) if mention.end <= position), default=-1)


def _forward_target(mentions: list[Mention], tokens: list[Token], position: int, blockers: frozenset[Tag]) -> list[int]:
    """The list right after an operator word, unless one of `blockers` stands in between."""
    first = _first_after(mentions, position)
    if first == len(mentions) or any(t.tag in blockers for t in tokens[position + 1:mentions[first].start]):
        return []
    return _list_run(mentions, tokens, first)


def _backward_target(mentions: list[Mention], tokens: list[Token], position: int) -> list[int]:
    """The object list right before an operator word in the same sentence: 'red and blue stay'."""
    last = _last_before(mentions, position)
    if last < 0 or isinstance(mentions[last], BinMention):
        return []
    if any(token.tag is Tag.BREAK for token in tokens[mentions[last].end:position]):
        return []
    return _run_back(mentions, tokens, last)


def _apply_leave(mentions: list[Mention], tokens: list[Token], position: int) -> None:
    """'leave X', 'ignore X' bind forward; 'X stays', 'X is left alone' bind backward; a bin may go unused."""
    forward = _forward_target(mentions, tokens, position, _NOT_TRANSPARENT)
    backward = _backward_target(mentions, tokens, position)
    for index in (forward or backward) if tokens[position].value == "forward" else (backward or forward):
        mention = mentions[index]
        if isinstance(mention, ObjectMention):
            mention.leave = True
        else:
            mention.optional = True


def _apply_not(mentions: list[Mention], tokens: list[Token], position: int) -> None:
    """Clause negation: bins in its scope are excluded; with no bin in scope the objects are not to be moved."""
    if tokens[position].value != "clause":
        return
    scope = _negation_scope(mentions, tokens, position)
    bins = [mentions[index] for index in scope if isinstance(mentions[index], BinMention)]
    for bin_mention in bins:
        bin_mention.negated = True
    for index in [] if bins else scope or _backward_target(mentions, tokens, position):
        mentions[index].leave = True


def _negation_scope(mentions: list[Mention], tokens: list[Token], position: int) -> list[int]:
    """Mentions a clause negation reaches: lists joined by prepositions or a plain 'and', up to a comma or 'but'."""
    scope = _forward_target(mentions, tokens, position, _SCOPE_BREAKERS | {Tag.AND})
    while scope and scope[-1] + 1 < len(mentions):
        gap = gap_between(tokens, mentions[scope[-1]], mentions[scope[-1] + 1])
        if gap.has(*_SCOPE_BREAKERS) or (gap.has(Tag.AND) and not gap.list_like):
            break
        scope += _list_run(mentions, tokens, scope[-1] + 1)
    return scope


def _apply_except(mentions: list[Mention], tokens: list[Token], position: int) -> None:
    """'except red', 'other than the cubes', 'instead of the left bin' take the next list out."""
    for index in _forward_target(mentions, tokens, position, frozenset({Tag.BREAK})):
        mention = mentions[index]
        if isinstance(mention, BinMention):
            mention.negated = not mention.negated       # "must not go anywhere except the left bin" is positive
        elif mention.negated is Negation.NONE:
            mention.negated = Negation.WHOLE


def _apply_void(mentions: list[Mention], tokens: list[Token], position: int) -> None:
    """'the center bin stays empty', 'an unused left bin': that bin takes no objects."""
    last, first = _last_before(mentions, position), _first_after(mentions, position)
    stated_before = last >= 0 and not any(t.tag in _SEPARATOR_TAGS for t in tokens[mentions[last].end:position])
    if stated_before and isinstance(mentions[last], BinMention):
        mentions[last].negated = True
    elif first < len(mentions) and isinstance(mentions[first], BinMention):
        if all(t.tag in (Tag.FILL, Tag.CONTAINER) for t in tokens[position + 1:mentions[first].start]):
            mentions[first].negated = True


def _apply_universal(mentions: list[Mention], tokens: list[Token], position: int) -> None:
    """'regardless of colour' strengthens the object list before it in its sentence, else the one after it."""
    def same_sentence(a: int, b: int) -> bool:
        return all(token.tag is not Tag.BREAK for token in tokens[a:b])
    objects = [i for i, mention in enumerate(mentions) if isinstance(mention, ObjectMention)]
    before = [i for i in objects if mentions[i].end <= position and same_sentence(mentions[i].end, position)]
    after = [i for i in objects if mentions[i].start > position and same_sentence(position, mentions[i].start)]
    forward = _list_run(mentions, tokens, after[0]) if after else []
    for index in _run_back(mentions, tokens, before[-1]) if before else forward:
        mentions[index].universal = True


def _apply_correction(mentions: list[Mention], tokens: list[Token], position: int) -> None:
    """'sorry, I meant the right bin': what the rest of that sentence says replaces what was said before."""
    first = _first_after(mentions, position)
    if first == len(mentions) or sum(t.tag is Tag.BREAK for t in tokens[position:mentions[first].start]) > 1:
        return
    start = mentions[first].start
    end = next((i for i in range(start, len(tokens)) if tokens[i].tag is Tag.BREAK), len(tokens))
    for mention in mentions[first:]:
        mention.corrects = mention.corrects or mention.start < end


_OPERATORS = {Tag.LEAVE: _apply_leave, Tag.NOT: _apply_not, Tag.EXCEPT: _apply_except, Tag.VOID: _apply_void,
              Tag.UNIVERSAL: _apply_universal, Tag.CORRECTION: _apply_correction}


# ---------------------------------------------------------------- cues
def _mark_cues(mentions: list[Mention], tokens: list[Token]) -> None:
    """Flag object mentions with 'too', 'as well', 'same for' beside them in their own clause."""
    boundaries = (Tag.BREAK, Tag.COMMA, Tag.CLAUSE)
    for index, mention in enumerate(mentions):
        if not isinstance(mention, ObjectMention):
            continue
        low = mentions[index - 1].end if index else 0
        high = mentions[index + 1].start if index + 1 < len(mentions) else len(tokens)
        before = _until_boundary(reversed(tokens[low:mention.start]), boundaries)
        after = _until_boundary(tokens[mention.end:high], boundaries)
        mention.add = any(token.tag is Tag.ADD for token in before + after)


def _mark_sentences(mentions: list[Mention], tokens: list[Token]) -> None:
    """Number each mention's sentence; a mention is descriptive when its sentence has none of the other kind."""
    sentence_of, topic_of, sentence, topic = [], [], 0, 0
    for token in tokens:
        sentence_of.append(sentence)
        topic_of.append(topic)
        sentence += token.tag is Tag.BREAK
        topic = topic if token.tag is not Tag.BREAK or token.text == ":" else sentence
    for mention in mentions:
        mention.sentence, mention.topic = sentence_of[mention.start], topic_of[mention.start]
    with_objects = {m.sentence for m in mentions if isinstance(m, ObjectMention)}
    with_bins = {m.sentence for m in mentions if isinstance(m, BinMention)}
    for mention in mentions:
        others = with_bins if isinstance(mention, ObjectMention) else with_objects
        mention.descriptive = mention.sentence not in others


def _until_boundary(tokens: Iterable[Token], boundaries: tuple[Tag, ...]) -> list[Token]:
    """Leading tokens up to the first clause boundary."""
    kept: list[Token] = []
    for token in tokens:
        if token.tag in boundaries:
            break
        kept.append(token)
    return kept


def _read_cubes_as_objects(mentions: list[Mention]) -> None:
    """'red cubes left, blue cubes right' with no other shape or rest clause: 'cubes' is the generic noun."""
    objects = [m for m in mentions if isinstance(m, ObjectMention) and m.known]
    shaped = [m for m in objects if m.shape]
    contrast = any(m.shape != "cube" or not m.colour or m.negated is not Negation.NONE for m in shaped)
    has_rest_clause = any(m.scope is Scope.REST for m in objects)
    if contrast or has_rest_clause or len({m.colour for m in shaped}) < 2:
        return
    for mention in shaped:
        mention.shape = None
    shaped[0].issues += (Issue("'cubes' named for every group: read as a generic word for objects",
                               tuning.GENERIC_CUBE),)
