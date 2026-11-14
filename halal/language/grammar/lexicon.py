"""Closed vocabulary of the sorting domain: colours, shapes, bin positions and the function words around them."""
from __future__ import annotations

from enum import Enum, auto

COLOURS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")
BINS = ("left", "center", "right")
DEFAULT_KEY = "default"


class Tag(Enum):
    """Role a token plays in a sorting instruction."""
    COLOUR = auto()
    SHAPE = auto()
    BIN = auto()
    OBJECT = auto()          # generic object noun: "objects", "ones", "blocks"
    CONTAINER = auto()       # generic bin noun: "bin", "box", "side"
    REST = auto()            # "everything else", "the others"
    ALL = auto()             # "everything", "each", "all"
    NOT = auto()             # value: clause | constituent
    EXCEPT = auto()
    LEAVE = auto()           # value: forward ("leave X") | backward ("X stays")
    VOID = auto()            # "empty", "unused"
    RESPECTIVELY = auto()
    PRONOUN = auto()
    ADD = auto()             # "too", "as well", "same for"
    UNIVERSAL = auto()       # "regardless of colour", "whatever its shape"
    SAME = auto()            # "the same bin", "the opposite side"; value: same | opposite
    COMPANION = auto()       # "blue goes with yellow", "red joins blue"
    CORRECTION = auto()      # "sorry, I meant", "actually", "make that"
    ORDINAL = auto()         # value: rank counted along the row of bins
    BREAK = auto()           # sentence boundary
    COMMA = auto()
    AND = auto()
    CLAUSE = auto()          # clause conjunction: "but", "while", "then"
    ARROW = auto()           # "->", ":", "="
    LINK = auto()            # known verb or preposition; value: directional | copula | ""
    FILL = auto()            # transparent word: determiner, adjective, number
    OTHER = auto()           # unknown word


def _entries(tag: Tag, groups: dict[str, str]) -> dict[str, tuple[Tag, str]]:
    """Expand {value: "space separated words"} into word -> (tag, value)."""
    return {word: (tag, value) for value, words in groups.items() for word in words.split()}


WORDS: dict[str, tuple[Tag, str]] = {
    **_entries(Tag.FILL, {"": (
        "the a an of this that some both either several many few two three four five six my your its their our "
        "small little big large tiny coloured colored shaped sized light dark bright pale plain wooden plastic "
        "printed correct proper appropriate corresponding respective matching designated own orange identical far "
        "near outer extreme furthest farthest very most just only simply always now please kindly finally lastly "
        "next again robot table counter desk workspace scene mat view perspective row line number nr what who "
        "instead there here way exactly single whole entire together respect regardless irrespective "
        "hand edge anywhere everywhere somewhere eg ie etc ok okay hi hello hey dear task job goal "
        "instruction instructions rule rules mapping follows following according per based depending new "
        "different separate individual present available visible given current possible total")}),
    **_entries(Tag.LINK, {
        "directional": "to into in inside onto toward towards within",
        "copula": "is are was were be been being am",
        "modal": "should must shall will would can could may might need needs have has had do does did",
        "": (
            "on at for by from as than about through via under over beside across out up down off back away "
            "go goes going gone belong belongs belonging put puts putting place places placed placing move moves "
            "moved moving drop drops dropped dropping sort sorts sorted sorting send sends sent deposit deposits "
            "deposited throw toss tossed bring brings carry carries carried transfer transfers transferred set "
            "sets lay stack store stored stow collect collects collected gather gathers pick picks picked grab "
            "grabs take takes taken taking get gets getting receive receives hold holds contain contains use uses "
            "used using fill fills filled load insert inserts assign assigned assigns end ends land lands live "
            "lives sit sits arrive deliver delivers delivered relocate relocated push slide dump dumps dumped "
            "want wants wanted needed make makes sure ensure see sees find finds found look looks "
            "done let lets like starting beginning counting reading numbered ordered "
            "you i we me us if when whenever so such how touch touching handle lift "
            "start begin finish complete arrange organise organize separate distribute allocate group groups "
            "share shares house houses accept accepts give gives given reserve reserved meant intended "
            "supposed destined bound headed come comes wind winds grasp remove removed clear")},
    ),
    **_entries(Tag.OBJECT, {"": (
        "object objects one ones item items piece pieces block blocks thing things part parts shape shapes toy "
        "toys brick bricks element elements figure figures stuff solid solids colour colours color colors kind "
        "kinds type types")}),
    **_entries(Tag.CONTAINER, {"": (
        "bin bins box boxes container containers tray trays crate crates basket baskets bucket buckets side "
        "sides receptacle receptacles tub tubs bowl bowls compartment compartments slot slots spot position "
        "location area zone pile")}),
    **_entries(Tag.COLOUR, {
        "red": "red reds reddish crimson scarlet ruby maroon vermilion",
        "yellow": "yellow yellows yellowish golden gold lemon",
        "blue": "blue blu blues bluish navy azure cobalt indigo",
        "": "green purple pink black white grey gray brown violet magenta cyan turquoise teal silver beige"}),
    **_entries(Tag.SHAPE, {
        "cube": "cube cubes cubic cubical cuboid cuboids square squares squared",
        "cylinder": ("cylinder cylinders cylindrical round rounded circular circle circles disc discs disk disks "
                     "cans tube tubes puck pucks"),
        "triangle": "triangle triangles triangular prism prisms wedge wedges pyramid pyramids"}),
    **_entries(Tag.BIN, {
        "left": "left lef leftmost lefthand leftward lhs",
        "center": "center centre central middle mid midmost centermost",
        "right": "right rightmost righthand rightward rhs"}),
    **_entries(Tag.REST, {"": "rest remaining remainder other others leftover leftovers otherwise else"}),
    **_entries(Tag.ALL, {"": "everything anything all every each any whatever whichever everyone"}),
    **_entries(Tag.NOT, {
        "clause": "no never nothing none neither without avoid avoids avoiding",
        "either": "not",
        "constituent": "non"}),
    **_entries(Tag.EXCEPT, {"": "except excluding besides barring unless"}),
    **_entries(Tag.LEAVE, {
        "forward": ("leave leaves leaving ignore ignores ignoring skip skips skipping keep keeps keeping forget "
                    "omit neglect disregard"),
        "backward": "stay stays staying remain remains untouched unmoved unsorted"}),
    **_entries(Tag.VOID, {"": "empty unused vacant"}),
    **_entries(Tag.RESPECTIVELY, {"": "respectively correspondingly"}),
    **_entries(Tag.PRONOUN, {"": "them they those these it which"}),
    **_entries(Tag.ADD, {"": "too also likewise ditto similarly"}),
    **_entries(Tag.SAME, {"same": "same", "opposite": "opposite"}),
    **_entries(Tag.COMPANION, {"": "with alongside join joins accompany accompanies where wherever"}),
    **_entries(Tag.ORDINAL, {"1": "first 1st", "2": "second 2nd", "3": "third 3rd last"}),
    **_entries(Tag.AND, {"": "and or nor plus then & / +"}),
    **_entries(Tag.CLAUSE, {"": "but however whereas while whilst though although yet"}),
    **_entries(Tag.CORRECTION, {"": "sorry oops whoops correction actually wait meant mean mistake wrong apologies "
                                    "pardon nope rather erratum"}),
    **_entries(Tag.BREAK, {"": ". ; ! ?"}),
    **_entries(Tag.COMMA, {"": ", ( )"}),
    **_entries(Tag.ARROW, {"": ": = →"}),
    "#": (Tag.FILL, ""),
}

PHRASES: dict[tuple[str, ...], tuple[Tag, str]] = {
    ("everything", "else"): (Tag.REST, ""),
    ("anything", "else"): (Tag.REST, ""),
    ("all", "else"): (Tag.REST, ""),
    ("all", "other"): (Tag.REST, ""),
    ("all", "others"): (Tag.REST, ""),
    ("all", "the", "other"): (Tag.REST, ""),
    ("all", "the", "others"): (Tag.REST, ""),
    ("all", "the", "rest"): (Tag.REST, ""),
    ("all", "remaining"): (Tag.REST, ""),
    ("any", "other"): (Tag.REST, ""),
    ("any", "remaining"): (Tag.REST, ""),
    ("every", "other"): (Tag.REST, ""),
    ("what", "remains"): (Tag.REST, ""),
    ("whatever", "remains"): (Tag.REST, ""),
    ("that", "remain"): (Tag.REST, ""),
    ("that", "remains"): (Tag.REST, ""),
    ("left", "over"): (Tag.REST, ""),
    ("left", "overs"): (Tag.REST, ""),
    ("in", "that", "order"): (Tag.RESPECTIVELY, ""),
    ("in", "this", "order"): (Tag.RESPECTIVELY, ""),
    ("in", "the", "same", "order"): (Tag.RESPECTIVELY, ""),
    ("in", "order"): (Tag.RESPECTIVELY, ""),
    ("in", "turn"): (Tag.RESPECTIVELY, ""),
    ("as", "well", "as"): (Tag.AND, ""),
    ("along", "with"): (Tag.AND, ""),
    ("together", "with"): (Tag.AND, ""),
    ("as", "well"): (Tag.ADD, ""),
    ("so", "do"): (Tag.ADD, ""),
    ("so", "does"): (Tag.ADD, ""),
    ("same", "for"): (Tag.ADD, ""),
    ("same", "goes", "for"): (Tag.ADD, ""),
    ("same", "applies", "to"): (Tag.ADD, ""),
    ("same", "with"): (Tag.ADD, ""),
    ("followed", "by"): (Tag.AND, ""),
    ("after", "that"): (Tag.AND, ""),
    ("make", "that"): (Tag.CORRECTION, ""),
    ("make", "it"): (Tag.CORRECTION, ""),
    ("scratch", "that"): (Tag.CORRECTION, ""),
    ("change", "that"): (Tag.CORRECTION, ""),
    ("just", "like"): (Tag.COMPANION, ""),
    ("same", "as"): (Tag.COMPANION, ""),
    ("other", "than"): (Tag.EXCEPT, ""),
    ("apart", "from"): (Tag.EXCEPT, ""),
    ("aside", "from"): (Tag.EXCEPT, ""),
    ("except", "for"): (Tag.EXCEPT, ""),
    ("with", "the", "exception", "of"): (Tag.EXCEPT, ""),
    ("instead", "of"): (Tag.EXCEPT, ""),
    ("rather", "than"): (Tag.EXCEPT, ""),
    ("anywhere", "but"): (Tag.EXCEPT, ""),
    ("everywhere", "but"): (Tag.EXCEPT, ""),
    ("left", "most"): (Tag.BIN, "left"),
    ("left", "hand"): (Tag.BIN, "left"),
    ("right", "most"): (Tag.BIN, "right"),
    ("right", "hand"): (Tag.BIN, "right"),
    ("stay", "put"): (Tag.LEAVE, "backward"),
    ("stays", "put"): (Tag.LEAVE, "backward"),
    ("three", "sided"): (Tag.SHAPE, "triangle"),
    ("point", "of", "view"): (Tag.FILL, ""),
    ("regardless", "of"): (Tag.UNIVERSAL, ""),
    ("irrespective", "of"): (Tag.UNIVERSAL, ""),
    ("no", "matter"): (Tag.UNIVERSAL, ""),
    ("whatever", "its"): (Tag.UNIVERSAL, ""),
    ("whatever", "their"): (Tag.UNIVERSAL, ""),
    ("whatever", "the"): (Tag.UNIVERSAL, ""),
    **{(preposition, quantifier, noun): (Tag.UNIVERSAL, "")
       for preposition in ("of", "in") for quantifier in ("any", "every", "all", "whatever", "whichever")
       for noun in ("colour", "colours", "color", "colors", "shape", "shapes")},
    ("make", "sure"): (Tag.LINK, ""),
}
MAX_PHRASE = max(len(phrase) for phrase in PHRASES)

# Typo correction is limited to these words; common English words one edit away from them are never corrected.
FUZZY_TARGETS = ("yellow", "blue", "cube", "cubes", "cylinder", "cylinders", "triangle", "triangles", "triangular",
                 "left", "right", "center", "centre", "middle", "leftmost", "rightmost")
NOT_TYPOS = frozenset(
    "lift loft heft deft weft cleft lefty lefts lent lest glue clue blur flue blued bluer cure cute lube "
    "rube cubs cubed curb night light might sight tight fight eight bright fright wright rights righty bight "
    "fellow mellow bellow yellowy enter centers centres centred centered canter renter meddle muddle riddle "
    "fiddle middles strangle triangled tubes cures".split())

NUMBERS = {"1": 1, "2": 2, "3": 3, "one": 1, "two": 2, "three": 3}
RELATIVE_PRONOUNS = frozenset({"that", "which", "who"})
WH_WORDS = frozenset({"what", "whatever", "that", "which", "whichever"})
NOUN_FILLERS = frozenset({"the", "of", "a", "an", "coloured", "colored", "shaped", "sized"})
SURFACES = frozenset({"table", "counter", "desk", "place", "workspace", "mat", "surface", "tabletop"})
ROW_STARTERS = frozenset("start starting begin beginning counting going reading moving numbered ordered".split())
ROW_JOINERS = frozenset({"to", "through", "till", "until", "→", ":"})
TRAVEL_WORDS = frozenset("going moving walking travelling traveling proceeding heading working sweeping scanning "
                         "counting reading stepping running".split())
NAMING_WORDS = frozenset("call called name named label labelled labeled denote denoted letter lettered".split())
DEFINING_WORDS = frozenset({"=", ":", "is", "means", "denotes", "marks"})
COLOUR_EMOJI = {"🔴": "red", "🟥": "red", "❤": "red", "🟡": "yellow", "🟨": "yellow", "💛": "yellow",
                "🔵": "blue", "🟦": "blue", "💙": "blue"}
STAY_WORDS = frozenset({"alone", "untouched", "behind", "out", "unsorted", "unmoved", "there", "where", "as"})
INTENSIFIED = frozenset({"away", "now", "here", "after", "before"})
PRONOUN_QUANTIFIERS = frozenset({"everything", "anything", "all", "everyone", "whatever"})
MODAL_FOLLOWERS = frozenset(
    "go be stay remain get have also then all either just simply move put place land sit belong fit only not".split())
