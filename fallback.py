"""What the board shows when the model, or the data, is unavailable.

The deterministic grid needs no model, which is why it doubles as the
cold-start renderer: the board is useful the instant the plugin is enabled.
The outage messages are gentle and never at the user's expense, and they
always carry a timestamp so a two-day outage is not mistaken for a
two-minute one.
"""

from .catalog import default_label
from .validation import _is_identifier_ref

# Values that mean "no reading". The prompt tells the model to skip these;
# the deterministic grid has no model, so it filters them itself.
_PLACEHOLDERS = frozenset({"UNKNOWN", "N/A", "NA", "NONE", "NULL", "--", "-", ""})
from .charset import truncate
from .layout import Geometry, Tile, fits, wrap_center

# (short, long) pairs. The long form is used when it fits; Notes get the short.
OUTAGE_MESSAGES: tuple[tuple[str, str], ...] = (
    ("HAMSTERS ON BREAK", "THE HAMSTERS HAVE STOPPED RUNNING ON THE WHEEL"),
    ("ROUTER UNPLUGGED?", "SOMEONE MAY HAVE UNPLUGGED SOMETHING SOMEWHERE"),
    ("NUMBERS NAPPING", "THE NUMBERS ARE TAKING A SHORT NAP"),
    ("NO WORD YET", "LOST CONTACT WITH THE OUTSIDE WORLD FOR NOW"),
    ("WHEEL STOPPED", "THE WHEEL STOPPED SPINNING - CHECK BACK SOON"),
)

_SETUP = ("ENABLE A PLUGIN", "NO PLUGIN DATA YET - ENABLE SOME PLUGINS IN SETTINGS")


def deterministic_tiles(
    order: list[str],
    labels: dict[str, str],
    values: dict[str, str],
    pinned: list[str],
) -> list[Tile]:
    """Lay the watchlist out in priority order without asking a model.

    Pinned variables first, then the user's own ordering. Nothing is coloured:
    no model has judged anything worth accenting.
    """
    pinned_set = set(pinned)
    ranked = [r for r in order if r in pinned_set] + [r for r in order if r not in pinned_set]
    return [
        Tile(label=labels.get(ref) or default_label(ref), value=values[ref])
        for ref in ranked
        if ref in values
        and values[ref].strip().upper() not in _PLACEHOLDERS
        # A currency pair's name or a ticker is not a stat, even with no
        # model around to say so — it was the face of the de facto board.
        and not (_is_identifier_ref(ref) and not any(c.isdigit() for c in values[ref]))
    ]


def _choose(short: str, long: str, cols: int, rows: int) -> str:
    """Prefer the long phrasing, fall back to the short one on small boards."""
    return long if fits(long, rows, cols) else short


def _footer(last_good: str | None, cols: int) -> str:
    """One honest line: how long it has been broken, or that it never worked."""
    if not last_good:
        return "NO DATA YET"
    return _choose(f"LAST {last_good}", f"LAST GOOD DATA {last_good}", cols, 1)


def outage_lines(geo: Geometry, last_good: str | None, index: int) -> list[str]:
    """Humour plus one honest line, filling the board exactly."""
    body_rows = max(1, geo.rows - 1)
    short, long = OUTAGE_MESSAGES[index % len(OUTAGE_MESSAGES)]
    body = wrap_center(_choose(short, long, geo.cols, body_rows), body_rows, geo.cols)
    footer = truncate(_footer(last_good, geo.cols), geo.cols)
    return (body + [footer.center(geo.cols).rstrip()])[: geo.rows]


def setup_lines(geo: Geometry) -> list[str]:
    """Actionable message for an unconfigured plugin. Not a joke."""
    short, long = _SETUP
    return wrap_center(_choose(short, long, geo.cols, geo.rows), geo.rows, geo.cols)
