"""Validating model output before it is allowed near the board.

Grid mode cannot produce a wrong number: the model names a variable and the
plugin substitutes the value it already fetched. Prose mode can, because the
numbers live inside the sentence — so every numeric token in the text must
appear verbatim in the supplied values, and the prompt tells the model not to
compute, round, or abbreviate.
"""

import re
from dataclasses import dataclass

from .catalog import default_label
from .charset import ACCENT_COLORS, cell_width, sanitize, truncate
from .layout import Geometry, Tile, fits

# A run of digits with internal separators: 42, 94,120, 1.25
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# {plugin.var} or {plugin.var|filter} inside prose. The dot is mandatory,
# which is what keeps these from ever colliding with {red}-style markers.
_PLACEHOLDER_RE = re.compile(r"\{([a-z0-9_]+\.[a-z0-9_]+)(?:\|([a-z0-9_]+))?\}")

# The whole expression language the model gets. Deliberately tiny: every
# filter must preserve truth, and each one is validated by name so a typo is
# rejected with an error the retry can fix. Grows one safe verb at a time.
_FILTERS = {
    # Truncation only ever omits precision — the same honesty rule the
    # number check applies to literals.
    "int": lambda v: v.split(".")[0] if v and v[0].isdigit() and "." in v else v,
}

# Longest unit worth appending. " MPH" is 4; anything longer is prose.
_MAX_SUFFIX = 5

# Ref-name words that mark a value as an identifier — a name for other data,
# never data itself. Deliberately narrow: condition/status/phase text are
# readings and must not be absorbed into a neighbour's label.
_IDENTIFIER_WORDS = frozenset({
    "symbol", "ticker", "name", "callsign", "station", "code", "id",
    "airport", "location", "base", "pair", "team", "city",
})

# Unit phrases a manifest description can carry, and the affix each implies.
# Inference is the floor; an affix the model chose always wins.
_UNIT_HINTS: tuple[tuple[str, str, str], ...] = (
    ("minute", "", "MIN"),
    ("mph", "", "MPH"),
    ("fahrenheit", "", "F"),
    ("percent", "", "%"),
    ("usd", "$", ""),
    ("dollar", "$", ""),
    ("mile", "", "MI"),
    ("km", "", "KM"),
)


def _is_identifier_ref(ref: str) -> bool:
    words = set(ref.partition(".")[2].split("_"))
    return bool(words & _IDENTIFIER_WORDS)


def infer_affixes(description: str) -> tuple[str, str]:
    """(prefix, suffix) implied by a manifest description, or two blanks."""
    lowered = description.lower()
    for hint, prefix, suffix in _UNIT_HINTS:
        if hint in lowered:
            return prefix, suffix
    return "", ""


# Enough room for genuine reasoning; enough of a cap that the log stays sane.
_MAX_THINKING = 600


def _thinking(data: dict) -> str:
    return " ".join(str(data.get("thinking") or "").split())[:_MAX_THINKING]


class ValidationError(Exception):
    """The model's answer cannot be shown, and should be retried or dropped."""


@dataclass(frozen=True)
class GridResult:
    tiles: list[Tile]
    refs: list[str]  # source ref per tile, parallel to ``tiles``
    banner: str
    banner_color: str | None
    subtitle: str
    layout: str
    thinking: str  # "auto" | "list"
    suffixes: dict[str, str]  # unit per ref, e.g. {"wx.temp": "F"}
    prefixes: dict[str, str]  # currency-style prefix per ref, e.g. {"st.p": "$"}
    headline: str
    reason: str
    log: str


@dataclass(frozen=True)
class ProseResult:
    text: str
    headline: str
    banner_color: str | None
    reason: str
    log: str
    thinking: str


def extract_numbers(text: str) -> list[str]:
    """Every numeric token in *text*, with separators preserved."""
    return _NUMBER_RE.findall(text)


def allowed_numbers(current: dict[str, str], previous: dict[str, str]) -> set[str]:
    """Numbers the model is permitted to write: those it was given.

    Every decimal also admits its truncations — "335.31" allows "335.3" and
    "335" — because dropping precision only omits, it invents nothing. A
    live board froze on this: the model shortened a price the same way on
    every retry, and the strict check rejected it forever. Rounding up and
    digit-mangling ("8604" for 0.8604) remain forbidden: those are numbers
    nobody supplied.
    """
    allowed: set[str] = set()
    for source in (current, previous):
        for value in source.values():
            for token in extract_numbers(str(value)):
                allowed.add(token)
                if "." in token:
                    whole, _, frac = token.partition(".")
                    allowed.add(whole)
                    for cut in range(1, len(frac)):
                        allowed.add(f"{whole}.{frac[:cut]}")
    return allowed


def clean_suffix(raw: object) -> str:
    """Sanitize a unit suffix so it can never alter the number it follows.

    Digits are removed outright: "62" + "9F" reading as 629F would be a
    number nobody measured, which is the one thing this plugin promises
    never to show.
    """
    text = sanitize(str(raw or ""))
    text = re.sub(r"\d", "", text).rstrip()
    return text[:_MAX_SUFFIX]


def trim_to_words(text: str, width: int) -> str:
    """Fit *text* into *width* cells, dropping whole words before letters.

    "EQUITY AND FOREX UPDAT" reads as generated; "EQUITY AND FOREX" reads as
    chosen. Only a single word too long for the board gets cut mid-word.
    """
    if cell_width(text) <= width:
        return text
    kept: list[str] = []
    for word in text.split():
        candidate = " ".join(kept + [word])
        if cell_width(candidate) > width:
            break
        kept.append(word)
    return " ".join(kept) if kept else truncate(text, width)


def apply_prefix(value: str, prefix: str) -> str:
    """Attach a currency-style prefix, but only to a bare number.

    "$339.08" reads as designed; "$CLEAR" reads as broken.
    """
    if not prefix or not value or not value[0].isdigit():
        return value
    return prefix + value


def apply_suffix(value: str, suffix: str) -> str:
    """Append a unit only where one is missing.

    Values that already end in letters carry their own unit ("6.2MI",
    "7:36 PM"); appending another produces 6.2MIMI. Only a value ending in a
    digit or a percent sign is bare enough to need one.
    """
    if not suffix or not value or value[-1] not in "0123456789":
        return value
    return value + suffix


def render_prose(template: str, values: dict[str, str]) -> str:
    """Substitute live values into a prose template.

    This is what keeps a sentence true for fifteen minutes: the model wrote
    {stocks.price}, not a number, so every render shows the current price —
    the same never-type-a-value trick the grid has always used. A value that
    has vanished shows as -- rather than a stale figure.
    """
    def _fill(match: "re.Match[str]") -> str:
        value = str(values.get(match.group(1), "--"))
        transform = _FILTERS.get(match.group(2) or "")
        return transform(value) if transform else value

    return _PLACEHOLDER_RE.sub(_fill, template)


def _as_dict(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValidationError("Response was not a JSON object")
    return payload


def validate_grid(
    payload: object,
    *,
    watchlist: list[str],
    pinned: list[str],
    values: dict[str, str],
    labels: dict[str, str],
    geo: Geometry,
    use_color: bool,
    previous: dict[str, str] | None = None,
    descriptions: dict[str, str] | None = None,
) -> GridResult:
    """Turn a model response into placeable tiles, or raise."""
    data = _as_dict(payload)
    raw_tiles = data.get("tiles")
    if not isinstance(raw_tiles, list):
        raise ValidationError("Response had no 'tiles' list")

    watched = set(watchlist)
    layout = str(data.get("layout") or "auto").lower()
    if layout not in ("auto", "list"):
        layout = "auto"
    banner = trim_to_words(sanitize(str(data.get("banner") or "")), geo.cols)
    subtitle = trim_to_words(sanitize(str(data.get("subtitle") or "")), geo.cols)
    banner_hue = str(data.get("banner_color") or "").lower()
    banner_color = banner_hue if (use_color and banner_hue in ACCENT_COLORS) else None

    # The tiles cannot show a wrong number — values are substituted, never
    # typed — but a banner saying "AQI 999 EMERGENCY" could. Header text obeys
    # the same rule as prose: only digits the stats actually contain.
    allowed = allowed_numbers(values, previous or {})
    for header in (banner, subtitle):
        for token in extract_numbers(header):
            if token not in allowed:
                raise ValidationError(
                    f"Header contains the number {token}, which was not supplied"
                )
    body_rows = geo.rows - (1 if banner else 0) - (1 if banner and subtitle else 0)
    capacity = max(1, geo.tile_columns * body_rows)

    chosen: list[tuple[str, Tile]] = []
    seen: set[str] = set()
    suffixes: dict[str, str] = {}
    prefixes: dict[str, str] = {}
    for entry in raw_tiles:
        if not isinstance(entry, dict):
            continue
        ref = str(entry.get("variable") or "")
        if ref not in watched or ref in seen or ref not in values:
            continue
        color = str(entry.get("color") or "").lower()
        label_text = (
            labels.get(ref)
            or sanitize(str(entry.get("label") or ""))
            or default_label(ref)
        )
        # "RAIN RAIN" tells you nothing twice, and "ALPHABE ALPHABET INC."
        # is the same echo wearing a truncation: for text values, a label
        # that repeats the start of its value (or vice versa) is dropped and
        # the value speaks for itself.
        value_text = str(values[ref]).strip().upper()
        label_up = label_text.strip().upper()
        if label_up and not value_text[:1].isdigit():
            if value_text.startswith(label_up) or label_up.startswith(value_text):
                label_text = ""
        suffix = clean_suffix(entry.get("suffix"))
        prefix = clean_suffix(entry.get("prefix"))[:2]
        if "$" in suffix:
            # "335.31$" appeared on a live board. Currency signs lead a
            # number; the model's slot choice does not change money notation.
            prefix = prefix or "$"
            suffix = suffix.replace("$", "")
        if not suffix and not prefix:
            # The model forgot the unit; the manifest often knows it. A board
            # that shows 62F on Tuesday and 62 on Wednesday looks generated —
            # inference makes units a property of the data, not of model mood.
            prefix, suffix = infer_affixes((descriptions or {}).get(ref, ""))
        if suffix and apply_suffix(values[ref], suffix) != values[ref]:
            suffixes[ref] = suffix
        else:
            suffix = ""
        if prefix and apply_prefix(values[ref], prefix) != values[ref]:
            prefixes[ref] = prefix
        else:
            prefix = ""
        chosen.append((
            ref,
            Tile(
                label=label_text,
                value=apply_prefix(apply_suffix(values[ref], suffix), prefix),
                color=color if (use_color and color in ACCENT_COLORS) else None,
            ),
        ))
        seen.add(ref)

    # Identifiers are never tiles: a digitless value whose ref names a
    # symbol/station/etc becomes the label of its plugin's first numeric
    # tile, or disappears. GOOG's tile and CHANGE's tile become one row
    # reading GOOG -4.10%, whichever order the model listed them in.
    identifiers: dict[str, str] = {}
    for ref, tile in chosen:
        if _is_identifier_ref(ref) and not any(c.isdigit() for c in tile.value):
            identifiers.setdefault(ref.partition(".")[0], tile.value)

    if identifiers:
        relabelled: list[tuple[str, Tile]] = []
        for ref, tile in chosen:
            plugin_id = ref.partition(".")[0]
            if _is_identifier_ref(ref) and not any(c.isdigit() for c in tile.value):
                continue  # the identifier itself never survives as a tile
            name = identifiers.get(plugin_id)
            if name and any(c.isdigit() for c in tile.value):
                tile = Tile(label=name, value=tile.value, color=tile.color)
                del identifiers[plugin_id]
            relabelled.append((ref, tile))
        chosen = relabelled
        seen = {ref for ref, _ in chosen}

    chosen = chosen[:capacity]

    # Pins are a promise; honour them even when the model forgot.
    pinned_refs = [r for r in pinned if r in values and r in watched]
    for ref in pinned_refs:
        if ref in seen:
            continue
        if len(chosen) >= capacity:
            for index in range(len(chosen) - 1, -1, -1):
                if chosen[index][0] not in pinned_refs:
                    chosen.pop(index)
                    break
            else:
                break
        chosen.append((
            ref,
            Tile(label=labels.get(ref) or default_label(ref), value=values[ref]),
        ))
        seen.add(ref)

    # A board where everything is colored says nothing: the first three
    # accents survive, the rest go plain. Enforced here because the reader
    # was promised restraint, and promises to readers are not left to the
    # model's mood.
    kept = 0
    for index, (ref, tile) in enumerate(chosen):
        if tile.color:
            kept += 1
            if kept > 5:
                chosen[index] = (ref, Tile(label=tile.label, value=tile.value, color=None))

    if not chosen:
        raise ValidationError("No usable tiles in response")

    return GridResult(
        tiles=[tile for _, tile in chosen],
        refs=[ref for ref, _ in chosen],
        banner=banner,
        banner_color=banner_color,
        subtitle=subtitle,
        layout=layout,
        suffixes=suffixes,
        prefixes=prefixes,
        headline=sanitize(str(data.get("headline") or "")),
        reason=sanitize(str(data.get("reason") or "")),
        # The log is prompt context, never board text, so it keeps its case
        # and punctuation.
        log=" ".join(str(data.get("log") or "").split()),
        thinking=_thinking(data),
    )


def validate_prose(
    payload: object,
    *,
    geo: Geometry,
    current: dict[str, str],
    previous: dict[str, str],
) -> ProseResult:
    """Check a generated sentence for fit, charset, and invented numbers."""
    data = _as_dict(payload)
    raw = data.get("text")
    if not isinstance(raw, str) or not raw.strip():
        raise ValidationError("Response had no 'text'")

    # Placeholders are verified against the stats — refs and filters both —
    # then protected from the sanitizer (which strips unknown braces) by
    # validating the *rendered* form while storing the template.
    for ref, flt in _PLACEHOLDER_RE.findall(raw):
        if ref not in current:
            raise ValidationError(
                f"Placeholder {{{ref}}} names a stat that was not supplied"
            )
        if flt and flt not in _FILTERS:
            raise ValidationError(
                f"Unknown filter |{flt}; the only filters are: "
                + ", ".join(sorted(_FILTERS))
            )

    rendered = sanitize(render_prose(raw, current)).strip()
    # Letters only: a period before a digit is a decimal point, and adding a
    # space there once split 335.31 into fragments this very validator then
    # rejected.
    rendered = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", rendered)
    if not rendered:
        raise ValidationError("Text was empty after removing unrenderable characters")
    # Store the template when placeholders exist so values stay live on the
    # board; plain text stores its cleaned form.
    text = " ".join(str(raw).split()) if _PLACEHOLDER_RE.search(raw) else rendered

    allowed = allowed_numbers(current, previous)
    for token in extract_numbers(rendered):
        if token not in allowed:
            raise ValidationError(
                f"Text contains the number {token}, which was not supplied"
            )

    # A framed colored headline takes the top row, so the text has one
    # fewer to wrap into — unbudgeted, it ends mid-thought on the board.
    headline = sanitize(str(data.get("headline") or ""))
    prose_hue = str(data.get("banner_color") or "").lower()
    rows = geo.rows - (1 if (headline and prose_hue in ACCENT_COLORS) else 0)
    if not fits(rendered, rows, geo.cols):
        raise ValidationError(f"Text does not fit {rows} rows of {geo.cols}")


    return ProseResult(
        text=text,
        headline=headline,
        banner_color=prose_hue if prose_hue in ACCENT_COLORS else None,
        reason=sanitize(str(data.get("reason") or "")),
        log=" ".join(str(data.get("log") or "").split()),
        thinking=_thinking(data),
    )


def validate_auto(
    payload: object,
    *,
    watchlist: list[str],
    pinned: list[str],
    values: dict[str, str],
    labels: dict[str, str],
    geo: Geometry,
    use_color: bool,
    previous: dict[str, str] | None = None,
    descriptions: dict[str, str] | None = None,
) -> "GridResult | ProseResult":
    """Route an auto-mode reply to whichever validator fits its shape.

    The declared "format" wins; a model that forgets it is judged by what it
    actually sent — text is prose, tiles are a grid.
    """
    data = _as_dict(payload)
    declared = str(data.get("format") or "").lower()
    is_prose = declared == "prose" or (declared != "grid" and "text" in data)
    if is_prose:
        return validate_prose(
            data, geo=geo, current=values, previous=previous or {}
        )
    return validate_grid(
        data, watchlist=watchlist, pinned=pinned, values=values, labels=labels,
        geo=geo, use_color=use_color, previous=previous, descriptions=descriptions,
    )
