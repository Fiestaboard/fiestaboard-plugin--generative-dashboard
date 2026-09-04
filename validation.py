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
from .charset import ACCENT_COLORS, sanitize, truncate
from .layout import Geometry, Tile, fits

# A run of digits with internal separators: 42, 94,120, 1.25
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Longest unit worth appending. " MPH" is 4; anything longer is prose.
_MAX_SUFFIX = 5


class ValidationError(Exception):
    """The model's answer cannot be shown, and should be retried or dropped."""


@dataclass(frozen=True)
class GridResult:
    tiles: list[Tile]
    refs: list[str]  # source ref per tile, parallel to ``tiles``
    banner: str
    banner_color: str | None
    suffixes: dict[str, str]  # unit per ref, e.g. {"wx.temp": "F"}
    headline: str
    reason: str
    log: str


@dataclass(frozen=True)
class ProseResult:
    text: str
    headline: str
    reason: str
    log: str


def extract_numbers(text: str) -> list[str]:
    """Every numeric token in *text*, with separators preserved."""
    return _NUMBER_RE.findall(text)


def allowed_numbers(current: dict[str, str], previous: dict[str, str]) -> set[str]:
    """Numbers the model is permitted to write: those it was given."""
    allowed: set[str] = set()
    for source in (current, previous):
        for value in source.values():
            allowed.update(extract_numbers(str(value)))
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


def apply_suffix(value: str, suffix: str) -> str:
    """Append a unit only where one is missing.

    Values that already end in letters carry their own unit ("6.2MI",
    "7:36 PM"); appending another produces 6.2MIMI. Only a value ending in a
    digit or a percent sign is bare enough to need one.
    """
    if not suffix or not value or value[-1] not in "0123456789":
        return value
    return value + suffix


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
) -> GridResult:
    """Turn a model response into placeable tiles, or raise."""
    data = _as_dict(payload)
    raw_tiles = data.get("tiles")
    if not isinstance(raw_tiles, list):
        raise ValidationError("Response had no 'tiles' list")

    watched = set(watchlist)
    banner = truncate(sanitize(str(data.get("banner") or "")), geo.cols)
    banner_hue = str(data.get("banner_color") or "").lower()
    banner_color = banner_hue if (use_color and banner_hue in ACCENT_COLORS) else None
    body_rows = geo.rows - (1 if banner else 0)
    capacity = max(1, geo.tile_columns * body_rows)

    chosen: list[tuple[str, Tile]] = []
    seen: set[str] = set()
    suffixes: dict[str, str] = {}
    for entry in raw_tiles:
        if not isinstance(entry, dict):
            continue
        ref = str(entry.get("variable") or "")
        if ref not in watched or ref in seen or ref not in values:
            continue
        color = str(entry.get("color") or "").lower()
        suffix = clean_suffix(entry.get("suffix"))
        if suffix and apply_suffix(values[ref], suffix) != values[ref]:
            suffixes[ref] = suffix
        else:
            suffix = ""
        chosen.append((
            ref,
            Tile(
                label=(
                    labels.get(ref)
                    or sanitize(str(entry.get("label") or ""))
                    or default_label(ref)
                ),
                value=apply_suffix(values[ref], suffix) if suffix else values[ref],
                color=color if (use_color and color in ACCENT_COLORS) else None,
            ),
        ))
        seen.add(ref)

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

    if not chosen:
        raise ValidationError("No usable tiles in response")

    return GridResult(
        tiles=[tile for _, tile in chosen],
        refs=[ref for ref, _ in chosen],
        banner=banner,
        banner_color=banner_color,
        suffixes=suffixes,
        headline=sanitize(str(data.get("headline") or "")),
        reason=sanitize(str(data.get("reason") or "")),
        # The log is prompt context, never board text, so it keeps its case
        # and punctuation.
        log=" ".join(str(data.get("log") or "").split()),
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

    text = sanitize(raw).strip()
    if not text:
        raise ValidationError("Text was empty after removing unrenderable characters")

    allowed = allowed_numbers(current, previous)
    for token in extract_numbers(text):
        if token not in allowed:
            raise ValidationError(
                f"Text contains the number {token}, which was not supplied"
            )

    if not fits(text, geo.rows, geo.cols):
        raise ValidationError(f"Text does not fit {geo.rows} rows of {geo.cols}")

    return ProseResult(
        text=text,
        headline=sanitize(str(data.get("headline") or "")),
        reason=sanitize(str(data.get("reason") or "")),
        log=" ".join(str(data.get("log") or "").split()),
    )
