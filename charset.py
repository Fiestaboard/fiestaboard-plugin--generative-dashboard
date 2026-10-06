"""Board charset rules.

The split-flap board renders uppercase letters, digits, a short punctuation
set, and solid colour tiles. Everything else has no flap and must be removed
before it reaches the board. Colour markers are single-brace tokens that each
occupy exactly one cell, so width accounting cannot use ``len()``.
"""

import re
from dataclasses import dataclass

VALID_COLORS = frozenset(
    {"red", "orange", "yellow", "green", "blue", "violet", "purple", "white", "black"}
)

# The subset a model may use to draw attention. White and black are the board's
# own background tones, so as "accents" they are invisible or merely noisy.
ACCENT_COLORS = ("red", "orange", "yellow", "green", "blue", "violet")

# Single-brace markers, matching src/text_to_board.py's COLOR_MARKER_PATTERN.
COLOR_MARKER_RE = re.compile(
    r"\{(?:red|orange|yellow|green|blue|violet|purple|white|black|6[3-9]|7[01])\}",
    re.IGNORECASE,
)

# Space plus the punctuation with a real flap. Deliberately excludes the
# degree sign (code 62), whose glyph depends on the hardware.
ALLOWED_PUNCTUATION = " !@#$()-+&=;:'\"%,./?"
_ALLOWED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" + ALLOWED_PUNCTUATION)


@dataclass(frozen=True)
class Style:
    """How text reaches the display this dashboard is drawn on.

    The default is the split-flap board every FiestaBoard core before 10
    assumed: capitals, the flap punctuation, colour as solid tiles. A display
    that says more (FiestaBoard 10's ``self.board.display``) gets lowercase,
    its own character set, and colour as coloured *text*.
    """

    color_text: bool = False
    mixed_case: bool = False
    chars: frozenset[str] | None = None
    # FiestaBoard core's own description of the display, for the prompt.
    brief: str = ""
    kind: str = "split-flap board"


FLAP = Style()


def style_for(display: object) -> Style:
    """The :class:`Style` for a board's ``display`` profile (``None`` = split-flap)."""
    if display is None:
        return FLAP
    try:
        color_text = bool(display.supports("color_text"))
        mixed_case = bool(display.supports("lowercase"))
        brief = str(display.ai_brief())
    except (AttributeError, TypeError, ValueError):
        return FLAP
    chars = getattr(display, "chars", "") or ""
    technology = getattr(display, "technology", "")
    kind = {
        "led_matrix": "full-colour LED pixel display" if getattr(display, "color", "") == "rgb" else "LED pixel display",
        "screen": "screen",
    }.get(technology, "split-flap board")
    if not (color_text or mixed_case):
        return Style(brief=brief, kind=kind)
    return Style(
        color_text=color_text,
        mixed_case=mixed_case,
        chars=frozenset(chars) if chars else None,
        brief=brief,
        kind=kind,
    )


def sanitize(text: str, style: Style = FLAP) -> str:
    """Drop anything the display cannot render; uppercase unless it draws lowercase.

    Colour markers survive intact even though braces are not renderable. On a
    display with its own character set, that set decides what survives.
    """
    allowed = (style.chars | {" "}) if style.chars else _ALLOWED
    out: list[str] = []
    pos = 0
    while pos < len(text):
        match = COLOR_MARKER_RE.match(text, pos)
        if match:
            out.append(match.group(0).lower())
            pos = match.end()
            continue
        char = text[pos] if style.mixed_case else text[pos].upper()
        if char in allowed:
            out.append(char)
        elif style.mixed_case and char.upper() in allowed:
            out.append(char.upper())
        pos += 1
    return "".join(out)


def cell_width(text: str) -> int:
    """Width of *text* in board cells, counting each colour marker as one."""
    return len(COLOR_MARKER_RE.sub("\x00", text))


def truncate(text: str, width: int) -> str:
    """Truncate *text* to *width* cells without splitting a colour marker."""
    if width <= 0:
        return ""
    out: list[str] = []
    used = 0
    pos = 0
    while pos < len(text) and used < width:
        match = COLOR_MARKER_RE.match(text, pos)
        if match:
            out.append(match.group(0))
            pos = match.end()
        else:
            out.append(text[pos])
            pos += 1
        used += 1
    return "".join(out)
