"""Board charset rules.

The split-flap board renders uppercase letters, digits, a short punctuation
set, and solid colour tiles. Everything else has no flap and must be removed
before it reaches the board. Colour markers are single-brace tokens that each
occupy exactly one cell, so width accounting cannot use ``len()``.
"""

import re

VALID_COLORS = frozenset(
    {"red", "orange", "yellow", "green", "blue", "violet", "purple", "white", "black"}
)

# The subset a model may use to draw attention. White and black are the board's
# own background tones, so as "accents" they are invisible or merely noisy.
ACCENT_COLORS = ("red", "orange", "yellow", "green", "blue", "violet")

# Single-brace markers, matching src/text_to_board.py's COLOR_MARKER_PATTERN.
COLOR_MARKER_RE = re.compile(
    r"\{(?:red|orange|yellow|green|blue|violet|purple|white|black)\}",
    re.IGNORECASE,
)

# Space plus the punctuation with a real flap. Deliberately excludes the
# degree sign (code 62), whose glyph depends on the hardware.
ALLOWED_PUNCTUATION = " !@#$()-+&=;:'\"%,./?"
_ALLOWED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" + ALLOWED_PUNCTUATION)


def sanitize(text: str) -> str:
    """Uppercase *text* and drop anything the board cannot render.

    Colour markers survive intact even though braces are not renderable.
    """
    out: list[str] = []
    pos = 0
    while pos < len(text):
        match = COLOR_MARKER_RE.match(text, pos)
        if match:
            out.append(match.group(0).lower())
            pos = match.end()
            continue
        char = text[pos].upper()
        if char in _ALLOWED:
            out.append(char)
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
