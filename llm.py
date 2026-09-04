"""Talking to an OpenAI-compatible endpoint, and telling it what we need.

Runs on a worker thread, never in the render path, so a generous timeout is
safe. The prompts carry the board's real geometry and budget rather than
letting the model infer them.
"""

import json
import logging
import re
from datetime import datetime

import requests

from .charset import ACCENT_COLORS
from .layout import Geometry, column_inner
from .when import describe_now

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

_CHARSET_RULES = (
    "The board renders UPPERCASE A-Z, digits, space, and only these symbols: "
    "! @ # $ ( ) - + & = ; : ' \" % , . / ? "
    "It has no lowercase, no arrows, no pipes, no asterisks, no brackets, and "
    "no degree sign. Never use them."
)


class LLMError(Exception):
    """The endpoint failed, or its answer was not usable JSON.

    ``retryable`` separates the two: a model that emitted broken JSON will
    often get it right when asked again, whereas an unreachable endpoint will
    still be unreachable a millisecond later.
    """

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class DashboardLLM:
    """Minimal chat-completions client for any OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        # Generous on purpose: this runs on a worker thread, never the
        # render path, and a shared local model under load can take a
        # while. A timeout costs a whole cycle; patience costs nothing.
        timeout: int = 90,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system: str, user: str) -> dict:
        """Send one prompt pair and return the parsed JSON object."""
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "temperature": self.temperature,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as exc:
            raise LLMError(f"Request failed: {exc}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"Malformed response: {exc}", retryable=True) from exc

        cleaned = _FENCE_RE.sub("", content).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Response was not JSON: {exc}", retryable=True) from exc
        if not isinstance(parsed, dict):
            raise LLMError("Response JSON was not an object", retryable=True)
        return parsed


def describe_variables(
    refs: list[str],
    labels: dict[str, str],
    notes: dict[str, str],
    current: dict[str, str],
    previous: dict[str, str],
    label_budget: int | None = None,
    wide_budget: int | None = None,
    descriptions: dict[str, str] | None = None,
) -> str:
    """One line per variable: name, label, note, what it did, and how much
    room its label has once the value is placed.

    ``label_budget`` is the tile's inner width. Without it the model writes
    "TIME" beside a seven-character clock in a ten-cell column, and the
    renderer has to cut it to "TI".
    """
    lines: list[str] = []
    for ref in refs:
        if ref not in current:
            continue
        parts = [ref, f'label="{labels.get(ref, "")}"', f'now="{current[ref]}"']
        if label_budget is not None:
            width = len(str(current[ref]))
            # A value too wide for a column is given the whole row, so its
            # label has the board's full width to work with.
            budget = label_budget
            if wide_budget is not None and width > label_budget - 4:
                budget = wide_budget
            parts.append(f"label_max={max(0, budget - width - 1)}")
        if ref in previous and previous[ref] != current[ref]:
            parts.append(f'was="{previous[ref]}"')
        if descriptions and descriptions.get(ref):
            parts.append(f'desc="{descriptions[ref]}"')
        if notes.get(ref):
            parts.append(f'meaning="{notes[ref]}"')
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def _context_block(
    geo: Geometry,
    refs: list[str],
    labels: dict[str, str],
    notes: dict[str, str],
    current: dict[str, str],
    previous: dict[str, str],
    previous_board: list[str],
    now: "datetime | None" = None,
    journal: str = "",
    descriptions: dict[str, str] | None = None,
) -> str:
    board = "\n".join(previous_board) if previous_board else "(nothing yet)"
    when = describe_now(now) if now is not None else ""
    # The narrowest column is the one that gives up a cell to the gutter.
    described = describe_variables(
        refs, labels, notes, current, previous,
        label_budget=column_inner(geo),
        wide_budget=geo.cols if geo.tile_columns > 1 else None,
        descriptions=descriptions,
    )
    return (
        (f"RIGHT NOW: {when}.\n\n" if when else "")
        + (f"EARLIER TODAY:\n{journal}\n\n" if journal else "")
        + f"BOARD: {geo.rows} rows of {geo.cols} columns.\n\n"
        f"AVAILABLE STATS:\n{described}\n\n"
        f"CURRENTLY ON THE BOARD:\n{board}\n"
    )


def _with_extra(system: str, extra_instructions: str) -> str:
    return system + (f"\n\n{extra_instructions}" if extra_instructions.strip() else "")


def build_grid_prompt(
    *,
    geo: Geometry,
    refs: list[str],
    labels: dict[str, str],
    notes: dict[str, str],
    current: dict[str, str],
    previous: dict[str, str],
    previous_board: list[str],
    use_color: bool,
    extra_instructions: str,
    now: datetime | None = None,
    journal: str = "",
    descriptions: dict[str, str] | None = None,
) -> tuple[str, str]:
    """System and user prompts for tile-based composition."""
    if use_color:
        colour_rule = (
            'Set "color" on a tile to show the *level* of that stat, not merely '
            "that it changed: green when a reading is good or low, yellow or "
            "orange as it climbs, red when it is bad or high, blue for cold. A "
            "UV index, an air quality number or a pollen count all read this way. "
            "Leave a stat uncolored when its level means nothing — a clock, a "
            "date and a ticker have no level. Color at most three tiles; a board "
            "where everything is colored says nothing at all.\n\n"
            'Set "banner_color" to color the title itself, which is where color '
            "reads best of all. Valid colors: "
            f"{', '.join(ACCENT_COLORS)}.\n\n"
        )
        example = (
            '{"tiles": [{"label": "AQI", "variable": "air.aqi", "color": "red"}, '
            '{"label": "NOW", "variable": "wx.temp", "suffix": "F"}], '
            '"banner": "AIR QUALITY", "banner_color": "red", '
            '"subtitle": "KEEP WINDOWS SHUT", "headline": "AQI 168", '
            '"reason": "why you changed it", '
            '"log": "one line on how things stand, for your future self"}'
        )
    else:
        colour_rule = ""
        example = (
            '{"tiles": [{"label": "AQI", "variable": "air.aqi"}], '
            '"banner": "AIR QUALITY", "headline": "AQI 168", '
            '"reason": "why you changed it", '
            '"log": "one line on how things stand, for your future self"}'
        )

    system = (
        "You lay out a stats dashboard for a split-flap board.\n\n"
        f"You may place at most {geo.tile_budget} tiles, arranged in "
        f"{geo.tile_columns} column(s) of {geo.tile_width} cells, reading left "
        "to right then down. The most important stat goes first.\n\n"
        "You choose WHICH stats appear, their order, and what they are called. "
        "You never type a value: name the variable and the value is filled in "
        "for you.\n\n"
        "Each stat below carries a label_max: the number of cells its label "
        "may use once its value is placed. Never exceed it — a longer label is "
        "cut off mid-word. Abbreviate to fit: PRESSURE at label_max=4 becomes "
        "PRES. A stat with a long value is given a whole row to itself, which "
        "is why some label_max values are generous.\n\n"
        "Skip any stat whose value is a placeholder — UNKNOWN, N/A, NONE, TEST, or an empty reading. Showing them wastes the board.\n\n"
        "Fill the board. Empty rows look broken, so use the slots you have "
        "unless there is genuinely nothing else worth showing.\n\n"
        'A bare number is ambiguous: 62 what? Set "suffix" on a tile to the '
        "unit its desc implies — F, %, MPH, KM, MI, MIN — so the board shows 62F "
        "and 7.2MPH. Time spans especially: a bare 4 could be four minutes or "
        "four trains, so write 4MIN. No suffix for unitless counts or indexes, "
        "and none for values that already carry their unit.\n\n"
        "Compose the board around ONE coherent theme — the weather story, the "
        "market story, the transit story — chosen for the time of day and "
        "what is actually happening. A themed board reads like a page; a "
        "grab-bag of unrelated stats reads like noise. Off-theme stats earn a "
        "slot only when they are genuinely urgent. Never spend a tile "
        "repeating something already in the title: if the banner says SAN "
        "FRANCISCO, no tile should say SAN FRANCISCO again.\n\n"
        "Keep labels in the same column the same length where you can — NOW, "
        "LIKE, HIGH, LOW — so the values line up beneath each other.\n\n"
        "Some values mean nothing on their own. A ticker symbol without its "
        "price, a station name without its arrival time, a moon phase name "
        "without its illumination — anything that merely identifies what "
        "another stat refers to. Put such a value directly beside the stat it "
        "belongs to, or fold it into the title, or leave it out. Never give it "
        "a tile alone. Use common sense about which values those are.\n\n"
        "Design the board like a made page, not a printout. Pick the pattern "
        "that fits the moment and fill it completely — on a six-row board use "
        "all six rows.\n\n"
        "PATTERN PAGE — the everyday themed board. Title, subtitle for the "
        "context line, then paired stats:\n"
        "  {blue}{blue} SAN FRANCISCO {blue}{blue}\n"
        "  {blue} LIGHT RAIN, 8 AM {blue}\n"
        "  NOW    62F RAIN   87%\n"
        "  LIKE   61F WIND 7.2MPH\n"
        "  HIGH   67F UV      0.1\n"
        "  LOW    59F SET 7:36 PM\n\n"
        "PATTERN ALERT — when one thing dominates. Title names the problem, "
        "subtitle says what to do, stats support it:\n"
        "  {red}{red} AIR QUALITY {red}{red}\n"
        "  {red} KEEP WINDOWS SHUT {red}\n"
        "  AQI    168 PM25    89\n"
        "  NOW    62F WIND 9.6MPH\n\n"
        'Set "subtitle" for that second header line. Note the ticker rule at '
        "work: a stock board would be titled GOOG with the price beneath, "
        "never a tile spending itself on the word GOOG.\n\n"
        "Numbers in the banner and subtitle obey the tile rule: only digits "
        "that appear in the stats you were given. Words are yours; numbers "
        "are not.\n\n"
        'Write one short "log" line describing how things stand — "fog thick '
        'since morning", "AQI climbing all afternoon". You will be shown your '
        "own recent log lines next time, and they are the only memory you have "
        "of what came before, so make them worth reading.\n\n"
        "Weight the board for the time and date given above. A weekday "
        "commute hour makes transit and traffic matter; a hot afternoon makes "
        "air quality matter; late at night almost nothing is urgent and "
        "something light is fine. A holiday or a notable date outranks routine "
        "numbers. Use your own judgement about what the date means.\n\n"

        f"{colour_rule}"
        'Optionally set "banner" to one short line across the top when '
        f"something deserves a sentence. A banner costs {geo.tile_columns} "
        "tile(s) of budget.\n\n"
        "Keep stats that have not changed in the positions they already "
        "occupy. Moving things for no reason makes the board noisy.\n\n"
        f"{_CHARSET_RULES}\n\n"
        f"Reply with JSON only:\n{example}"
    )
    return (
        _with_extra(system, extra_instructions),
        _context_block(
            geo, refs, labels, notes, current, previous, previous_board, now, journal,
            descriptions,
        ),
    )


def build_prose_prompt(
    *,
    geo: Geometry,
    refs: list[str],
    labels: dict[str, str],
    notes: dict[str, str],
    current: dict[str, str],
    previous: dict[str, str],
    previous_board: list[str],
    extra_instructions: str,
    now: datetime | None = None,
    journal: str = "",
    descriptions: dict[str, str] | None = None,
) -> tuple[str, str]:
    """System and user prompts for sentence composition."""
    system = (
        "You write a very short status summary for a split-flap board.\n\n"
        f"Aim for roughly {int(geo.prose_budget * 0.8)} characters and never "
        f"exceed {geo.prose_budget}. It wraps at {geo.cols} columns across "
        f"{geo.rows} rows. Two or three short sentences fill a board well; a "
        "single stub line wastes it.\n\n"
        "Shape it like a bulletin: lead with what changed and what to do "
        "about it, then a line of context — how the day has run, or one more "
        "stat worth knowing. EXAMPLE: AQI JUMPED FROM 31 TO 168 THIS "
        "AFTERNOON. KEEP THE WINDOWS SHUT. OTHERWISE MILD AT 62F WITH LIGHT "
        "WIND.\n\n"
        "Say what changed and why it matters. Lead with the most important "
        "thing. Skip anything that has not moved unless there is room.\n\n"
        'Write one short "log" line describing how things stand — "fog thick '
        'since morning", "AQI climbing all afternoon". You will be shown your '
        "own recent log lines next time, and they are the only memory you have "
        "of what came before, so make them worth reading.\n\n"
        "Weight the board for the time and date given above. A weekday "
        "commute hour makes transit and traffic matter; a hot afternoon makes "
        "air quality matter; late at night almost nothing is urgent and "
        "something light is fine. A holiday or a notable date outranks routine "
        "numbers. Use your own judgement about what the date means.\n\n"

        "Skip any stat whose value is a placeholder — UNKNOWN, N/A, NONE, TEST, or an empty reading. Showing them wastes the board.\n\n"
        "Numbers must also keep their meaning, not just their digits: a price "
        "is a price, a total is a total. Never say something rose, fell, is up "
        "or is down unless its was/now values actually show that direction — "
        "writing UP next to a plain price turns a true number into a false "
        "sentence.\n\n"
        "CRITICAL: use the supplied values exactly as written. Do not compute, "
        "do not round, do not abbreviate, and do not invent any number. If a "
        "value reads 94,120 then write 94,120. Never state a percentage or a "
        "difference that was not given to you.\n\n"
        f"{_CHARSET_RULES}\n\n"
        "Reply with JSON only:\n"
        '{"text": "AQI ROSE FROM 31 TO 168.", "headline": "AQI 168", '
        '"reason": "why you changed it", '
        '"log": "one line on how things stand, for your future self"}'
    )
    return (
        _with_extra(system, extra_instructions),
        _context_block(
            geo, refs, labels, notes, current, previous, previous_board, now, journal,
            descriptions,
        ),
    )
