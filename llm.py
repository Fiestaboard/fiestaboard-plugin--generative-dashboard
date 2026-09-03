"""Talking to an OpenAI-compatible endpoint, and telling it what we need.

Runs on a worker thread, never in the render path, so a generous timeout is
safe. The prompts carry the board's real geometry and budget rather than
letting the model infer them.
"""

import json
import logging
import re

import requests

from .layout import Geometry

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

_CHARSET_RULES = (
    "The board renders UPPERCASE A-Z, digits, space, and only these symbols: "
    "! @ # $ ( ) - + & = ; : ' \" % , . / ? "
    "It has no lowercase, no arrows, no pipes, no asterisks, no brackets, and "
    "no degree sign. Never use them."
)


class LLMError(Exception):
    """The endpoint failed, or its answer was not usable JSON."""


class DashboardLLM:
    """Minimal chat-completions client for any OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        timeout: int = 30,
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
            raise LLMError(f"Malformed response: {exc}") from exc

        cleaned = _FENCE_RE.sub("", content).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Response was not JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMError("Response JSON was not an object")
        return parsed


def describe_variables(
    refs: list[str],
    labels: dict[str, str],
    notes: dict[str, str],
    current: dict[str, str],
    previous: dict[str, str],
) -> str:
    """One line per variable: name, label, note, and what it did."""
    lines: list[str] = []
    for ref in refs:
        if ref not in current:
            continue
        parts = [ref, f'label="{labels.get(ref, "")}"', f'now="{current[ref]}"']
        if ref in previous and previous[ref] != current[ref]:
            parts.append(f'was="{previous[ref]}"')
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
) -> str:
    board = "\n".join(previous_board) if previous_board else "(nothing yet)"
    return (
        f"BOARD: {geo.rows} rows of {geo.cols} columns.\n\n"
        f"AVAILABLE STATS:\n{describe_variables(refs, labels, notes, current, previous)}\n\n"
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
) -> tuple[str, str]:
    """System and user prompts for tile-based composition."""
    if use_color:
        colour_rule = (
            'Set "color" on at most one or two tiles, and only when a value has '
            "moved enough to deserve attention. Valid colors: red, orange, "
            "yellow, green, blue, violet, white, black.\n\n"
        )
        example = (
            '{"tiles": [{"label": "AQI", "variable": "air.aqi", "color": "red"}], '
            '"banner": "", "headline": "AQI 168", "reason": "why you changed it"}'
        )
    else:
        colour_rule = ""
        example = (
            '{"tiles": [{"label": "AQI", "variable": "air.aqi"}], '
            '"banner": "", "headline": "AQI 168", "reason": "why you changed it"}'
        )

    system = (
        "You lay out a stats dashboard for a split-flap board.\n\n"
        f"You may place at most {geo.tile_budget} tiles, arranged in "
        f"{geo.tile_columns} column(s) of {geo.tile_width} cells, reading left "
        "to right then down. The most important stat goes first.\n\n"
        "You choose WHICH stats appear, their order, and what they are called. "
        "You never type a value: name the variable and the value is filled in "
        "for you. Labels should be short, a few characters at most.\n\n"
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
        _context_block(geo, refs, labels, notes, current, previous, previous_board),
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
) -> tuple[str, str]:
    """System and user prompts for sentence composition."""
    system = (
        "You write a very short status summary for a split-flap board.\n\n"
        f"Write at most {geo.prose_budget} characters. It will be wrapped at "
        f"{geo.cols} columns across {geo.rows} rows, so be brief.\n\n"
        "Say what changed and why it matters. Lead with the most important "
        "thing. Skip anything that has not moved unless there is room.\n\n"
        "CRITICAL: use the supplied values exactly as written. Do not compute, "
        "do not round, do not abbreviate, and do not invent any number. If a "
        "value reads 94,120 then write 94,120. Never state a percentage or a "
        "difference that was not given to you.\n\n"
        f"{_CHARSET_RULES}\n\n"
        "Reply with JSON only:\n"
        '{"text": "AQI ROSE FROM 31 TO 168.", "headline": "AQI 168", '
        '"reason": "why you changed it"}'
    )
    return (
        _with_extra(system, extra_instructions),
        _context_block(geo, refs, labels, notes, current, previous, previous_board),
    )
