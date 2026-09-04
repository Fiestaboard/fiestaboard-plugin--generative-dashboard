"""Score generated boards the way a skill eval scores transcripts.

Fixed scenarios (deterministic inputs) x N runs against the live model, each
board judged by a rubric of the quality rules the prompts promise: units on
bare numbers, one coherent theme, no orphan identifiers, color restraint, no
title repetition, decent fill, honest labels. The score makes prompt changes
measurable instead of vibed.

Run inside the dev container:
  docker exec -e PYTHONPATH=/app -w /tmp/gd fiestaboard-dev \
      python scripts/eval_boards.py [--runs 3] [--endpoint URL] [--model NAME]
"""

import argparse
import json
import re
import sys
from collections import Counter

sys.path.insert(0, ".")

from plugins.generative_dashboard.charset import COLOR_MARKER_RE  # noqa: E402
from plugins.generative_dashboard.layout import geometry, render_grid, wrap_center  # noqa: E402
from plugins.generative_dashboard.llm import (  # noqa: E402
    DashboardLLM,
    LLMError,
    build_grid_prompt,
    build_prose_prompt,
)
from plugins.generative_dashboard.validation import (  # noqa: E402
    ValidationError,
    validate_grid,
    validate_prose,
)
from datetime import datetime  # noqa: E402

# --- scenarios: deterministic inputs, the model is the only variable --------

WEATHER = {
    "weather.temperature": "62",
    "weather.feels_like": "61",
    "weather.humidity": "98",
    "weather.wind_speed": "9.6",
    "weather.visibility": "6.2MI",
    "weather.sunset": "7:36 PM",
    "weather.uv_index": "0.2",
}
DESC = {
    "weather.temperature": "Current temperature in Fahrenheit",
    "weather.feels_like": "Feels-like temperature in Fahrenheit",
    "weather.humidity": "Relative humidity percent",
    "weather.wind_speed": "Wind speed in MPH",
    "weather.visibility": "Visibility distance",
    "weather.uv_index": "UV index (0-11)",
    "air.aqi": "US Air Quality Index",
    "air.pm25": "Fine particulate concentration",
    "stocks.symbol": "Ticker symbol of the tracked stock",
    "stocks.price": "Current share price in USD",
    "stocks.change_pct": "Percent change today",
    "transit.next_train": "Minutes until the next train",
    "transit.delay": "Line delay in minutes",
}
NOTES = {"air.aqi": "over 100 is unhealthy, over 150 keep the windows shut"}

SCENARIOS = [
    {
        "name": "aqi_spike",
        "mode": "grid",
        "geo": (6, 22),
        "current": dict(WEATHER, **{"air.aqi": "168", "air.pm25": "89"}),
        "previous": dict(WEATHER, **{"air.aqi": "31", "air.pm25": "12"}),
        "identifiers": [],
        "when": datetime(2026, 9, 3, 14, 30),
    },
    {
        "name": "market_open",
        "mode": "grid",
        "geo": (6, 22),
        "current": dict(WEATHER, **{
            "stocks.symbol": "GOOG", "stocks.price": "339.08",
            "stocks.change_pct": "-4.10",
        }),
        "previous": dict(WEATHER, **{
            "stocks.symbol": "GOOG", "stocks.price": "353.60",
            "stocks.change_pct": "0.12",
        }),
        "identifiers": ["stocks.symbol"],
        "when": datetime(2026, 9, 3, 9, 5),
    },
    {
        "name": "commute_delay",
        "mode": "grid",
        "geo": (6, 22),
        "current": dict(WEATHER, **{"transit.next_train": "4", "transit.delay": "18"}),
        "previous": dict(WEATHER, **{"transit.next_train": "6", "transit.delay": "0"}),
        "identifiers": [],
        "when": datetime(2026, 9, 3, 8, 10),
    },
    {
        "name": "note_board",
        "mode": "grid",
        "geo": (3, 15),
        "current": dict(WEATHER, **{"air.aqi": "168"}),
        "previous": dict(WEATHER, **{"air.aqi": "31"}),
        "identifiers": [],
        "when": datetime(2026, 9, 3, 14, 30),
    },
    {
        "name": "prose_spike",
        "mode": "prose",
        "geo": (6, 22),
        "current": dict(WEATHER, **{"air.aqi": "168"}),
        "previous": dict(WEATHER, **{"air.aqi": "31"}),
        "identifiers": [],
        "when": datetime(2026, 9, 3, 14, 30),
    },
    {
        "name": "prose_quiet",
        "mode": "prose",
        "geo": (6, 22),
        "current": dict(WEATHER),
        "previous": dict(WEATHER),
        "identifiers": [],
        "when": datetime(2026, 9, 3, 21, 40),
    },
]

# --- rubric -----------------------------------------------------------------

def _plain(line: str) -> str:
    return COLOR_MARKER_RE.sub("", line)


def judge_grid(result, lines, scenario, geo):
    """Each check is one promise the prompts make. True = kept."""
    checks = {}
    content = [l for l in lines if l.strip()]
    checks["fill"] = len(content) >= max(2, geo.rows // 2)
    # The bar is "nobody guesses it is generated": a designed page uses the
    # whole board, and no tile echoes its own value.
    checks["full_board"] = len(content) >= geo.rows - 1
    # Row rhythm: once the board leaves a shape it never returns to it.
    def _shape(line):
        stripped = _plain(line)
        return "full" if len(stripped) >= geo.cols - 1 and not stripped[geo.tile_width - 1 : geo.tile_width + 1].strip() else "pair"
    stat_lines = [l for l in lines if l.strip() and "{" not in l[:2] and not l.startswith(" ")]
    shapes = [_shape(l) for l in stat_lines]
    checks["rhythm"] = sum(1 for a, b in zip(shapes, shapes[1:]) if a != b) <= 1
    checks["no_echo_label"] = not any(
        t.label.strip() and t.label.strip().upper() == t.value.strip().upper()
        for t in result.tiles
    )
    checks["title"] = bool(result.banner)
    # Header framing is design, not signalling — only tile accents count.
    checks["color_restraint"] = sum(1 for t in result.tiles if t.color) <= 3
    # A unit is owed only where the description names one; AQI and UV are
    # honestly unitless.
    unit_words = re.compile(r"MPH|Fahrenheit|percent|USD|minutes|distance", re.I)
    owed = [
        t for t, r in zip(result.tiles, result.refs)
        if unit_words.search(DESC.get(r, "")) and t.value
        and t.value[-1].isdigit() and t.value[0].isdigit()
    ]
    checks["units"] = len(owed) <= 1
    # UV is an honest two-character label; one character is a truncation stub.
    checks["labels_honest"] = all(
        len(t.label.strip()) >= 2 for t in result.tiles if t.label.strip()
    )
    checks["no_double_unit"] = not any(
        re.search(r"(MI|MPH|KM|F|%)\1$", t.value) for t in result.tiles
    )
    if result.banner:
        banner_words = set(_plain(result.banner).upper().split())
        checks["no_title_repeat"] = not any(
            t.value.upper() in banner_words and not any(c.isdigit() for c in t.value)
            for t in result.tiles
        )
    else:
        checks["no_title_repeat"] = True
    # The rule tightened: an identifier is never a tile at all — it becomes
    # the label of its companion stat or part of the title.
    checks["no_orphan_identifier"] = not any(
        ident in result.refs for ident in scenario["identifiers"]
    )
    themes = Counter(r.split(".")[0] for r in result.refs)
    checks["coherent_theme"] = len(themes) <= 3 if geo.rows >= 6 else len(themes) <= 2
    return checks


def judge_prose(result, scenario, geo):
    checks = {}
    changed = [r for r in scenario["current"]
               if scenario["current"][r] != scenario["previous"].get(r)]
    if changed:
        nums = {scenario["current"][r] for r in changed}
        checks["mentions_change"] = any(n in result.text for n in nums)
    else:
        checks["mentions_change"] = True
    checks["not_a_stub"] = len(result.text) >= geo.prose_budget * 0.3
    checks["title"] = True
    return checks


# --- driver -----------------------------------------------------------------

def run(endpoint, model, runs):
    client = DashboardLLM(endpoint, "eval", model, 0.4)
    totals = Counter()
    counts = Counter()
    failures = []

    for scenario in SCENARIOS:
        geo = geometry(*scenario["geo"])
        refs = sorted(scenario["current"])
        for i in range(runs):
            kwargs = dict(
                geo=geo, refs=refs, labels={}, notes=NOTES,
                current=scenario["current"], previous=scenario["previous"],
                previous_board=[], extra_instructions="",
                now=scenario["when"], descriptions=DESC,
            )
            try:
                if scenario["mode"] == "prose":
                    system, user = build_prose_prompt(**kwargs)
                    payload = client.complete(system, user)
                    res = validate_prose(payload, geo=geo,
                                         current=scenario["current"],
                                         previous=scenario["previous"])
                    checks = judge_prose(res, scenario, geo)
                    board = wrap_center(res.text, geo.rows, geo.cols)
                else:
                    system, user = build_grid_prompt(use_color=True, **kwargs)
                    payload = client.complete(system, user)
                    res = validate_grid(payload, watchlist=refs, pinned=[],
                                        values=scenario["current"], labels={},
                                        geo=geo, use_color=True,
                                        previous=scenario["previous"],
                                        descriptions=DESC)
                    board = render_grid(res.tiles, geo, banner=res.banner,
                                        banner_color=res.banner_color,
                                        subtitle=res.subtitle)
                    checks = judge_grid(res, board, scenario, geo)
                checks["validated"] = True
            except (LLMError, ValidationError) as exc:
                checks = {"validated": False}
                board = [f"<rejected: {exc}>"]

            key = scenario["name"]
            for name, ok in checks.items():
                totals[f"{key}.{name}"] += int(bool(ok))
                counts[f"{key}.{name}"] += 1
                if not ok:
                    failures.append((key, i, name, board))

    print(f"\n{'scenario.check':40} pass")
    print("-" * 52)
    scenario_scores = Counter()
    scenario_counts = Counter()
    for key in sorted(counts):
        rate = totals[key] / counts[key]
        flag = "  " if rate == 1 else ("~ " if rate >= 0.5 else "X ")
        print(f"{flag}{key:40} {totals[key]}/{counts[key]}")
        scen = key.split(".")[0]
        scenario_scores[scen] += totals[key]
        scenario_counts[scen] += counts[key]
    overall = sum(totals.values()) / max(1, sum(counts.values()))
    print("-" * 52)
    print(f"OVERALL {overall:.0%}  ({sum(totals.values())}/{sum(counts.values())} checks)")

    if failures:
        print(f"\n--- {len(failures)} failing boards (first 4) ---")
        for key, i, name, board in failures[:4]:
            print(f"[{key} run {i}] failed {name}:")
            for line in board:
                print("   |" + str(line) + "|")
    return overall


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://host.docker.internal:8080/v1")
    ap.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    run(args.endpoint, args.model, args.runs)
