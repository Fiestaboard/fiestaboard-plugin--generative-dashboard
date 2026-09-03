# Generative Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FiestaBoard plugin that composes a full-board dashboard from a curated pool of other plugins' variables, using an LLM to decide what matters right now, regenerating only when the numbers actually move.

**Architecture:** Pure-function modules (charset, layout, gate, validation, fallback) carry all the logic and are tested without mocks. A thin plugin class wires them together, reads other plugins' values through the core registry, and runs LLM calls on a one-shot worker thread so the render path never blocks. Two output modes — `grid` (model returns tiles, plugin places characters) and `prose` (model writes a sentence, plugin validates every number in it).

**Tech Stack:** Python 3.14, `requests`, pytest. Runs against FiestaBoard core `>=2.10.0` via `src.plugins.base.PluginBase`.

**Spec:** `.superpowers/specs/2026-09-03-generative-dashboard-design.md`

## Global Constraints

- Plugin id is `generative_dashboard`. Repo is `fiestaboard-plugin--generative-dashboard`.
- Board charset is uppercase only: `A-Z`, `0-9`, space, and `! @ # $ ( ) - + & = ; : ' " % , . / ?`. Nothing else renders. Never emit code 62 (degree/heart — hardware-dependent glyph).
- Color markers are single-brace and case-insensitive: `{red} {orange} {yellow} {green} {blue} {violet} {purple} {white} {black}`. Each occupies **exactly one board cell**. Never emit `{filled}` (code 71 is unavailable on the local API).
- Board dimensions always come from `self.board` (`BoardContext.rows` / `.cols`). Never hardcode 6x22.
- `fetch_data()` must never block on the network. Core drops plugins that exceed **15 seconds** during `build_template_context` (`registry.py:1533`).
- Never call `registry.build_template_context()` from `fetch_data()`. Use `registry.fetch_plugin_data(pid, board)` per plugin, with `generative_dashboard` always removed from the id set.
- Generation is only ever started when `self.board is not None`. A `None` board means this is not a real render (e.g. the settings dialog fanning out) and must not trigger a paid API call.
- Watchlist hard cap: **100** entries.
- `get_options()` must work with no API key, while disabled, and must never call the LLM.

## Test Loop

The host has no pytest. Tests run in the already-running FiestaBoard dev container, which has pytest 9.1.1, Python 3.14, and `/app/src`:

```bash
cd /Users/jeffre/workspace/fiestaboard-plugin--generative-dashboard
docker exec fiestaboard-dev rm -rf /tmp/gd
docker cp . fiestaboard-dev:/tmp/gd
docker exec -e PYTHONPATH=/app -w /tmp/gd fiestaboard-dev python -m pytest tests/ -q
```

This leaves no footprint on the host and needs no container restart. Save it as `scripts/test.sh`.

## File Structure

| File | Responsibility |
| --- | --- |
| `manifest.json` | Settings schema (incl. picker widget), emitted variables, demo pages |
| `__init__.py` | `GenerativeDashboardPlugin`: lifecycle, state, gate orchestration, worker, `get_options` |
| `charset.py` | Sanitising, cell-width accounting, marker-safe truncation |
| `layout.py` | Grid geometry, tile placement, prose wrap/centre |
| `gate.py` | Significance comparison |
| `catalog.py` | Registry access: variable catalog and watchlist value reads |
| `llm.py` | OpenAI-compatible client and prompt construction |
| `validation.py` | Grid and prose response validation |
| `fallback.py` | Deterministic grid, humour pool, setup message |
| `tests/` | One test module per source module |

Flat modules rather than a package: the loader imports the plugin's `__init__.py`, and the art plugin's precedent is top-level modules alongside it.

---

### Task 1: Charset handling

**Files:**
- Create: `charset.py`
- Create: `tests/test_charset.py`
- Create: `scripts/test.sh`

**Interfaces:**
- Consumes: nothing
- Produces: `sanitize(text: str) -> str`, `cell_width(text: str) -> int`, `truncate(text: str, width: int) -> str`, `COLOR_MARKER_RE`, `VALID_COLORS: frozenset[str]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_charset.py
"""Charset sanitising and cell-width accounting."""

from plugins.generative_dashboard.charset import (
    cell_width,
    sanitize,
    truncate,
)


def test_sanitize_uppercases_lowercase_letters():
    assert sanitize("Temp") == "TEMP"


def test_sanitize_drops_characters_the_board_cannot_render():
    # Pipes, arrows, asterisks and underscores have no flap.
    assert sanitize("A|B*C_D") == "ABCD"


def test_sanitize_keeps_allowed_punctuation():
    assert sanitize("CPU: 42% (OK)") == "CPU: 42% (OK)"


def test_sanitize_preserves_color_markers_despite_braces():
    # Braces are not renderable, but markers must survive intact.
    assert sanitize("{red}AQI") == "{red}AQI"


def test_sanitize_drops_the_degree_sign():
    # Code 62 renders as degree or heart depending on hardware.
    assert sanitize("61°F") == "61F"


def test_cell_width_counts_a_color_marker_as_one_cell():
    assert cell_width("{red}AQI") == 4


def test_cell_width_of_plain_text_is_its_length():
    assert cell_width("AQI 168") == 7


def test_truncate_never_splits_a_color_marker():
    # Cutting at 3 cells would land inside the marker; drop it whole.
    assert truncate("{red}AQI", 3) == "AQI"


def test_truncate_keeps_marker_when_it_fits():
    assert truncate("{red}AQIXX", 4) == "{red}AQI"


def test_truncate_leaves_short_text_alone():
    assert truncate("AQI", 10) == "AQI"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test.sh tests/test_charset.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'plugins.generative_dashboard.charset'`

- [ ] **Step 3: Write the implementation**

```python
# charset.py
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

# Single-brace markers, matching src/text_to_board.py's COLOR_MARKER_PATTERN.
COLOR_MARKER_RE = re.compile(
    r"\{(?:red|orange|yellow|green|blue|violet|purple|white|black)\}",
    re.IGNORECASE,
)

# Space plus the punctuation with a real flap. Deliberately excludes the
# degree sign (code 62), whose glyph depends on the hardware.
ALLOWED_PUNCTUATION = " !@#$()-+&=;:'\"%,./?"
_ALLOWED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" + ALLOWED_PUNCTUATION
)


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
```

```bash
# scripts/test.sh
#!/usr/bin/env bash
# Run the plugin's tests inside the FiestaBoard dev container.
# The host has no pytest, and FiestaBoard's CLAUDE.md forbids local installs.
set -euo pipefail
cd "$(dirname "$0")/.."
docker exec fiestaboard-dev rm -rf /tmp/gd
docker cp . fiestaboard-dev:/tmp/gd >/dev/null
docker exec -e PYTHONPATH=/app -w /tmp/gd fiestaboard-dev \
    python -m pytest "${@:-tests/}" -q
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chmod +x scripts/test.sh && ./scripts/test.sh tests/test_charset.py`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add charset.py tests/test_charset.py scripts/test.sh
git commit -m "feat: board charset sanitising and cell-width accounting"
```

---

### Task 2: Layout geometry and rendering

**Files:**
- Create: `layout.py`
- Create: `tests/test_layout.py`

**Interfaces:**
- Consumes: `charset.sanitize`, `charset.cell_width`, `charset.truncate`
- Produces: `Geometry` (frozen dataclass: `rows`, `cols`, `tile_columns`, `tile_width`, `tile_budget`, `prose_budget`), `geometry(rows: int, cols: int) -> Geometry`, `Tile` (frozen dataclass: `label: str`, `value: str`, `color: str | None = None`), `render_grid(tiles: list[Tile], geo: Geometry, banner: str = "", use_color: bool = True) -> list[str]`, `wrap_center(text: str, rows: int, cols: int) -> list[str]`, `fits(text: str, rows: int, cols: int) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_layout.py
"""Grid geometry, tile placement, and prose wrapping."""

import pytest

from plugins.generative_dashboard.layout import (
    Geometry,
    Tile,
    fits,
    geometry,
    render_grid,
    wrap_center,
)


def test_flagship_geometry_is_two_columns_of_eleven():
    geo = geometry(6, 22)
    assert (geo.tile_columns, geo.tile_width, geo.tile_budget) == (2, 11, 12)


def test_note_geometry_is_one_narrow_column():
    geo = geometry(3, 15)
    assert (geo.tile_columns, geo.tile_width, geo.tile_budget) == (1, 15, 3)


def test_tall_note_array_keeps_one_column_but_more_rows():
    geo = geometry(12, 15)
    assert (geo.tile_columns, geo.tile_budget) == (1, 12)


def test_wide_note_array_gets_two_columns():
    geo = geometry(6, 30)
    assert (geo.tile_columns, geo.tile_width, geo.tile_budget) == (2, 15, 12)


def test_prose_budget_discounts_for_wrap_waste():
    assert geometry(6, 22).prose_budget == 112


def test_render_grid_returns_exactly_one_string_per_row():
    geo = geometry(6, 22)
    lines = render_grid([Tile("CPU", "42%")], geo)
    assert len(lines) == 6


def test_render_grid_never_exceeds_the_board_width():
    geo = geometry(6, 22)
    tiles = [Tile("LABEL", "VALUE") for _ in range(12)]
    for line in render_grid(tiles, geo):
        assert len(line) <= 22


def test_render_grid_places_tiles_left_to_right_then_down():
    geo = geometry(6, 22)
    lines = render_grid([Tile("CPU", "42%"), Tile("TEMP", "61F")], geo)
    assert lines[0].startswith("CPU")
    assert "TEMP" in lines[0]


def test_render_grid_right_aligns_the_value():
    geo = geometry(3, 15)
    lines = render_grid([Tile("CPU", "42%")], geo)
    assert lines[0] == "CPU         42%"


def test_render_grid_leaves_a_gutter_between_columns():
    geo = geometry(6, 22)
    lines = render_grid([Tile("A", "1"), Tile("B", "2")], geo)
    # First column is inset by one cell so the columns do not touch.
    assert lines[0] == "A        1 B        2"


def test_render_grid_prefixes_a_color_marker_when_a_tile_is_accented():
    geo = geometry(3, 15)
    lines = render_grid([Tile("AQI", "168", "red")], geo)
    assert lines[0].startswith("{red}")


def test_render_grid_omits_color_when_color_is_disabled():
    geo = geometry(3, 15)
    lines = render_grid([Tile("AQI", "168", "red")], geo, use_color=False)
    assert "{red}" not in lines[0]


def test_colored_tile_still_occupies_exactly_the_board_width():
    from plugins.generative_dashboard.charset import cell_width

    geo = geometry(3, 15)
    lines = render_grid([Tile("AQI", "168", "red")], geo, use_color=True)
    assert cell_width(lines[0]) == 15


def test_render_grid_puts_the_banner_on_the_first_row():
    geo = geometry(6, 22)
    lines = render_grid([Tile("CPU", "42%")], geo, banner="AIR QUALITY BAD")
    assert lines[0].strip() == "AIR QUALITY BAD"
    assert "CPU" in lines[1]


def test_banner_costs_one_row_of_tiles():
    geo = geometry(6, 22)
    tiles = [Tile(f"L{i}", str(i)) for i in range(12)]
    lines = render_grid(tiles, geo, banner="HEADS UP")
    # Twelve tiles no longer fit in the ten remaining slots; the last is dropped.
    assert "L11" not in "".join(lines)


def test_render_grid_pads_unused_rows_with_blanks():
    geo = geometry(6, 22)
    lines = render_grid([Tile("CPU", "42%")], geo)
    assert lines[5].strip() == ""


def test_wrap_center_returns_exactly_the_row_count():
    assert len(wrap_center("HELLO WORLD", 6, 22)) == 6


def test_wrap_center_breaks_on_word_boundaries():
    lines = wrap_center("AQI JUMPED FROM 31 TO 168 TODAY", 6, 12)
    assert all(len(line) <= 12 for line in lines)
    assert "JUMPED" in "".join(lines)


def test_wrap_center_vertically_centers_the_block():
    lines = wrap_center("HI", 5, 10)
    assert lines[0].strip() == ""
    assert lines[2].strip() == "HI"


def test_fits_rejects_text_that_needs_too_many_rows():
    assert not fits("WORD " * 40, 3, 15)


def test_fits_accepts_text_that_wraps_within_the_board():
    assert fits("SHORT MESSAGE", 3, 15)


def test_wrap_center_drops_overflow_rather_than_returning_extra_rows():
    lines = wrap_center("WORD " * 40, 3, 15)
    assert len(lines) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test.sh tests/test_layout.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'plugins.generative_dashboard.layout'`

- [ ] **Step 3: Write the implementation**

```python
# layout.py
"""Board geometry and character placement.

The plugin owns every character position. The model chooses what appears and
in what order; it never types spacing. That is what makes alignment identical
between cycles and width violations impossible.
"""

from dataclasses import dataclass

from .charset import cell_width, sanitize, truncate

# Minimum sensible width for a label/value pair. Below this a second column
# would leave no room for either half.
_MIN_TILE_WIDTH = 11

# Word wrapping wastes ragged-right space, so the advertised prose budget is
# discounted. The real check is fits(), which actually wraps.
_PROSE_FILL = 0.85


@dataclass(frozen=True)
class Geometry:
    """Derived layout numbers for one board size."""

    rows: int
    cols: int
    tile_columns: int
    tile_width: int
    tile_budget: int
    prose_budget: int


@dataclass(frozen=True)
class Tile:
    """One label/value stat, optionally accented with a colour tile."""

    label: str
    value: str
    color: str | None = None


def geometry(rows: int, cols: int) -> Geometry:
    """Compute the layout budget for a board of *rows* x *cols*."""
    tile_columns = max(1, cols // _MIN_TILE_WIDTH)
    tile_width = cols // tile_columns
    return Geometry(
        rows=rows,
        cols=cols,
        tile_columns=tile_columns,
        tile_width=tile_width,
        tile_budget=tile_columns * rows,
        prose_budget=int(rows * cols * _PROSE_FILL),
    )


def _render_tile(tile: Tile, width: int, use_color: bool) -> str:
    """Render one tile into exactly *width* cells."""
    prefix = ""
    if use_color and tile.color:
        prefix = "{" + tile.color.lower() + "}"

    inner = width - cell_width(prefix)
    value = truncate(sanitize(tile.value), inner)
    # The label yields first: a wrong-looking name beats a wrong number.
    label = truncate(sanitize(tile.label), max(0, inner - cell_width(value) - 1))
    gap = inner - cell_width(label) - cell_width(value)
    return prefix + label + (" " * max(0, gap)) + value


def render_grid(
    tiles: list[Tile],
    geo: Geometry,
    banner: str = "",
    use_color: bool = True,
) -> list[str]:
    """Place *tiles* into the board grid, returning exactly ``geo.rows`` lines."""
    lines: list[str] = []
    banner_text = truncate(sanitize(banner), geo.cols) if banner else ""
    if banner_text:
        lines.append(banner_text.center(geo.cols).rstrip())

    grid_rows = geo.rows - len(lines)
    capacity = geo.tile_columns * grid_rows
    placed = list(tiles[:capacity])

    for row in range(grid_rows):
        chunk = placed[row * geo.tile_columns : (row + 1) * geo.tile_columns]
        if not chunk:
            lines.append("")
            continue
        cells: list[str] = []
        for column, tile in enumerate(chunk):
            # Every column but the last gives up one cell as a gutter.
            is_last = column == geo.tile_columns - 1
            width = geo.tile_width if is_last else geo.tile_width - 1
            cells.append(_render_tile(tile, width, use_color))
        lines.append(" ".join(cells).rstrip())

    return lines[: geo.rows]


def _wrap(text: str, cols: int) -> list[str]:
    """Greedy word wrap at *cols* cells, breaking over-long words."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if cell_width(candidate) <= cols:
            current = candidate
            continue
        if current:
            lines.append(current)
        while cell_width(word) > cols:
            lines.append(truncate(word, cols))
            word = word[cols:]
        current = word
    if current:
        lines.append(current)
    return lines


def fits(text: str, rows: int, cols: int) -> bool:
    """Whether *text* wraps into at most *rows* lines of *cols* cells."""
    return len(_wrap(sanitize(text), cols)) <= rows


def wrap_center(text: str, rows: int, cols: int) -> list[str]:
    """Wrap *text* and centre the block, returning exactly *rows* lines."""
    wrapped = _wrap(sanitize(text), cols)[:rows]
    top = (rows - len(wrapped)) // 2
    out = [""] * top + [line.center(cols).rstrip() for line in wrapped]
    return (out + [""] * rows)[:rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./scripts/test.sh tests/test_layout.py`
Expected: PASS, 21 tests

- [ ] **Step 5: Commit**

```bash
git add layout.py tests/test_layout.py
git commit -m "feat: board geometry, tile placement, and prose wrapping"
```

---

### Task 3: The significance gate

**Files:**
- Create: `gate.py`
- Create: `tests/test_gate.py`

**Interfaces:**
- Consumes: nothing
- Produces: `as_number(value: object) -> float | None`, `is_material(old: object, new: object, threshold_pct: float) -> bool`, `material_changes(old: dict[str, str], new: dict[str, str], thresholds: dict[str, float], default_pct: float) -> list[str]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gate.py
"""Significance gating: the thing that keeps the board still."""

from plugins.generative_dashboard.gate import (
    as_number,
    is_material,
    material_changes,
)


def test_as_number_parses_a_plain_integer():
    assert as_number("42") == 42.0


def test_as_number_strips_grouping_commas_and_percent_signs():
    assert as_number("94,120") == 94120.0
    assert as_number("42%") == 42.0


def test_as_number_returns_none_for_text():
    assert as_number("PARTLY CLOUDY") is None


def test_change_below_the_threshold_is_not_material():
    assert not is_material("100", "104", 5.0)


def test_change_above_the_threshold_is_material():
    assert is_material("100", "106", 5.0)


def test_change_exactly_at_the_threshold_is_material():
    assert is_material("100", "105", 5.0)


def test_any_change_to_a_text_value_is_material():
    assert is_material("SUNNY", "RAINY", 50.0)


def test_identical_text_is_not_material():
    assert not is_material("SUNNY", "SUNNY", 5.0)


def test_growth_from_zero_is_material_and_does_not_divide_by_zero():
    assert is_material("0", "5", 5.0)


def test_zero_to_zero_is_not_material():
    assert not is_material("0", "0", 5.0)


def test_a_variable_appearing_is_material():
    assert material_changes({}, {"a.b": "1"}, {}, 5.0) == ["a.b"]


def test_a_variable_disappearing_is_material():
    assert material_changes({"a.b": "1"}, {}, {}, 5.0) == ["a.b"]


def test_no_changes_yields_an_empty_list():
    values = {"a.b": "1", "c.d": "SUNNY"}
    assert material_changes(values, dict(values), {}, 5.0) == []


def test_per_variable_threshold_overrides_the_global_default():
    old = {"a.b": "100"}
    new = {"a.b": "102"}
    # 2% moves, global default would ignore it, but this variable is sensitive.
    assert material_changes(old, new, {"a.b": 1.0}, 5.0) == ["a.b"]


def test_global_default_applies_when_no_override_exists():
    assert material_changes({"a.b": "100"}, {"a.b": "102"}, {}, 5.0) == []


def test_only_the_changed_variables_are_reported():
    old = {"a.b": "100", "c.d": "1"}
    new = {"a.b": "100", "c.d": "99"}
    assert material_changes(old, new, {}, 5.0) == ["c.d"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test.sh tests/test_gate.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'plugins.generative_dashboard.gate'`

- [ ] **Step 3: Write the implementation**

```python
# gate.py
"""Deciding whether anything actually happened.

This is the plugin's primary anti-jitter mechanism and its primary cost
control: when nothing has moved materially, no LLM call is made and the board
is left exactly as it is.
"""

# Guards the percentage division when the previous value was zero.
_EPSILON = 1e-9


def as_number(value: object) -> float | None:
    """Parse *value* as a number, tolerating grouping commas and a percent sign.

    Returns ``None`` when the value is not numeric, which the caller treats as
    "compare as text".
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None
    cleaned = value.strip().replace(",", "").rstrip("%").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_material(old: object, new: object, threshold_pct: float) -> bool:
    """Whether the move from *old* to *new* is worth redrawing the board for."""
    old_num = as_number(old)
    new_num = as_number(new)

    if old_num is None or new_num is None:
        return str(old) != str(new)

    delta = abs(new_num - old_num)
    if delta == 0:
        return False
    return (delta / max(abs(old_num), _EPSILON)) * 100 >= threshold_pct


def material_changes(
    old: dict[str, str],
    new: dict[str, str],
    thresholds: dict[str, float],
    default_pct: float,
) -> list[str]:
    """Return the variable refs that moved materially between two snapshots.

    A variable appearing or disappearing is always material — the board's
    contents change either way.
    """
    changed: list[str] = []
    for ref in sorted(set(old) | set(new)):
        if ref not in old or ref not in new:
            changed.append(ref)
            continue
        threshold = thresholds.get(ref, default_pct)
        if is_material(old[ref], new[ref], threshold):
            changed.append(ref)
    return changed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./scripts/test.sh tests/test_gate.py`
Expected: PASS, 16 tests

- [ ] **Step 5: Commit**

```bash
git add gate.py tests/test_gate.py
git commit -m "feat: significance gate for change detection"
```

---
### Task 4: Registry catalog and value reads

**Files:**
- Create: `catalog.py`
- Create: `tests/test_catalog.py`

**Interfaces:**
- Consumes: nothing
- Produces: `VariableChoice` (frozen dataclass: `ref`, `label`, `description`, `group`, `preview`, `disabled: bool = False`, `disabled_reason: str = ""`), `variable_catalog(exclude_plugin_id: str, max_value_width: int, query: str = "") -> list[VariableChoice]`, `read_values(refs, board, exclude_plugin_id: str) -> dict[str, str]`, `default_label(ref: str) -> str`, `_registry()`

Tests replace `catalog._registry` with a fake, so no real registry is needed.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_catalog.py
"""Reading other plugins' variables through the core registry."""

from types import SimpleNamespace

import pytest

from plugins.generative_dashboard import catalog
from plugins.generative_dashboard.catalog import (
    default_label,
    read_values,
    variable_catalog,
)


class FakeResult:
    def __init__(self, data, available=True):
        self.available = available
        self.data = data
        self.error = None


class FakeRegistry:
    def __init__(self, metadata=None, data=None, names=None):
        self._metadata = metadata or {}
        self._data = data or {}
        self._names = names or {}
        self.fetched = []

    def get_all_variables_with_metadata(self):
        return self._metadata

    def get_manifest(self, plugin_id):
        if plugin_id not in self._names:
            return None
        return SimpleNamespace(name=self._names[plugin_id])

    def fetch_plugin_data(self, plugin_id, board=None):
        self.fetched.append(plugin_id)
        if plugin_id not in self._data:
            return FakeResult(None, available=False)
        return FakeResult(self._data[plugin_id])


@pytest.fixture
def fake(monkeypatch):
    def install(registry):
        monkeypatch.setattr(catalog, "_registry", lambda: registry)
        return registry

    return install


def test_default_label_uses_the_variable_name_without_its_plugin():
    assert default_label("weather.temp_f") == "TEMP F"


def test_catalog_lists_one_choice_per_variable(fake):
    fake(FakeRegistry(
        metadata={"weather": {"temp_f": {"description": "Temp", "max_length": 6,
                                         "group": "", "preview": "61F"}}},
        names={"weather": "Weather"},
    ))
    choices = variable_catalog("generative_dashboard", 15)
    assert [c.ref for c in choices] == ["weather.temp_f"]


def test_catalog_exposes_the_live_value_as_the_preview(fake):
    fake(FakeRegistry(
        metadata={"weather": {"temp_f": {"description": "", "max_length": 6,
                                         "group": "", "preview": "61F"}}},
        names={"weather": "Weather"},
    ))
    assert variable_catalog("generative_dashboard", 15)[0].preview == "61F"


def test_catalog_groups_choices_under_the_source_plugin_name(fake):
    fake(FakeRegistry(
        metadata={"weather": {"temp_f": {"description": "", "max_length": 6,
                                         "group": "", "preview": ""}}},
        names={"weather": "Weather"},
    ))
    assert variable_catalog("generative_dashboard", 15)[0].group == "Weather"


def test_catalog_disables_our_own_variables_to_prevent_recursion(fake):
    fake(FakeRegistry(
        metadata={"generative_dashboard": {"headline": {"description": "", "max_length": 20,
                                                        "group": "", "preview": ""}}},
        names={"generative_dashboard": "Generative Dashboard"},
    ))
    choice = variable_catalog("generative_dashboard", 15)[0]
    assert choice.disabled
    assert "itself" in choice.disabled_reason.lower()


def test_catalog_disables_variables_too_wide_for_a_tile(fake):
    fake(FakeRegistry(
        metadata={"art": {"art": {"description": "", "max_length": 512,
                                  "group": "", "preview": ""}}},
        names={"art": "Art"},
    ))
    choice = variable_catalog("generative_dashboard", 15)[0]
    assert choice.disabled
    assert "512" in choice.disabled_reason


def test_catalog_keeps_disabled_choices_visible_rather_than_hiding_them(fake):
    fake(FakeRegistry(
        metadata={"art": {"art": {"description": "", "max_length": 512,
                                  "group": "", "preview": ""}}},
        names={"art": "Art"},
    ))
    assert len(variable_catalog("generative_dashboard", 15)) == 1


def test_catalog_query_filters_case_insensitively(fake):
    fake(FakeRegistry(
        metadata={"weather": {
            "temp_f": {"description": "", "max_length": 6, "group": "", "preview": ""},
            "humidity": {"description": "", "max_length": 6, "group": "", "preview": ""},
        }},
        names={"weather": "Weather"},
    ))
    refs = [c.ref for c in variable_catalog("generative_dashboard", 15, query="TEMP")]
    assert refs == ["weather.temp_f"]


def test_read_values_returns_current_values_keyed_by_ref(fake):
    fake(FakeRegistry(data={"weather": {"temp_f": 61}}))
    assert read_values(["weather.temp_f"], None, "generative_dashboard") == {"weather.temp_f": "61"}


def test_read_values_fetches_each_plugin_only_once(fake):
    registry = fake(FakeRegistry(data={"weather": {"temp_f": 61, "humidity": 40}}))
    read_values(["weather.temp_f", "weather.humidity"], None, "generative_dashboard")
    assert registry.fetched == ["weather"]


def test_read_values_never_fetches_our_own_plugin(fake):
    registry = fake(FakeRegistry(data={"generative_dashboard": {"headline": "X"}}))
    values = read_values(["generative_dashboard.headline"], None, "generative_dashboard")
    assert registry.fetched == []
    assert values == {}


def test_read_values_omits_unavailable_plugins(fake):
    fake(FakeRegistry(data={}))
    assert read_values(["weather.temp_f"], None, "generative_dashboard") == {}


def test_read_values_omits_variables_the_plugin_did_not_supply(fake):
    fake(FakeRegistry(data={"weather": {"humidity": 40}}))
    assert read_values(["weather.temp_f"], None, "generative_dashboard") == {}


def test_read_values_ignores_malformed_refs(fake):
    fake(FakeRegistry(data={"weather": {"temp_f": 61}}))
    assert read_values(["nodot"], None, "generative_dashboard") == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test.sh tests/test_catalog.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'plugins.generative_dashboard.catalog'`

- [ ] **Step 3: Write the implementation**

```python
# catalog.py
"""Reading the variable catalog and live values from the core registry.

Deliberately never calls ``registry.build_template_context()``: that fans out
to every enabled plugin on a thread pool, this plugin included, so calling it
from a fetch would re-enter this plugin on another thread. Fetching only the
plugins the watchlist names avoids the problem by construction.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariableChoice:
    """One selectable variable, as shown in the settings picker."""

    ref: str
    label: str
    description: str
    group: str
    preview: str
    disabled: bool = False
    disabled_reason: str = ""


def _registry() -> Any:
    """The core plugin registry. Indirected so tests can replace it."""
    from src.plugins import get_plugin_registry

    return get_plugin_registry()


def default_label(ref: str) -> str:
    """Human label for a ``plugin.var`` ref when the user has not set one.

    Underscores have no flap on the board, so they become spaces.
    """
    name = ref.split(".", 1)[-1]
    return name.replace("_", " ").upper()


def _plugin_name(registry: Any, plugin_id: str) -> str:
    """Display name of a plugin, falling back to its id."""
    try:
        manifest = registry.get_manifest(plugin_id)
    except Exception:
        return plugin_id
    return getattr(manifest, "name", None) or plugin_id


def variable_catalog(
    exclude_plugin_id: str,
    max_value_width: int,
    query: str = "",
) -> list[VariableChoice]:
    """Every variable the user could watch, including ones they should not.

    Unsuitable variables come back ``disabled`` with a reason rather than
    omitted, so the picker can explain itself instead of silently hiding them.
    """
    registry = _registry()
    metadata = registry.get_all_variables_with_metadata()
    needle = query.strip().lower()

    choices: list[VariableChoice] = []
    for plugin_id in sorted(metadata):
        group = _plugin_name(registry, plugin_id)
        for name in sorted(metadata[plugin_id]):
            meta = metadata[plugin_id][name] or {}
            ref = f"{plugin_id}.{name}"
            description = str(meta.get("description") or "")

            if needle and needle not in f"{ref} {description}".lower():
                continue

            disabled = False
            reason = ""
            if plugin_id == exclude_plugin_id:
                disabled = True
                reason = "A dashboard cannot watch itself."
            else:
                max_length = meta.get("max_length")
                if isinstance(max_length, int) and max_length > max_value_width:
                    disabled = True
                    reason = (
                        f"Up to {max_length} characters — too wide for a "
                        f"{max_value_width}-cell tile."
                    )

            choices.append(
                VariableChoice(
                    ref=ref,
                    label=default_label(ref),
                    description=description,
                    group=group,
                    preview=str(meta.get("preview") or ""),
                    disabled=disabled,
                    disabled_reason=reason,
                )
            )
    return choices


def read_values(
    refs: Sequence[str],
    board: Any,
    exclude_plugin_id: str,
) -> dict[str, str]:
    """Current values for *refs*, fetching each source plugin exactly once.

    Refs whose plugin is unavailable, or whose variable is absent, are omitted
    rather than blanked — the gate reads that absence as a real change.
    """
    wanted: dict[str, list[str]] = defaultdict(list)
    for ref in refs:
        plugin_id, _, name = ref.partition(".")
        if not plugin_id or not name or plugin_id == exclude_plugin_id:
            continue
        wanted[plugin_id].append(name)

    if not wanted:
        return {}

    registry = _registry()
    values: dict[str, str] = {}
    for plugin_id, names in wanted.items():
        try:
            result = registry.fetch_plugin_data(plugin_id, board)
        except Exception:
            logger.warning("Failed reading %s for the dashboard", plugin_id, exc_info=True)
            continue
        if not getattr(result, "available", False) or not getattr(result, "data", None):
            continue
        data = result.data
        for name in names:
            raw = data.get(name)
            if raw is None or isinstance(raw, list | dict):
                continue
            values[f"{plugin_id}.{name}"] = str(raw)
    return values
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./scripts/test.sh tests/test_catalog.py`
Expected: PASS, 14 tests

- [ ] **Step 5: Commit**

```bash
git add catalog.py tests/test_catalog.py
git commit -m "feat: registry-backed variable catalog and value reads"
```

---

### Task 5: Fallback rendering

**Files:**
- Create: `fallback.py`
- Create: `tests/test_fallback.py`

**Interfaces:**
- Consumes: `layout.Geometry`, `layout.Tile`, `layout.wrap_center`, `layout.fits`, `catalog.default_label`
- Produces: `OUTAGE_MESSAGES: tuple[tuple[str, str], ...]`, `deterministic_tiles(order, labels, values, pinned) -> list[Tile]`, `outage_lines(geo, last_good: str | None, index: int) -> list[str]`, `setup_lines(geo) -> list[str]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fallback.py
"""The failure ladder: what the board shows when things go wrong."""

from plugins.generative_dashboard.fallback import (
    OUTAGE_MESSAGES,
    deterministic_tiles,
    outage_lines,
    setup_lines,
)
from plugins.generative_dashboard.layout import geometry


def test_deterministic_tiles_follow_watchlist_order():
    tiles = deterministic_tiles(
        ["a.x", "b.y"], {}, {"a.x": "1", "b.y": "2"}, []
    )
    assert [t.label for t in tiles] == ["X", "Y"]


def test_deterministic_tiles_put_pinned_variables_first():
    tiles = deterministic_tiles(
        ["a.x", "b.y"], {}, {"a.x": "1", "b.y": "2"}, ["b.y"]
    )
    assert [t.label for t in tiles] == ["Y", "X"]


def test_deterministic_tiles_use_configured_labels():
    tiles = deterministic_tiles(["a.aqi"], {"a.aqi": "AIR"}, {"a.aqi": "68"}, [])
    assert tiles[0].label == "AIR"


def test_deterministic_tiles_skip_variables_with_no_value():
    tiles = deterministic_tiles(["a.x", "b.y"], {}, {"a.x": "1"}, [])
    assert [t.label for t in tiles] == ["X"]


def test_deterministic_tiles_are_never_colored():
    # No model ran, so nothing has been judged worth accenting.
    tiles = deterministic_tiles(["a.x"], {}, {"a.x": "1"}, [])
    assert tiles[0].color is None


def test_outage_lines_fill_the_whole_board():
    geo = geometry(6, 22)
    assert len(outage_lines(geo, "14:32", 0)) == 6


def test_outage_lines_report_when_data_was_last_good():
    lines = outage_lines(geometry(6, 22), "14:32", 0)
    assert "14:32" in "".join(lines)


def test_outage_lines_say_so_when_there_has_never_been_data():
    lines = outage_lines(geometry(6, 22), None, 0)
    assert "NO DATA YET" in "".join(lines)


def test_outage_lines_rotate_through_the_message_pool():
    geo = geometry(6, 22)
    first = outage_lines(geo, "14:32", 0)
    second = outage_lines(geo, "14:32", 1)
    assert first != second


def test_outage_lines_fit_a_small_note_board():
    geo = geometry(3, 15)
    for index in range(len(OUTAGE_MESSAGES)):
        lines = outage_lines(geo, "14:32", index)
        assert len(lines) == 3
        assert all(len(line) <= 15 for line in lines)


def test_outage_lines_fit_a_flagship_board():
    geo = geometry(6, 22)
    for index in range(len(OUTAGE_MESSAGES)):
        lines = outage_lines(geo, "14:32", index)
        assert len(lines) == 6
        assert all(len(line) <= 22 for line in lines)


def test_setup_lines_tell_the_user_what_to_do():
    joined = " ".join(setup_lines(geometry(6, 22)))
    assert "SETTINGS" in joined


def test_setup_lines_fit_a_note_board():
    lines = setup_lines(geometry(3, 15))
    assert len(lines) == 3
    assert all(len(line) <= 15 for line in lines)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test.sh tests/test_fallback.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'plugins.generative_dashboard.fallback'`

- [ ] **Step 3: Write the implementation**

```python
# fallback.py
"""What the board shows when the model, or the data, is unavailable.

The deterministic grid needs no model, which is why it doubles as the
cold-start renderer: the board is useful the instant the plugin is enabled.
The outage messages are gentle and never at the user's expense, and they
always carry a timestamp so a two-day outage is not mistaken for a two-minute
one.
"""

from .catalog import default_label
from .layout import Geometry, Tile, fits, wrap_center

# (short, long) pairs. The long form is used when it fits; Notes get the short.
OUTAGE_MESSAGES: tuple[tuple[str, str], ...] = (
    ("HAMSTERS ON BREAK", "THE HAMSTERS HAVE STOPPED RUNNING ON THE WHEEL"),
    ("ROUTER UNPLUGGED?", "SOMEONE MAY HAVE UNPLUGGED SOMETHING SOMEWHERE"),
    ("NUMBERS NAPPING", "THE NUMBERS ARE TAKING A SHORT NAP"),
    ("NO WORD YET", "LOST CONTACT WITH THE OUTSIDE WORLD FOR NOW"),
    ("WHEEL STOPPED", "THE WHEEL STOPPED SPINNING - CHECK BACK SOON"),
)

_SETUP = ("PICK SOME STATS", "NO STATS WATCHED YET - ADD VARIABLES IN SETTINGS")


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
    ]


def _choose(short: str, long: str, geo: Geometry, body_rows: int) -> str:
    """Prefer the long phrasing, fall back to the short one on small boards."""
    return long if fits(long, body_rows, geo.cols) else short


def outage_lines(geo: Geometry, last_good: str | None, index: int) -> list[str]:
    """Humour plus one honest line, filling the board exactly."""
    body_rows = max(1, geo.rows - 1)
    short, long = OUTAGE_MESSAGES[index % len(OUTAGE_MESSAGES)]
    body = wrap_center(_choose(short, long, geo, body_rows), body_rows, geo.cols)
    footer = f"LAST GOOD DATA {last_good}" if last_good else "NO DATA YET"
    return (body + [footer.center(geo.cols).rstrip()])[: geo.rows]


def setup_lines(geo: Geometry) -> list[str]:
    """Actionable message for an unconfigured plugin. Not a joke."""
    short, long = _SETUP
    return wrap_center(_choose(short, long, geo, geo.rows), geo.rows, geo.cols)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./scripts/test.sh tests/test_fallback.py`
Expected: PASS, 13 tests

- [ ] **Step 5: Commit**

```bash
git add fallback.py tests/test_fallback.py
git commit -m "feat: deterministic fallback grid and outage messaging"
```

---
### Task 6: Response validation

**Files:**
- Create: `validation.py`
- Create: `tests/test_validation.py`

**Interfaces:**
- Consumes: `layout.Geometry`, `layout.Tile`, `layout.fits`, `charset.sanitize`, `charset.VALID_COLORS`, `catalog.default_label`
- Produces: `ValidationError`, `GridResult` (frozen dataclass: `tiles: list[Tile]`, `refs: list[str]`, `banner: str`, `headline: str`, `reason: str`), `ProseResult` (frozen dataclass: `text: str`, `headline: str`, `reason: str`), `extract_numbers(text: str) -> list[str]`, `allowed_numbers(current: dict[str, str], previous: dict[str, str]) -> set[str]`, `validate_grid(payload, *, watchlist, pinned, values, labels, geo, use_color) -> GridResult`, `validate_prose(payload, *, geo, current, previous) -> ProseResult`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_validation.py
"""Validating what the model sends back."""

import pytest

from plugins.generative_dashboard.layout import geometry
from plugins.generative_dashboard.validation import (
    ValidationError,
    allowed_numbers,
    extract_numbers,
    validate_grid,
    validate_prose,
)

GEO = geometry(6, 22)
WATCHLIST = ["air.aqi", "wx.temp", "sys.cpu"]
VALUES = {"air.aqi": "168", "wx.temp": "61F", "sys.cpu": "42%"}


def _grid(**overrides):
    payload = {"tiles": [{"label": "AQI", "variable": "air.aqi"}]}
    payload.update(overrides)
    return payload


def _validate(payload, **kwargs):
    options = {
        "watchlist": WATCHLIST,
        "pinned": [],
        "values": VALUES,
        "labels": {},
        "geo": GEO,
        "use_color": True,
    }
    options.update(kwargs)
    return validate_grid(payload, **options)


# --- grid ---------------------------------------------------------------

def test_grid_builds_tiles_from_the_models_choices():
    result = _validate(_grid())
    assert [(t.label, t.value) for t in result.tiles] == [("AQI", "168")]


def test_grid_takes_the_value_from_our_data_not_the_model():
    # The model is not permitted to type a number; it names a variable.
    payload = {"tiles": [{"label": "AQI", "variable": "air.aqi", "value": "999"}]}
    assert _validate(payload).tiles[0].value == "168"


def test_grid_drops_variables_that_are_not_watched():
    payload = {"tiles": [{"label": "X", "variable": "evil.hack"},
                         {"label": "AQI", "variable": "air.aqi"}]}
    assert [t.label for t in _validate(payload).tiles] == ["AQI"]


def test_grid_drops_watched_variables_with_no_current_value():
    payload = {"tiles": [{"label": "GONE", "variable": "air.aqi"}]}
    with pytest.raises(ValidationError):
        _validate(payload, values={})


def test_grid_deduplicates_a_repeated_variable():
    payload = {"tiles": [{"label": "AQI", "variable": "air.aqi"},
                         {"label": "AQI AGAIN", "variable": "air.aqi"}]}
    assert len(_validate(payload).tiles) == 1


def test_grid_truncates_to_the_tile_budget():
    payload = {"tiles": [{"label": f"L{i}", "variable": WATCHLIST[i % 3]} for i in range(30)]}
    assert len(_validate(payload).tiles) <= GEO.tile_budget


def test_grid_reinstates_a_pinned_variable_the_model_forgot():
    result = _validate(_grid(), pinned=["sys.cpu"])
    assert "sys.cpu" in [t.label for t in result.tiles] or any(
        t.value == "42%" for t in result.tiles
    )


def test_grid_evicts_an_unpinned_tile_to_make_room_for_a_pin():
    geo = geometry(3, 15)  # budget of 3
    payload = {"tiles": [{"label": "A", "variable": "air.aqi"},
                         {"label": "B", "variable": "wx.temp"},
                         {"label": "C", "variable": "sys.cpu"}]}
    values = dict(VALUES, **{"z.pin": "9"})
    result = validate_grid(
        payload, watchlist=WATCHLIST + ["z.pin"], pinned=["z.pin"],
        values=values, labels={}, geo=geo, use_color=True,
    )
    assert len(result.tiles) == 3
    assert any(t.value == "9" for t in result.tiles)


def test_grid_uses_the_configured_label_over_the_models():
    result = _validate(_grid(), labels={"air.aqi": "AIR"})
    assert result.tiles[0].label == "AIR"


def test_grid_falls_back_to_a_derived_label_when_the_model_omits_one():
    payload = {"tiles": [{"variable": "air.aqi"}]}
    assert _validate(payload).tiles[0].label == "AQI"


def test_grid_keeps_a_valid_color():
    payload = {"tiles": [{"label": "AQI", "variable": "air.aqi", "color": "red"}]}
    assert _validate(payload).tiles[0].color == "red"


def test_grid_discards_an_invented_color():
    payload = {"tiles": [{"label": "AQI", "variable": "air.aqi", "color": "chartreuse"}]}
    assert _validate(payload).tiles[0].color is None


def test_grid_strips_color_when_color_is_disabled():
    payload = {"tiles": [{"label": "AQI", "variable": "air.aqi", "color": "red"}]}
    assert _validate(payload, use_color=False).tiles[0].color is None


def test_grid_sanitizes_the_banner():
    # The pipe has no flap on the board.
    assert _validate(_grid(banner="Air|Quality")).banner == "AIRQUALITY"


def test_grid_rejects_a_payload_that_is_not_an_object():
    with pytest.raises(ValidationError):
        _validate([])


def test_grid_rejects_a_payload_with_no_usable_tiles():
    with pytest.raises(ValidationError):
        _validate({"tiles": []})


def test_grid_reports_the_source_ref_for_each_tile():
    # The plugin needs the ref to re-render live values between generations.
    result = _validate(_grid())
    assert result.refs == ["air.aqi"]
    assert len(result.refs) == len(result.tiles)


def test_grid_carries_through_the_models_reason():
    result = _validate(_grid(reason="AQI ROSE"))
    assert result.reason == "AQI ROSE"


# --- prose --------------------------------------------------------------

def test_extract_numbers_finds_plain_integers():
    assert extract_numbers("AQI 168 AND CPU 42") == ["168", "42"]


def test_extract_numbers_keeps_grouping_commas_together():
    assert extract_numbers("BTC 94,120") == ["94,120"]


def test_extract_numbers_keeps_decimals_together():
    assert extract_numbers("LOAD 1.25") == ["1.25"]


def test_allowed_numbers_includes_current_and_previous_values():
    allowed = allowed_numbers({"a.x": "168"}, {"a.x": "31"})
    assert {"168", "31"} <= allowed


def test_prose_accepts_a_sentence_using_supplied_numbers():
    result = validate_prose(
        {"text": "AQI ROSE FROM 31 TO 168."},
        geo=GEO, current={"a.x": "168"}, previous={"a.x": "31"},
    )
    assert "168" in result.text


def test_prose_rejects_a_fabricated_number():
    with pytest.raises(ValidationError) as excinfo:
        validate_prose(
            {"text": "AQI ROSE 400 PERCENT."},
            geo=GEO, current={"a.x": "168"}, previous={"a.x": "31"},
        )
    assert "400" in str(excinfo.value)


def test_prose_rejects_text_too_long_for_the_board():
    with pytest.raises(ValidationError):
        validate_prose(
            {"text": "WORD " * 200},
            geo=GEO, current={}, previous={},
        )


def test_prose_uppercases_and_strips_unrenderable_characters():
    result = validate_prose(
        {"text": "all quiet | today"},
        geo=GEO, current={}, previous={},
    )
    assert result.text == "ALL QUIET  TODAY"


def test_prose_rejects_a_missing_text_field():
    with pytest.raises(ValidationError):
        validate_prose({"headline": "X"}, geo=GEO, current={}, previous={})


def test_prose_rejects_empty_text():
    with pytest.raises(ValidationError):
        validate_prose({"text": "   "}, geo=GEO, current={}, previous={})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test.sh tests/test_validation.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'plugins.generative_dashboard.validation'`

- [ ] **Step 3: Write the implementation**

```python
# validation.py
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
from .charset import VALID_COLORS, sanitize
from .layout import Geometry, Tile, fits

# A run of digits with internal separators: 42, 94,120, 1.25
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


class ValidationError(Exception):
    """The model's answer cannot be shown, and should be retried or dropped."""


@dataclass(frozen=True)
class GridResult:
    tiles: list[Tile]
    refs: list[str]  # source ref per tile, parallel to ``tiles``
    banner: str
    headline: str
    reason: str


@dataclass(frozen=True)
class ProseResult:
    text: str
    headline: str
    reason: str


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
    banner = sanitize(str(data.get("banner") or ""))[: geo.cols]
    body_rows = geo.rows - (1 if banner else 0)
    capacity = max(1, geo.tile_columns * body_rows)

    chosen: list[tuple[str, Tile]] = []
    seen: set[str] = set()
    for entry in raw_tiles:
        if not isinstance(entry, dict):
            continue
        ref = str(entry.get("variable") or "")
        if ref not in watched or ref in seen or ref not in values:
            continue
        color = str(entry.get("color") or "").lower()
        chosen.append((
            ref,
            Tile(
                label=labels.get(ref) or sanitize(str(entry.get("label") or "")) or default_label(ref),
                value=values[ref],
                color=color if (use_color and color in VALID_COLORS) else None,
            ),
        ))
        seen.add(ref)

    chosen = chosen[:capacity]

    # Pins are a promise; honour them even when the model forgot.
    pinned_set = [r for r in pinned if r in values and r in watched]
    for ref in pinned_set:
        if ref in seen:
            continue
        tile = Tile(label=labels.get(ref) or default_label(ref), value=values[ref])
        if len(chosen) >= capacity:
            for index in range(len(chosen) - 1, -1, -1):
                if chosen[index][0] not in pinned_set:
                    chosen.pop(index)
                    break
            else:
                break
        chosen.append((ref, tile))
        seen.add(ref)

    if not chosen:
        raise ValidationError("No usable tiles in response")

    return GridResult(
        tiles=[tile for _, tile in chosen],
        refs=[ref for ref, _ in chosen],
        banner=banner,
        headline=sanitize(str(data.get("headline") or "")),
        reason=sanitize(str(data.get("reason") or "")),
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
            raise ValidationError(f"Text contains the number {token}, which was not supplied")

    if not fits(text, geo.rows, geo.cols):
        raise ValidationError(f"Text does not fit {geo.rows} rows of {geo.cols}")

    return ProseResult(
        text=text,
        headline=sanitize(str(data.get("headline") or "")),
        reason=sanitize(str(data.get("reason") or "")),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./scripts/test.sh tests/test_validation.py`
Expected: PASS, 28 tests

- [ ] **Step 5: Commit**

```bash
git add validation.py tests/test_validation.py
git commit -m "feat: grid and prose response validation"
```

---

### Task 7: LLM client and prompts

**Files:**
- Create: `llm.py`
- Create: `tests/test_llm.py`

**Interfaces:**
- Consumes: `layout.Geometry`
- Produces: `LLMError`, `DashboardLLM(base_url, api_key, model, temperature, timeout=30)` with `.complete(system: str, user: str) -> dict`, `describe_variables(refs, labels, notes, current, previous) -> str`, `build_grid_prompt(...) -> tuple[str, str]`, `build_prose_prompt(...) -> tuple[str, str]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm.py
"""LLM transport and prompt construction."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from plugins.generative_dashboard.layout import geometry
from plugins.generative_dashboard.llm import (
    DashboardLLM,
    LLMError,
    build_grid_prompt,
    build_prose_prompt,
    describe_variables,
)

GEO = geometry(6, 22)


def _client():
    return DashboardLLM("https://api.test/v1", "sk-test", "gpt-4o-mini", 0.3)


def _response(content):
    reply = MagicMock()
    reply.status_code = 200
    reply.json.return_value = {"choices": [{"message": {"content": content}}]}
    reply.raise_for_status.return_value = None
    return reply


def test_complete_parses_the_models_json():
    with patch("requests.post", return_value=_response('{"tiles": []}')):
        assert _client().complete("sys", "user") == {"tiles": []}


def test_complete_strips_a_markdown_code_fence():
    fenced = '```json\n{"tiles": []}\n```'
    with patch("requests.post", return_value=_response(fenced)):
        assert _client().complete("sys", "user") == {"tiles": []}


def test_complete_raises_on_unparseable_output():
    with patch("requests.post", return_value=_response("sorry, no")):
        with pytest.raises(LLMError):
            _client().complete("sys", "user")


def test_complete_raises_on_a_network_failure():
    with patch("requests.post", side_effect=requests.RequestException("boom")):
        with pytest.raises(LLMError):
            _client().complete("sys", "user")


def test_complete_sends_the_bearer_token():
    with patch("requests.post", return_value=_response("{}")) as post:
        _client().complete("sys", "user")
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-test"


def test_complete_posts_to_the_chat_completions_path():
    with patch("requests.post", return_value=_response("{}")) as post:
        _client().complete("sys", "user")
    assert post.call_args.args[0] == "https://api.test/v1/chat/completions"


def test_complete_does_not_double_a_trailing_slash():
    client = DashboardLLM("https://api.test/v1/", "k", "m", 0.3)
    with patch("requests.post", return_value=_response("{}")) as post:
        client.complete("sys", "user")
    assert post.call_args.args[0] == "https://api.test/v1/chat/completions"


def test_describe_variables_shows_current_and_previous_values():
    text = describe_variables(
        ["a.aqi"], {"a.aqi": "AQI"}, {}, {"a.aqi": "168"}, {"a.aqi": "31"}
    )
    assert "168" in text and "31" in text


def test_describe_variables_includes_the_users_note():
    text = describe_variables(
        ["a.aqi"], {}, {"a.aqi": "over 100 is unhealthy"}, {"a.aqi": "168"}, {}
    )
    assert "unhealthy" in text


def test_describe_variables_omits_variables_with_no_value():
    text = describe_variables(["a.x", "b.y"], {}, {}, {"a.x": "1"}, {})
    assert "b.y" not in text


def test_grid_prompt_states_the_exact_tile_budget():
    system, _ = build_grid_prompt(
        geo=GEO, refs=["a.x"], labels={}, notes={}, current={"a.x": "1"},
        previous={}, previous_board=[], use_color=True, extra_instructions="",
    )
    assert "12" in system


def test_grid_prompt_forbids_typing_values():
    system, _ = build_grid_prompt(
        geo=GEO, refs=["a.x"], labels={}, notes={}, current={"a.x": "1"},
        previous={}, previous_board=[], use_color=True, extra_instructions="",
    )
    assert "never" in system.lower()


def test_grid_prompt_omits_color_guidance_when_color_is_off():
    system, _ = build_grid_prompt(
        geo=GEO, refs=["a.x"], labels={}, notes={}, current={"a.x": "1"},
        previous={}, previous_board=[], use_color=False, extra_instructions="",
    )
    assert "color" not in system.lower()


def test_grid_prompt_includes_the_previous_board_for_continuity():
    _, user = build_grid_prompt(
        geo=GEO, refs=["a.x"], labels={}, notes={}, current={"a.x": "1"},
        previous={}, previous_board=["CPU 42%"], use_color=True, extra_instructions="",
    )
    assert "CPU 42%" in user


def test_grid_prompt_appends_extra_instructions():
    system, _ = build_grid_prompt(
        geo=GEO, refs=["a.x"], labels={}, notes={}, current={"a.x": "1"},
        previous={}, previous_board=[], use_color=True,
        extra_instructions="FAVOUR NETWORK STATS",
    )
    assert "FAVOUR NETWORK STATS" in system


def test_prose_prompt_states_the_character_budget():
    system, _ = build_prose_prompt(
        geo=GEO, refs=["a.x"], labels={}, notes={}, current={"a.x": "1"},
        previous={}, previous_board=[], extra_instructions="",
    )
    assert str(GEO.prose_budget) in system


def test_prose_prompt_forbids_arithmetic():
    system, _ = build_prose_prompt(
        geo=GEO, refs=["a.x"], labels={}, notes={}, current={"a.x": "1"},
        previous={}, previous_board=[], extra_instructions="",
    )
    lowered = system.lower()
    assert "do not compute" in lowered or "not compute" in lowered
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test.sh tests/test_llm.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'plugins.generative_dashboard.llm'`

- [ ] **Step 3: Write the implementation**

```python
# llm.py
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
        except (KeyError, IndexError, ValueError) as exc:
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
        parts = [f"{ref}", f'label="{labels.get(ref, "")}"', f'now="{current[ref]}"']
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
    color_rule = (
        'Set "color" on at most one or two tiles, and only when a value has '
        "moved enough to deserve attention. Valid colors: red, orange, yellow, "
        "green, blue, violet, white, black.\n"
        if use_color
        else ""
    )
    system = (
        "You lay out a stats dashboard for a split-flap board.\n\n"
        f"You may place at most {geo.tile_budget} tiles, arranged in "
        f"{geo.tile_columns} column(s) of {geo.tile_width} cells, reading left "
        "to right then down. The most important stat goes first.\n\n"
        "You choose WHICH stats appear, their order, and what they are called. "
        "You never type a value: name the variable and the value is filled in "
        "for you. Labels should be short — a few characters.\n\n"
        f"{color_rule}"
        'Optionally set "banner" to one short line across the top when '
        f"something deserves a sentence. A banner costs {geo.tile_columns} "
        "tile(s) of budget.\n\n"
        "Keep stats that have not changed in the positions they already "
        "occupy. Moving things for no reason makes the board noisy.\n\n"
        f"{_CHARSET_RULES}\n\n"
        "Reply with JSON only:\n"
        '{"tiles": [{"label": "AQI", "variable": "air.aqi", "color": "red"}], '
        '"banner": "", "headline": "AQI 168", "reason": "why you changed it"}'
        + (f"\n\n{extra_instructions}" if extra_instructions.strip() else "")
    )
    user = _context_block(geo, refs, labels, notes, current, previous, previous_board)
    return system, user


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
        "round, abbreviate, or invent any number. If a value reads 94,120 write "
        "94,120. Never state a percentage or difference that was not given.\n\n"
        f"{_CHARSET_RULES}\n\n"
        "Reply with JSON only:\n"
        '{"text": "AQI ROSE FROM 31 TO 168.", "headline": "AQI 168", '
        '"reason": "why you changed it"}'
        + (f"\n\n{extra_instructions}" if extra_instructions.strip() else "")
    )
    user = _context_block(geo, refs, labels, notes, current, previous, previous_board)
    return system, user
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./scripts/test.sh tests/test_llm.py`
Expected: PASS, 17 tests

- [ ] **Step 5: Commit**

```bash
git add llm.py tests/test_llm.py
git commit -m "feat: OpenAI-compatible client and mode-specific prompts"
```

---
### Task 8: The plugin class

**Files:**
- Create: `__init__.py`
- Create: `conftest.py`
- Create: `tests/conftest.py`
- Create: `tests/test_plugin.py`

**Interfaces:**
- Consumes: every module above
- Produces: `GenerativeDashboardPlugin`, `BoardState`, `TileSpec`

**Design notes the implementer must not "simplify" away:**

1. **Grid mode re-renders live values on every fetch.** State holds the model's
   *tile choice* (`TileSpec`: ref, label, colour), not finished lines. Values
   are substituted fresh each render. The model still recomposes on every
   material change — but between generations the numbers stay live while the
   positions stay put. Caching finished lines would display stale numbers.
2. **Only update the gate snapshot when a generation is actually triggered.**
   If a change is detected but generation is not permitted yet, the snapshot
   must not advance — otherwise a series of sub-threshold drifts silently
   resets and never accumulates into a real change.
3. **Never generate when `self.board is None`.** A `None` board means this is
   not a render (`get_all_variables_with_metadata()` fans out that way from
   the settings dialog). Generating there would bill the user for a keystroke.
4. **Prose falls back only after an actual failure**, never while a worker is
   in flight, or the board flashes a different layout for a few seconds.

- [ ] **Step 1: Write the failing tests**

```python
# conftest.py  (repo root)
"""Keep pytest from importing plugin modules as top-level modules.

They are reached through the ``plugins/`` symlink as
``plugins.generative_dashboard.*``, where relative imports resolve.
"""

collect_ignore = [
    "__init__.py",
    "catalog.py",
    "charset.py",
    "fallback.py",
    "gate.py",
    "layout.py",
    "llm.py",
    "validation.py",
]
```

```python
# tests/conftest.py
"""Shared fixtures."""

import pytest


@pytest.fixture
def manifest():
    return {
        "id": "generative_dashboard",
        "name": "Generative Dashboard",
        "version": "1.0.0",
        "description": "Test",
        "author": "FiestaBoard Team",
        "live_data": True,
        "settings_schema": {
            "type": "object",
            "properties": {
                "refresh_seconds": {"type": "integer", "default": 300, "minimum": 120}
            },
        },
    }


@pytest.fixture
def config():
    return {
        "enabled": True,
        "api_key": "sk-test",
        "api_base_url": "https://api.test/v1",
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "output_mode": "grid",
        "refresh_seconds": 300,
        "default_threshold_pct": 5,
        "use_color": True,
        "watchlist": ["air.aqi", "wx.temp"],
        "labels": {},
        "pinned": [],
        "notes": {},
        "thresholds": {},
    }
```

```python
# tests/test_plugin.py
"""Plugin lifecycle, gating, worker behaviour, and the failure ladder."""

from unittest.mock import patch

import pytest
from src.devices import BoardContext

from plugins.generative_dashboard import GenerativeDashboardPlugin, catalog

FLAGSHIP = BoardContext("flagship", rows=6, cols=22)
NOTE = BoardContext("note", rows=3, cols=15)


@pytest.fixture
def plugin(manifest, config, monkeypatch):
    instance = GenerativeDashboardPlugin(manifest)
    instance.config = config
    monkeypatch.setattr(
        catalog, "read_values",
        lambda refs, board, exclude: {"air.aqi": "68", "wx.temp": "61F"},
    )
    # Generation is exercised explicitly; default to never spawning a worker.
    monkeypatch.setattr(instance, "_spawn", lambda *a, **k: None)
    return instance


def _fetch(plugin, board=FLAGSHIP):
    with plugin._bound_board(board):
        return plugin.fetch_data()


def test_unconfigured_watchlist_shows_the_setup_message(plugin):
    plugin.config = dict(plugin.config, watchlist=[])
    result = _fetch(plugin)
    assert "SETTINGS" in " ".join(result.data["rows"])
    assert result.data["degraded"] == "unconfigured"


def test_cold_start_renders_a_deterministic_grid_without_any_llm_call(plugin):
    result = _fetch(plugin)
    joined = " ".join(result.data["rows"])
    assert "68" in joined and "61F" in joined
    assert result.data["degraded"] == "no_llm"


def test_rows_match_the_board_height(plugin):
    assert len(_fetch(plugin, FLAGSHIP).data["rows"]) == 6
    assert len(_fetch(plugin, NOTE).data["rows"]) == 3


def test_no_watchlist_values_shows_humor_and_a_timestamp(plugin, monkeypatch):
    monkeypatch.setattr(catalog, "read_values", lambda refs, board, exclude: {})
    result = _fetch(plugin)
    assert result.data["degraded"] == "no_data"
    assert "NO DATA YET" in " ".join(result.data["rows"])


def test_a_board_with_data_reports_when_data_was_last_good(plugin, monkeypatch):
    _fetch(plugin)
    monkeypatch.setattr(catalog, "read_values", lambda refs, board, exclude: {})
    result = _fetch(plugin)
    assert "LAST GOOD DATA" in " ".join(result.data["rows"])


def test_each_board_size_keeps_its_own_composition(plugin):
    _fetch(plugin, FLAGSHIP)
    _fetch(plugin, NOTE)
    assert set(plugin._states) == {"flagship", "note"}


def test_generation_is_never_started_without_a_board(plugin, monkeypatch):
    calls = []
    monkeypatch.setattr(plugin, "_spawn", lambda *a, **k: calls.append(a))
    plugin.fetch_data()  # no _bound_board: self.board is None
    assert calls == []


def test_generation_starts_when_a_board_is_present(plugin, monkeypatch):
    calls = []
    monkeypatch.setattr(plugin, "_spawn", lambda *a, **k: calls.append(a))
    _fetch(plugin)
    assert calls


def test_a_quiet_cycle_does_not_start_a_generation(plugin, monkeypatch):
    _fetch(plugin)
    plugin._states["flagship"].tiles = [object()]  # pretend the model answered
    calls = []
    monkeypatch.setattr(plugin, "_spawn", lambda *a, **k: calls.append(a))
    _fetch(plugin)
    assert calls == []


def test_a_material_change_starts_a_generation(plugin, monkeypatch):
    _fetch(plugin)
    state = plugin._states["flagship"]
    state.tiles = [object()]
    state.last_generated = 0.0
    monkeypatch.setattr(
        catalog, "read_values",
        lambda refs, board, exclude: {"air.aqi": "168", "wx.temp": "61F"},
    )
    calls = []
    monkeypatch.setattr(plugin, "_spawn", lambda *a, **k: calls.append(a))
    _fetch(plugin)
    assert calls


def test_the_snapshot_does_not_advance_when_generation_is_blocked(plugin, monkeypatch):
    _fetch(plugin)
    state = plugin._states["flagship"]
    state.tiles = [object()]
    plugin._inflight.add("flagship")  # a worker is already running
    monkeypatch.setattr(
        catalog, "read_values",
        lambda refs, board, exclude: {"air.aqi": "168", "wx.temp": "61F"},
    )
    _fetch(plugin)
    # Snapshot still holds the pre-change value, so the change stays pending.
    assert state.values["air.aqi"] == "68"


def test_live_values_are_re_rendered_between_generations(plugin, monkeypatch):
    from plugins.generative_dashboard import TileSpec

    _fetch(plugin)
    plugin._states["flagship"].tiles = [TileSpec("air.aqi", "AQI", None)]
    monkeypatch.setattr(
        catalog, "read_values",
        lambda refs, board, exclude: {"air.aqi": "999", "wx.temp": "61F"},
    )
    result = _fetch(plugin)
    # The model has not run again, but the number on the board is current.
    assert "999" in " ".join(result.data["rows"])


def test_validate_config_requires_an_api_key(plugin):
    assert any("API key" in e for e in plugin.validate_config({"watchlist": []}))


def test_validate_config_rejects_an_oversized_watchlist(plugin):
    errors = plugin.validate_config(
        {"api_key": "k", "watchlist": [f"a.v{i}" for i in range(101)]}
    )
    assert any("100" in e for e in errors)


def test_validate_config_accepts_a_hundred_entries(plugin):
    errors = plugin.validate_config(
        {"api_key": "k", "watchlist": [f"a.v{i}" for i in range(100)]}
    )
    assert not any("100" in e for e in errors)


def test_validate_config_rejects_an_unknown_output_mode(plugin):
    errors = plugin.validate_config({"api_key": "k", "output_mode": "interpretive_dance"})
    assert any("output_mode" in e for e in errors)


def test_config_change_discards_an_in_flight_result(plugin):
    generation = plugin._config_generation
    plugin.on_config_change({}, {})
    assert plugin._config_generation != generation


def test_cleanup_stops_further_swaps(plugin):
    plugin.cleanup()
    assert plugin._stopped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test.sh tests/test_plugin.py`
Expected: FAIL — `ImportError: cannot import name 'GenerativeDashboardPlugin'`

- [ ] **Step 3: Write the implementation**

```python
# __init__.py
"""Generative Dashboard plugin for FiestaBoard.

Composes a full-board dashboard from a curated pool of other plugins'
variables, asking an LLM which of them matter right now. Regenerates only when
the underlying numbers move materially, so the board stays still when the
world is quiet.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

from src.plugins.base import Option, OptionsRequest, OptionsResult, OptionsUnavailable, PluginBase, PluginResult

try:
    from . import catalog, fallback, gate
    from .charset import sanitize
    from .layout import Geometry, Tile, geometry, render_grid, wrap_center
    from .llm import DashboardLLM, LLMError, build_grid_prompt, build_prose_prompt
    from .validation import ValidationError, validate_grid, validate_prose
except ImportError:  # pragma: no cover - direct module import fallback
    import catalog, fallback, gate  # type: ignore[no-redef]
    from charset import sanitize  # type: ignore[no-redef]
    from layout import Geometry, Tile, geometry, render_grid, wrap_center  # type: ignore[no-redef]
    from llm import DashboardLLM, LLMError, build_grid_prompt, build_prose_prompt  # type: ignore[no-redef]
    from validation import ValidationError, validate_grid, validate_prose  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

MAX_WATCHLIST = 100
DEFAULT_BOARD_ROWS = 6
DEFAULT_BOARD_COLS = 22
OUTPUT_MODES = ("grid", "prose")


@dataclass(frozen=True)
class TileSpec:
    """The model's choice for one tile. The value is filled in at render time."""

    ref: str
    label: str
    color: str | None


@dataclass
class BoardState:
    """Everything the plugin remembers for one board size."""

    tiles: list[TileSpec] = field(default_factory=list)
    banner: str = ""
    prose: str = ""
    values: dict[str, str] = field(default_factory=dict)
    previous: dict[str, str] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)
    headline: str = ""
    reason: str = ""
    degraded: str = "no_llm"
    generated_at: str = ""
    last_good: str = ""
    last_generated: float = 0.0
    failures: int = 0
    pending: bool = False


class GenerativeDashboardPlugin(PluginBase):
    """See the module docstring. Composition happens off the render path."""

    def __init__(self, manifest: dict) -> None:
        super().__init__(manifest)
        self._states: dict[str, BoardState] = {}
        self._lock = threading.Lock()
        self._inflight: set[str] = set()
        self._stopped = False
        self._config_generation = 0
        self._outage_index = 0

    @property
    def plugin_id(self) -> str:
        return "generative_dashboard"

    # -- configuration ----------------------------------------------------

    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        if not config.get("api_key"):
            errors.append("API key is required")

        base_url = config.get("api_base_url", "")
        if base_url and not base_url.startswith(("http://", "https://")):
            errors.append("API base URL must start with http:// or https://")

        temperature = config.get("temperature", 0.3)
        if not isinstance(temperature, int | float) or not 0 <= temperature <= 2:
            errors.append("temperature must be a number between 0 and 2")

        mode = config.get("output_mode", "grid")
        if mode not in OUTPUT_MODES:
            errors.append(f"output_mode must be one of {', '.join(OUTPUT_MODES)}")

        watchlist = config.get("watchlist") or []
        if not isinstance(watchlist, list):
            errors.append("watchlist must be a list of variable references")
        elif len(watchlist) > MAX_WATCHLIST:
            errors.append(
                f"watchlist has {len(watchlist)} entries; the maximum is {MAX_WATCHLIST}"
            )

        errors.extend(self._validate_refresh_seconds(config))
        return errors

    def on_config_change(self, old_config: dict, new_config: dict) -> None:
        """Drop every composition and orphan any in-flight worker's result."""
        with self._lock:
            self._states.clear()
            self._config_generation += 1

    def cleanup(self) -> None:
        self._stopped = True

    # -- helpers ----------------------------------------------------------

    def _geometry(self) -> Geometry:
        board = self.board
        if board is None:
            return geometry(DEFAULT_BOARD_ROWS, DEFAULT_BOARD_COLS)
        return geometry(board.rows, board.cols)

    def _state_key(self) -> str:
        board = self.board
        if board is None:
            return "_default"
        if board.device_type == "note_array":
            return f"note_array:{board.cols}x{board.rows}"
        return board.device_type

    def _watchlist(self, config: dict) -> list[str]:
        raw = config.get("watchlist") or []
        if not isinstance(raw, list):
            return []
        return [str(r) for r in raw if isinstance(r, str) and "." in r][:MAX_WATCHLIST]

    # -- the render path --------------------------------------------------

    def fetch_data(self) -> PluginResult:
        config = self.config
        geo = self._geometry()
        key = self._state_key()
        state = self._states.setdefault(key, BoardState())

        watchlist = self._watchlist(config)
        if not watchlist:
            return self._result(state, fallback.setup_lines(geo), "unconfigured", geo)

        values = catalog.read_values(watchlist, self.board, self.plugin_id)
        if not values:
            self._outage_index += 1
            lines = fallback.outage_lines(geo, state.last_good or None, self._outage_index)
            return self._result(state, lines, "no_data", geo)

        state.last_good = datetime.now().strftime("%H:%M")

        changed = gate.material_changes(
            state.values,
            values,
            self._thresholds(config),
            float(config.get("default_threshold_pct", 5) or 5),
        )
        wants_generation = bool(changed) or not (state.tiles or state.prose)

        if wants_generation and self._may_generate(key, state, config):
            state.previous = dict(state.values)
            state.values = dict(values)
            state.pending = True
            state.last_generated = time.monotonic()
            self._spawn(key, geo, config, watchlist, dict(values), dict(state.previous),
                        list(state.lines), self._config_generation)

        lines, degraded = self._compose(state, geo, config, values, watchlist)
        return self._result(state, lines, degraded, geo)

    def _may_generate(self, key: str, state: BoardState, config: dict) -> bool:
        """Whether a worker may start right now."""
        if self.board is None:
            # Not a render — the settings dialog fans out with board=None.
            return False
        if self._stopped or not config.get("api_key"):
            return False
        with self._lock:
            if key in self._inflight:
                return False
        if not (state.tiles or state.prose):
            return True  # cold start: go immediately
        interval = float(config.get("refresh_seconds", 300) or 300)
        return (time.monotonic() - state.last_generated) >= interval

    def _compose(
        self,
        state: BoardState,
        geo: Geometry,
        config: dict,
        values: dict[str, str],
        watchlist: list[str],
    ) -> tuple[list[str], str]:
        """Build the lines to show, using live values wherever possible."""
        labels = self._labels(config)
        use_color = bool(config.get("use_color", True))
        mode = config.get("output_mode", "grid")

        if mode == "prose" and state.prose and not (state.pending and state.failures):
            return wrap_center(state.prose, geo.rows, geo.cols), state.degraded

        if state.tiles:
            tiles = [
                Tile(label=spec.label, value=values[spec.ref], color=spec.color)
                for spec in state.tiles
                if spec.ref in values
            ]
            if tiles:
                return (
                    render_grid(tiles, geo, banner=state.banner, use_color=use_color),
                    state.degraded,
                )

        pinned = [r for r in (config.get("pinned") or []) if isinstance(r, str)]
        tiles = fallback.deterministic_tiles(watchlist, labels, values, pinned)
        return render_grid(tiles, geo, use_color=False), "no_llm"

    def _result(
        self, state: BoardState, lines: list[str], degraded: str, geo: Geometry
    ) -> PluginResult:
        stat_count = sum(1 for line in lines if line.strip())
        return PluginResult(
            available=True,
            formatted_lines=lines,
            data={
                "rows": lines,
                "prose": state.prose,
                "headline": state.headline,
                "reason": state.reason,
                "stat_count": len(state.tiles) or stat_count,
                "degraded": degraded,
                "generated_at": state.generated_at,
                "model": str(self.config.get("model", "")),
            },
        )

    def get_formatted_display(self) -> list[str] | None:
        result = self.get_data(self.board)
        if result.available and result.data:
            return result.data.get("rows")
        return None

    # -- generation, off the render path ----------------------------------

    def _spawn(self, key, geo, config, watchlist, current, previous, previous_board, generation) -> None:
        with self._lock:
            if key in self._inflight:
                return
            self._inflight.add(key)
        thread = threading.Thread(
            target=self._run,
            args=(key, geo, config, watchlist, current, previous, previous_board, generation),
            name=f"generative-dashboard-{key}",
            daemon=True,
        )
        thread.start()

    def _run(self, key, geo, config, watchlist, current, previous, previous_board, generation) -> None:
        try:
            outcome = self._generate(geo, config, watchlist, current, previous, previous_board)
        except Exception:
            logger.exception("Dashboard generation failed")
            outcome = None
        finally:
            with self._lock:
                self._inflight.discard(key)

        if self._stopped or generation != self._config_generation:
            return

        with self._lock:
            state = self._states.get(key)
            if state is None:
                return
            state.pending = False
            if outcome is None:
                state.failures += 1
                state.degraded = "no_llm"
                return
            tiles, banner, prose, headline, reason = outcome
            state.tiles = tiles
            state.banner = banner
            state.prose = prose
            state.headline = headline
            state.reason = reason
            state.failures = 0
            state.degraded = ""
            state.generated_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    def _generate(self, geo, config, watchlist, current, previous, previous_board):
        """One generation attempt, with a single stricter retry."""
        client = DashboardLLM(
            base_url=str(config.get("api_base_url", "https://api.openai.com/v1")),
            api_key=str(config.get("api_key", "")),
            model=str(config.get("model", "gpt-4o-mini")),
            temperature=float(config.get("temperature", 0.3) or 0.3),
        )
        mode = config.get("output_mode", "grid")
        labels = self._labels(config)
        notes = self._notes(config)
        pinned = [r for r in (config.get("pinned") or []) if isinstance(r, str)]
        extra = str(config.get("extra_instructions", "") or "")

        for attempt in range(2):
            suffix = "" if attempt == 0 else (
                "\n\nYour previous reply was rejected. Follow the rules exactly."
            )
            if mode == "prose":
                system, user = build_prose_prompt(
                    geo=geo, refs=watchlist, labels=labels, notes=notes,
                    current=current, previous=previous, previous_board=previous_board,
                    extra_instructions=extra + suffix,
                )
            else:
                system, user = build_grid_prompt(
                    geo=geo, refs=watchlist, labels=labels, notes=notes,
                    current=current, previous=previous, previous_board=previous_board,
                    use_color=bool(config.get("use_color", True)),
                    extra_instructions=extra + suffix,
                )
            try:
                payload = client.complete(system, user)
            except LLMError as exc:
                logger.warning("Dashboard LLM call failed: %s", exc)
                return None

            try:
                if mode == "prose":
                    result = validate_prose(payload, geo=geo, current=current, previous=previous)
                    return [], "", result.text, result.headline, result.reason
                grid = validate_grid(
                    payload, watchlist=watchlist, pinned=pinned, values=current,
                    labels=labels, geo=geo, use_color=bool(config.get("use_color", True)),
                )
                specs = [
                    TileSpec(ref=ref, label=tile.label, color=tile.color)
                    for ref, tile in zip(grid.refs, grid.tiles, strict=True)
                ]
                return specs, grid.banner, "", grid.headline, grid.reason
            except ValidationError as exc:
                logger.warning("Dashboard response rejected (attempt %d): %s", attempt + 1, exc)

        return None

    def _labels(self, config: dict) -> dict[str, str]:
        raw = config.get("labels") or {}
        return {k: sanitize(str(v)) for k, v in raw.items()} if isinstance(raw, dict) else {}

    def _notes(self, config: dict) -> dict[str, str]:
        raw = config.get("notes") or {}
        return {k: str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}

    def _thresholds(self, config: dict) -> dict[str, float]:
        raw = config.get("thresholds") or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, float] = {}
        for key, value in raw.items():
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./scripts/test.sh tests/test_plugin.py`
Expected: PASS, 18 tests

- [ ] **Step 5: Commit**

```bash
git add __init__.py conftest.py tests/conftest.py tests/test_plugin.py
git commit -m "feat: plugin lifecycle, gating, and off-render-path generation"
```

---
### Task 9: The variable picker and manifest

**Files:**
- Create: `manifest.json`
- Modify: `__init__.py` (add `get_options`)
- Create: `tests/test_options.py`
- Create: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `catalog.variable_catalog`, `catalog.default_label`, `layout.geometry`
- Produces: `GenerativeDashboardPlugin.get_options(request) -> OptionsResult`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_options.py
"""The settings picker: choosing which variables the dashboard may use."""

import pytest
from src.plugins.base import OptionsRequest, OptionsUnavailable

from plugins.generative_dashboard import GenerativeDashboardPlugin, catalog
from plugins.generative_dashboard.catalog import VariableChoice


@pytest.fixture
def plugin(manifest, config, monkeypatch):
    instance = GenerativeDashboardPlugin(manifest)
    instance.config = config
    monkeypatch.setattr(
        catalog, "variable_catalog",
        lambda exclude, width, query="": [
            VariableChoice("wx.temp", "TEMP", "Outside temperature", "Weather", "61F"),
            VariableChoice("gd.headline", "HEADLINE", "", "Generative Dashboard", "",
                           True, "A dashboard cannot watch itself."),
        ],
    )
    return instance


def _options(plugin, options_id, **kwargs):
    return plugin.get_options(OptionsRequest(options_id=options_id, **kwargs))


def test_variables_picker_returns_one_option_per_choice(plugin):
    assert len(_options(plugin, "variables").options) == 2


def test_variables_picker_stores_the_ref_as_the_value(plugin):
    assert _options(plugin, "variables").options[0].value == "wx.temp"


def test_variables_picker_shows_the_live_value_as_a_preview(plugin):
    assert _options(plugin, "variables").options[0].preview == "61F"


def test_variables_picker_groups_by_source_plugin(plugin):
    assert _options(plugin, "variables").options[0].group == "Weather"


def test_variables_picker_marks_unusable_choices_disabled(plugin):
    assert _options(plugin, "variables").options[1].disabled


def test_a_disabled_choice_explains_itself(plugin):
    assert "itself" in _options(plugin, "variables").options[1].description.lower()


def test_watched_picker_offers_only_already_watched_variables(plugin):
    values = [o.value for o in _options(plugin, "watched").options]
    assert values == ["air.aqi", "wx.temp"]


def test_watched_picker_is_empty_when_nothing_is_watched(plugin):
    plugin.config = dict(plugin.config, watchlist=[])
    assert _options(plugin, "watched").options == []


def test_an_unknown_options_id_is_refused(plugin):
    with pytest.raises(OptionsUnavailable):
        _options(plugin, "nonsense")


def test_the_picker_works_with_no_api_key_configured(plugin):
    plugin.config = dict(plugin.config, api_key="")
    assert _options(plugin, "variables").options


def test_the_picker_never_calls_the_llm(plugin, monkeypatch):
    import plugins.generative_dashboard.llm as llm

    def explode(*args, **kwargs):
        raise AssertionError("get_options must not call the LLM")

    monkeypatch.setattr(llm.DashboardLLM, "complete", explode)
    _options(plugin, "variables")
```

```python
# tests/test_manifest.py
"""The manifest is the plugin's contract with core; check it parses."""

import json
import pathlib

import pytest
from src.plugins.manifest import load_manifest

MANIFEST_PATH = pathlib.Path(__file__).resolve().parent.parent / "manifest.json"


@pytest.fixture(scope="module")
def raw():
    return json.loads(MANIFEST_PATH.read_text())


def test_manifest_is_accepted_by_core(tmp_path):
    # load_manifest returns None on any validation error.
    assert load_manifest(MANIFEST_PATH.parent) is not None


def test_plugin_id_matches_the_class(raw):
    assert raw["id"] == "generative_dashboard"


def test_watchlist_uses_the_remote_options_picker(raw):
    field = raw["settings_schema"]["properties"]["watchlist"]
    assert field["ui:widget"] == "remote-options"
    assert field["type"] == "array"


def test_watchlist_picker_is_multiple_searchable_and_reorderable(raw):
    options = raw["settings_schema"]["properties"]["watchlist"]["ui:options"]
    assert options["multiple"] and options["searchable"] and options["reorderable"]


def test_watchlist_picker_writes_labels_to_a_real_sibling(raw):
    properties = raw["settings_schema"]["properties"]
    labels_field = properties["watchlist"]["ui:options"]["labels_field"]
    assert labels_field in properties


def test_pinned_picker_depends_on_the_watchlist(raw):
    options = raw["settings_schema"]["properties"]["pinned"]["ui:options"]
    assert options["depends_on"] == ["watchlist"]


def test_watchlist_is_capped_at_one_hundred(raw):
    assert raw["settings_schema"]["properties"]["watchlist"]["maxItems"] == 100


def test_live_data_is_enabled_so_values_never_go_stale(raw):
    assert raw["live_data"] is True


def test_every_emitted_variable_is_declared(raw):
    declared = set(raw["variables"]["simple"]) | set(raw["variables"].get("arrays", {}))
    assert {"prose", "headline", "reason", "stat_count", "degraded",
            "generated_at", "model", "rows"} <= declared


def test_api_key_is_marked_secret(raw):
    assert raw["settings_schema"]["properties"]["api_key"]["secret"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test.sh tests/test_options.py tests/test_manifest.py`
Expected: FAIL — `AttributeError: 'GenerativeDashboardPlugin' object has no attribute 'get_options'` and `FileNotFoundError: manifest.json`

- [ ] **Step 3: Write the implementation**

Add to `__init__.py`:

```python
    # -- settings picker --------------------------------------------------

    def get_options(self, request: OptionsRequest) -> OptionsResult:
        """Browse variables for the settings picker.

        Runs on a throwaway instance with draft config applied, on every
        keystroke, whether or not the plugin is enabled. It must therefore
        never need an API key and never call the LLM.
        """
        if request.options_id == "variables":
            return OptionsResult(options=self._variable_options(request))
        if request.options_id == "watched":
            return OptionsResult(options=self._watched_options(request))
        raise OptionsUnavailable(f"Unknown options id: {request.options_id}")

    def _variable_options(self, request: OptionsRequest) -> list[Option]:
        # The settings dialog has no board, so judge tile width by a Flagship.
        width = geometry(DEFAULT_BOARD_ROWS, DEFAULT_BOARD_COLS).tile_width
        choices = catalog.variable_catalog(self.plugin_id, width, request.query)
        return [
            Option(
                value=choice.ref,
                label=choice.label,
                description=choice.disabled_reason or choice.description,
                group=choice.group,
                preview=choice.preview or None,
                disabled=choice.disabled,
            )
            for choice in choices[: request.limit]
        ]

    def _watched_options(self, request: OptionsRequest) -> list[Option]:
        """Only variables already on the watchlist can be pinned."""
        labels = self._labels(self.config)
        needle = request.query.strip().lower()
        return [
            Option(
                value=ref,
                label=labels.get(ref) or catalog.default_label(ref),
                description=ref,
            )
            for ref in self._watchlist(self.config)
            if not needle or needle in ref.lower()
        ][: request.limit]
```

Create `manifest.json`:

```json
{
  "id": "generative_dashboard",
  "name": "Generative Dashboard",
  "version": "1.0.0",
  "description": "Composes a full-board dashboard from variables you pick across your other plugins, using an LLM to surface whatever matters right now.",
  "author": "FiestaBoard Team",
  "repository": "https://github.com/Fiestaboard/fiestaboard-plugin--generative-dashboard",
  "documentation": "README.md",
  "icon": "layout-dashboard",
  "category": "data",
  "live_data": true,
  "min_refresh_seconds": 120,
  "fiestaboard_version": ">=2.10.0",
  "settings_schema": {
    "type": "object",
    "properties": {
      "enabled": {"type": "boolean", "title": "Enable Generative Dashboard", "default": false},
      "watchlist": {
        "type": "array",
        "title": "Watched Variables",
        "description": "The pool the dashboard chooses from. Watch more than fits — picking what matters is the point.",
        "items": {"type": "string"},
        "maxItems": 100,
        "default": [],
        "ui:widget": "remote-options",
        "ui:options": {
          "options_id": "variables",
          "multiple": true,
          "searchable": true,
          "server_search": true,
          "reorderable": true,
          "labels_field": "labels",
          "cache_seconds": 30,
          "placeholder": "Search variables from your enabled plugins"
        }
      },
      "labels": {
        "type": "object",
        "title": "Display Labels",
        "description": "Short board label per variable. Filled in by the picker.",
        "default": {}
      },
      "pinned": {
        "type": "array",
        "title": "Always Show",
        "description": "Variables that must appear even on a quiet day.",
        "items": {"type": "string"},
        "default": [],
        "ui:widget": "remote-options",
        "ui:options": {
          "options_id": "watched",
          "multiple": true,
          "searchable": true,
          "depends_on": ["watchlist"],
          "placeholder": "Pick from your watched variables"
        }
      },
      "notes": {
        "type": "object",
        "title": "Variable Notes",
        "description": "What a variable means, per variable — e.g. 'over 100 is unhealthy'. This is what turns the ranking from a guess into a judgement.",
        "default": {}
      },
      "thresholds": {
        "type": "object",
        "title": "Per-Variable Sensitivity",
        "description": "Percent change needed before a variable counts as having moved. Overrides the global default.",
        "default": {}
      },
      "output_mode": {
        "type": "string",
        "title": "Output Mode",
        "description": "grid lays stats out in tiles; prose writes a short sentence about what changed.",
        "enum": ["grid", "prose"],
        "default": "grid"
      },
      "api_base_url": {
        "type": "string",
        "title": "API Base URL",
        "description": "Any OpenAI-compatible chat completions endpoint (e.g. https://api.openai.com/v1, or http://localhost:11434/v1 for Ollama).",
        "default": "https://api.openai.com/v1"
      },
      "api_key": {
        "type": "string",
        "title": "API Key",
        "description": "Use any value for local endpoints that do not check it.",
        "secret": true
      },
      "model": {"type": "string", "title": "Model", "default": "gpt-4o-mini"},
      "temperature": {
        "type": "number",
        "title": "Temperature",
        "description": "Low values keep the layout steady. This is layout, not art.",
        "default": 0.3,
        "minimum": 0,
        "maximum": 2
      },
      "refresh_seconds": {
        "type": "integer",
        "title": "Minimum Seconds Between Generations",
        "description": "A floor, not a schedule. Nothing is generated unless a watched value has actually moved.",
        "default": 300,
        "minimum": 120
      },
      "default_threshold_pct": {
        "type": "number",
        "title": "Change Sensitivity (%)",
        "description": "How far a number must move before the board is worth redrawing.",
        "default": 5,
        "minimum": 0,
        "maximum": 100
      },
      "use_color": {
        "type": "boolean",
        "title": "Colour Accents",
        "description": "Let the model flag an outlier with a single coloured tile.",
        "default": true
      },
      "extra_instructions": {
        "type": "string",
        "title": "Extra Prompt Instructions",
        "format": "textarea",
        "default": ""
      }
    },
    "required": ["api_key"]
  },
  "env_vars": [
    {"name": "GENERATIVE_DASHBOARD_ENABLED", "required": false, "description": "Enable the plugin", "default": "false"},
    {"name": "GENERATIVE_DASHBOARD_API_KEY", "required": false, "description": "API key for the endpoint"},
    {"name": "GENERATIVE_DASHBOARD_API_BASE_URL", "required": false, "description": "Chat completions base URL", "default": "https://api.openai.com/v1"},
    {"name": "GENERATIVE_DASHBOARD_MODEL", "required": false, "description": "Model name", "default": "gpt-4o-mini"},
    {"name": "GENERATIVE_DASHBOARD_OUTPUT_MODE", "required": false, "description": "grid or prose", "default": "grid"}
  ],
  "variables": {
    "groups": {
      "display": {"label": "Display"},
      "meta": {"label": "Metadata"}
    },
    "simple": {
      "prose": {"description": "The generated sentence, in prose mode", "type": "string", "max_length": 200, "group": "display", "example": "AQI ROSE FROM 31 TO 168."},
      "headline": {"description": "The single most important stat right now", "type": "string", "max_length": 40, "group": "display", "example": "AQI 168"},
      "reason": {"description": "Why the board last changed", "type": "string", "max_length": 120, "group": "meta", "example": "AQI ROSE 31 TO 168"},
      "stat_count": {"description": "How many stats are currently shown", "type": "number", "max_length": 3, "group": "meta", "example": "8"},
      "degraded": {"description": "Empty when healthy, otherwise why the board is degraded", "type": "string", "max_length": 20, "group": "meta", "example": "no_llm"},
      "generated_at": {"description": "When the composition was last generated", "type": "string", "max_length": 20, "group": "meta", "example": "2026-09-03T14:30:00"},
      "model": {"description": "Model that composed the board", "type": "string", "max_length": 40, "group": "meta", "example": "gpt-4o-mini"}
    },
    "arrays": {
      "rows": {"description": "The composed board, one entry per row", "type": "string", "max_length": 22, "group": "display", "example": "CPU  42%   TEMP  61F"}
    }
  },
  "demo": {
    "flagship": {
      "name": "Generative Dashboard (Flagship)",
      "template": ["{{generative_dashboard.rows[0]}}", "{{generative_dashboard.rows[1]}}", "{{generative_dashboard.rows[2]}}", "{{generative_dashboard.rows[3]}}", "{{generative_dashboard.rows[4]}}", "{{generative_dashboard.rows[5]}}"],
      "line_metadata": [
        {"alignment": "left", "wrap": false},
        {"alignment": "left", "wrap": false},
        {"alignment": "left", "wrap": false},
        {"alignment": "left", "wrap": false},
        {"alignment": "left", "wrap": false},
        {"alignment": "left", "wrap": false}
      ]
    },
    "note": {
      "name": "Generative Dashboard (Note)",
      "template": ["{{generative_dashboard.rows[0]}}", "{{generative_dashboard.rows[1]}}", "{{generative_dashboard.rows[2]}}"],
      "line_metadata": [
        {"alignment": "left", "wrap": false},
        {"alignment": "left", "wrap": false},
        {"alignment": "left", "wrap": false}
      ]
    }
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./scripts/test.sh tests/test_options.py tests/test_manifest.py`
Expected: PASS, 21 tests

- [ ] **Step 5: Commit**

```bash
git add manifest.json __init__.py tests/test_options.py tests/test_manifest.py
git commit -m "feat: variable picker and plugin manifest"
```

---

### Task 10: Documentation and full-suite verification

**Files:**
- Create: `README.md`
- Create: `docs/SETUP.md`

- [ ] **Step 1: Write `README.md`**

Cover: what it does, the two output modes with a worked example of each, how
the gate keeps the board still, the failure ladder table from the spec, the
settings reference, and a note that watching more variables than fit is the
intended usage.

- [ ] **Step 2: Write `docs/SETUP.md`**

Cover: getting an API key, pointing at a local Ollama endpoint instead,
picking variables, writing useful notes, and tuning sensitivity.

- [ ] **Step 3: Run the entire suite**

Run: `./scripts/test.sh`
Expected: PASS, all tests across all modules

- [ ] **Step 4: Prove the new tests are not vacuous**

FiestaBoard's testing bar requires this. For at least the gate, the prose
number validator, and the reentrancy guard: break the production behaviour,
confirm the specific test fails, restore it. Record the observed failures.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/SETUP.md
git commit -m "docs: usage and setup guide"
```

---

## Self-Review

**Spec coverage:** watchlist/picker (T4, T9), gate (T3), board sizing (T2),
grid contract (T6, T7), prose contract (T6, T7), failure ladder (T5, T8),
reentrancy avoidance (T4, T8), non-blocking execution (T8), emitted variables
and manifest (T9), charset rules (T1), testing bar (T10).

**Deferred from the spec, deliberately:** `auto` mode, trend glyphs, history
deeper than one snapshot, and per-variable value formatters all remain out of
scope, as the spec's Deferred section states.

**Type consistency:** `Geometry`, `Tile`, `TileSpec`, `GridResult`,
`ProseResult`, `VariableChoice`, and `BoardState` are each defined once and
referenced with the same field names throughout. `GridResult.refs` is
parallel to `.tiles` and is what Task 8 zips against.
