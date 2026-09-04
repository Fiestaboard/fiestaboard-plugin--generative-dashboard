"""Generative Dashboard plugin for FiestaBoard.

Composes a full-board dashboard from a curated pool of other plugins'
variables, asking an LLM which of them matter right now. Regenerates only when
the underlying numbers move materially, so the board stays still when the
world is quiet.
"""

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# An in-place update re-executes this file but leaves the package's other
# modules in sys.modules from the previous version. A new __init__ then binds
# to a stale catalog/layout/llm and the plugin dies with AttributeError on its
# first fetch — which is what a user sees as "???" filling the board. Drop them
# so the imports below always take the code that shipped with this file.
for _stale in [_m for _m in list(sys.modules) if _m.startswith(__name__ + ".")]:
    del sys.modules[_stale]

from src.plugins.base import (
    Option,
    OptionsRequest,
    OptionsResult,
    OptionsUnavailable,
    PluginBase,
    PluginResult,
)

from . import catalog, fallback, gate
from .charset import sanitize
from .layout import Geometry, Tile, geometry, placed_count, render_grid, wrap_center
from .llm import DashboardLLM, LLMError, build_grid_prompt, build_prose_prompt
from .validation import ValidationError, validate_grid, validate_prose

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
    headline: str = ""
    reason: str = ""
    degraded: str = "no_llm"
    generated_at: str = ""
    last_good: str = ""
    recent: dict[str, str] = field(default_factory=dict)
    last_generated: float = 0.0
    outage_index: int = -1
    failures: int = 0
    stale: bool = False
    lines: list[str] = field(default_factory=list)


class GenerativeDashboardPlugin(PluginBase):
    """See the module docstring. Composition happens off the render path."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        super().__init__(manifest)
        self._states: dict[str, BoardState] = {}
        # Reentrant: the generation decision holds the lock across _spawn,
        # which takes it again to claim the in-flight slot.
        self._lock = threading.RLock()
        self._inflight: set[str] = set()
        self._stopped = False
        self._config_generation = 0
        self._outage_index = 0

    @property
    def plugin_id(self) -> str:
        return "generative_dashboard"

    # -- configuration ----------------------------------------------------

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        if not config.get("api_key"):
            errors.append("API key is required")

        base_url = config.get("api_base_url", "")
        if base_url and not base_url.startswith(("http://", "https://")):
            errors.append("API base URL must start with http:// or https://")

        temperature = config.get("temperature", 0.3)
        if not isinstance(temperature, int | float) or isinstance(temperature, bool) or not 0 <= temperature <= 2:
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

    def on_config_change(self, old_config: dict[str, Any], new_config: dict[str, Any]) -> None:
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

    def _watchlist(self, config: dict[str, Any]) -> list[str]:
        """The candidate pool: what the user chose, or everything if they did not.

        An empty watchlist means "watch everything", not "watch nothing".
        Curating a pool by hand duplicates the model's job, so the picker is
        there to narrow the default, not to construct it.
        """
        raw = config.get("watchlist") or []
        chosen = (
            [r for r in raw if isinstance(r, str) and "." in r][:MAX_WATCHLIST]
            if isinstance(raw, list)
            else []
        )
        if chosen:
            return chosen
        # Judged against the whole board: a long value gets a full row to
        # itself, so a single column's width is no longer the limit.
        return catalog.eligible_refs(self.plugin_id, DEFAULT_BOARD_COLS)

    def _labels(self, config: dict[str, Any]) -> dict[str, str]:
        raw = config.get("labels") or {}
        if not isinstance(raw, dict):
            return {}
        return {str(k): sanitize(str(v)) for k, v in raw.items()}

    @staticmethod
    def _rows_to_map(raw: Any, value_key: str) -> dict[str, str]:
        """Read a {variable, <value_key>} row list into a map.

        The settings form cannot render an open-ended object — it has no
        key/value editor — so these arrive as rows. A plain map is still
        accepted, since that is what older configs hold.
        """
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
        if not isinstance(raw, list):
            return {}
        out: dict[str, str] = {}
        for row in raw:
            if not isinstance(row, dict):
                continue
            ref = str(row.get("variable") or "").strip()
            value = row.get(value_key)
            if ref and value not in (None, ""):
                out[ref] = str(value)
        return out

    def _notes(self, config: dict[str, Any]) -> dict[str, str]:
        return self._rows_to_map(config.get("notes"), "note")

    def _pinned(self, config: dict[str, Any]) -> list[str]:
        raw = config.get("pinned") or []
        if not isinstance(raw, list):
            return []
        return [r for r in raw if isinstance(r, str)]

    def _thresholds(self, config: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for ref, value in self._rows_to_map(config.get("thresholds"), "percent").items():
            try:
                out[ref] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    # -- the render path --------------------------------------------------

    def fetch_data(self) -> PluginResult:
        config = self.config
        geo = self._geometry()
        key = self._state_key()
        with self._lock:
            state = self._states.setdefault(key, BoardState())

        watchlist = self._watchlist(config)
        if not watchlist:
            return self._result(state, fallback.setup_lines(geo), "unconfigured", 0)

        values = catalog.read_values(
            watchlist, self.board, self.plugin_id, fallback=state.recent
        )
        if values:
            # Remember what we saw, so a slow source next cycle shows its
            # last value instead of vanishing off the board.
            state.recent = dict(values)
        if not values:
            # Pick the message once per outage, not once per render: with
            # live_data the board redraws constantly, and a joke that changes
            # every few seconds is the noise this plugin exists to avoid.
            if state.outage_index < 0:
                with self._lock:
                    self._outage_index += 1
                    state.outage_index = self._outage_index
            lines = fallback.outage_lines(geo, state.last_good or None, state.outage_index)
            return self._result(state, lines, "no_data", 0)

        state.outage_index = -1
        state.last_good = datetime.now().strftime("%H:%M")

        changed = gate.material_changes(
            state.values,
            values,
            self._thresholds(config),
            float(config.get("default_threshold_pct", 5) or 5),
        )
        # Decide and record atomically. Two render threads on one board must
        # not both advance the snapshot, or the second erases the "was" value
        # the prompt needs to explain what moved.
        with self._lock:
            if (changed or not (state.tiles or state.prose)) and self._may_generate(
                key, state, config
            ):
                state.previous = dict(state.values)
                state.values = dict(values)
                state.last_generated = time.monotonic()
                self._spawn(
                    key, geo, config, watchlist, dict(values), dict(state.previous),
                    list(state.lines), self._config_generation,
                )

        lines, degraded, stat_count = self._compose(state, geo, config, values, watchlist)
        state.lines = lines
        return self._result(state, lines, degraded, stat_count)

    def _may_generate(self, key: str, state: BoardState, config: dict[str, Any]) -> bool:
        """Whether a worker may start right now."""
        if self.board is None:
            # Not a render. get_all_variables_with_metadata() fans out with
            # board=None from the settings dialog; generating there would bill
            # the user for a keystroke.
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
        config: dict[str, Any],
        values: dict[str, str],
        watchlist: list[str],
    ) -> tuple[list[str], str, int]:
        """Build the lines to show, using live values wherever possible."""
        use_color = bool(config.get("use_color", True))

        # Read the worker-owned fields as one consistent snapshot: _run swaps
        # them together, and rendering half of an old composition against half
        # of a new one would show a board that never existed.
        with self._lock:
            specs = list(state.tiles)
            banner, prose = state.banner, state.prose
            stale, degraded = state.stale, state.degraded

        if config.get("output_mode", "grid") == "prose" and prose and not stale:
            return wrap_center(prose, geo.rows, geo.cols), degraded, 0

        if specs:
            tiles = [
                Tile(label=spec.label, value=values[spec.ref], color=spec.color)
                for spec in specs
                if spec.ref in values
            ]
            if tiles:
                lines = render_grid(tiles, geo, banner=banner, use_color=use_color)
                return lines, degraded, placed_count(tiles, geo, banner)

        tiles = fallback.deterministic_tiles(
            watchlist, self._labels(config), values, self._pinned(config)
        )
        # Nothing has been asked of the model yet when no board has rendered
        # this plugin; calling that an LLM failure reads as an outage.
        reason = "no_llm" if self.board is not None else "awaiting_board"
        return render_grid(tiles, geo, use_color=False), reason, placed_count(tiles, geo)

    def _result(
        self, state: BoardState, lines: list[str], degraded: str, stat_count: int
    ) -> PluginResult:
        return PluginResult(
            available=True,
            formatted_lines=lines,
            data={
                # Arrays are lists of objects in core's variable schema,
                # reachable from a template as ``rows.0.text``. The plain
                # strings go out via ``formatted_lines``.
                "rows": [{"text": line} for line in lines],
                "prose": state.prose,
                "headline": state.headline,
                "reason": state.reason,
                "stat_count": stat_count,
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
        threading.Thread(
            target=self._run,
            args=(key, geo, config, watchlist, current, previous, previous_board, generation),
            name=f"generative-dashboard-{key}",
            daemon=True,
        ).start()

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
            if outcome is None:
                state.failures += 1
                state.degraded = "no_llm"
                # The numbers moved but the composition did not follow, so any
                # frozen prose is now describing a board that no longer exists.
                state.stale = True
                return
            tiles, banner, prose, headline, reason = outcome
            state.tiles = tiles
            state.banner = banner
            state.prose = prose
            state.headline = headline
            state.reason = reason
            state.failures = 0
            state.stale = False
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
        pinned = self._pinned(config)
        use_color = bool(config.get("use_color", True))
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
                    use_color=use_color, extra_instructions=extra + suffix,
                )
            try:
                payload = client.complete(system, user)
            except LLMError as exc:
                logger.warning(
                    "Dashboard LLM call failed (attempt %d/2): %s", attempt + 1, exc
                )
                if exc.retryable and attempt == 0:
                    continue
                return None

            try:
                if mode == "prose":
                    result = validate_prose(
                        payload, geo=geo, current=current, previous=previous
                    )
                    return [], "", result.text, result.headline, result.reason
                grid = validate_grid(
                    payload, watchlist=watchlist, pinned=pinned, values=current,
                    labels=labels, geo=geo, use_color=use_color,
                )
                specs = [
                    TileSpec(ref=ref, label=tile.label, color=tile.color)
                    for ref, tile in zip(grid.refs, grid.tiles, strict=True)
                ]
                return specs, grid.banner, "", grid.headline, grid.reason
            except ValidationError as exc:
                logger.warning(
                    "Dashboard response rejected (attempt %d/2): %s", attempt + 1, exc
                )

        return None

    # -- settings picker --------------------------------------------------

    def get_options(self, request: OptionsRequest) -> OptionsResult:
        """Browse variables for the settings picker.

        Runs on a throwaway instance with draft config applied, on every
        keystroke, whether or not the plugin is enabled. It must therefore
        never need an API key and never call the LLM.
        """
        if request.options_id == "variables":
            return self._variable_options(request)
        if request.options_id == "watched":
            options = self._watched_options(request)
            if not options:
                # An empty picker with no reason reads as broken.
                return OptionsResult(
                    options=[],
                    error="Nothing to pin yet — enable some plugins so their "
                    "variables become available.",
                )
            return OptionsResult(options=options)
        raise OptionsUnavailable(f"Unknown options id: {request.options_id}")

    def _variable_options(self, request: OptionsRequest) -> OptionsResult:
        # The settings dialog has no board, so judge by a Flagship's width.
        width = DEFAULT_BOARD_COLS
        choices = catalog.variable_catalog(self.plugin_id, width, request.query)
        shown = choices[: request.limit]
        return OptionsResult(
            options=[
                Option(
                    value=choice.ref,
                    label=choice.label,
                    # Keep the ref in the description even when disabled: the
                    # search box filters on label + description only.
                    description=" · ".join(
                        part
                        for part in (choice.disabled_reason, choice.description)
                        if part
                    ),
                    group=choice.group,
                    preview=choice.preview or None,
                    disabled=choice.disabled,
                )
                for choice in shown
            ],
            # Silent truncation reads as "that is everything", which is exactly
            # how a whole plugin's variables appear to go missing.
            has_more=len(choices) > len(shown),
            total=len(choices),
        )

    def _watched_options(self, request: OptionsRequest) -> list[Option]:
        """Only variables already on the watchlist can be pinned."""
        labels = self._labels(self.config)
        needle = request.query.strip().lower()
        return [
            Option(
                # The widget draws only the label, so the owning plugin has to
                # be in it or two variables called "temp" look identical.
                value=ref,
                label=catalog.picker_label(
                    ref, catalog.plugin_display_name(ref.split(".", 1)[0]), labels.get(ref, "")
                ),
                description=ref,
            )
            for ref in self._watchlist(self.config)
            if not needle or needle in ref.lower()
        ][: request.limit]
