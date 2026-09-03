"""Plugin lifecycle, gating, worker behaviour, and the failure ladder."""

import time

import pytest
from src.devices import BoardContext

from plugins.generative_dashboard import (
    GenerativeDashboardPlugin,
    TileSpec,
    catalog,
)

FLAGSHIP = BoardContext("flagship", rows=6, cols=22)
NOTE = BoardContext("note", rows=3, cols=15)
VALUES = {"air.aqi": "68", "wx.temp": "61F"}


@pytest.fixture
def plugin(manifest, config, monkeypatch):
    instance = GenerativeDashboardPlugin(manifest)
    instance.config = config
    monkeypatch.setattr(catalog, "read_values", lambda refs, board, exclude: dict(VALUES))
    # Generation is exercised explicitly; default to never spawning a worker.
    monkeypatch.setattr(instance, "_spawn", lambda *a, **k: None)
    return instance


def _fetch(plugin, board=FLAGSHIP):
    with plugin._bound_board(board):
        return plugin.fetch_data()


def _values(monkeypatch, values):
    monkeypatch.setattr(catalog, "read_values", lambda refs, board, exclude: dict(values))


def _spy_spawn(plugin, monkeypatch):
    calls = []
    monkeypatch.setattr(plugin, "_spawn", lambda *a, **k: calls.append(a))
    return calls


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


def test_formatted_lines_are_offered_to_core(plugin):
    assert len(_fetch(plugin).formatted_lines) == 6


def test_no_watchlist_values_shows_humor_and_a_timestamp(plugin, monkeypatch):
    _values(monkeypatch, {})
    result = _fetch(plugin)
    assert result.data["degraded"] == "no_data"
    assert "NO DATA YET" in " ".join(result.data["rows"])


def test_a_board_with_data_reports_when_data_was_last_good(plugin, monkeypatch):
    _fetch(plugin)
    _values(monkeypatch, {})
    assert "LAST GOOD DATA" in " ".join(_fetch(plugin).data["rows"])


def test_each_board_size_keeps_its_own_composition(plugin):
    _fetch(plugin, FLAGSHIP)
    _fetch(plugin, NOTE)
    assert set(plugin._states) == {"flagship", "note"}


def test_note_arrays_are_keyed_by_their_dimensions(plugin):
    _fetch(plugin, BoardContext("note_array", rows=6, cols=30))
    assert "note_array:30x6" in plugin._states


def test_generation_is_never_started_without_a_board(plugin, monkeypatch):
    calls = _spy_spawn(plugin, monkeypatch)
    plugin.fetch_data()  # no _bound_board: self.board is None
    assert calls == []


def test_generation_starts_when_a_board_is_present(plugin, monkeypatch):
    calls = _spy_spawn(plugin, monkeypatch)
    _fetch(plugin)
    assert calls


def test_generation_is_skipped_without_an_api_key(plugin, monkeypatch):
    plugin.config = dict(plugin.config, api_key="")
    calls = _spy_spawn(plugin, monkeypatch)
    _fetch(plugin)
    assert calls == []


def test_a_quiet_cycle_does_not_start_a_generation(plugin, monkeypatch):
    _fetch(plugin)
    plugin._states["flagship"].tiles = [TileSpec("air.aqi", "AQI", None)]
    calls = _spy_spawn(plugin, monkeypatch)
    _fetch(plugin)
    assert calls == []


def test_a_material_change_starts_a_generation(plugin, monkeypatch):
    _fetch(plugin)
    state = plugin._states["flagship"]
    state.tiles = [TileSpec("air.aqi", "AQI", None)]
    state.last_generated = 0.0
    _values(monkeypatch, {"air.aqi": "168", "wx.temp": "61F"})
    calls = _spy_spawn(plugin, monkeypatch)
    _fetch(plugin)
    assert calls


def test_a_change_within_the_refresh_floor_waits(plugin, monkeypatch):
    _fetch(plugin)
    state = plugin._states["flagship"]
    state.tiles = [TileSpec("air.aqi", "AQI", None)]
    state.last_generated = time.monotonic()  # just generated
    _values(monkeypatch, {"air.aqi": "168", "wx.temp": "61F"})
    calls = _spy_spawn(plugin, monkeypatch)
    _fetch(plugin)
    assert calls == []


def test_the_snapshot_does_not_advance_when_generation_is_blocked(plugin, monkeypatch):
    _fetch(plugin)
    state = plugin._states["flagship"]
    state.tiles = [TileSpec("air.aqi", "AQI", None)]
    plugin._inflight.add("flagship")  # a worker is already running
    _values(monkeypatch, {"air.aqi": "168", "wx.temp": "61F"})
    _fetch(plugin)
    # Snapshot still holds the pre-change value, so the change stays pending
    # and small drifts keep accumulating instead of silently resetting.
    assert state.values["air.aqi"] == "68"


def test_live_values_are_re_rendered_between_generations(plugin, monkeypatch):
    _fetch(plugin)
    plugin._states["flagship"].tiles = [TileSpec("air.aqi", "AQI", None)]
    _values(monkeypatch, {"air.aqi": "999", "wx.temp": "61F"})
    result = _fetch(plugin)
    # The model has not run again, but the number on the board is current.
    assert "999" in " ".join(result.data["rows"])


def test_prose_is_shown_when_the_model_has_produced_some(plugin):
    plugin.config = dict(plugin.config, output_mode="prose")
    _fetch(plugin)
    plugin._states["flagship"].prose = "ALL QUIET TODAY."
    assert "ALL QUIET" in " ".join(_fetch(plugin).data["rows"])


def test_stale_prose_gives_way_to_a_grid_of_live_values(plugin, monkeypatch):
    plugin.config = dict(plugin.config, output_mode="prose")
    _fetch(plugin)
    state = plugin._states["flagship"]
    state.prose = "AQI IS 68."
    state.stale = True  # a generation failed after the numbers moved
    _values(monkeypatch, {"air.aqi": "999", "wx.temp": "61F"})
    joined = " ".join(_fetch(plugin).data["rows"])
    assert "999" in joined and "AQI IS 68" not in joined


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


def test_validate_config_rejects_an_out_of_range_temperature(plugin):
    errors = plugin.validate_config({"api_key": "k", "temperature": 9})
    assert any("temperature" in e for e in errors)


def test_config_change_discards_an_in_flight_result(plugin):
    generation = plugin._config_generation
    plugin.on_config_change({}, {})
    assert plugin._config_generation != generation


def test_cleanup_stops_further_swaps(plugin):
    plugin.cleanup()
    assert plugin._stopped
