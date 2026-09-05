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
    monkeypatch.setattr(catalog, "read_values", lambda refs, board, exclude, **kw: dict(VALUES))
    # Generation is exercised explicitly; default to never spawning a worker.
    monkeypatch.setattr(instance, "_spawn", lambda *a, **k: None)
    return instance


def _fetch(plugin, board=FLAGSHIP):
    with plugin._bound_board(board):
        return plugin.fetch_data()


def _values(monkeypatch, values):
    monkeypatch.setattr(catalog, "read_values", lambda refs, board, exclude, **kw: dict(values))


def _spy_spawn(plugin, monkeypatch):
    calls = []
    monkeypatch.setattr(plugin, "_spawn", lambda *a, **k: calls.append(a))
    return calls


def test_cold_start_renders_a_deterministic_grid_without_any_llm_call(plugin):
    result = _fetch(plugin)
    joined = " ".join(result.formatted_lines)
    assert "68" in joined and "61F" in joined
    assert result.data["degraded"] == "no_llm"


def test_rows_match_the_board_height(plugin):
    assert len(_fetch(plugin, FLAGSHIP).formatted_lines) == 6
    assert len(_fetch(plugin, NOTE).formatted_lines) == 3


def test_formatted_lines_are_offered_to_core(plugin):
    assert len(_fetch(plugin).formatted_lines) == 6


def test_no_watchlist_values_shows_humor_and_a_timestamp(plugin, monkeypatch):
    _values(monkeypatch, {})
    result = _fetch(plugin)
    assert result.data["degraded"] == "no_data"
    assert "NO DATA YET" in " ".join(result.formatted_lines)


def test_a_board_with_data_reports_when_data_was_last_good(plugin, monkeypatch):
    _fetch(plugin)
    _values(monkeypatch, {})
    assert "LAST GOOD DATA" in " ".join(_fetch(plugin).formatted_lines)


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
    import time as _t; plugin._inflight["flagship"] = _t.monotonic()  # a worker is already running
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
    assert "999" in " ".join(result.formatted_lines)


def test_prose_is_shown_when_the_model_has_produced_some(plugin):
    plugin.config = dict(plugin.config, output_mode="prose")
    _fetch(plugin)
    plugin._states["flagship"].prose = "ALL QUIET TODAY."
    assert "ALL QUIET" in " ".join(_fetch(plugin).formatted_lines)


def test_stale_prose_gives_way_to_a_grid_of_live_values(plugin, monkeypatch):
    plugin.config = dict(plugin.config, output_mode="prose")
    _fetch(plugin)
    state = plugin._states["flagship"]
    state.prose = "AQI IS 68."
    state.stale = True  # a generation failed after the numbers moved
    _values(monkeypatch, {"air.aqi": "999", "wx.temp": "61F"})
    joined = " ".join(_fetch(plugin).formatted_lines)
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


def test_generation_resumes_after_a_disable_enable_cycle(plugin, monkeypatch):
    # Disable calls cleanup(); a one-way stop latch bricked a live board
    # until full reload. Cleanup must orphan in-flight results, not end the
    # instance's working life.
    plugin.cleanup()
    calls = _spy_spawn(plugin, monkeypatch)
    _fetch(plugin)
    assert calls, "a render after cleanup must be able to spawn again"


def test_a_wedged_inflight_marker_expires(plugin, monkeypatch):
    # A worker killed mid-flight (e.g. by a plugin update) left its marker
    # behind forever; every later spawn was silently refused — the frozen
    # board of 2026-09-05.
    import time as _time

    plugin._inflight["flagship"] = _time.monotonic() - 700  # long dead
    calls = _spy_spawn(plugin, monkeypatch)
    _fetch(plugin)
    assert calls


def test_a_live_inflight_marker_still_blocks(plugin, monkeypatch):
    import time as _time

    plugin._inflight["flagship"] = _time.monotonic()  # genuinely running
    calls = _spy_spawn(plugin, monkeypatch)
    _fetch(plugin)
    assert calls == []


def test_rows_are_emitted_in_the_shape_the_manifest_declares(plugin):
    # variables.arrays models lists of objects; templates read rows.0.text.
    rows = _fetch(plugin).data["rows"]
    assert all(set(row) == {"text"} for row in rows)
    assert [r["text"] for r in rows] == _fetch(plugin).formatted_lines


def test_a_repeated_spawn_for_one_board_is_ignored_while_in_flight(plugin, monkeypatch):
    started = []
    monkeypatch.setattr(
        "threading.Thread",
        lambda **kw: type("T", (), {"start": lambda self: started.append(kw["name"])})(),
    )
    # The fixture stubs _spawn, so reach past it to the real implementation.
    spawn = GenerativeDashboardPlugin._spawn
    args = ("flagship", None, plugin.config, [], {}, {}, [], 0)
    spawn(plugin, *args)
    spawn(plugin, *args)
    assert len(started) == 1


def test_the_outage_message_does_not_change_on_every_render(plugin, monkeypatch):
    # live_data means fetch_data runs on every render. Rotating the joke each
    # time would make the outage screen flicker — the exact noise we avoid.
    _values(monkeypatch, {})
    first = _fetch(plugin).formatted_lines
    for _ in range(5):
        assert _fetch(plugin).formatted_lines == first


def test_a_new_outage_episode_picks_a_new_message(plugin, monkeypatch):
    _values(monkeypatch, {})
    first = _fetch(plugin).formatted_lines
    _values(monkeypatch, VALUES)
    _fetch(plugin)  # recovered
    _values(monkeypatch, {})
    assert _fetch(plugin).formatted_lines != first


def test_concurrent_fetches_do_not_lose_the_previous_snapshot(plugin, monkeypatch):
    # Two render threads hitting the same board must not overwrite the
    # pre-change snapshot with the post-change one; the prompt needs the
    # "was" value to explain what moved.
    _fetch(plugin)
    state = plugin._states["flagship"]
    state.tiles = [TileSpec("air.aqi", "AQI", None)]
    state.last_generated = 0.0
    captured = []
    monkeypatch.setattr(
        plugin, "_spawn",
        lambda *a, **k: captured.append(a[5]),  # the `previous` argument
    )
    _values(monkeypatch, {"air.aqi": "168", "wx.temp": "61F"})
    _fetch(plugin)
    _fetch(plugin)
    assert captured[0]["air.aqi"] == "68"


def test_an_empty_watchlist_watches_everything_eligible(plugin, monkeypatch):
    # Curating a pool duplicates the model's own job. Empty means "everything".
    monkeypatch.setattr(
        catalog, "eligible_refs",
        lambda exclude, width: ["air.aqi", "wx.temp", "sys.cpu"],
    )
    plugin.config = dict(plugin.config, watchlist=[])
    assert plugin._watchlist(plugin.config) == ["air.aqi", "wx.temp", "sys.cpu"]


def test_an_explicit_watchlist_still_narrows(plugin, monkeypatch):
    monkeypatch.setattr(
        catalog, "eligible_refs",
        lambda exclude, width: ["air.aqi", "wx.temp", "sys.cpu"],
    )
    plugin.config = dict(plugin.config, watchlist=["air.aqi"])
    assert plugin._watchlist(plugin.config) == ["air.aqi"]


def test_with_nothing_enabled_anywhere_it_asks_for_plugins_not_variables(plugin, monkeypatch):
    monkeypatch.setattr(catalog, "eligible_refs", lambda exclude, width: [])
    plugin.config = dict(plugin.config, watchlist=[])
    result = _fetch(plugin)
    assert result.data["degraded"] == "unconfigured"
    assert "PLUGIN" in " ".join(result.formatted_lines)


def test_notes_are_read_from_the_row_form(plugin):
    # The settings form cannot render an open-ended object, so notes are rows.
    plugin.config = dict(plugin.config, notes=[
        {"variable": "air.aqi", "note": "over 100 is unhealthy"},
        {"variable": "sys.cpu", "note": "rarely matters"},
    ])
    assert plugin._notes(plugin.config) == {
        "air.aqi": "over 100 is unhealthy",
        "sys.cpu": "rarely matters",
    }


def test_notes_still_accept_the_old_map_form(plugin):
    plugin.config = dict(plugin.config, notes={"air.aqi": "over 100 is unhealthy"})
    assert plugin._notes(plugin.config) == {"air.aqi": "over 100 is unhealthy"}


def test_incomplete_note_rows_are_ignored(plugin):
    plugin.config = dict(plugin.config, notes=[
        {"variable": "", "note": "orphan"}, {"variable": "a.b"}, "nonsense",
    ])
    assert plugin._notes(plugin.config) == {}


def test_thresholds_are_read_from_the_row_form(plugin):
    plugin.config = dict(plugin.config, thresholds=[
        {"variable": "air.aqi", "percent": 2},
        {"variable": "sys.cpu", "percent": "25"},
    ])
    assert plugin._thresholds(plugin.config) == {"air.aqi": 2.0, "sys.cpu": 25.0}


def test_thresholds_still_accept_the_old_map_form(plugin):
    plugin.config = dict(plugin.config, thresholds={"air.aqi": 2})
    assert plugin._thresholds(plugin.config) == {"air.aqi": 2.0}


def test_unparseable_threshold_rows_are_ignored(plugin):
    plugin.config = dict(plugin.config, thresholds=[{"variable": "a.b", "percent": "soon"}])
    assert plugin._thresholds(plugin.config) == {}


def test_stat_count_reports_tiles_actually_placed(plugin, monkeypatch):
    # Not the number of candidates: a 6x22 board fits 12, however many are watched.
    many = {f"p.v{i}": str(i) for i in range(40)}
    _values(monkeypatch, many)
    plugin.config = dict(plugin.config, watchlist=list(many))
    result = _fetch(plugin)
    assert result.data["stat_count"] <= 12
    assert result.data["stat_count"] == len([r for r in result.formatted_lines if r.strip()]) * 2


def test_never_rendered_is_reported_differently_from_a_failed_model(plugin):
    # board=None means no page has displayed this yet, so nothing was asked of
    # the model. Calling that "no_llm" reads as an outage.
    result = plugin.fetch_data()
    assert result.data["degraded"] == "awaiting_board"


def test_reloading_the_package_refreshes_its_submodules():
    """An in-place update re-executes __init__.py but not its submodules.

    Python keeps the old ones in sys.modules, so a new __init__ pairs with a
    stale catalog and the plugin dies with AttributeError on the first fetch.
    """
    import importlib
    import sys

    pkg = "plugins.generative_dashboard"
    sys.modules[f"{pkg}.catalog"].__marker__ = "stale"
    importlib.reload(sys.modules[pkg])
    assert not hasattr(sys.modules[f"{pkg}.catalog"], "__marker__")


def test_suffixes_survive_the_live_re_render(plugin, monkeypatch):
    # The unit must stay attached when values refresh between generations.
    _fetch(plugin)
    state = plugin._states["flagship"]
    state.tiles = [TileSpec("air.aqi", "TEMP", None, suffix="F")]
    _values(monkeypatch, {"air.aqi": "71", "wx.temp": "61F"})
    assert "71F" in " ".join(_fetch(plugin).formatted_lines)


def test_prose_with_a_colored_headline_renders_it_as_a_framed_title(plugin):
    plugin.config = dict(plugin.config, output_mode="prose")
    _fetch(plugin)
    state = plugin._states["flagship"]
    state.prose = "STOCKS SLIPPING LATE."
    state.headline = "MARKET WATCH"
    state.banner_color = "red"
    lines = _fetch(plugin).formatted_lines
    top = next(l for l in lines if l.strip())
    assert "{red}" in top and "MARKET WATCH" in top
    assert any("SLIPPING" in l for l in lines)


def test_prose_without_a_color_stays_a_plain_block(plugin):
    plugin.config = dict(plugin.config, output_mode="prose")
    _fetch(plugin)
    state = plugin._states["flagship"]
    state.prose = "ALL QUIET."
    state.headline = "CALM"
    state.banner_color = None
    assert not any("{" in l for l in _fetch(plugin).formatted_lines)
