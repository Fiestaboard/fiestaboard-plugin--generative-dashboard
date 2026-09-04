"""End-to-end generation: prompt out, response in, board on the other side."""

import json
from unittest.mock import MagicMock, patch

import pytest
from src.devices import BoardContext

from plugins.generative_dashboard import (
    GenerativeDashboardPlugin,
    catalog,
)
from plugins.generative_dashboard.layout import geometry

FLAGSHIP = BoardContext("flagship", rows=6, cols=22)
GEO = geometry(6, 22)
CURRENT = {"air.aqi": "168", "wx.temp": "61F"}
PREVIOUS = {"air.aqi": "31", "wx.temp": "61F"}
WATCHLIST = ["air.aqi", "wx.temp"]


@pytest.fixture
def plugin(manifest, config, monkeypatch):
    instance = GenerativeDashboardPlugin(manifest)
    instance.config = config
    monkeypatch.setattr(catalog, "read_values", lambda refs, board, exclude: dict(CURRENT))
    # _run and _generate are driven explicitly here; never spawn a real thread.
    monkeypatch.setattr(instance, "_spawn", lambda *a, **k: None)
    return instance


def _reply(payload):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": json.dumps(payload)}}]
    }
    return response


def _reply_text(content):
    """A 200 response whose body is not the JSON we asked for."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return response


def _generate(plugin, config=None):
    return plugin._generate(
        GEO, config or plugin.config, WATCHLIST, CURRENT, PREVIOUS, []
    )


GOOD_GRID = {
    "tiles": [
        {"label": "AQI", "variable": "air.aqi", "color": "red"},
        {"label": "TEMP", "variable": "wx.temp"},
    ],
    "banner": "",
    "headline": "AQI 168",
    "reason": "AQI ROSE",
}


def test_a_good_grid_response_becomes_tile_specs(plugin):
    with patch("requests.post", return_value=_reply(GOOD_GRID)):
        specs, banner, prose, headline, reason = _generate(plugin)
    assert [s.ref for s in specs] == ["air.aqi", "wx.temp"]
    assert specs[0].color == "red"
    assert headline == "AQI 168"
    assert prose == ""


def test_a_good_prose_response_becomes_text(plugin):
    config = dict(plugin.config, output_mode="prose")
    payload = {"text": "AQI ROSE FROM 31 TO 168.", "headline": "AQI 168", "reason": "R"}
    with patch("requests.post", return_value=_reply(payload)):
        specs, banner, prose, headline, reason = _generate(plugin, config)
    assert prose == "AQI ROSE FROM 31 TO 168."
    assert specs == []


def test_a_rejected_response_is_retried_once(plugin):
    bad = _reply({"tiles": []})
    with patch("requests.post", return_value=bad) as post:
        assert _generate(plugin) is None
    assert post.call_count == 2


def test_the_retry_prompt_tells_the_model_it_was_rejected(plugin):
    with patch("requests.post", return_value=_reply({"tiles": []})) as post:
        _generate(plugin)
    retry_system = post.call_args_list[1].kwargs["json"]["messages"][0]["content"]
    assert "rejected" in retry_system.lower()


def test_a_retry_that_succeeds_is_used(plugin):
    with patch("requests.post", side_effect=[_reply({"tiles": []}), _reply(GOOD_GRID)]):
        specs, _, _, headline, _ = _generate(plugin)
    assert headline == "AQI 168"


def test_a_network_failure_gives_up_without_retrying(plugin):
    import requests

    with patch("requests.post", side_effect=requests.RequestException("down")) as post:
        assert _generate(plugin) is None
    # A dead endpoint will still be dead a millisecond later.
    assert post.call_count == 1


def test_prose_that_invents_a_number_is_refused_end_to_end(plugin):
    config = dict(plugin.config, output_mode="prose")
    payload = {"text": "AQI IS UP 442 PERCENT.", "headline": "", "reason": ""}
    with patch("requests.post", return_value=_reply(payload)):
        assert _generate(plugin, config) is None


def test_a_successful_run_puts_tiles_on_the_board(plugin):
    with plugin._bound_board(FLAGSHIP):
        plugin.fetch_data()  # creates the board state the worker swaps into
    with patch("requests.post", return_value=_reply(GOOD_GRID)):
        plugin._run("flagship", GEO, plugin.config, WATCHLIST, CURRENT, PREVIOUS, [],
                    plugin._config_generation)
    with plugin._bound_board(FLAGSHIP):
        lines = plugin.fetch_data().formatted_lines
    joined = " ".join(lines)
    assert "{red}" in joined and "168" in joined and "61F" in joined


def test_a_failed_run_marks_the_board_degraded(plugin):
    import requests

    with plugin._bound_board(FLAGSHIP):
        plugin.fetch_data()
    with patch("requests.post", side_effect=requests.RequestException("down")):
        plugin._run("flagship", GEO, plugin.config, WATCHLIST, CURRENT, PREVIOUS, [],
                    plugin._config_generation)
    with plugin._bound_board(FLAGSHIP):
        result = plugin.fetch_data()
    assert result.data["degraded"] == "no_llm"


def test_a_result_from_an_old_config_generation_is_discarded(plugin):
    with plugin._bound_board(FLAGSHIP):
        plugin.fetch_data()  # create the state
    stale_generation = plugin._config_generation
    plugin._config_generation += 1  # config changed while the worker ran
    with patch("requests.post", return_value=_reply(GOOD_GRID)):
        plugin._run("flagship", GEO, plugin.config, WATCHLIST, CURRENT, PREVIOUS, [],
                    stale_generation)
    assert plugin._states["flagship"].tiles == []


def test_a_result_arriving_after_cleanup_is_discarded(plugin):
    with plugin._bound_board(FLAGSHIP):
        plugin.fetch_data()
    plugin.cleanup()
    with patch("requests.post", return_value=_reply(GOOD_GRID)):
        plugin._run("flagship", GEO, plugin.config, WATCHLIST, CURRENT, PREVIOUS, [],
                    plugin._config_generation)
    assert plugin._states["flagship"].tiles == []


def test_the_worker_always_releases_its_in_flight_slot(plugin):
    import requests

    plugin._inflight.add("flagship")
    with patch("requests.post", side_effect=requests.RequestException("down")):
        plugin._run("flagship", GEO, plugin.config, WATCHLIST, CURRENT, PREVIOUS, [],
                    plugin._config_generation)
    assert "flagship" not in plugin._inflight


def test_an_unexpected_exception_does_not_wedge_the_plugin(plugin):
    with patch("requests.post", side_effect=RuntimeError("something odd")):
        plugin._run("flagship", GEO, plugin.config, WATCHLIST, CURRENT, PREVIOUS, [],
                    plugin._config_generation)
    assert "flagship" not in plugin._inflight


def test_malformed_json_is_retried_before_giving_up(plugin):
    with patch("requests.post", side_effect=[_reply_text("nonsense"), _reply(GOOD_GRID)]):
        _, _, _, headline, _ = _generate(plugin)
    assert headline == "AQI 168"


def test_malformed_json_twice_gives_up(plugin):
    with patch("requests.post", return_value=_reply_text("nonsense")) as post:
        assert _generate(plugin) is None
    assert post.call_count == 2
