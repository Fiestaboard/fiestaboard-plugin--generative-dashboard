"""The shared board-geometry conformance suite, run against this plugin.

Core is on ``PYTHONPATH`` in this repo's CI, so the suite is imported from
core rather than copied: the definition of "supports every board" then moves
in one edit for every plugin at once.
"""

import json
import pathlib

import pytest
from src.plugins.geometry_conformance import assert_board_conformance, note_array

from plugins.generative_dashboard import GenerativeDashboardPlugin, catalog

MANIFEST = json.loads(
    (pathlib.Path(__file__).resolve().parent.parent / "manifest.json").read_text()
)

# Enough stats to saturate the shortest rung of the growth ladder (a 3-row
# Note). The suite can only demand that a taller board render more rows when
# the shorter one was actually full, so a thin fixture would make the growth
# check pass vacuously.
# 37 rather than a round 40 on purpose: 37 does not divide into the ten tile
# columns of a 120-cell panel, so the last seven land in the ledger section —
# the code path that used to give each of them a whole 120-cell row.
VALUES = {f"src.stat{index:02d}": str(100 + index) for index in range(37)}

CONFIG = {
    "enabled": True,
    "api_key": "sk-test",
    "api_base_url": "https://api.test/v1",
    "model": "gpt-4o-mini",
    "temperature": 0.3,
    "output_mode": "grid",
    "refresh_seconds": 300,
    "default_threshold_pct": 5,
    "use_color": True,
    "watchlist": list(VALUES),
    "labels": {},
    "pinned": [],
    "notes": {},
    "thresholds": {},
}


@pytest.fixture
def factory(monkeypatch):
    """A callable returning a fresh, configured, network-free plugin."""
    monkeypatch.setattr(
        catalog, "read_values", lambda refs, board, exclude, **kw: dict(VALUES)
    )

    def make_plugin():
        plugin = GenerativeDashboardPlugin(MANIFEST)
        plugin.config = dict(CONFIG)
        # The suite renders many times and must never reach the network. The
        # composition worker is the only thing here that would.
        plugin._spawn = lambda *args, **kwargs: None
        return plugin

    return make_plugin


def test_renders_on_every_board_shape(factory):
    assert_board_conformance(
        factory,
        manifest=MANIFEST,
        strict_growth=True,
        require_note_array_preview=True,
    )


def test_the_widest_panel_is_not_one_stat_per_row(factory):
    """A 120-cell board must not spend a whole row on one label/value pair.

    This is the failure the suite cannot see: every row was inside its
    bounds, so nothing was violated — the board simply had a label in cell 1
    and its value in cell 115 with a hundred dead cells between, repeated
    down the page.
    """
    plugin = factory()
    lines = plugin.get_data(note_array(8, 8)).formatted_lines
    filled = [line for line in lines if line.strip()]
    assert filled, "the panel rendered nothing at all"
    # 40 stats over a 24x120 panel: anything close to one per row means the
    # leftovers each took a full-width row again.
    assert len(filled) <= 8, filled


def test_a_wide_board_shows_every_stat_it_was_given(factory):
    plugin = factory()
    joined = " ".join(plugin.get_data(note_array(8, 8)).formatted_lines)
    assert all(value in joined for value in VALUES.values())


def test_the_hook_returns_board_lines_not_row_objects(factory):
    """``get_formatted_display`` is the documented contract: ``list[str]``."""
    plugin = factory()
    with plugin._bound_board(note_array(2, 4)):
        lines = plugin.get_formatted_display()
    assert lines and all(isinstance(line, str) for line in lines)
