"""The failure ladder: what the board shows when things go wrong."""

from plugins.generative_dashboard.fallback import (
    OUTAGE_MESSAGES,
    deterministic_tiles,
    outage_lines,
    setup_lines,
)
from plugins.generative_dashboard.layout import geometry

BOARDS = [geometry(6, 22), geometry(3, 15), geometry(12, 15), geometry(6, 30)]


def test_deterministic_tiles_follow_watchlist_order():
    tiles = deterministic_tiles(["a.x", "b.y"], {}, {"a.x": "1", "b.y": "2"}, [])
    assert [t.label for t in tiles] == ["X", "Y"]


def test_deterministic_tiles_put_pinned_variables_first():
    tiles = deterministic_tiles(["a.x", "b.y"], {}, {"a.x": "1", "b.y": "2"}, ["b.y"])
    assert [t.label for t in tiles] == ["Y", "X"]


def test_deterministic_tiles_use_configured_labels():
    tiles = deterministic_tiles(["a.aqi"], {"a.aqi": "AIR"}, {"a.aqi": "68"}, [])
    assert tiles[0].label == "AIR"


def test_deterministic_tiles_skip_variables_with_no_value():
    tiles = deterministic_tiles(["a.x", "b.y"], {}, {"a.x": "1"}, [])
    assert [t.label for t in tiles] == ["X"]


def test_deterministic_tiles_are_never_colored():
    # No model ran, so nothing has been judged worth accenting.
    assert deterministic_tiles(["a.x"], {}, {"a.x": "1"}, [])[0].color is None


def test_outage_lines_fill_the_whole_board():
    assert len(outage_lines(geometry(6, 22), "14:32", 0)) == 6


def test_outage_lines_report_when_data_was_last_good():
    assert "14:32" in "".join(outage_lines(geometry(6, 22), "14:32", 0))


def test_outage_lines_say_so_when_there_has_never_been_data():
    assert "NO DATA YET" in "".join(outage_lines(geometry(6, 22), None, 0))


def test_outage_lines_rotate_through_the_message_pool():
    geo = geometry(6, 22)
    assert outage_lines(geo, "14:32", 0) != outage_lines(geo, "14:32", 1)


def test_outage_lines_fit_every_supported_board():
    for geo in BOARDS:
        for index in range(len(OUTAGE_MESSAGES)):
            for last_good in ("14:32", None):
                lines = outage_lines(geo, last_good, index)
                assert len(lines) == geo.rows
                assert all(len(line) <= geo.cols for line in lines)


def test_outage_lines_still_carry_the_time_on_a_narrow_board():
    assert "14:32" in "".join(outage_lines(geometry(3, 15), "14:32", 0))


def test_setup_lines_tell_the_user_what_to_do():
    assert "PLUGIN" in " ".join(setup_lines(geometry(6, 22)))


def test_setup_lines_fit_every_supported_board():
    for geo in BOARDS:
        lines = setup_lines(geo)
        assert len(lines) == geo.rows
        assert all(len(line) <= geo.cols for line in lines)


def test_setup_lines_stay_actionable_even_on_a_narrow_board():
    assert "ENABLE" in " ".join(setup_lines(geometry(3, 15)))
