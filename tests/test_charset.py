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
    # Three cells: the marker plus two letters. The marker is never cut in half.
    assert truncate("{red}AQI", 3) == "{red}AQ"


def test_truncate_keeps_marker_when_it_fits():
    assert truncate("{red}AQIXX", 4) == "{red}AQI"


def test_truncate_leaves_short_text_alone():
    assert truncate("AQI", 10) == "AQI"
