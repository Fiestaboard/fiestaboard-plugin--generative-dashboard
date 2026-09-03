"""Grid geometry, tile placement, and prose wrapping."""

from plugins.generative_dashboard.charset import cell_width
from plugins.generative_dashboard.layout import (
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
    assert len(render_grid([Tile("CPU", "42%")], geometry(6, 22))) == 6


def test_render_grid_never_exceeds_the_board_width():
    geo = geometry(6, 22)
    tiles = [Tile("LABEL", "VALUE") for _ in range(12)]
    for line in render_grid(tiles, geo):
        assert cell_width(line) <= 22


def test_render_grid_places_tiles_left_to_right_then_down():
    lines = render_grid([Tile("CPU", "42%"), Tile("TEMP", "61F")], geometry(6, 22))
    assert lines[0].startswith("CPU")
    assert "TEMP" in lines[0]


def test_render_grid_right_aligns_the_value():
    lines = render_grid([Tile("CPU", "42%")], geometry(3, 15))
    assert lines[0] == "CPU" + " " * 9 + "42%"


def test_render_grid_leaves_a_gutter_between_columns():
    lines = render_grid([Tile("A", "1"), Tile("B", "2")], geometry(6, 22))
    # First column gives up one cell so the columns cannot touch.
    assert lines[0] == "A" + " " * 8 + "1" + " " + "B" + " " * 9 + "2"


def test_render_grid_prefixes_a_color_marker_when_a_tile_is_accented():
    lines = render_grid([Tile("AQI", "168", "red")], geometry(3, 15))
    assert lines[0].startswith("{red}")


def test_render_grid_omits_color_when_color_is_disabled():
    lines = render_grid([Tile("AQI", "168", "red")], geometry(3, 15), use_color=False)
    assert "{red}" not in lines[0]


def test_colored_tile_still_occupies_exactly_the_board_width():
    lines = render_grid([Tile("AQI", "168", "red")], geometry(3, 15), use_color=True)
    assert cell_width(lines[0]) == 15


def test_render_grid_puts_the_banner_on_the_first_row():
    lines = render_grid([Tile("CPU", "42%")], geometry(6, 22), banner="AIR QUALITY BAD")
    assert lines[0].strip() == "AIR QUALITY BAD"
    assert "CPU" in lines[1]


def test_banner_costs_one_row_of_tiles():
    geo = geometry(6, 22)
    tiles = [Tile(f"L{i}", str(i)) for i in range(12)]
    lines = render_grid(tiles, geo, banner="HEADS UP")
    # Twelve tiles no longer fit in the ten remaining slots.
    assert "L11" not in "".join(lines)


def test_render_grid_pads_unused_rows_with_blanks():
    lines = render_grid([Tile("CPU", "42%")], geometry(6, 22))
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
    assert len(wrap_center("WORD " * 40, 3, 15)) == 3
