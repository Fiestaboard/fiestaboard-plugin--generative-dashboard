"""Grid geometry, tile placement, and prose wrapping."""

from plugins.generative_dashboard.charset import cell_width
from plugins.generative_dashboard.layout import (
    Tile,
    fits,
    geometry,
    render_grid,
    wrap_center,
)


def _content(lines):
    """The non-blank lines, since the block is centred rather than top-aligned."""
    return [line for line in lines if line.strip()]


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
    lines = _content(render_grid([Tile("CPU", "42%"), Tile("TEMP", "61F")], geometry(6, 22)))
    assert lines[0].startswith("CPU")
    assert "TEMP" in lines[0]


def test_render_grid_right_aligns_the_value():
    lines = render_grid([Tile("CPU", "42%")], geometry(3, 15))
    assert _content(lines)[0] == "CPU" + " " * 9 + "42%"


def test_render_grid_leaves_a_gutter_between_columns():
    lines = render_grid([Tile("A", "1"), Tile("B", "2")], geometry(6, 22))
    # First column gives up one cell so the columns cannot touch.
    assert _content(lines)[0] == "A" + " " * 8 + "1" + " " + "B" + " " * 9 + "2"


def test_render_grid_prefixes_a_color_marker_when_a_tile_is_accented():
    lines = render_grid([Tile("AQI", "168", "red")], geometry(3, 15))
    assert _content(lines)[0].startswith("{red}")


def test_render_grid_omits_color_when_color_is_disabled():
    lines = render_grid([Tile("AQI", "168", "red")], geometry(3, 15), use_color=False)
    assert "{red}" not in lines[0]


def test_colored_tile_still_occupies_exactly_the_board_width():
    lines = render_grid([Tile("AQI", "168", "red")], geometry(3, 15), use_color=True)
    assert cell_width(_content(lines)[0]) == 15


def test_render_grid_puts_the_banner_above_the_tiles():
    lines = render_grid([Tile("CPU", "42%")], geometry(6, 22), banner="AIR QUALITY BAD")
    content = _content(lines)
    assert content[0].strip() == "AIR QUALITY BAD"
    assert "CPU" in content[1]


def test_banner_costs_one_row_of_tiles():
    geo = geometry(6, 22)
    tiles = [Tile(f"L{i}", str(i)) for i in range(12)]
    lines = render_grid(tiles, geo, banner="HEADS UP")
    # Twelve tiles no longer fit in the ten remaining slots.
    assert "L11" not in "".join(lines)


def test_render_grid_pads_unused_rows_with_blanks():
    lines = render_grid([Tile("CPU", "42%")], geometry(6, 22))
    assert len(lines) == 6
    assert len(_content(lines)) == 1


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


def test_render_grid_centers_the_content_block_vertically():
    # Content clustered at the top with dead rows beneath reads like a bug.
    lines = render_grid([Tile("CPU", "42%"), Tile("TEMP", "61F")], geometry(6, 22))
    assert lines[0].strip() == ""
    assert "CPU" in lines[2]
    assert lines[5].strip() == ""


def test_a_full_grid_is_not_padded():
    geo = geometry(6, 22)
    tiles = [Tile(f"L{i}", str(i)) for i in range(12)]
    lines = render_grid(tiles, geo)
    assert all(line.strip() for line in lines)


def test_banner_stays_with_the_block_it_heads():
    lines = render_grid([Tile("CPU", "42%")], geometry(6, 22), banner="HEADS UP")
    banner_row = next(i for i, l in enumerate(lines) if "HEADS UP" in l)
    tile_row = next(i for i, l in enumerate(lines) if "CPU" in l)
    assert tile_row == banner_row + 1


def test_a_single_tile_row_sits_in_the_middle():
    lines = render_grid([Tile("CPU", "42%")], geometry(3, 15))
    assert lines[1].startswith("CPU")


def test_a_tile_whose_value_crowds_out_its_label_takes_a_full_row():
    # "09/03/26" in a 10-cell column leaves one cell for the label, which
    # renders as "D 09/03/26". Giving it the whole row keeps it labelled.
    geo = geometry(6, 22)
    lines = _content(render_grid([Tile("DATE", "09/03/26")], geo))
    assert lines[0].startswith("DATE")
    assert lines[0].endswith("09/03/26")
    assert cell_width(lines[0]) == 22


def test_short_values_still_share_a_row():
    geo = geometry(6, 22)
    lines = _content(render_grid([Tile("CPU", "42%"), Tile("AQI", "168")], geo))
    assert len(lines) == 1


def test_a_wide_tile_does_not_swallow_its_neighbours():
    geo = geometry(6, 22)
    tiles = [Tile("CPU", "42%"), Tile("DATE", "09/03/26"), Tile("AQI", "168")]
    lines = _content(render_grid(tiles, geo))
    assert len(lines) == 3
    assert "CPU" in lines[0] and "DATE" in lines[1] and "AQI" in lines[2]


def test_a_single_column_board_never_needs_widening():
    # A Note's column is already the full width.
    geo = geometry(3, 15)
    lines = _content(render_grid([Tile("DATE", "09/03/26")], geo))
    assert lines[0] == "DATE" + " " * 3 + "09/03/26"


def test_a_value_too_long_for_the_board_is_dropped_never_truncated():
    # Truncating "123,456,789.0123" to "123,456,789.012" puts a number on the
    # board that is not the real number. Showing nothing is the safe failure.
    geo = geometry(3, 15)
    lines = _content(render_grid([Tile("X", "123,456,789.0123")], geo))
    assert lines == []


def test_a_value_that_exactly_fills_the_row_is_kept():
    geo = geometry(3, 15)
    lines = _content(render_grid([Tile("X", "12,345,678.9012")], geo))
    assert lines[0] == "12,345,678.9012"


def test_an_over_long_value_does_not_take_its_neighbours_down():
    geo = geometry(3, 15)
    tiles = [Tile("X", "123,456,789.0123"), Tile("CPU", "42%")]
    lines = _content(render_grid(tiles, geo))
    assert len(lines) == 1
    assert "42%" in lines[0]


def test_a_tile_with_no_value_is_dropped():
    # "AIR COLOR" with nothing beside it is noise, not a stat.
    geo = geometry(6, 22)
    lines = _content(render_grid([Tile("AIR COLOR", ""), Tile("CPU", "42%")], geo))
    assert len(lines) == 1
    assert "AIR COLOR" not in lines[0]


def test_a_whitespace_only_value_is_also_dropped():
    geo = geometry(6, 22)
    assert _content(render_grid([Tile("X", "   ")], geo)) == []


def test_a_banner_can_be_flanked_by_color_tiles():
    # Color reads best around a title, framing it rather than sitting inside it.
    geo = geometry(6, 22)
    line = _content(render_grid([Tile("CPU", "42%")], geo, banner="AIR QUALITY",
                                banner_color="red"))[0]
    assert line.strip().startswith("{red}")
    assert line.strip().endswith("{red}")
    assert "AIR QUALITY" in line
    assert cell_width(line) <= 22


def test_a_banner_without_a_color_is_just_centered():
    geo = geometry(6, 22)
    line = _content(render_grid([Tile("CPU", "42%")], geo, banner="AIR QUALITY"))[0]
    assert "{" not in line
    assert line.strip() == "AIR QUALITY"


def test_banner_color_is_dropped_when_color_is_off():
    geo = geometry(6, 22)
    line = _content(render_grid([Tile("CPU", "42%")], geo, banner="AIR QUALITY",
                                banner_color="red", use_color=False))[0]
    assert "{red}" not in line


def test_a_banner_too_wide_to_frame_keeps_its_text():
    # Losing a word to make room for decoration would be the wrong trade.
    geo = geometry(3, 15)
    line = _content(render_grid([Tile("CPU", "42%")], geo, banner="AIR QUALITY BAD",
                                banner_color="red"))[0]
    assert "AIR QUALITY BAD" in line
    assert cell_width(line) <= 15
