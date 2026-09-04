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


def test_color_renders_as_a_status_dot_after_the_value():
    # Color is data — an indicator light beside the reading — never part of
    # the label. A prefix shifts colored labels out of column with the rest.
    lines = render_grid([Tile("AQI", "168", "red")], geometry(3, 15))
    assert _content(lines)[0].endswith("168{red}")
    assert not _content(lines)[0].startswith("{red}")


def test_render_grid_omits_color_when_color_is_disabled():
    lines = render_grid([Tile("AQI", "168", "red")], geometry(3, 15), use_color=False)
    assert "{red}" not in lines[0]


def test_colored_tile_still_occupies_exactly_the_board_width():
    lines = render_grid([Tile("AQI", "168", "red")], geometry(3, 15), use_color=True)
    assert cell_width(_content(lines)[0]) == 15


def test_the_dot_column_is_reserved_uniformly_once_any_tile_is_colored():
    # If one reading gets a dot, every value shifts one cell left so the
    # numbers still sit in a straight column — presence of color must never
    # change where the data lives.
    geo = geometry(3, 15)
    lines = _content(render_grid([Tile("AQI", "168", "red"), Tile("NOW", "62F")], geo))
    dotted = lines[0]
    plain = lines[1]
    assert dotted.endswith("{red}")
    # Same rightmost data column on both rows: strip the dot, compare ends.
    assert cell_width(dotted) - 1 == len(plain.rstrip())


def test_no_colors_means_no_reserved_column():
    geo = geometry(3, 15)
    lines = _content(render_grid([Tile("CPU", "42%")], geo))
    assert lines[0].endswith("42%")
    assert len(lines[0]) == 15


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


def test_a_wide_tile_moves_to_its_section_instead_of_splitting_a_pair():
    # Sections beat strict ordering: the two short stats share a row and the
    # wide one takes a ledger row, instead of three ragged lines.
    geo = geometry(6, 22)
    tiles = [Tile("CPU", "42%"), Tile("DATE", "09/03/26"), Tile("AQI", "168")]
    lines = _content(render_grid(tiles, geo))
    assert len(lines) == 2
    assert "CPU" in lines[0] and "AQI" in lines[0]
    assert "DATE" in lines[1] and len(lines[1]) == 22


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


def test_a_banner_gets_double_frame_tiles_when_there_is_room():
    # The handmade reference frames its title {69}{69} ... {69}{69} — the
    # double frame is part of why it reads as designed rather than generated.
    geo = geometry(6, 22)
    line = _content(render_grid([Tile("CPU", "42%")], geo, banner="WEATHER",
                                banner_color="blue"))[0].strip()
    assert line.startswith("{blue}{blue}")
    assert line.endswith("{blue}{blue}")


def test_a_long_banner_falls_back_to_single_frame_then_plain():
    geo = geometry(3, 15)
    line = _content(render_grid([Tile("A", "1")], geo, banner="AIR QUALITY",
                                banner_color="red"))[0].strip()
    assert line.startswith("{red}") and not line.startswith("{red}{red}")


def test_a_subtitle_renders_under_the_banner_with_single_frame():
    geo = geometry(6, 22)
    lines = _content(render_grid([Tile("NOW", "62F")], geo, banner="SAN FRANCISCO",
                                 banner_color="blue", subtitle="THU 8:15 AM"))
    assert "SAN FRANCISCO" in lines[0]
    assert "THU 8:15 AM" in lines[1]
    assert lines[1].strip().startswith("{blue}") and not lines[1].strip().startswith("{blue}{blue}")
    assert "NOW" in lines[2]


def test_a_subtitle_without_a_banner_is_ignored():
    geo = geometry(6, 22)
    lines = _content(render_grid([Tile("NOW", "62F")], geo, subtitle="ORPHAN"))
    assert all("ORPHAN" not in l for l in lines)


def test_header_rows_reduce_the_tile_capacity():
    geo = geometry(6, 22)
    tiles = [Tile(f"L{i}", str(i)) for i in range(12)]
    lines = render_grid(tiles, geo, banner="HEAD", subtitle="SUB")
    assert "L8" not in "".join(lines)  # 4 rows x 2 columns = 8 tiles max


def test_a_leftover_tile_spans_the_full_row_instead_of_half():
    # A row that is 80% blank in mid-board is what reads as generated.
    geo = geometry(6, 22)
    lines = _content(render_grid(
        [Tile("CPU", "42%"), Tile("AQI", "168"), Tile("NET", "412")], geo))
    assert len(lines) == 2
    solo = lines[1]
    assert solo.startswith("NET")
    assert solo.endswith("412")
    assert len(solo) == 22  # label left, value right, spanning the board


def test_money_rows_render_full_width_and_consistent():
    # $339.08 cannot share an 11-cell column with a label, so both stats go
    # full-width — label left, value right, the handmade stocks-page shape.
    # What must never happen is one full row and one half-empty row.
    geo = geometry(6, 22)
    lines = _content(render_grid([Tile("GOOG", "$339.08"), Tile("CHG", "1.59%")], geo))
    assert len(lines) == 2
    assert lines[0] == "GOOG" + " " * 11 + "$339.08"
    assert lines[1].startswith("CHG") and lines[1].endswith("1.59%")
    assert len(lines[0]) == len(lines[1]) == 22


def test_single_column_boards_keep_their_normal_row_shape():
    geo = geometry(3, 15)
    lines = _content(render_grid([Tile("CPU", "42%")], geo))
    assert lines[0] == "CPU" + " " * 9 + "42%"


def test_row_shapes_never_interleave():
    # A human never alternates pair rows and ledger rows; shape is grouped
    # into sections. Wide values gather in one block, pairs in another.
    geo = geometry(6, 22)
    tiles = [
        Tile("GOOG", "$339.08"),   # wide
        Tile("VIS", "6.2MI"),      # narrow
        Tile("CHANGE", "1.59%"),   # wide (needs full row? 6ch value fits pair)
        Tile("TEMP", "60F"),       # narrow
    ]
    lines = _content(render_grid(tiles, geo))
    shapes = ["pair" if len(l.split()) >= 4 or ("  " in l.strip() and len(l) <= 22 and l.count(" ") < 8) else "full" for l in lines]
    # classify by structure instead: a full row spans 22 with one label+value
    def shape(line):
        return "full" if len(line) == 22 and line[:11].strip() and not line[10:12].strip() else "pair"
    shapes = [shape(l) for l in lines]
    # once we leave a shape we never return to it
    transitions = sum(1 for a, b in zip(shapes, shapes[1:]) if a != b)
    assert transitions <= 1, (shapes, lines)


def test_the_lead_story_decides_which_section_comes_first():
    geo = geometry(6, 22)
    wide_first = _content(render_grid(
        [Tile("GOOG", "$339.08"), Tile("VIS", "6.2MI"), Tile("TEMP", "60F")], geo))
    assert "GOOG" in wide_first[0]
    narrow_first = _content(render_grid(
        [Tile("VIS", "6.2MI"), Tile("TEMP", "60F"), Tile("GOOG", "$339.08")], geo))
    assert "VIS" in narrow_first[0] and "GOOG" in narrow_first[-1]


def test_list_layout_makes_every_row_a_ledger_row():
    # The handmade stocks page is all full-width rows; the model can ask for
    # that shape outright instead of hoping values force it.
    geo = geometry(6, 22)
    lines = _content(render_grid(
        [Tile("GOOG", "$339.08"), Tile("VIS", "6.2MI"), Tile("TEMP", "60F")],
        geo, layout="list"))
    assert len(lines) == 3
    assert all(len(l) == 22 for l in lines)
    assert lines[1] == "VIS" + " " * 14 + "6.2MI"
