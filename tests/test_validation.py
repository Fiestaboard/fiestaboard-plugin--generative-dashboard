"""Validating what the model sends back."""

import pytest

from plugins.generative_dashboard.layout import geometry
from plugins.generative_dashboard.validation import (
    ValidationError,
    allowed_numbers,
    extract_numbers,
    validate_grid,
    validate_prose,
)

GEO = geometry(6, 22)
WATCHLIST = ["air.aqi", "wx.temp", "sys.cpu"]
VALUES = {"air.aqi": "168", "wx.temp": "61F", "sys.cpu": "42%"}


def _grid(**overrides):
    payload = {"tiles": [{"label": "AQI", "variable": "air.aqi"}]}
    payload.update(overrides)
    return payload


def _validate(payload, **kwargs):
    options = {
        "watchlist": WATCHLIST,
        "pinned": [],
        "values": VALUES,
        "labels": {},
        "geo": GEO,
        "use_color": True,
    }
    options.update(kwargs)
    return validate_grid(payload, **options)


# --- grid ---------------------------------------------------------------

def test_grid_builds_tiles_from_the_models_choices():
    assert [(t.label, t.value) for t in _validate(_grid()).tiles] == [("AQI", "168")]


def test_grid_takes_the_value_from_our_data_not_the_model():
    # The model is not permitted to type a number; it names a variable.
    payload = {"tiles": [{"label": "AQI", "variable": "air.aqi", "value": "999"}]}
    assert _validate(payload).tiles[0].value == "168"


def test_grid_drops_variables_that_are_not_watched():
    payload = {"tiles": [{"label": "X", "variable": "evil.hack"},
                         {"label": "AQI", "variable": "air.aqi"}]}
    assert [t.label for t in _validate(payload).tiles] == ["AQI"]


def test_grid_drops_watched_variables_with_no_current_value():
    with pytest.raises(ValidationError):
        _validate(_grid(), values={})


def test_grid_deduplicates_a_repeated_variable():
    payload = {"tiles": [{"label": "AQI", "variable": "air.aqi"},
                         {"label": "AQI AGAIN", "variable": "air.aqi"}]}
    assert len(_validate(payload).tiles) == 1


def test_grid_truncates_to_the_tile_budget():
    geo = geometry(3, 15)  # budget of 3
    values = {f"a.v{i}": str(i) for i in range(10)}
    payload = {"tiles": [{"label": f"L{i}", "variable": f"a.v{i}"} for i in range(10)]}
    result = validate_grid(payload, watchlist=list(values), pinned=[], values=values,
                           labels={}, geo=geo, use_color=True)
    assert len(result.tiles) == 3


def test_grid_reinstates_a_pinned_variable_the_model_forgot():
    result = _validate(_grid(), pinned=["sys.cpu"])
    assert "42%" in [t.value for t in result.tiles]


def test_grid_evicts_an_unpinned_tile_to_make_room_for_a_pin():
    geo = geometry(3, 15)  # budget of 3
    payload = {"tiles": [{"label": "A", "variable": "air.aqi"},
                         {"label": "B", "variable": "wx.temp"},
                         {"label": "C", "variable": "sys.cpu"}]}
    values = dict(VALUES, **{"z.pin": "9"})
    result = validate_grid(payload, watchlist=WATCHLIST + ["z.pin"], pinned=["z.pin"],
                           values=values, labels={}, geo=geo, use_color=True)
    assert len(result.tiles) == 3
    assert "9" in [t.value for t in result.tiles]


def test_grid_uses_the_configured_label_over_the_models():
    assert _validate(_grid(), labels={"air.aqi": "AIR"}).tiles[0].label == "AIR"


def test_grid_falls_back_to_a_derived_label_when_the_model_omits_one():
    assert _validate({"tiles": [{"variable": "air.aqi"}]}).tiles[0].label == "AQI"


def test_grid_keeps_a_valid_color():
    payload = {"tiles": [{"label": "AQI", "variable": "air.aqi", "color": "red"}]}
    assert _validate(payload).tiles[0].color == "red"


def test_grid_discards_an_invented_color():
    payload = {"tiles": [{"label": "AQI", "variable": "air.aqi", "color": "chartreuse"}]}
    assert _validate(payload).tiles[0].color is None


def test_grid_strips_color_when_color_is_disabled():
    payload = {"tiles": [{"label": "AQI", "variable": "air.aqi", "color": "red"}]}
    assert _validate(payload, use_color=False).tiles[0].color is None


def test_grid_sanitizes_the_banner():
    # The pipe has no flap on the board.
    assert _validate(_grid(banner="Air|Quality")).banner == "AIRQUALITY"


def test_grid_reports_the_source_ref_for_each_tile():
    # The plugin needs the ref to re-render live values between generations.
    result = _validate(_grid())
    assert result.refs == ["air.aqi"]
    assert len(result.refs) == len(result.tiles)


def test_grid_rejects_a_payload_that_is_not_an_object():
    with pytest.raises(ValidationError):
        _validate([])


def test_grid_rejects_a_payload_with_no_tiles_key():
    with pytest.raises(ValidationError):
        _validate({"banner": "HI"})


def test_grid_rejects_a_payload_with_no_usable_tiles():
    with pytest.raises(ValidationError):
        _validate({"tiles": []})


def test_grid_ignores_non_object_entries_in_the_tile_list():
    payload = {"tiles": ["nonsense", {"label": "AQI", "variable": "air.aqi"}]}
    assert len(_validate(payload).tiles) == 1


def test_grid_carries_through_the_models_reason():
    assert _validate(_grid(reason="AQI ROSE")).reason == "AQI ROSE"


# --- prose --------------------------------------------------------------

def test_extract_numbers_finds_plain_integers():
    assert extract_numbers("AQI 168 AND CPU 42") == ["168", "42"]


def test_extract_numbers_keeps_grouping_commas_together():
    assert extract_numbers("BTC 94,120") == ["94,120"]


def test_extract_numbers_keeps_decimals_together():
    assert extract_numbers("LOAD 1.25") == ["1.25"]


def test_allowed_numbers_includes_current_and_previous_values():
    assert {"168", "31"} <= allowed_numbers({"a.x": "168"}, {"a.x": "31"})


def test_prose_accepts_a_sentence_using_supplied_numbers():
    result = validate_prose({"text": "AQI ROSE FROM 31 TO 168."},
                            geo=GEO, current={"a.x": "168"}, previous={"a.x": "31"})
    assert "168" in result.text


def test_prose_rejects_a_fabricated_number():
    with pytest.raises(ValidationError) as excinfo:
        validate_prose({"text": "AQI ROSE 400 PERCENT."},
                       geo=GEO, current={"a.x": "168"}, previous={"a.x": "31"})
    assert "400" in str(excinfo.value)


def test_prose_rejects_text_too_long_for_the_board():
    with pytest.raises(ValidationError):
        validate_prose({"text": "WORD " * 200}, geo=GEO, current={}, previous={})


def test_prose_uppercases_and_strips_unrenderable_characters():
    result = validate_prose({"text": "all quiet | today"}, geo=GEO, current={}, previous={})
    assert result.text == "ALL QUIET  TODAY"


def test_prose_rejects_a_missing_text_field():
    with pytest.raises(ValidationError):
        validate_prose({"headline": "X"}, geo=GEO, current={}, previous={})


def test_prose_rejects_empty_text():
    with pytest.raises(ValidationError):
        validate_prose({"text": "   "}, geo=GEO, current={}, previous={})


def test_prose_rejects_text_that_is_empty_after_sanitising():
    with pytest.raises(ValidationError):
        validate_prose({"text": "|||"}, geo=GEO, current={}, previous={})


def test_grid_rejects_white_and_black_as_tile_accents():
    for colour in ("white", "black"):
        payload = {"tiles": [{"label": "AQI", "variable": "air.aqi", "color": colour}]}
        assert _validate(payload).tiles[0].color is None


def test_grid_accepts_a_banner_color():
    assert _validate(_grid(banner="AIR QUALITY", banner_color="red")).banner_color == "red"


def test_grid_discards_an_invalid_banner_color():
    assert _validate(_grid(banner="X", banner_color="taupe")).banner_color is None


def test_banner_color_is_ignored_when_color_is_disabled():
    result = _validate(_grid(banner="X", banner_color="red"), use_color=False)
    assert result.banner_color is None


def test_a_tile_suffix_becomes_part_of_the_value():
    # "62" is ambiguous; "62F" is a temperature. The model supplies the unit.
    payload = {"tiles": [{"label": "TEMP", "variable": "wx.temp", "suffix": "F"}]}
    result = _validate(payload, values={"wx.temp": "62"}, watchlist=["wx.temp"])
    assert result.tiles[0].value == "62F"
    assert result.suffixes == {"wx.temp": "F"}


def test_a_suffix_may_never_contain_a_digit():
    # "62" + "9" would read as 629 — a number nobody measured.
    payload = {"tiles": [{"label": "TEMP", "variable": "wx.temp", "suffix": "9F"}]}
    result = _validate(payload, values={"wx.temp": "62"}, watchlist=["wx.temp"])
    assert result.tiles[0].value == "62F"


def test_a_suffix_is_sanitized_and_kept_short():
    payload = {"tiles": [{"label": "W", "variable": "wx.w", "suffix": " miles/hour"}]}
    result = _validate(payload, values={"wx.w": "7"}, watchlist=["wx.w"])
    assert len(result.tiles[0].value) <= 1 + 5


def test_no_suffix_leaves_the_value_alone():
    result = _validate(_grid())
    assert result.tiles[0].value == "168"
    assert result.suffixes == {}


def test_a_suffix_is_not_applied_when_the_value_already_has_a_unit():
    # weather.visibility is "6.2MI"; adding MI again reads 6.2MIMI.
    from plugins.generative_dashboard.validation import apply_suffix

    assert apply_suffix("6.2MI", "MI") == "6.2MI"
    assert apply_suffix("7:36 PM", "PM") == "7:36 PM"
    assert apply_suffix("61F", "F") == "61F"


def test_a_suffix_applies_only_to_values_ending_in_a_digit():
    from plugins.generative_dashboard.validation import apply_suffix

    assert apply_suffix("62", "F") == "62F"
    assert apply_suffix("100", "%") == "100%"
    assert apply_suffix("CLEAR", "F") == "CLEAR"


def test_banner_numbers_must_come_from_the_supplied_values():
    # The tiles cannot lie, but a banner saying AQI 999 EMERGENCY could.
    with pytest.raises(ValidationError):
        _validate(_grid(banner="AQI 999 EMERGENCY"))


def test_banner_numbers_that_match_a_value_pass():
    assert _validate(_grid(banner="AQI 168 WARNING")).banner == "AQI 168 WARNING"


def test_subtitle_is_validated_the_same_way():
    result = _validate(_grid(banner="AIR", subtitle="AQI NOW 168"))
    assert result.subtitle == "AQI NOW 168"
    with pytest.raises(ValidationError):
        _validate(_grid(banner="AIR", subtitle="AQI 999"))


def test_previous_values_are_fair_game_in_a_banner():
    result = _validate(_grid(banner="AQI ROSE FROM 31"), previous={"air.aqi": "31"})
    assert "31" in result.banner


def test_a_label_that_echoes_its_text_value_is_blanked():
    # RAIN RAIN tells you nothing twice.
    payload = {"tiles": [{"label": "RAIN", "variable": "wx.sky"}]}
    result = _validate(payload, watchlist=["wx.sky"], values={"wx.sky": "RAIN"})
    assert result.tiles[0].label == ""
    assert result.tiles[0].value == "RAIN"


def test_a_currency_prefix_attaches_before_the_value():
    from plugins.generative_dashboard.validation import apply_prefix

    assert apply_prefix("339.08", "$") == "$339.08"
    assert apply_prefix("CLEAR", "$") == "CLEAR"  # only bare numbers take one


def test_a_prefix_may_never_contain_a_digit():
    payload = {"tiles": [{"label": "PRICE", "variable": "st.p", "prefix": "1$"}]}
    result = _validate(payload, watchlist=["st.p"], values={"st.p": "339.08"})
    assert result.tiles[0].value == "$339.08"


def test_prefix_and_suffix_can_combine():
    payload = {"tiles": [{"label": "P", "variable": "st.p", "prefix": "$", "suffix": "M"}]}
    result = _validate(payload, watchlist=["st.p"], values={"st.p": "42"})
    assert result.tiles[0].value == "$42M"


def test_headers_truncate_at_word_boundaries():
    # "EQUITY AND FOREX UPDAT" reads as generated; a designed page would
    # drop the word it cannot fit.
    result = _validate(_grid(banner="EQUITY AND FOREX UPDATE FOR TODAY"))
    assert not result.banner.endswith("UPDAT")
    assert len(result.banner) <= 22
    assert result.banner == "EQUITY AND FOREX"


def test_a_header_with_one_giant_word_still_fits():
    result = _validate(_grid(banner="A" * 30))
    assert len(result.banner) == 22


def test_a_label_that_is_a_prefix_of_its_value_is_blanked():
    # "HEAVY HEAVY RAIN" — a truncated label echoing a reading is blanked.
    # (Identifier refs like st.name are absorbed or dropped instead.)
    payload = {"tiles": [{"label": "HEAVY", "variable": "wx.summary"}]}
    result = _validate(payload, watchlist=["wx.summary"], values={"wx.summary": "HEAVY RAIN"})
    assert result.tiles[0].label == ""


def test_a_numeric_value_keeps_a_coincidental_label(  ):
    payload = {"tiles": [{"label": "NOW", "variable": "wx.t", "suffix": "F"}]}
    result = _validate(payload, watchlist=["wx.t"], values={"wx.t": "62"})
    assert result.tiles[0].label == "NOW"


def test_an_identifier_tile_is_absorbed_as_its_companions_label():
    # Gemma keeps giving GOOG a tile despite instructions; the fix is
    # deterministic. A digitless tile whose ref names an identifier becomes
    # the label of its plugin's first numeric tile.
    payload = {"tiles": [
        {"label": "GOOG", "variable": "stocks.symbol"},
        {"label": "CHA", "variable": "stocks.change_pct"},
    ]}
    result = _validate(
        payload,
        watchlist=["stocks.symbol", "stocks.change_pct"],
        values={"stocks.symbol": "GOOG", "stocks.change_pct": "-4.10"},
    )
    assert [t.label for t in result.tiles] == ["GOOG"]
    assert result.tiles[0].value == "-4.10"
    assert "stocks.symbol" not in result.refs


def test_an_identifier_with_no_companion_is_dropped_not_stranded():
    payload = {"tiles": [{"label": "GOOG", "variable": "stocks.symbol"},
                         {"label": "AQI", "variable": "air.aqi"}]}
    result = _validate(
        payload, watchlist=["stocks.symbol", "air.aqi"],
        values={"stocks.symbol": "GOOG", "air.aqi": "168"},
    )
    assert result.refs == ["air.aqi"]


def test_condition_text_is_not_mistaken_for_an_identifier():
    # weather.condition "RAIN" is a reading, not a name — absorbing it as
    # the temperature's label would caption 62F with the word RAIN.
    payload = {"tiles": [
        {"label": "SKY", "variable": "wx.condition"},
        {"label": "NOW", "variable": "wx.temperature"},
    ]}
    result = _validate(
        payload, watchlist=["wx.condition", "wx.temperature"],
        values={"wx.condition": "RAIN", "wx.temperature": "62"},
    )
    assert len(result.tiles) == 2


def test_units_are_inferred_from_the_description_when_the_model_forgets():
    payload = {"tiles": [
        {"label": "DELAY", "variable": "tr.delay"},
        {"label": "PRICE", "variable": "st.price"},
    ]}
    result = _validate(
        payload, watchlist=["tr.delay", "st.price"],
        values={"tr.delay": "18", "st.price": "339.08"},
        descriptions={"tr.delay": "Line delay in minutes",
                      "st.price": "Current share price in USD"},
    )
    assert result.tiles[0].value == "18MIN"
    assert result.tiles[1].value == "$339.08"


def test_the_models_own_affix_wins_over_inference():
    payload = {"tiles": [{"label": "D", "variable": "tr.delay", "suffix": "M"}]}
    result = _validate(
        payload, watchlist=["tr.delay"], values={"tr.delay": "18"},
        descriptions={"tr.delay": "Line delay in minutes"},
    )
    assert result.tiles[0].value == "18M"


def test_at_most_three_tiles_keep_their_color():
    # "At most three" is a promise to the reader, so the validator enforces
    # it: the first three colored tiles keep their dots, the rest go plain.
    payload = {"tiles": [
        {"label": f"L{i}", "variable": f"p.v{i}", "color": "red"} for i in range(5)
    ]}
    values = {f"p.v{i}": str(i) for i in range(5)}
    result = _validate(payload, watchlist=list(values), values=values)
    assert sum(1 for t in result.tiles if t.color) == 3
    assert [t.color for t in result.tiles[:3]] == ["red", "red", "red"]


def test_the_model_may_choose_the_list_layout():
    result = _validate(_grid(layout="list"))
    assert result.layout == "list"


def test_an_unknown_layout_falls_back_to_auto():
    assert _validate(_grid(layout="mosaic")).layout == "auto"
