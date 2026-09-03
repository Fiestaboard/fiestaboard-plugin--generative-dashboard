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
