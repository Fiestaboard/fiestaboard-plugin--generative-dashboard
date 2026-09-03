"""Significance gating: the thing that keeps the board still."""

from plugins.generative_dashboard.gate import (
    as_number,
    is_material,
    material_changes,
)


def test_as_number_parses_a_plain_integer():
    assert as_number("42") == 42.0


def test_as_number_strips_grouping_commas_and_percent_signs():
    assert as_number("94,120") == 94120.0
    assert as_number("42%") == 42.0


def test_as_number_returns_none_for_text():
    assert as_number("PARTLY CLOUDY") is None


def test_as_number_treats_booleans_as_text():
    assert as_number(True) is None


def test_change_below_the_threshold_is_not_material():
    assert not is_material("100", "104", 5.0)


def test_change_above_the_threshold_is_material():
    assert is_material("100", "106", 5.0)


def test_change_exactly_at_the_threshold_is_material():
    assert is_material("100", "105", 5.0)


def test_any_change_to_a_text_value_is_material():
    assert is_material("SUNNY", "RAINY", 50.0)


def test_identical_text_is_not_material():
    assert not is_material("SUNNY", "SUNNY", 5.0)


def test_growth_from_zero_is_material_and_does_not_divide_by_zero():
    assert is_material("0", "5", 5.0)


def test_zero_to_zero_is_not_material():
    assert not is_material("0", "0", 5.0)


def test_a_variable_appearing_is_material():
    assert material_changes({}, {"a.b": "1"}, {}, 5.0) == ["a.b"]


def test_a_variable_disappearing_is_material():
    assert material_changes({"a.b": "1"}, {}, {}, 5.0) == ["a.b"]


def test_no_changes_yields_an_empty_list():
    values = {"a.b": "1", "c.d": "SUNNY"}
    assert material_changes(values, dict(values), {}, 5.0) == []


def test_per_variable_threshold_overrides_the_global_default():
    # 2% moves; the global default would ignore it, but this one is sensitive.
    assert material_changes({"a.b": "100"}, {"a.b": "102"}, {"a.b": 1.0}, 5.0) == ["a.b"]


def test_global_default_applies_when_no_override_exists():
    assert material_changes({"a.b": "100"}, {"a.b": "102"}, {}, 5.0) == []


def test_only_the_changed_variables_are_reported():
    old = {"a.b": "100", "c.d": "1"}
    new = {"a.b": "100", "c.d": "99"}
    assert material_changes(old, new, {}, 5.0) == ["c.d"]
