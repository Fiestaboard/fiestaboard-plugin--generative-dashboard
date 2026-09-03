"""Reading other plugins' variables through the core registry."""

from types import SimpleNamespace

import pytest

from plugins.generative_dashboard import catalog
from plugins.generative_dashboard.catalog import (
    default_label,
    read_values,
    variable_catalog,
)


class FakeResult:
    def __init__(self, data, available=True):
        self.available = available
        self.data = data
        self.error = None


class FakeRegistry:
    def __init__(self, metadata=None, data=None, names=None):
        self._metadata = metadata or {}
        self._data = data or {}
        self._names = names or {}
        self.fetched = []

    def get_all_variables_with_metadata(self):
        return self._metadata

    def get_manifest(self, plugin_id):
        if plugin_id not in self._names:
            return None
        return SimpleNamespace(name=self._names[plugin_id])

    def fetch_plugin_data(self, plugin_id, board=None):
        self.fetched.append(plugin_id)
        if plugin_id not in self._data:
            return FakeResult(None, available=False)
        return FakeResult(self._data[plugin_id])


@pytest.fixture
def fake(monkeypatch):
    def install(registry):
        monkeypatch.setattr(catalog, "_registry", lambda: registry)
        return registry

    return install


def test_default_label_uses_the_variable_name_without_its_plugin():
    assert default_label("weather.temp_f") == "TEMP F"


def test_catalog_lists_one_choice_per_variable(fake):
    fake(FakeRegistry(
        metadata={"weather": {"temp_f": {"description": "Temp", "max_length": 6,
                                         "group": "", "preview": "61F"}}},
        names={"weather": "Weather"},
    ))
    assert [c.ref for c in variable_catalog("generative_dashboard", 15)] == ["weather.temp_f"]


def test_catalog_exposes_the_live_value_as_the_preview(fake):
    fake(FakeRegistry(
        metadata={"weather": {"temp_f": {"description": "", "max_length": 6,
                                         "group": "", "preview": "61F"}}},
        names={"weather": "Weather"},
    ))
    assert variable_catalog("generative_dashboard", 15)[0].preview == "61F"


def test_catalog_groups_choices_under_the_source_plugin_name(fake):
    fake(FakeRegistry(
        metadata={"weather": {"temp_f": {"description": "", "max_length": 6,
                                         "group": "", "preview": ""}}},
        names={"weather": "Weather"},
    ))
    assert variable_catalog("generative_dashboard", 15)[0].group == "Weather"


def test_catalog_falls_back_to_the_plugin_id_when_it_has_no_manifest(fake):
    fake(FakeRegistry(
        metadata={"weather": {"temp_f": {"description": "", "max_length": 6,
                                         "group": "", "preview": ""}}},
    ))
    assert variable_catalog("generative_dashboard", 15)[0].group == "weather"


def test_catalog_disables_our_own_variables_to_prevent_recursion(fake):
    fake(FakeRegistry(
        metadata={"generative_dashboard": {"headline": {"description": "", "max_length": 20,
                                                        "group": "", "preview": ""}}},
        names={"generative_dashboard": "Generative Dashboard"},
    ))
    choice = variable_catalog("generative_dashboard", 15)[0]
    assert choice.disabled
    assert "itself" in choice.disabled_reason.lower()


def test_catalog_disables_variables_too_wide_for_a_tile(fake):
    fake(FakeRegistry(
        metadata={"art": {"art": {"description": "", "max_length": 512,
                                  "group": "", "preview": ""}}},
        names={"art": "Art"},
    ))
    choice = variable_catalog("generative_dashboard", 15)[0]
    assert choice.disabled
    assert "512" in choice.disabled_reason


def test_catalog_keeps_disabled_choices_visible_rather_than_hiding_them(fake):
    fake(FakeRegistry(
        metadata={"art": {"art": {"description": "", "max_length": 512,
                                  "group": "", "preview": ""}}},
        names={"art": "Art"},
    ))
    assert len(variable_catalog("generative_dashboard", 15)) == 1


def test_catalog_allows_a_variable_with_no_declared_max_length(fake):
    fake(FakeRegistry(
        metadata={"weather": {"temp_f": {"description": "", "max_length": None,
                                         "group": "", "preview": ""}}},
        names={"weather": "Weather"},
    ))
    assert not variable_catalog("generative_dashboard", 15)[0].disabled


def test_catalog_query_filters_case_insensitively(fake):
    fake(FakeRegistry(
        metadata={"weather": {
            "temp_f": {"description": "", "max_length": 6, "group": "", "preview": ""},
            "humidity": {"description": "", "max_length": 6, "group": "", "preview": ""},
        }},
        names={"weather": "Weather"},
    ))
    refs = [c.ref for c in variable_catalog("generative_dashboard", 15, query="TEMP")]
    assert refs == ["weather.temp_f"]


def test_read_values_returns_current_values_keyed_by_ref(fake):
    fake(FakeRegistry(data={"weather": {"temp_f": 61}}))
    assert read_values(["weather.temp_f"], None, "generative_dashboard") == {"weather.temp_f": "61"}


def test_read_values_fetches_each_plugin_only_once(fake):
    registry = fake(FakeRegistry(data={"weather": {"temp_f": 61, "humidity": 40}}))
    read_values(["weather.temp_f", "weather.humidity"], None, "generative_dashboard")
    assert registry.fetched == ["weather"]


def test_read_values_never_fetches_our_own_plugin(fake):
    registry = fake(FakeRegistry(data={"generative_dashboard": {"headline": "X"}}))
    values = read_values(["generative_dashboard.headline"], None, "generative_dashboard")
    assert registry.fetched == []
    assert values == {}


def test_read_values_omits_unavailable_plugins(fake):
    fake(FakeRegistry(data={}))
    assert read_values(["weather.temp_f"], None, "generative_dashboard") == {}


def test_read_values_omits_variables_the_plugin_did_not_supply(fake):
    fake(FakeRegistry(data={"weather": {"humidity": 40}}))
    assert read_values(["weather.temp_f"], None, "generative_dashboard") == {}


def test_read_values_ignores_malformed_refs(fake):
    fake(FakeRegistry(data={"weather": {"temp_f": 61}}))
    assert read_values(["nodot"], None, "generative_dashboard") == {}


def test_read_values_skips_list_and_dict_values(fake):
    fake(FakeRegistry(data={"weather": {"hourly": [1, 2, 3]}}))
    assert read_values(["weather.hourly"], None, "generative_dashboard") == {}


def test_read_values_survives_a_plugin_that_raises(fake):
    class Exploding(FakeRegistry):
        def fetch_plugin_data(self, plugin_id, board=None):
            raise RuntimeError("upstream is on fire")

    fake(Exploding())
    assert read_values(["weather.temp_f"], None, "generative_dashboard") == {}
