"""The settings picker: choosing which variables the dashboard may use."""

import pytest
from src.plugins.base import OptionsRequest, OptionsUnavailable

from plugins.generative_dashboard import GenerativeDashboardPlugin, catalog
from plugins.generative_dashboard.catalog import VariableChoice


@pytest.fixture
def plugin(manifest, config, monkeypatch):
    instance = GenerativeDashboardPlugin(manifest)
    instance.config = config
    monkeypatch.setattr(
        catalog, "variable_catalog",
        lambda exclude, width, query="": [
            VariableChoice("wx.temp", "TEMP", "Outside temperature", "Weather", "61F"),
            VariableChoice("gd.headline", "HEADLINE", "", "Generative Dashboard", "",
                           True, "A dashboard cannot watch itself."),
        ],
    )
    return instance


def _options(plugin, options_id, **kwargs):
    return plugin.get_options(OptionsRequest(options_id=options_id, **kwargs))


def test_variables_picker_returns_one_option_per_choice(plugin):
    assert len(_options(plugin, "variables").options) == 2


def test_variables_picker_stores_the_ref_as_the_value(plugin):
    assert _options(plugin, "variables").options[0].value == "wx.temp"


def test_variables_picker_shows_the_live_value_as_a_preview(plugin):
    assert _options(plugin, "variables").options[0].preview == "61F"


def test_variables_picker_groups_by_source_plugin(plugin):
    assert _options(plugin, "variables").options[0].group == "Weather"


def test_variables_picker_marks_unusable_choices_disabled(plugin):
    assert _options(plugin, "variables").options[1].disabled


def test_a_disabled_choice_explains_itself(plugin):
    assert "itself" in _options(plugin, "variables").options[1].description.lower()


def test_variables_picker_respects_the_requested_limit(plugin):
    assert len(_options(plugin, "variables", limit=1).options) == 1


def test_watched_picker_offers_only_already_watched_variables(plugin):
    assert [o.value for o in _options(plugin, "watched").options] == ["air.aqi", "wx.temp"]


def test_watched_picker_is_empty_when_nothing_is_watched(plugin):
    plugin.config = dict(plugin.config, watchlist=[])
    assert _options(plugin, "watched").options == []


def test_watched_picker_filters_on_the_query(plugin):
    assert [o.value for o in _options(plugin, "watched", query="aqi").options] == ["air.aqi"]


def test_watched_picker_uses_configured_labels(plugin):
    plugin.config = dict(plugin.config, labels={"air.aqi": "AIR"})
    assert _options(plugin, "watched").options[0].label == "AIR"


def test_an_unknown_options_id_is_refused(plugin):
    with pytest.raises(OptionsUnavailable):
        _options(plugin, "nonsense")


def test_the_picker_works_with_no_api_key_configured(plugin):
    plugin.config = dict(plugin.config, api_key="")
    assert _options(plugin, "variables").options


def test_the_picker_never_calls_the_llm(plugin, monkeypatch):
    from plugins.generative_dashboard import llm

    def explode(*args, **kwargs):
        raise AssertionError("get_options must not call the LLM")

    monkeypatch.setattr(llm.DashboardLLM, "complete", explode)
    _options(plugin, "variables")
