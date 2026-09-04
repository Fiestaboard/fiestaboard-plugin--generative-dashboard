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
        catalog, "plugin_display_name",
        lambda pid: {"air": "Air Quality", "wx": "Weather"}.get(pid, pid),
    )
    monkeypatch.setattr(
        catalog, "variable_catalog",
        lambda exclude, width, query="": [
            VariableChoice("wx.temp", "Temp (Weather)",
                           "61F · Outside temperature · wx.temp", "Weather", "61F"),
            VariableChoice("gd.headline", "Headline (Generative Dashboard)",
                           "gd.headline", "Generative Dashboard", "",
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


def test_watched_picker_offers_everything_when_the_watchlist_is_empty(plugin, monkeypatch):
    # Empty means "watching everything", so everything is pinnable.
    monkeypatch.setattr(catalog, "eligible_refs", lambda exclude, width: ["a.b", "c.d"])
    plugin.config = dict(plugin.config, watchlist=[])
    assert [o.value for o in _options(plugin, "watched").options] == ["a.b", "c.d"]


def test_watched_picker_filters_on_the_query(plugin):
    assert [o.value for o in _options(plugin, "watched", query="aqi").options] == ["air.aqi"]


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


def test_watched_picker_names_the_owning_plugin(plugin):
    # Same widget, same blind spot: only `label` is drawn.
    labels = [o.label for o in _options(plugin, "watched").options]
    assert labels == ["Aqi (Air Quality)", "Temp (Weather)"]


def test_watched_picker_keeps_a_custom_label_but_still_attributes_it(plugin):
    plugin.config = dict(plugin.config, labels={"air.aqi": "AIR"})
    assert _options(plugin, "watched").options[0].label == "AIR (Air Quality)"


def test_a_disabled_option_keeps_its_ref_searchable(plugin):
    # The reason matters most, but dropping the ref would make the row
    # impossible to find by typing its name.
    option = _options(plugin, "variables").options[1]
    assert "itself" in option.description.lower()
    assert "gd.headline" in option.description


def test_a_truncated_catalog_tells_the_ui_there_is_more(plugin, monkeypatch):
    # Silent truncation reads as "that's everything", which is how a whole
    # plugin's variables can appear to be missing.
    monkeypatch.setattr(
        catalog, "variable_catalog",
        lambda exclude, width, query="": [
            VariableChoice(f"p.v{i}", f"V{i} (P)", f"p.v{i}", "P", "") for i in range(50)
        ],
    )
    result = _options(plugin, "variables", limit=10)
    assert len(result.options) == 10
    assert result.has_more
    assert result.total == 50


def test_a_complete_catalog_does_not_claim_there_is_more(plugin):
    result = _options(plugin, "variables", limit=100)
    assert not result.has_more
    assert result.total == 2


def test_the_pin_picker_explains_an_empty_list(plugin, monkeypatch):
    # "No options available" with no reason reads as a broken picker.
    monkeypatch.setattr(catalog, "eligible_refs", lambda exclude, width: [])
    plugin.config = dict(plugin.config, watchlist=[])
    result = _options(plugin, "watched")
    assert result.options == []
    assert result.error and "enable" in result.error.lower()
