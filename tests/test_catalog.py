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


class FakeVariables:
    def __init__(self, simple):
        # {name: {"max_length": int|None, "description": str}} or {name: {}}
        self.simple = simple

    def get_variable_metadata(self, name):
        meta = self.simple.get(name) or {}
        return SimpleNamespace(
            description=meta.get("description", ""), type="string",
            max_length=meta.get("max_length", 6), group="", example="",
        )


class FakeRegistry:
    def __init__(self, metadata=None, data=None, names=None, installed=None):
        self._metadata = metadata or {}
        self._data = data or {}
        self._names = names or {}
        # {plugin_id: [declared variable names]} for plugins that are installed
        # but not enabled, which expose nothing through the metadata call.
        self._installed = installed or {}
        self.fetched = []

    @property
    def enabled_plugins(self):
        return {p: object() for p in getattr(self, "_enabled_ids", set(self._metadata))}

    @property
    def plugins(self):
        return {**{p: object() for p in self._metadata}, **{p: object() for p in self._installed}}

    def get_all_variables_with_metadata(self):
        return self._metadata

    def get_manifest(self, plugin_id):
        if plugin_id not in self._names:
            return None
        # Enabled plugins declare the same variables in their manifest as they
        # publish through the metadata call; the fake mirrors that.
        declared = self._installed.get(plugin_id) or self._metadata.get(plugin_id) or {}
        return SimpleNamespace(name=self._names[plugin_id], variables=FakeVariables(declared))

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


def test_choice_label_names_the_owning_plugin(fake):
    # The picker widget renders only `label` — it ignores `group` entirely —
    # so attribution has to live in the label or it is invisible. The variable
    # leads because the chosen-row list truncates on overflow.
    fake(FakeRegistry(
        metadata={"weather": {"temp_f": {"description": "", "max_length": 6,
                                         "group": "", "preview": "61F"}}},
        names={"weather": "Weather"},
    ))
    assert variable_catalog("generative_dashboard", 15)[0].label == "Temp F (Weather)"


def test_choice_description_carries_the_ref_so_search_can_match_it(fake):
    # The search box filters on label + description only.
    fake(FakeRegistry(
        metadata={"weather": {"temp_f": {"description": "Outside temp", "max_length": 6,
                                         "group": "", "preview": "61F"}}},
        names={"weather": "Weather"},
    ))
    description = variable_catalog("generative_dashboard", 15)[0].description
    assert "weather.temp_f" in description
    assert "Outside temp" in description


def test_choice_description_shows_the_current_value(fake):
    fake(FakeRegistry(
        metadata={"weather": {"temp_f": {"description": "", "max_length": 6,
                                         "group": "", "preview": "61F"}}},
        names={"weather": "Weather"},
    ))
    assert "61F" in variable_catalog("generative_dashboard", 15)[0].description


def test_two_plugins_with_the_same_variable_name_are_distinguishable(fake):
    fake(FakeRegistry(
        metadata={
            "weather": {"temp": {"description": "", "max_length": 6, "group": "", "preview": "61F"}},
            "home_assistant": {"temp": {"description": "", "max_length": 6, "group": "", "preview": "68F"}},
        },
        names={"weather": "Weather", "home_assistant": "Home Assistant"},
    ))
    labels = [c.label for c in variable_catalog("generative_dashboard", 15)]
    assert len(set(labels)) == 2
    assert "Temp (Home Assistant)" in labels and "Temp (Weather)" in labels


def test_a_disabled_reason_still_wins_the_description(fake):
    fake(FakeRegistry(
        metadata={"art": {"art": {"description": "Art blob", "max_length": 512,
                                  "group": "", "preview": ""}}},
        names={"art": "Art"},
    ))
    choice = variable_catalog("generative_dashboard", 15)[0]
    assert "512" in choice.disabled_reason


def test_searching_by_plugin_name_matches_the_label(fake):
    fake(FakeRegistry(
        metadata={"weather": {"temp_f": {"description": "", "max_length": 6,
                                         "group": "", "preview": ""}}},
        names={"weather": "Weather"},
    ))
    assert len(variable_catalog("generative_dashboard", 15, query="weather")) == 1


def test_searching_by_the_plugins_display_name_finds_its_variables(fake):
    # "Star Trek Quotes" is the name a user reads; the ref says
    # "star_trek_quotes", so matching only the ref misses the spaced form.
    fake(FakeRegistry(
        metadata={"star_trek_quotes": {
            "character": {"description": "", "max_length": 6, "group": "", "preview": ""},
            "series": {"description": "", "max_length": 6, "group": "", "preview": ""},
        }},
        names={"star_trek_quotes": "Star Trek Quotes"},
    ))
    hits = variable_catalog("generative_dashboard", 15, query="star trek")
    assert len(hits) == 2


def test_an_installed_but_disabled_plugin_still_shows_its_variables(fake):
    # You cannot decide whether to enable Dad Jokes if you cannot see what it
    # offers. Core's variable catalog covers enabled plugins only.
    fake(FakeRegistry(
        metadata={},
        installed={"dad_jokes": {"joke": {}, "punchline": {}}},
        names={"dad_jokes": "Dad Jokes"},
    ))
    choices = variable_catalog("generative_dashboard", 15)
    assert [c.ref for c in choices] == ["dad_jokes.joke", "dad_jokes.punchline"]


def test_a_disabled_plugins_variables_say_to_enable_it(fake):
    fake(FakeRegistry(
        metadata={},
        installed={"dad_jokes": {"joke": {}}},
        names={"dad_jokes": "Dad Jokes"},
    ))
    choice = variable_catalog("generative_dashboard", 15)[0]
    assert choice.disabled
    assert "enable" in choice.disabled_reason.lower()
    assert "Dad Jokes" in choice.disabled_reason


def test_enabled_variables_are_listed_before_disabled_ones(fake):
    fake(FakeRegistry(
        metadata={"weather": {"temp": {"description": "", "max_length": 6,
                                       "group": "", "preview": "61F"}}},
        installed={"aardvark_facts": {"fact": {}}},
        names={"weather": "Weather", "aardvark_facts": "Aardvark Facts"},
    ))
    choices = variable_catalog("generative_dashboard", 15)
    # Alphabetically 'aardvark' sorts first, but usable options come first.
    assert [c.ref for c in choices] == ["weather.temp", "aardvark_facts.fact"]


def test_a_plugin_is_not_listed_twice_when_enabled(fake):
    fake(FakeRegistry(
        metadata={"weather": {"temp": {"description": "", "max_length": 6,
                                       "group": "", "preview": ""}}},
        installed={"weather": {"temp": {}}},
        names={"weather": "Weather"},
    ))
    assert len(variable_catalog("generative_dashboard", 15)) == 1


def test_our_own_plugin_says_it_cannot_watch_itself_even_when_disabled(fake):
    fake(FakeRegistry(
        metadata={},
        installed={"generative_dashboard": {"headline": {}}},
        names={"generative_dashboard": "Generative Dashboard"},
    ))
    choice = variable_catalog("generative_dashboard", 15)[0]
    assert "itself" in choice.disabled_reason.lower()


def test_a_disabled_plugin_is_searchable_by_name(fake):
    fake(FakeRegistry(
        metadata={},
        installed={"dad_jokes": {"joke": {}}},
        names={"dad_jokes": "Dad Jokes"},
    ))
    assert len(variable_catalog("generative_dashboard", 15, query="dad")) == 1


def test_eligible_refs_returns_everything_watchable(fake):
    fake(FakeRegistry(
        metadata={"weather": {
            "temp": {"description": "", "max_length": 6, "group": "", "preview": "61F"},
            "forecast": {"description": "", "max_length": 200, "group": "", "preview": "x"},
        }},
        names={"weather": "Weather"},
    ))
    from plugins.generative_dashboard.catalog import eligible_refs

    # The 200-char forecast is wider than the board.
    assert eligible_refs("generative_dashboard", 11) == ["weather.temp"]


def test_eligible_refs_never_includes_our_own_variables(fake):
    fake(FakeRegistry(
        metadata={"generative_dashboard": {
            "headline": {"description": "", "max_length": 6, "group": "", "preview": ""}}},
        names={"generative_dashboard": "Generative Dashboard"},
    ))
    from plugins.generative_dashboard.catalog import eligible_refs

    assert eligible_refs("generative_dashboard", 11) == []


def test_eligible_refs_ignores_disabled_plugins(fake):
    fake(FakeRegistry(
        metadata={},
        installed={"dad_jokes": {"joke": {}}},
        names={"dad_jokes": "Dad Jokes"},
    ))
    from plugins.generative_dashboard.catalog import eligible_refs

    # A disabled plugin publishes no values, so there is nothing to watch.
    assert eligible_refs("generative_dashboard", 11) == []


def test_eligibility_is_judged_against_the_board_not_one_column(fake):
    # A long value gets a full row to itself, so a single column's width is
    # no longer the limit on what can be watched.
    fake(FakeRegistry(
        metadata={"stocks": {
            "price": {"description": "", "max_length": 16, "group": "", "preview": ""}}},
        names={"stocks": "Stocks"},
    ))
    from plugins.generative_dashboard.catalog import eligible_refs

    assert eligible_refs("generative_dashboard", 22) == ["stocks.price"]
    assert eligible_refs("generative_dashboard", 11) == []


def test_eligible_refs_never_builds_a_template_context(monkeypatch):
    """The render path must not fan out to every plugin.

    ``get_all_variables_with_metadata`` calls ``build_template_context``, which
    dispatches to a thread pool that calls *this* plugin's ``get_data`` again.
    Reaching it from ``fetch_data`` is an infinite recursion that hangs the
    whole API, not just this plugin.
    """

    class Exploding(FakeRegistry):
        def get_all_variables_with_metadata(self):
            raise AssertionError("eligible_refs must not build a template context")

    registry = Exploding(
        installed={"weather": {"temp": {}}},
        names={"weather": "Weather"},
    )
    registry._enabled_ids = {"weather"}
    monkeypatch.setattr(catalog, "_registry", lambda: registry)

    from plugins.generative_dashboard.catalog import eligible_refs

    assert eligible_refs("generative_dashboard", 22) == ["weather.temp"]


def test_read_values_fetches_plugins_concurrently_with_a_deadline(fake):
    """58 enabled plugins fetched one after another blows core's 15s budget.

    A plugin that exceeds the budget is dropped from the render context, and
    every one of its variables renders as "???" on the board.
    """
    import time

    class Slow(FakeRegistry):
        def fetch_plugin_data(self, plugin_id, board=None):
            time.sleep(0.4)
            return FakeResult({"v": plugin_id})

    fake(Slow(data={f"p{i}": {"v": str(i)} for i in range(8)}))
    refs = [f"p{i}.v" for i in range(8)]
    start = time.monotonic()
    values = read_values(refs, None, "generative_dashboard", timeout=5.0)
    elapsed = time.monotonic() - start
    assert len(values) == 8
    assert elapsed < 2.0, f"sequential fetch would take 3.2s, took {elapsed:.1f}s"


def test_read_values_returns_what_arrived_before_the_deadline(fake):
    import time

    class Stuck(FakeRegistry):
        def fetch_plugin_data(self, plugin_id, board=None):
            if plugin_id == "slow":
                time.sleep(30)
            return FakeResult({"v": "ok"})

    fake(Stuck(data={"fast": {"v": "ok"}, "slow": {"v": "ok"}}))
    start = time.monotonic()
    values = read_values(["fast.v", "slow.v"], None, "generative_dashboard", timeout=1.0)
    assert time.monotonic() - start < 3.0
    assert values == {"fast.v": "ok"}
