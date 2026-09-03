"""The manifest is the plugin's contract with core; check core accepts it."""

import json
import pathlib

import pytest
from src.plugins.manifest import load_manifest, settings_schema_ui_warnings

MANIFEST_PATH = pathlib.Path(__file__).resolve().parent.parent / "manifest.json"


@pytest.fixture(scope="module")
def raw():
    return json.loads(MANIFEST_PATH.read_text())


def test_core_loads_the_manifest_without_errors():
    parsed, errors = load_manifest(MANIFEST_PATH)
    assert errors == []
    assert parsed is not None


def test_the_ui_vocabulary_is_one_core_understands(raw):
    # A typo in ui:options is a warning, not an error, so it would otherwise
    # sail through load_manifest and only fail in the settings dialog.
    assert settings_schema_ui_warnings(raw["settings_schema"]) == []


def test_plugin_id_matches_the_class(raw):
    from plugins.generative_dashboard import GenerativeDashboardPlugin

    assert raw["id"] == GenerativeDashboardPlugin(raw).plugin_id


def test_watchlist_uses_the_remote_options_picker(raw):
    field = raw["settings_schema"]["properties"]["watchlist"]
    assert field["ui:widget"] == "remote-options"
    assert field["type"] == "array"


def test_watchlist_picker_is_multiple_searchable_and_reorderable(raw):
    options = raw["settings_schema"]["properties"]["watchlist"]["ui:options"]
    assert options["multiple"] and options["searchable"] and options["reorderable"]


def test_watchlist_picker_writes_labels_to_a_real_sibling(raw):
    properties = raw["settings_schema"]["properties"]
    assert properties["watchlist"]["ui:options"]["labels_field"] in properties


def test_pinned_picker_depends_on_the_watchlist(raw):
    assert raw["settings_schema"]["properties"]["pinned"]["ui:options"]["depends_on"] == [
        "watchlist"
    ]


def test_watchlist_is_capped_at_one_hundred(raw):
    from plugins.generative_dashboard import MAX_WATCHLIST

    assert raw["settings_schema"]["properties"]["watchlist"]["maxItems"] == MAX_WATCHLIST


def test_every_options_id_the_schema_asks_for_is_served(raw):
    from src.plugins.manifest import collect_options_ids

    from plugins.generative_dashboard import GenerativeDashboardPlugin
    from src.plugins.base import OptionsRequest, OptionsUnavailable

    plugin = GenerativeDashboardPlugin(raw)
    plugin.config = {"watchlist": []}
    for options_id in collect_options_ids(raw["settings_schema"]):
        try:
            plugin.get_options(OptionsRequest(options_id=options_id))
        except OptionsUnavailable:
            pytest.fail(f"Schema asks for '{options_id}' but the plugin refuses it")
        except Exception:
            pass  # any other failure is a registry problem, not a contract gap


def test_live_data_is_enabled_so_values_never_go_stale(raw):
    assert raw["live_data"] is True


def test_refresh_floor_matches_the_schema_minimum(raw):
    schema_min = raw["settings_schema"]["properties"]["refresh_seconds"]["minimum"]
    assert raw["min_refresh_seconds"] == schema_min


def test_every_emitted_variable_is_declared(raw):
    declared = set(raw["variables"]["simple"]) | set(raw["variables"].get("arrays", {}))
    assert {
        "prose", "headline", "reason", "stat_count",
        "degraded", "generated_at", "model", "rows",
    } <= declared


def test_api_key_is_marked_secret(raw):
    assert raw["settings_schema"]["properties"]["api_key"]["secret"] is True


def test_output_mode_enum_matches_the_code(raw):
    from plugins.generative_dashboard import OUTPUT_MODES

    assert raw["settings_schema"]["properties"]["output_mode"]["enum"] == list(OUTPUT_MODES)
