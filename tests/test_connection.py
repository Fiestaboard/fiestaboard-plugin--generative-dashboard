"""Where the model comes from: FiestaBoard's AI providers, or a separate pasted key."""

import json
import pathlib

import pytest

from plugins.generative_dashboard import GenerativeDashboardPlugin, catalog, connection
from plugins.generative_dashboard.connection import (
    UPDATE_MESSAGE,
    ConnectionUnavailable,
    build_client,
)
from plugins.generative_dashboard.llm import CoreLLM, DashboardLLM, LLMError

MANIFEST_PATH = pathlib.Path(__file__).resolve().parent.parent / "manifest.json"
VALUES = {"air.aqi": "68", "wx.temp": "61F"}
REPLY = '{"tiles": []}'


class Completion:
    def __init__(self, text, model="test-model"):
        self.text = text
        self.model = model
        self.provider_id = "p1"
        self.usage = {}
        self.data = None

    def __str__(self):
        return self.text


class FakeCore:
    """A plugin on FiestaBoard 9.9.0: ai_complete exists."""

    def __init__(self, replies=None):
        self.replies = list(replies or [Completion(REPLY)])
        self.calls = []

    def ai_complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class OldCore:
    """A plugin on a core before 9.9.0: no ai_complete."""


def _core_config(config, **extra):
    return dict(config, api_key="", **extra)


# -- the separate API key: byte-for-byte the old client, and it wins ----------


def test_a_saved_api_key_builds_the_old_client_unchanged(config):
    client = build_client(config, FakeCore(), temperature=0.3, max_tokens=900)
    assert isinstance(client, DashboardLLM)
    assert (client.base_url, client.api_key, client.model, client.temperature, client.max_tokens) == (
        "https://api.test/v1", "sk-test", "gpt-4o-mini", 0.3, 900,
    )


def test_a_saved_api_key_wins_over_a_chosen_provider(config):
    core = FakeCore()
    client = build_client(dict(config, ai_provider="p1", ai_model="m"), core,
                          temperature=0.3, max_tokens=None)
    assert isinstance(client, DashboardLLM)
    assert core.calls == []


def test_a_saved_api_key_keeps_the_old_defaults():
    client = build_client({"api_key": "k"}, FakeCore(), temperature=0.3, max_tokens=None)
    assert client.base_url == "https://api.openai.com/v1"
    assert client.model == "gpt-4o-mini"


def test_a_saved_api_key_works_on_a_core_without_ai_complete(config):
    assert isinstance(build_client(config, OldCore(), temperature=0.3, max_tokens=None), DashboardLLM)


# -- FiestaBoard's AI providers ----------------------------------------------


def test_without_a_key_the_board_uses_fiestaboards_ai(config):
    client = build_client(_core_config(config), FakeCore(), temperature=0.3, max_tokens=900)
    assert isinstance(client, CoreLLM)


def test_a_blank_provider_and_model_mean_fiestabots_defaults(config):
    core = FakeCore()
    build_client(_core_config(config), core, temperature=0.3, max_tokens=900).complete("sys", "user")
    messages, kwargs = core.calls[0]
    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]
    assert kwargs["provider_id"] is None
    assert kwargs["model"] is None
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 900


def test_the_chosen_provider_and_model_are_passed_on(config):
    core = FakeCore()
    client = build_client(_core_config(config, ai_provider="p1", ai_model="m1"), core,
                          temperature=0.5, max_tokens=None)
    client.complete("sys", "user")
    assert core.calls[0][1]["provider_id"] == "p1"
    assert core.calls[0][1]["model"] == "m1"


def test_the_plugins_old_model_setting_does_not_leak_into_core(config):
    core = FakeCore()
    build_client(_core_config(config, model="gpt-4o-mini"), core,
                 temperature=0.3, max_tokens=None).complete("s", "u")
    assert core.calls[0][1]["model"] is None


def test_the_reply_is_parsed_like_the_old_client(config):
    core = FakeCore([Completion('```json\n{"tiles": [1,],}\n```', model="m-used")])
    client = build_client(_core_config(config), core, temperature=0.3, max_tokens=None)
    assert client.complete("s", "u") == {"tiles": [1]}
    assert client.model == "m-used"


def test_a_non_json_reply_is_retryable(config):
    core = FakeCore([Completion("not json")])
    client = build_client(_core_config(config), core, temperature=0.3, max_tokens=None)
    with pytest.raises(LLMError) as err:
        client.complete("s", "u")
    assert err.value.retryable


def test_a_core_ai_error_becomes_a_final_llm_error(config):
    from src.plugins.base import AINotConfiguredError, AIProviderError, AIRejectedError

    for exc in (AINotConfiguredError("AI is off"), AIRejectedError("sign in again"),
                AIProviderError("unreachable")):
        client = build_client(_core_config(config), FakeCore([exc]), temperature=0.3, max_tokens=None)
        with pytest.raises(LLMError) as err:
            client.complete("s", "u")
        assert not err.value.retryable
        assert str(exc) in str(err.value)


def test_the_real_core_says_when_ai_is_off(config, monkeypatch):
    from src.ai import plugin_api

    monkeypatch.setattr(plugin_api, "_providers_block", lambda: {"enabled": False, "providers": []})
    instance = GenerativeDashboardPlugin({"id": "generative_dashboard"})
    client = build_client(_core_config(config), instance, temperature=0.3, max_tokens=None)
    with pytest.raises(LLMError):
        client.complete("s", "u")


def test_an_old_core_without_a_key_says_update_or_paste(config):
    with pytest.raises(ConnectionUnavailable) as err:
        build_client(_core_config(config), OldCore(), temperature=0.3, max_tokens=None)
    assert str(err.value) == UPDATE_MESSAGE
    assert UPDATE_MESSAGE == "Update FiestaBoard to use its AI providers, or paste an API key"


def test_readiness(config):
    assert connection.is_ready(config, OldCore()) is True
    assert connection.is_ready(_core_config(config), FakeCore()) is True
    assert connection.is_ready(_core_config(config), OldCore()) is False


# -- the plugin ----------------------------------------------------------------


@pytest.fixture
def plugin(manifest, config, monkeypatch):
    instance = GenerativeDashboardPlugin(manifest)
    instance.config = config
    monkeypatch.setattr(catalog, "read_values", lambda refs, board, exclude, **kw: dict(VALUES))
    return instance


def _generate(plugin):
    with plugin._bound_board(None):
        return plugin._generate(plugin._geometry(), plugin.config, ["air.aqi"], VALUES, {}, [], "", "grid")


def test_generation_without_a_key_asks_fiestaboards_ai(plugin, monkeypatch):
    plugin.config = _core_config(plugin.config, ai_provider="p1")
    core = FakeCore([Completion("nope"), Completion("still nope")])
    monkeypatch.setattr(plugin, "ai_complete", core.ai_complete)
    assert _generate(plugin) is None
    assert len(core.calls) == 2  # one stricter retry, as with the pasted key
    assert core.calls[0][1]["provider_id"] == "p1"
    assert core.calls[0][1]["temperature"] == 0.3


def test_a_refused_provider_is_not_retried(plugin, monkeypatch):
    from src.plugins.base import AIRejectedError

    plugin.config = _core_config(plugin.config)
    core = FakeCore([AIRejectedError("sign in again")])
    monkeypatch.setattr(plugin, "ai_complete", core.ai_complete)
    assert _generate(plugin) is None
    assert len(core.calls) == 1


def test_generation_with_a_key_never_calls_core(plugin, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(plugin, "ai_complete", boom)
    seen = []

    class FakeClient:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def complete(self, system, user):
            raise LLMError("down")

    monkeypatch.setattr("plugins.generative_dashboard.connection.DashboardLLM", FakeClient)
    _generate(plugin)
    assert (seen[0]["base_url"], seen[0]["api_key"], seen[0]["model"]) == (
        "https://api.test/v1", "sk-test", "gpt-4o-mini",
    )


def test_generation_on_an_old_core_without_a_key_gives_up_quietly(plugin, monkeypatch):
    plugin.config = _core_config(plugin.config)
    monkeypatch.setattr(plugin, "ai_complete", None)
    assert _generate(plugin) is None
    assert connection.is_ready(plugin.config, plugin) is False


def test_the_model_variable_names_the_model_core_used(plugin, monkeypatch):
    plugin.config = _core_config(plugin.config)
    reply = Completion(REPLY, model="core-model")
    monkeypatch.setattr(plugin, "ai_complete", FakeCore([reply, reply]).ai_complete)
    _generate(plugin)
    with plugin._bound_board(None):
        state = plugin._hydrate(plugin._state_key())
        assert plugin._result(state, [""], "", 0).data["model"] == "core-model"


def test_the_model_variable_is_the_setting_with_a_key(plugin):
    with plugin._bound_board(None):
        state = plugin._hydrate(plugin._state_key())
        assert plugin._result(state, [""], "", 0).data["model"] == "gpt-4o-mini"


def test_validate_config_needs_no_key_with_fiestaboards_ai(plugin):
    assert not any("API key" in e for e in plugin.validate_config({}))


def test_validate_config_on_an_old_core_asks_to_update_or_paste(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "ai_complete", None)
    assert UPDATE_MESSAGE in plugin.validate_config({})
    assert UPDATE_MESSAGE not in plugin.validate_config({"api_key": "k"})


def test_the_provider_picker_is_cores(plugin, monkeypatch):
    from src.plugins.base import OptionsRequest, OptionsResult

    sentinel = OptionsResult(options=[])
    monkeypatch.setattr(plugin, "ai_provider_options", lambda request: sentinel)
    assert plugin.get_options(OptionsRequest(options_id="ai_providers")) is sentinel


# -- manifest -----------------------------------------------------------------


@pytest.fixture
def raw():
    return json.loads(MANIFEST_PATH.read_text())


def test_the_manifest_has_no_plugin_sign_in(raw):
    assert "oauth" not in raw


def test_the_manifest_needs_a_core_with_ai_complete(raw):
    assert raw["fiestaboard_version"] == ">=9.9.0"


def test_the_manifest_offers_cores_provider_picker(raw):
    props = raw["settings_schema"]["properties"]
    picker = props["ai_provider"]
    assert picker["ui:widget"] == "remote-options"
    assert picker["ui:options"]["options_id"] == "ai_providers"
    assert picker["default"] == ""
    assert props["ai_model"]["default"] == ""
    for gone in ("llm_source", "fiestabot_provider"):
        assert gone not in props


def test_existing_settings_keys_are_all_still_there(raw):
    props = raw["settings_schema"]["properties"]
    for key in ("api_key", "api_base_url", "model", "temperature", "output_mode"):
        assert key in props
    assert props["api_key"]["secret"] is True
    assert props["api_base_url"]["default"] == "https://api.openai.com/v1"
    assert props["model"]["default"] == "gpt-4o-mini"


def test_the_api_key_is_not_required_by_the_schema(raw):
    assert "api_key" not in raw["settings_schema"].get("required", [])
