"""Where the model connection comes from: a pasted key, OpenRouter sign-in, or FiestaBot."""

import json
import pathlib
from unittest.mock import patch

import pytest
import requests

from plugins.generative_dashboard import GenerativeDashboardPlugin, catalog, connection
from plugins.generative_dashboard.connection import (
    OPENROUTER_BASE_URL,
    ConnectionUnavailable,
    resolve,
)
from plugins.generative_dashboard.llm import DashboardLLM, LLMError

MANIFEST_PATH = pathlib.Path(__file__).resolve().parent.parent / "manifest.json"
VALUES = {"air.aqi": "68", "wx.temp": "61F"}


class FakePlugin:
    def __init__(self, token=None, refreshed=None):
        self.token = token
        self.refreshed = refreshed
        self.rejected = 0

    def get_oauth_token(self):
        return self.token

    def report_oauth_rejected(self):
        self.rejected += 1
        return self.refreshed


def _providers(monkeypatch, providers, default=None):
    block = {"enabled": True, "providers": providers, "default_provider_id": default}
    monkeypatch.setattr(connection, "_ai_providers", lambda: block)


# -- api_key: exactly today's behaviour ----------------------------------------


def test_a_config_without_a_source_uses_the_pasted_key_unchanged(config):
    endpoint = resolve(config, FakePlugin(token="test_ignored"))
    assert (endpoint.base_url, endpoint.api_key, endpoint.model) == (
        "https://api.test/v1", "sk-test", "gpt-4o-mini",
    )
    assert endpoint.source == "api_key"


def test_the_pasted_key_source_never_asks_for_a_sign_in(config):
    class Boom:
        def get_oauth_token(self):
            raise AssertionError("must not be called")

    assert resolve(dict(config, llm_source="api_key"), Boom()).api_key == "sk-test"


def test_the_pasted_key_source_keeps_the_default_base_url():
    endpoint = resolve({"api_key": "k"}, FakePlugin())
    assert endpoint.base_url == "https://api.openai.com/v1"
    assert endpoint.model == "gpt-4o-mini"


def test_an_unknown_source_falls_back_to_the_pasted_key(config):
    assert resolve(dict(config, llm_source="nonsense"), FakePlugin()).source == "api_key"


# -- OpenRouter sign-in ------------------------------------------------------


def test_openrouter_uses_the_signed_in_key_and_openrouter(config):
    endpoint = resolve(dict(config, llm_source="openrouter"), FakePlugin(token="test_or_key"))
    assert endpoint.api_key == "test_or_key"
    assert endpoint.base_url == OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


def test_openrouter_names_a_bare_model_as_an_openai_one(config):
    endpoint = resolve(dict(config, llm_source="openrouter"), FakePlugin(token="t"))
    assert endpoint.model == "openai/gpt-4o-mini"


def test_openrouter_keeps_a_model_that_already_names_its_vendor(config):
    cfg = dict(config, llm_source="openrouter", model="anthropic/claude-haiku")
    assert resolve(cfg, FakePlugin(token="t")).model == "anthropic/claude-haiku"


def test_openrouter_without_a_sign_in_says_so(config):
    with pytest.raises(ConnectionUnavailable, match="Sign in to OpenRouter"):
        resolve(dict(config, llm_source="openrouter"), FakePlugin(token=None))


def test_openrouter_on_a_core_without_sign_in_says_so(config):
    with pytest.raises(ConnectionUnavailable, match="9.9"):
        resolve(dict(config, llm_source="openrouter"), object())


def test_a_rejected_openrouter_key_is_reported_and_the_refresh_returned(config):
    plugin = FakePlugin(token="t", refreshed="test_new")
    endpoint = resolve(dict(config, llm_source="openrouter"), plugin)
    assert endpoint.on_rejected() == "test_new"
    assert plugin.rejected == 1


def test_a_rejected_pasted_key_reports_nothing(config):
    plugin = FakePlugin(token="t")
    assert resolve(config, plugin).on_rejected() is None
    assert plugin.rejected == 0


# -- FiestaBot provider ------------------------------------------------------


OPENAI_PROVIDER = {
    "id": "p1", "name": "My OpenAI", "base_url": "https://api.example.com/v1",
    "api_key": "test_fb_key", "protocol": "openai", "default_model": "gpt-4.1-mini",
}


def test_fiestabot_reuses_the_named_provider(config, monkeypatch):
    _providers(monkeypatch, [OPENAI_PROVIDER])
    cfg = dict(config, llm_source="fiestabot", fiestabot_provider="p1")
    endpoint = resolve(cfg, FakePlugin())
    assert (endpoint.base_url, endpoint.api_key, endpoint.model) == (
        "https://api.example.com/v1", "test_fb_key", "gpt-4.1-mini",
    )


def test_fiestabot_without_a_choice_uses_the_default_provider(config, monkeypatch):
    other = dict(OPENAI_PROVIDER, id="p2", api_key="test_other")
    _providers(monkeypatch, [OPENAI_PROVIDER, other], default="p2")
    assert resolve(dict(config, llm_source="fiestabot"), FakePlugin()).api_key == "test_other"


def test_fiestabot_falls_back_to_the_first_provider(config, monkeypatch):
    _providers(monkeypatch, [OPENAI_PROVIDER])
    assert resolve(dict(config, llm_source="fiestabot"), FakePlugin()).api_key == "test_fb_key"


def test_fiestabot_model_falls_back_to_the_providers_list_then_the_setting(config, monkeypatch):
    listed = {k: v for k, v in OPENAI_PROVIDER.items() if k != "default_model"}
    _providers(monkeypatch, [dict(listed, models=["m-listed"])])
    assert resolve(dict(config, llm_source="fiestabot"), FakePlugin()).model == "m-listed"
    _providers(monkeypatch, [listed])
    assert resolve(dict(config, llm_source="fiestabot"), FakePlugin()).model == "gpt-4o-mini"


def test_fiestabot_with_no_providers_says_so(config, monkeypatch):
    _providers(monkeypatch, [])
    with pytest.raises(ConnectionUnavailable, match="FiestaBot"):
        resolve(dict(config, llm_source="fiestabot"), FakePlugin())


def test_fiestabot_with_a_missing_provider_says_so(config, monkeypatch):
    _providers(monkeypatch, [OPENAI_PROVIDER])
    with pytest.raises(ConnectionUnavailable, match="gone"):
        resolve(dict(config, llm_source="fiestabot", fiestabot_provider="nope"), FakePlugin())


@pytest.mark.parametrize("protocol", ["anthropic", "openai_responses"])
def test_fiestabot_refuses_a_provider_that_is_not_openai_compatible(config, monkeypatch, protocol):
    _providers(monkeypatch, [dict(OPENAI_PROVIDER, protocol=protocol)])
    with pytest.raises(ConnectionUnavailable, match="OpenAI-compatible"):
        resolve(dict(config, llm_source="fiestabot"), FakePlugin())


def test_fiestabot_signed_in_provider_gets_its_current_token(config, monkeypatch):
    signed = dict(OPENAI_PROVIDER, api_key="", base_url="", sign_in={"preset": "openrouter"})
    _providers(monkeypatch, [signed])
    seen = {}

    def fake_resolve(provider):
        seen["provider"] = provider
        return dict(provider, api_key="test_signed", base_url="https://openrouter.ai/api/v1")

    monkeypatch.setattr(connection, "_resolve_provider_auth", fake_resolve)
    endpoint = resolve(dict(config, llm_source="fiestabot"), FakePlugin())
    assert seen["provider"]["id"] == "p1"
    assert (endpoint.api_key, endpoint.base_url) == ("test_signed", "https://openrouter.ai/api/v1")


def test_fiestabot_signed_out_provider_says_so(config, monkeypatch):
    _providers(monkeypatch, [dict(OPENAI_PROVIDER, sign_in={"preset": "openrouter"})])

    def signed_out(provider):
        raise RuntimeError("Sign in to OpenRouter again in Settings → AI.")

    monkeypatch.setattr(connection, "_resolve_provider_auth", signed_out)
    with pytest.raises(ConnectionUnavailable, match="Sign in to OpenRouter again"):
        resolve(dict(config, llm_source="fiestabot"), FakePlugin())


def test_fiestabot_rejection_is_reported_for_a_signed_in_provider(config, monkeypatch):
    _providers(monkeypatch, [dict(OPENAI_PROVIDER, sign_in={"preset": "openrouter"})])
    monkeypatch.setattr(connection, "_resolve_provider_auth", lambda p: dict(p, api_key="t"))
    reported = []
    monkeypatch.setattr(connection, "_report_ai_rejected", lambda pid: reported.append(pid) or "test_new")
    endpoint = resolve(dict(config, llm_source="fiestabot"), FakePlugin())
    assert endpoint.on_rejected() == "test_new"
    assert reported == ["p1"]


def test_fiestabot_rejection_of_a_pasted_provider_key_reports_nothing(config, monkeypatch):
    _providers(monkeypatch, [OPENAI_PROVIDER])
    monkeypatch.setattr(connection, "_report_ai_rejected", lambda pid: pytest.fail("reported"))
    assert resolve(dict(config, llm_source="fiestabot"), FakePlugin()).on_rejected() is None


CHATGPT_SIGNED_IN = {
    # Saved by Settings → AI without a protocol; core fills openai_responses.
    "id": "c1", "name": "ChatGPT", "api_key": "", "base_url": "",
    "sign_in": {"preset": "openai_chatgpt"},
}


def test_a_chatgpt_sign_in_without_a_stored_protocol_is_not_openai_compatible():
    assert not connection.is_openai_compatible(CHATGPT_SIGNED_IN)
    assert connection.is_openai_compatible(
        dict(CHATGPT_SIGNED_IN, sign_in={"preset": "openrouter"})
    )


def test_fiestabot_refuses_a_chatgpt_sign_in_without_a_stored_protocol(config, monkeypatch):
    _providers(monkeypatch, [CHATGPT_SIGNED_IN])
    monkeypatch.setattr(connection, "_resolve_provider_auth", lambda p: pytest.fail("resolved"))
    with pytest.raises(ConnectionUnavailable, match="OpenAI-compatible"):
        resolve(dict(config, llm_source="fiestabot"), FakePlugin())
    assert not connection.is_ready(dict(config, llm_source="fiestabot"), FakePlugin())


def test_fiestabot_refuses_a_provider_whose_resolved_protocol_is_not_openai(config, monkeypatch):
    _providers(monkeypatch, [dict(OPENAI_PROVIDER, protocol="", sign_in={"preset": "x"})])
    monkeypatch.setattr(
        connection, "_resolve_provider_auth",
        lambda p: dict(p, api_key="t", protocol="openai_responses"),
    )
    with pytest.raises(ConnectionUnavailable, match="OpenAI-compatible"):
        resolve(dict(config, llm_source="fiestabot"), FakePlugin())


def test_the_real_core_resolver_leaves_an_api_key_provider_alone():
    # Runs against the core checkout: the api_key path must be the identity.
    assert connection._resolve_provider_auth(OPENAI_PROVIDER) is OPENAI_PROVIDER


# -- the HTTP client reports the status it got -----------------------------------


def test_the_client_reports_a_401_status():
    reply = requests.Response()
    reply.status_code = 401
    with patch("requests.post", return_value=reply), pytest.raises(LLMError) as caught:
        DashboardLLM("https://api.test/v1", "k", "m", 0.3).complete("s", "u")
    assert caught.value.status == 401
    assert not caught.value.retryable


# -- the plugin wiring --------------------------------------------------------


@pytest.fixture
def plugin(manifest, config, monkeypatch):
    instance = GenerativeDashboardPlugin(manifest)
    instance.config = config
    monkeypatch.setattr(catalog, "read_values", lambda refs, board, exclude, **kw: dict(VALUES))
    return instance


def _fake_client(monkeypatch, replies):
    seen = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.api_key = kwargs["api_key"]
            seen.append(self)

        def complete(self, system, user):
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

    monkeypatch.setattr("plugins.generative_dashboard.DashboardLLM", FakeClient)
    return seen


def _generate(plugin):
    with plugin._bound_board(None):
        return plugin._generate(plugin._geometry(), plugin.config, ["air.aqi"], VALUES, {}, [], "", "grid")


def test_generation_with_a_pasted_key_builds_the_client_as_before(plugin, monkeypatch):
    seen = _fake_client(monkeypatch, [LLMError("down")])
    _generate(plugin)
    kwargs = seen[0].kwargs
    assert (kwargs["base_url"], kwargs["api_key"], kwargs["model"]) == (
        "https://api.test/v1", "sk-test", "gpt-4o-mini",
    )


def test_generation_signed_in_to_openrouter_uses_the_token(plugin, monkeypatch):
    plugin.config = dict(plugin.config, llm_source="openrouter", api_key="")
    monkeypatch.setattr(plugin, "get_oauth_token", lambda: "test_or", raising=False)
    seen = _fake_client(monkeypatch, [LLMError("down")])
    _generate(plugin)
    assert seen[0].kwargs["api_key"] == "test_or"
    assert seen[0].kwargs["base_url"] == OPENROUTER_BASE_URL


def test_a_401_is_reported_and_retried_once_with_the_new_key(plugin, monkeypatch):
    plugin.config = dict(plugin.config, llm_source="openrouter")
    monkeypatch.setattr(plugin, "get_oauth_token", lambda: "test_old", raising=False)
    monkeypatch.setattr(plugin, "report_oauth_rejected", lambda: "test_new", raising=False)

    class SecondCall(Exception):
        pass

    seen = _fake_client(monkeypatch, [LLMError("401", status=401), SecondCall()])
    with pytest.raises(SecondCall):
        _generate(plugin)
    assert seen[0].api_key == "test_new"


def test_a_401_with_no_refresh_gives_up_without_a_second_call(plugin, monkeypatch):
    plugin.config = dict(plugin.config, llm_source="openrouter")
    monkeypatch.setattr(plugin, "get_oauth_token", lambda: "test_old", raising=False)
    calls = []
    monkeypatch.setattr(plugin, "report_oauth_rejected", lambda: calls.append(1), raising=False)
    _fake_client(monkeypatch, [LLMError("401", status=401)])
    assert _generate(plugin) is None
    assert calls == [1]


def test_generation_is_skipped_while_signed_out(plugin, monkeypatch):
    plugin.config = dict(plugin.config, llm_source="openrouter", api_key="")
    monkeypatch.setattr(plugin, "get_oauth_token", lambda: None, raising=False)
    assert plugin._connection_ready(plugin.config) is False


def test_generation_may_start_once_signed_in(plugin, monkeypatch):
    plugin.config = dict(plugin.config, llm_source="openrouter", api_key="")
    monkeypatch.setattr(plugin, "get_oauth_token", lambda: "test_or", raising=False)
    assert plugin._connection_ready(plugin.config) is True


def test_generation_may_start_with_a_fiestabot_provider(plugin, monkeypatch):
    _providers(monkeypatch, [OPENAI_PROVIDER])
    plugin.config = dict(plugin.config, llm_source="fiestabot", api_key="")
    assert plugin._connection_ready(plugin.config) is True
    _providers(monkeypatch, [])
    assert plugin._connection_ready(plugin.config) is False


def test_validate_config_needs_no_key_when_signed_in(plugin):
    assert not any("API key" in e for e in plugin.validate_config({"llm_source": "openrouter"}))
    assert not any("API key" in e for e in plugin.validate_config({"llm_source": "fiestabot"}))


def test_validate_config_still_needs_a_key_by_default(plugin):
    assert any("API key" in e for e in plugin.validate_config({}))
    assert any("API key" in e for e in plugin.validate_config({"llm_source": "api_key"}))


def test_validate_config_rejects_an_unknown_source(plugin):
    assert any("llm_source" in e for e in plugin.validate_config({"api_key": "k", "llm_source": "x"}))


def test_the_provider_picker_lists_fiestabot_providers(plugin, monkeypatch):
    from src.plugins.base import OptionsRequest

    _providers(monkeypatch, [OPENAI_PROVIDER, dict(OPENAI_PROVIDER, id="a", name="Claude", protocol="anthropic")])
    result = plugin.get_options(OptionsRequest(options_id="ai_providers", query="", limit=50))
    by_value = {o.value: o for o in result.options}
    assert by_value["p1"].label == "My OpenAI" and not by_value["p1"].disabled
    assert by_value["a"].disabled


def test_the_provider_picker_disables_a_chatgpt_sign_in(plugin, monkeypatch):
    from src.plugins.base import OptionsRequest

    _providers(monkeypatch, [CHATGPT_SIGNED_IN])
    result = plugin.get_options(OptionsRequest(options_id="ai_providers", query="", limit=50))
    assert result.options[0].disabled


def test_the_provider_picker_explains_an_empty_list(plugin, monkeypatch):
    from src.plugins.base import OptionsRequest

    _providers(monkeypatch, [])
    result = plugin.get_options(OptionsRequest(options_id="ai_providers", query="", limit=50))
    assert result.options == [] and "FiestaBot" in result.error


# -- manifest ------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw():
    return json.loads(MANIFEST_PATH.read_text())


def test_the_manifest_declares_openrouter_key_exchange(raw):
    oauth = raw["oauth"]
    assert oauth["flows"] == ["key_exchange"]
    assert oauth["authorization_url"] == "https://openrouter.ai/auth"
    assert oauth["token_url"] == "https://openrouter.ai/api/v1/auth/keys"
    assert "client_id" not in oauth


def test_core_parses_the_oauth_block(raw):
    from src.oauth.provider import parse_provider_block, validate_provider_block

    assert validate_provider_block(raw["oauth"], raw["settings_schema"]) == []
    assert parse_provider_block(raw["oauth"], raw["name"], raw["settings_schema"]) is not None


def test_the_manifest_needs_a_core_with_key_exchange(raw):
    assert raw["fiestaboard_version"] == ">=9.9.0"


def test_existing_settings_keys_are_all_still_there(raw):
    props = raw["settings_schema"]["properties"]
    for key in ("api_key", "api_base_url", "model", "temperature", "output_mode"):
        assert key in props
    assert props["api_key"]["secret"] is True


def test_the_source_defaults_to_the_pasted_key(raw):
    field = raw["settings_schema"]["properties"]["llm_source"]
    assert field["default"] == "api_key"
    assert set(field["enum"]) == {"api_key", "openrouter", "fiestabot"}


def test_the_api_key_is_no_longer_required_by_the_schema(raw):
    assert "api_key" not in raw["settings_schema"].get("required", [])
