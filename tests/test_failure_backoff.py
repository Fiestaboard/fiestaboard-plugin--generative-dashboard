"""A failing model: back off instead of asking on every render, and say why."""

import pytest
from src.devices import BoardContext

from plugins.generative_dashboard import RETRY_BASE_SECONDS, GenerativeDashboardPlugin, catalog

FLAGSHIP = BoardContext("flagship", rows=6, cols=22)
VALUES = {"air.aqi": "68", "wx.temp": "61F"}
GOOD = (
    '{"tiles": [{"label": "AQI", "variable": "air.aqi"}, {"label": "TEMP", "variable": "wx.temp"}],'
    ' "banner": "", "headline": "AQI 68", "reason": "TEST"}'
)


class Completion:
    def __init__(self, text, model="test-model"):
        self.text = text
        self.model = model


class FakeCore:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def ai_complete(self, messages, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def plugin(manifest, config, monkeypatch):
    instance = GenerativeDashboardPlugin(manifest)
    instance.config = dict(config, api_key="")
    monkeypatch.setattr(catalog, "read_values", lambda refs, board, exclude, **kw: dict(VALUES))
    monkeypatch.setattr(instance, "ai_providers", lambda: [{"id": "p1"}])
    # Run each worker inline so a render's generation has finished when it returns.
    monkeypatch.setattr(instance, "_spawn", lambda *a, **k: instance._run(*a, **k))
    return instance


def _fetch(plugin):
    with plugin._bound_board(FLAGSHIP):
        return plugin.fetch_data()


def _age(plugin, seconds):
    """Pretend the last attempt was *seconds* longer ago than it was."""
    state = plugin._states["flagship"]
    state.last_generated -= seconds


def test_a_failing_cold_board_does_not_ask_the_model_on_every_render(plugin, monkeypatch):
    from src.plugins.base import AIProviderError

    core = FakeCore(AIProviderError("test provider down"))
    monkeypatch.setattr(plugin, "ai_complete", core.ai_complete)
    for _ in range(10):
        _fetch(plugin)
    assert len(core.calls) == 1


def test_a_failing_cold_board_tries_again_after_the_backoff(plugin, monkeypatch):
    from src.plugins.base import AIProviderError

    core = FakeCore(AIProviderError("test provider down"))
    monkeypatch.setattr(plugin, "ai_complete", core.ai_complete)
    _fetch(plugin)
    _age(plugin, RETRY_BASE_SECONDS + 1)
    _fetch(plugin)
    assert len(core.calls) == 2


def test_the_backoff_doubles_with_each_failure_up_to_the_refresh_interval(plugin, monkeypatch):
    from src.plugins.base import AIProviderError

    core = FakeCore(AIProviderError("test provider down"))
    monkeypatch.setattr(plugin, "ai_complete", core.ai_complete)
    _fetch(plugin)
    _age(plugin, RETRY_BASE_SECONDS + 1)
    _fetch(plugin)  # second failure: now wait twice as long
    _age(plugin, RETRY_BASE_SECONDS + 1)
    _fetch(plugin)
    assert len(core.calls) == 2
    _age(plugin, RETRY_BASE_SECONDS)
    _fetch(plugin)
    assert len(core.calls) == 3
    plugin._states["flagship"].failures = 50
    _age(plugin, plugin.config["refresh_seconds"] + 1)
    _fetch(plugin)
    assert len(core.calls) == 4


def test_the_error_variable_says_why_the_model_did_not_answer(plugin, monkeypatch):
    from src.plugins.base import AINotConfiguredError

    core = FakeCore(AINotConfiguredError("test pick a model"))
    monkeypatch.setattr(plugin, "ai_complete", core.ai_complete)
    _fetch(plugin)
    data = _fetch(plugin).data
    assert data["degraded"] == "no_llm"
    assert "test pick a model" in data["error"]


def test_the_error_variable_clears_once_the_model_answers(plugin, monkeypatch):
    from src.plugins.base import AIProviderError

    core = FakeCore(AIProviderError("test provider down"), Completion(GOOD))
    monkeypatch.setattr(plugin, "ai_complete", core.ai_complete)
    _fetch(plugin)
    _age(plugin, RETRY_BASE_SECONDS + 1)
    _fetch(plugin)
    data = _fetch(plugin).data
    assert data["degraded"] == ""
    assert data["error"] == ""


def test_a_temperature_of_zero_is_sent_as_zero(plugin, monkeypatch):
    core = FakeCore(Completion(GOOD))
    monkeypatch.setattr(plugin, "ai_complete", core.ai_complete)
    plugin.config = dict(plugin.config, temperature=0)
    _fetch(plugin)
    assert core.calls[0]["temperature"] == 0
