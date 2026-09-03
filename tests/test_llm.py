"""LLM transport and prompt construction."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from plugins.generative_dashboard.layout import geometry
from plugins.generative_dashboard.llm import (
    DashboardLLM,
    LLMError,
    build_grid_prompt,
    build_prose_prompt,
    describe_variables,
)

GEO = geometry(6, 22)


def _client():
    return DashboardLLM("https://api.test/v1", "sk-test", "gpt-4o-mini", 0.3)


def _response(content):
    reply = MagicMock()
    reply.status_code = 200
    reply.json.return_value = {"choices": [{"message": {"content": content}}]}
    reply.raise_for_status.return_value = None
    return reply


def _prompt(builder, **overrides):
    kwargs = {
        "geo": GEO, "refs": ["a.x"], "labels": {}, "notes": {},
        "current": {"a.x": "1"}, "previous": {}, "previous_board": [],
        "extra_instructions": "",
    }
    if builder is build_grid_prompt:
        kwargs["use_color"] = True
    kwargs.update(overrides)
    return builder(**kwargs)


def test_complete_parses_the_models_json():
    with patch("requests.post", return_value=_response('{"tiles": []}')):
        assert _client().complete("sys", "user") == {"tiles": []}


def test_complete_strips_a_markdown_code_fence():
    with patch("requests.post", return_value=_response('```json\n{"tiles": []}\n```')):
        assert _client().complete("sys", "user") == {"tiles": []}


def test_complete_raises_on_unparseable_output():
    with patch("requests.post", return_value=_response("sorry, no")):
        with pytest.raises(LLMError):
            _client().complete("sys", "user")


def test_complete_raises_when_the_json_is_not_an_object():
    with patch("requests.post", return_value=_response("[1, 2]")):
        with pytest.raises(LLMError):
            _client().complete("sys", "user")


def test_complete_raises_on_a_network_failure():
    with patch("requests.post", side_effect=requests.RequestException("boom")):
        with pytest.raises(LLMError):
            _client().complete("sys", "user")


def test_complete_raises_on_a_malformed_envelope():
    reply = MagicMock()
    reply.json.return_value = {"unexpected": True}
    reply.raise_for_status.return_value = None
    with patch("requests.post", return_value=reply):
        with pytest.raises(LLMError):
            _client().complete("sys", "user")


def test_complete_sends_the_bearer_token():
    with patch("requests.post", return_value=_response("{}")) as post:
        _client().complete("sys", "user")
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-test"


def test_complete_posts_to_the_chat_completions_path():
    with patch("requests.post", return_value=_response("{}")) as post:
        _client().complete("sys", "user")
    assert post.call_args.args[0] == "https://api.test/v1/chat/completions"


def test_complete_does_not_double_a_trailing_slash():
    client = DashboardLLM("https://api.test/v1/", "k", "m", 0.3)
    with patch("requests.post", return_value=_response("{}")) as post:
        client.complete("sys", "user")
    assert post.call_args.args[0] == "https://api.test/v1/chat/completions"


def test_describe_variables_shows_current_and_previous_values():
    text = describe_variables(["a.aqi"], {"a.aqi": "AQI"}, {}, {"a.aqi": "168"}, {"a.aqi": "31"})
    assert "168" in text and "31" in text


def test_describe_variables_includes_the_users_note():
    text = describe_variables(["a.aqi"], {}, {"a.aqi": "over 100 is unhealthy"},
                              {"a.aqi": "168"}, {})
    assert "unhealthy" in text


def test_describe_variables_omits_variables_with_no_value():
    assert "b.y" not in describe_variables(["a.x", "b.y"], {}, {}, {"a.x": "1"}, {})


def test_grid_prompt_states_the_exact_tile_budget():
    system, _ = _prompt(build_grid_prompt)
    assert "12" in system


def test_grid_prompt_forbids_typing_values():
    system, _ = _prompt(build_grid_prompt)
    assert "never" in system.lower()


def test_grid_prompt_mentions_color_when_it_is_enabled():
    system, _ = _prompt(build_grid_prompt, use_color=True)
    assert "color" in system.lower()


def test_grid_prompt_omits_color_entirely_when_color_is_off():
    # Including a colour key in the example would invite it back.
    system, _ = _prompt(build_grid_prompt, use_color=False)
    assert "color" not in system.lower()


def test_grid_prompt_includes_the_previous_board_for_continuity():
    _, user = _prompt(build_grid_prompt, previous_board=["CPU 42%"])
    assert "CPU 42%" in user


def test_grid_prompt_appends_extra_instructions():
    system, _ = _prompt(build_grid_prompt, extra_instructions="FAVOUR NETWORK STATS")
    assert "FAVOUR NETWORK STATS" in system


def test_prose_prompt_states_the_character_budget():
    system, _ = _prompt(build_prose_prompt)
    assert str(GEO.prose_budget) in system


def test_prose_prompt_forbids_arithmetic():
    system, _ = _prompt(build_prose_prompt)
    assert "do not compute" in system.lower()


def test_both_prompts_state_the_board_size():
    for builder in (build_grid_prompt, build_prose_prompt):
        _, user = _prompt(builder)
        assert "6" in user and "22" in user
