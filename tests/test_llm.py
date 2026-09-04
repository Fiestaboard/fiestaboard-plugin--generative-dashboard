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


def test_describe_variables_states_how_much_room_a_label_has():
    # Without this the model writes "TIME" for a 7-char value in a 10-cell
    # column, and the renderer truncates it to "TI".
    text = describe_variables(
        ["a.clock"], {}, {}, {"a.clock": "5:34 PM"}, {}, label_budget=10
    )
    # 10 cells, minus a 7-char value, minus the gap, leaves 2.
    assert "label_max=2" in text


def test_describe_variables_label_budget_is_optional():
    text = describe_variables(["a.x"], {}, {}, {"a.x": "1"}, {})
    assert "label_max" not in text


def test_grid_prompt_tells_the_model_to_use_the_space():
    system, _ = _prompt(build_grid_prompt)
    assert "fill" in system.lower() or "use the" in system.lower()


def test_a_malformed_json_error_is_worth_retrying():
    # The model slipped; asking again often works.
    with patch("requests.post", return_value=_response("not json at all")):
        with pytest.raises(LLMError) as excinfo:
            _client().complete("sys", "user")
    assert excinfo.value.retryable


def test_a_network_error_is_not_worth_retrying():
    # The endpoint is down; it will still be down a millisecond later.
    with patch("requests.post", side_effect=requests.RequestException("boom")):
        with pytest.raises(LLMError) as excinfo:
            _client().complete("sys", "user")
    assert not excinfo.value.retryable


def test_the_grid_prompt_does_not_offer_white_or_black_as_accents():
    # A white tile on a white board is not an accent, it is noise.
    system, _ = _prompt(build_grid_prompt, use_color=True)
    colours = system.split("Valid colors:")[1].split("\n")[0]
    assert "white" not in colours and "black" not in colours


def test_label_budget_accounts_for_a_value_that_gets_its_own_row():
    # A long value is given the full board width, so its label has more room
    # than the narrow-column arithmetic would suggest.
    text = describe_variables(
        ["a.date"], {}, {}, {"a.date": "09/03/26"}, {},
        label_budget=10, wide_budget=22,
    )
    assert "label_max=13" in text


def test_the_prompt_tells_the_model_when_now_is():
    from datetime import datetime

    _, user = _prompt(build_grid_prompt, now=datetime(2026, 12, 24, 18, 30))
    assert "RIGHT NOW" in user
    assert "Thursday 24 December 2026" in user
    assert "evening" in user


def test_the_prompt_omits_the_time_line_when_no_clock_is_given():
    _, user = _prompt(build_grid_prompt)
    assert "RIGHT NOW" not in user


def test_both_prompts_ask_the_model_to_weight_by_time_of_day():
    for builder in (build_grid_prompt, build_prose_prompt):
        system, _ = _prompt(builder)
        assert "commute" in system.lower()


def test_the_prompt_carries_what_has_been_happening():
    _, user = _prompt(build_grid_prompt, journal="- 09:00 Fog thick\n- 15:00 Cleared")
    assert "Fog thick" in user and "Cleared" in user
    assert "EARLIER" in user.upper()


def test_the_prompt_omits_the_history_block_when_there_is_none():
    _, user = _prompt(build_grid_prompt)
    assert "EARLIER" not in user.upper()


def test_the_model_is_asked_to_record_what_it_sees():
    system, _ = _prompt(build_grid_prompt)
    assert '"log"' in system


def test_the_grid_prompt_shows_an_example_of_a_good_board():
    # Superseded by named patterns; the exemplars are the examples now.
    system, _ = _prompt(build_grid_prompt)
    assert "PATTERN" in system.upper()


def test_the_prompt_explains_color_as_intensity_not_just_change():
    system, _ = _prompt(build_grid_prompt, use_color=True)
    lowered = system.lower()
    assert "green" in lowered and "red" in lowered
    assert "banner_color" in system


def test_the_prompt_forbids_showing_an_identifier_on_its_own():
    # A ticker with no price, a station name with no arrival time.
    system, _ = _prompt(build_grid_prompt)
    lowered = system.lower()
    assert "its own" in lowered
    assert "symbol" in lowered


def test_the_prompt_asks_for_restraint_with_color():
    system, _ = _prompt(build_grid_prompt, use_color=True)
    assert "at most" in system.lower()


def test_describe_variables_includes_the_manifest_description():
    text = describe_variables(
        ["wx.wind"], {}, {}, {"wx.wind": "7.2"}, {},
        descriptions={"wx.wind": "Wind speed in MPH"},
    )
    assert "Wind speed in MPH" in text


def test_the_prompt_asks_for_units_on_bare_numbers():
    system, _ = _prompt(build_grid_prompt)
    assert "suffix" in system.lower()
    assert "62F" in system or "unit" in system.lower()


def test_the_prompt_asks_for_one_coherent_theme():
    system, _ = _prompt(build_grid_prompt)
    assert "theme" in system.lower() or "coherent" in system.lower()


def test_the_prompt_forbids_repeating_the_title_in_a_tile():
    system, _ = _prompt(build_grid_prompt)
    assert "repeat" in system.lower() or "already in the title" in system.lower()


def test_prose_is_told_never_to_invent_direction():
    # "GOOG UP $339.08" — the number was real (the price) but UP was a lie.
    system, _ = _prompt(build_prose_prompt)
    lowered = system.lower()
    assert "direction" in lowered or "rose or fell" in lowered


def test_both_modes_are_told_to_skip_placeholder_values():
    for builder in (build_grid_prompt, build_prose_prompt):
        system, _ = _prompt(builder)
        assert "UNKNOWN" in system or "placeholder" in system.lower()


def test_the_default_timeout_tolerates_a_busy_local_model():
    # Off the render path, so waiting is free; a timeout wastes a whole cycle.
    assert DashboardLLM("http://x/v1", "k", "m", 0.3).timeout >= 60


def test_prose_is_given_a_target_length_not_just_a_cap():
    # "At most 112, be brief" produced 26-character stubs.
    system, _ = _prompt(build_prose_prompt)
    assert "aim for" in system.lower() or "two or three" in system.lower()


def test_the_grid_prompt_offers_named_design_patterns():
    system, _ = _prompt(build_grid_prompt)
    assert "PAGE" in system and "ALERT" in system


def test_the_grid_prompt_shows_a_two_row_header_in_its_example():
    system, _ = _prompt(build_grid_prompt)
    assert "subtitle" in system.lower()


def test_the_grid_prompt_extends_the_number_rule_to_headers():
    system, _ = _prompt(build_grid_prompt)
    lowered = system.lower()
    assert "banner" in lowered and ("digits" in lowered or "numbers" in lowered)


def test_label_budget_accounts_for_the_status_dot_column():
    # Telling the model label_max=4 and then reserving a dot cell is how
    # CHANGE became CHA on a live board. With color on, every column may
    # lose a cell to the dot, so the advertised budget must assume it.
    _, user_on = _prompt(build_grid_prompt, use_color=True,
                         current={"a.chg": "1.59%"}, refs=["a.chg"])
    _, user_off = _prompt(build_grid_prompt, use_color=False,
                          current={"a.chg": "1.59%"}, refs=["a.chg"])
    assert "label_max=3" in user_on
    assert "label_max=4" in user_off


def test_the_prompt_says_an_identifier_becomes_a_label():
    system, _ = _prompt(build_grid_prompt)
    assert 'label "GOOG"' in system or "as the label" in system.lower()


def test_the_prompt_explains_the_layout_control():
    system, _ = _prompt(build_grid_prompt)
    assert '"layout"' in system and "list" in system


def test_the_prompt_calls_a_date_tile_filler():
    # The header places the board in time; DATE 09/03/26 as a stat row is
    # dead weight unless time is the story.
    system, _ = _prompt(build_grid_prompt)
    assert "filler" in system.lower()


def test_a_trailing_comma_is_repaired_rather_than_rejected():
    # Gemma's most common JSON slip; a retry costs 3s, a regex costs nothing.
    fenced = '{"tiles": [{"label": "A", "variable": "a.b"},], "banner": "X",}'
    with patch("requests.post", return_value=_response(fenced)):
        result = _client().complete("sys", "user")
    assert result["banner"] == "X"
    assert len(result["tiles"]) == 1
