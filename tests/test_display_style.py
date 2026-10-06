"""The dashboard draws for the display it is on, not for a Vestaboard.

FiestaBoard 10 tells a plugin what the board's display can draw
(``self.board.display``). These tests fake a full-colour LED display (a Divoom
Pixoo 64: lowercase, coloured text, highlighted backgrounds, icons) with the
same duck-typed surface core's DisplayProfile has, and check that a split-flap
board — or a core that says nothing — gets exactly what it always got.
"""

from src.devices import BoardContext

from plugins.generative_dashboard import GenerativeDashboardPlugin
from plugins.generative_dashboard.charset import FLAP, sanitize, style_for
from plugins.generative_dashboard.layout import Tile, geometry, render_banner, render_grid
from plugins.generative_dashboard.llm import _CHARSET_RULES, build_grid_prompt, build_prose_prompt

LED_BRIEF = "THE DISPLAY: a full-colour LED pixel display.\nLowercase is drawn as lowercase."
LED_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$()-+&=;:'\"%,./?°♥"


class FakeLedDisplay:
    """What core's DisplayProfile answers for a Pixoo 64."""

    technology = "led_matrix"
    color = "rgb"
    chars = LED_CHARS
    key = "divoom_pixoo64|led_3x5|rgb|gap"

    def supports(self, feature):
        return feature in {"lowercase", "color_text", "background", "tiles", "icons", "rgb"}

    def ai_brief(self):
        return LED_BRIEF


class FakeFlapDisplay(FakeLedDisplay):
    technology = "split_flap"
    color = "tiles"
    key = "vestaboard_flagship|vestaboard_v1|tiles|None"

    def supports(self, feature):
        return feature == "tiles"

    def ai_brief(self):
        return "THE DISPLAY: a split-flap board."


LED = style_for(FakeLedDisplay())
PIXOO = geometry(10, 16)
TILES = [Tile("Temp", "63F", "green"), Tile("Wind", "6.3mph", "green"), Tile("AQI", "60", "yellow")]


# ── what a display gets ──────────────────────────────────────────────────────


def test_no_display_and_a_split_flap_display_both_get_the_split_flap_style():
    assert style_for(None) == FLAP
    flap = style_for(FakeFlapDisplay())
    assert not (flap.color_text or flap.mixed_case)


def test_an_led_display_keeps_lowercase_and_its_own_characters():
    assert sanitize("Night walk 63°F ♥", LED) == "Night walk 63°F ♥"
    assert sanitize("Night walk 63°F ♥") == "NIGHT WALK 63F "  # a flap has no degree or heart flap here


# ── how it is drawn ──────────────────────────────────────────────────────────


def test_on_an_led_display_a_level_colours_the_value_itself_not_a_square_beside_it():
    lines = [line for line in render_grid(TILES, PIXOO, style=LED) if line.strip()]
    # No cell is reserved for a square: the value runs to the right edge.
    assert lines[0] == "Temp         {green:63F}"
    assert lines[2] == "AQI           {yellow:60}"
    assert not any(line.rstrip().endswith(("{green}", "{yellow}")) for line in lines)


def test_on_a_split_flap_board_a_level_is_still_a_status_square():
    lines = [line for line in render_grid(TILES, PIXOO) if line.strip()]
    assert lines[0] == "TEMP        63F{green}"


def test_on_an_led_display_the_title_is_set_in_its_colour_unframed():
    assert render_banner("Night walk", "blue", 16, style=LED) == "   {blue:Night walk}"
    assert render_banner("Night walk", "blue", 16) == "{blue}{blue} NIGHT WALK {blue}{blue}"


# ── what the model is told ───────────────────────────────────────────────────


def _grid_prompt(**kw):
    return build_grid_prompt(
        geo=PIXOO, refs=["wx.temp"], labels={}, notes={}, current={"wx.temp": "63"},
        previous={}, previous_board=[], use_color=True, extra_instructions="", **kw,
    )[0]


def test_an_led_prompt_describes_the_led_with_cores_own_brief_and_no_flap_rules():
    system = _grid_prompt(style=LED)
    assert "stats dashboard for a full-colour LED pixel display" in system
    assert LED_BRIEF in system and _CHARSET_RULES not in system
    assert "colors the value itself" in system and "status tile beside the value" not in system
    assert "Now" in system and "NOW " not in system  # the worked example is drawn in the display's case


def test_a_split_flap_prompt_is_byte_for_byte_what_it_was():
    assert _grid_prompt(style=FLAP) == _grid_prompt()
    assert "stats dashboard for a split-flap board" in _grid_prompt()
    assert _CHARSET_RULES in _grid_prompt()


def test_a_prose_prompt_also_describes_the_display():
    system = build_prose_prompt(
        geo=PIXOO, refs=["wx.temp"], labels={}, notes={}, current={"wx.temp": "63"},
        previous={}, previous_board=[], extra_instructions="", style=LED,
    )[0]
    assert "status summary for a full-colour LED pixel display" in system and LED_BRIEF in system


# ── one composition per display ──────────────────────────────────────────────


def test_two_panels_of_different_sizes_or_displays_keep_their_own_composition(manifest):
    plugin = GenerativeDashboardPlugin(manifest)
    keys = set()
    for board in (
        BoardContext("panel", rows=10, cols=16),
        BoardContext("panel", rows=9, cols=22),
    ):
        with plugin._bound_board(board):
            keys.add(plugin._state_key())
    assert keys == {"panel:16x10", "panel:22x9"}


def test_the_state_key_names_the_display_when_core_says_what_it_is(manifest):
    plugin = GenerativeDashboardPlugin(manifest)
    board = BoardContext("panel", rows=10, cols=16)
    object.__setattr__(board, "display", FakeLedDisplay())  # a 10.0 core's BoardContext carries it
    with plugin._bound_board(board):
        assert plugin._state_key() == "panel:16x10|divoom_pixoo64|led_3x5|rgb|gap"
        assert plugin._style() == LED


def test_a_flagship_keeps_its_old_key(manifest):
    plugin = GenerativeDashboardPlugin(manifest)
    with plugin._bound_board(BoardContext("flagship", rows=6, cols=22)):
        assert plugin._state_key() == "flagship"
