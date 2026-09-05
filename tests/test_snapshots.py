"""Board VRTs: golden whole-board snapshots, the way FiestaUI snapshots pixels.

The renderer is deterministic, so a fixed composition must reproduce these
boards byte for byte. Goldens were captured from real renderer output and
eyeballed before freezing. Read a diff here like a failed visual test: if
the new board is *better*, update the golden and say why in the commit.
"""

from plugins.generative_dashboard.layout import Tile, geometry, render_grid, wrap_center

FLAGSHIP = geometry(6, 22)
NOTE = geometry(3, 15)


def test_golden_page_with_header_dots_and_pairs():
    tiles = [
        Tile("NOW", "62F", "blue"),
        Tile("RAIN", "87%"),
        Tile("LIKE", "61F"),
        Tile("WIND", "7.2MPH"),
    ]
    got = render_grid(tiles, FLAGSHIP, banner="SAN FRANCISCO",
                      banner_color="blue", subtitle="LIGHT RAIN, 8 AM")
    assert got == [
        "",
        " {blue}{blue} SAN FRANCISCO {blue}{blue}",
        " {blue} LIGHT RAIN, 8 AM {blue}",
        "NOW   62F{blue} RAIN   87%",
        # WIND clips to WIN: the model ignored the advertised label budget
        # and the renderer truncated honestly rather than overflow.
        "LIKE  61F  WIN 7.2MPH",
        "",
    ]


def test_golden_ledger_board():
    tiles = [
        Tile("GOOG", "$335.31", "red"),
        Tile("CHANGE", "-1.11%", "red"),
        Tile("EUR", "0.8604"),
    ]
    got = render_grid(tiles, FLAGSHIP, banner="MARKET WATCH",
                      banner_color="red", layout="list")
    assert got == [
        "",
        "  {red}{red} MARKET WATCH {red}{red}",
        "GOOG          $335.31{red}",
        "CHANGE         -1.11%{red}",
        "EUR            0.8604",
        "",
    ]


def test_golden_note_board():
    got = render_grid([Tile("TEMP", "62F"), Tile("AQI", "43", "green")], NOTE)
    assert got == [
        "TEMP       62F",
        "AQI         43{green}",
        "",
    ]


def test_golden_prose_wrap():
    got = wrap_center(
        "AQI JUMPED FROM 31 TO 168 THIS AFTERNOON. KEEP THE WINDOWS SHUT.",
        6, 22,
    )
    assert got == [
        "",
        "AQI JUMPED FROM 31 TO",
        " 168 THIS AFTERNOON.",
        "KEEP THE WINDOWS SHUT.",
        "",
        "",
    ]
