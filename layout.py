"""Board geometry and character placement.

The plugin owns every character position. The model chooses what appears and
in what order; it never types spacing. That is what makes alignment identical
between cycles and width violations impossible.
"""

from dataclasses import dataclass

from .charset import FLAP, Style, cell_width, sanitize, truncate

# Minimum sensible width for a label/value pair. Below this a second column
# would leave no room for either half.
_MIN_TILE_WIDTH = 11

# Shortest label worth printing. Below this a label is a stub — "D" for DATE
# tells you nothing — so the tile is given more than one column instead.
_MIN_LABEL = 3

# How many tile columns one ledger cell spans by default. Two is the shape the
# Flagship established — on a 22-cell board two columns *is* the whole width —
# and it stays two however wide the board gets. Spanning the full width of a
# 120-cell panel would put the label in cell 1 and its value in cell 115 with a
# hundred dead cells between: not a row, but two rows sharing a line.
_LEDGER_COLUMNS = 2

# Word wrapping wastes ragged-right space, so the advertised prose budget is
# discounted. The real check is fits(), which actually wraps.
_PROSE_FILL = 0.85


@dataclass(frozen=True)
class Geometry:
    """Derived layout numbers for one board size."""

    rows: int
    cols: int
    tile_columns: int
    tile_width: int
    tile_budget: int
    prose_budget: int


@dataclass(frozen=True)
class Tile:
    """One label/value stat, optionally accented with a colour tile."""

    label: str
    value: str
    color: str | None = None


def column_inner(geo: "Geometry") -> int:
    """Usable width of one tile column, after the gutter is taken."""
    return geo.tile_width - 1 if geo.tile_columns > 1 else geo.tile_width


def fits_board(value: str, geo: "Geometry") -> bool:
    """Whether *value* can be shown at all without being cut.

    A truncated value is a wrong value — "123,456,789.0123" becoming
    "123,456,789.012" puts a number on the board that was never true. The tile
    is dropped instead.
    """
    return cell_width(sanitize(value)) <= geo.cols


def needs_full_row(value: str, geo: "Geometry") -> bool:
    """Whether *value* leaves too little room for a real label in one column."""
    if geo.tile_columns == 1:
        return False
    return cell_width(sanitize(value)) > column_inner(geo) - _MIN_LABEL - 1


def geometry(rows: int, cols: int) -> Geometry:
    """Compute the layout budget for a board of *rows* x *cols*."""
    tile_columns = max(1, cols // _MIN_TILE_WIDTH)
    tile_width = cols // tile_columns
    return Geometry(
        rows=rows,
        cols=cols,
        tile_columns=tile_columns,
        tile_width=tile_width,
        tile_budget=tile_columns * rows,
        prose_budget=int(rows * cols * _PROSE_FILL),
    )


def _render_tile(
    tile: Tile, width: int, use_color: bool, reserve_dot: bool = False, style: Style = FLAP
) -> str:
    """Render one tile into exactly *width* cells.

    Color is data, not label decoration. On a split-flap board it renders as
    a status dot after the value, the way an indicator light sits beside a
    reading; when any tile in the grid is colored, *every* tile reserves the
    dot cell — presence of color must never change which column the numbers
    sit in. On a display that draws coloured text the value itself takes the
    colour, so no cell is spent on a dot at all.
    """
    dot = "{" + tile.color.lower() + "}" if (use_color and tile.color and reserve_dot) else ""

    inner = width - (1 if reserve_dot else 0)
    value = truncate(sanitize(tile.value, style), inner)
    # The label yields first: a shortened name beats a shortened number.
    label = truncate(sanitize(tile.label, style), max(0, inner - cell_width(value) - 1))
    gap = inner - cell_width(label) - cell_width(value)
    if style.color_text and use_color and tile.color and value:
        # Widths were measured on the plain text; the span costs no cells.
        value = "{" + tile.color.lower() + ":" + value + "}"
    return label + (" " * max(0, gap)) + value + (dot or (" " if reserve_dot else ""))


def render_banner(text: str, color: str | None, cols: int, weight: int = 2, style: Style = FLAP) -> str:
    """Centre a title, framed by color tiles when there is room for them.

    A double frame each side is what the best handmade pages use — it gives
    the title weight. Falls back to a single frame, then to plain text: if
    framing would cost a word, the words win. On a display that draws
    coloured text the title is simply set in its colour, unframed.
    """
    body = truncate(sanitize(text, style), cols)
    if not body:
        return ""
    if style.color_text:
        pad = (cols - cell_width(body)) // 2
        return (" " * pad) + ("{" + color.lower() + ":" + body + "}" if color else body)
    if color:
        marker = "{" + color.lower() + "}"
        for n in range(max(1, weight), 0, -1):
            framed = f"{marker * n} {body} {marker * n}"
            if cell_width(framed) <= cols:
                pad = (cols - cell_width(framed)) // 2
                return (" " * pad) + framed
    return body.center(cols).rstrip()


def ledger_span(tiles: list[Tile], geo: Geometry) -> int:
    """How many tile columns one ledger cell occupies.

    One span serves the whole ledger section, so its values stay in a single
    straight column the way a handmade prices page lines its numbers up. It is
    :data:`_LEDGER_COLUMNS` columns wide by default, widened only when some
    value genuinely needs the room, and never wider than the board.
    """
    if geo.tile_columns <= 1:
        return 1
    needed = max(
        (cell_width(sanitize(t.value)) + _MIN_LABEL + 1 for t in tiles),
        default=0,
    )
    span = max(_LEDGER_COLUMNS, -(-needed // geo.tile_width))
    return min(geo.tile_columns, span)


def _ledger_rows(tiles: list[Tile], geo: Geometry) -> list[tuple[int, list[Tile]]]:
    """Chunk ledger tiles into rows of as many ledger cells as the board holds.

    On a Flagship a ledger row holds exactly one pair, because two columns is
    the whole board. On a wide panel the same rule puts five or six pairs side
    by side instead of stranding one pair per row — same shape, more of it.
    """
    if not tiles:
        return []
    span = ledger_span(tiles, geo)
    per_row = max(1, geo.tile_columns // span)
    return [(span, tiles[i : i + per_row]) for i in range(0, len(tiles), per_row)]


def ledger_cell_width(value: str, geo: Geometry) -> int:
    """Cells a ledger row would give one label/value pair carrying *value*.

    The prompt advertises label room from this, so it has to be the number
    the renderer will actually use. Advertising the board's full width on a
    120-cell panel promised a label a hundred cells that the two-column
    ledger cell never had.
    """
    span = ledger_span([Tile(label="", value=value)], geo)
    return _cell_widths(span, 1, geo)[0]


def _cell_widths(span: int, count: int, geo: Geometry) -> list[int]:
    """Exact cell widths for a row of *count* cells, each *span* columns wide.

    Every cell but the last gives up one cell as a gutter. A row that reaches
    the board's last column hands the remainder to its final cell, so the row
    ends on the right edge instead of leaving the columns that did not divide
    evenly dark. A partial row stays left-packed on the column grid.
    """
    widths = [span * geo.tile_width - 1] * count
    if count and span * count >= geo.tile_columns:
        widths[-1] = geo.cols - sum(widths[:-1]) - (count - 1)
    return widths


def _pack(
    tiles: list[Tile], geo: Geometry, layout: str = "auto"
) -> list[tuple[int, list[Tile]]]:
    """Group tiles into rows with a single rhythm.

    A human never alternates row shapes mid-board: the handmade weather page
    is all pairs, the handmade stocks page is all ledger rows. So rows of the
    same shape are gathered into sections — the model's first tile decides
    which section leads — and ``layout="list"`` forces the all-ledger shape
    outright. Narrow tiles left over from the last pair row join the ledger
    section rather than leaving a half-empty row anywhere.

    Each row is returned as ``(span, tiles)``: how many tile columns one cell
    of that row occupies, and the tiles in it.
    """
    usable = [
        t for t in tiles
        if sanitize(t.value).strip() and fits_board(t.value, geo)
    ]
    if not usable:
        return []
    if geo.tile_columns == 1:
        return [(1, [t]) for t in usable]
    if layout == "list":
        return _ledger_rows(usable, geo)

    wide = [t for t in usable if needs_full_row(t.value, geo)]
    narrow = [t for t in usable if not needs_full_row(t.value, geo)]

    whole = len(narrow) - len(narrow) % geo.tile_columns
    pairs = [
        (1, narrow[i : i + geo.tile_columns])
        for i in range(0, whole, geo.tile_columns)
    ]
    ledger = _ledger_rows(wide + narrow[whole:], geo)

    if wide and needs_full_row(usable[0].value, geo):
        return ledger + pairs
    return pairs + ledger


def placed_count(
    tiles: list[Tile], geo: Geometry, banner: str = "", subtitle: str = "",
    layout: str = "auto",
) -> int:
    """How many of *tiles* actually reach the board."""
    rows = geo.rows - (1 if banner else 0) - (1 if banner and subtitle else 0)
    return sum(len(row) for _, row in _pack(tiles, geo, layout)[: max(0, rows)])


def render_grid(
    tiles: list[Tile],
    geo: Geometry,
    banner: str = "",
    use_color: bool = True,
    banner_color: str | None = None,
    subtitle: str = "",
    layout: str = "auto",
    style: Style = FLAP,
) -> list[str]:
    """Place *tiles* into the board grid, returning exactly ``geo.rows`` lines."""
    lines: list[str] = []
    hue = banner_color if use_color else None
    banner_text = render_banner(banner, hue, geo.cols, weight=2, style=style)
    if banner_text:
        lines.append(banner_text)
        # A subtitle only makes sense beneath a title; framed lighter, the way
        # the reference page frames its date line under the city name. On a
        # coloured-text display it is plain: the title carries the colour.
        subtitle_text = render_banner(subtitle, None if style.color_text else hue, geo.cols, weight=1, style=style)
        if subtitle_text:
            lines.append(subtitle_text)

    grid_rows = geo.rows - len(lines)
    packed = _pack(tiles, geo, layout)[: max(0, grid_rows)]
    reserve_dot = use_color and not style.color_text and any(t.color for _, row in packed for t in row)

    for span, row in packed:
        cells = [
            _render_tile(tile, width, use_color, reserve_dot, style)
            for tile, width in zip(row, _cell_widths(span, len(row), geo), strict=True)
        ]
        lines.append(" ".join(cells).rstrip())

    # Centre the block, banner included, rather than letting it cling to the
    # top with dead rows beneath it — that reads as a bug rather than a layout.
    top = (geo.rows - len(lines)) // 2
    return ([""] * top + lines + [""] * geo.rows)[: geo.rows]


def _wrap(text: str, cols: int) -> list[str]:
    """Greedy word wrap at *cols* cells, breaking over-long words."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if cell_width(candidate) <= cols:
            current = candidate
            continue
        if current:
            lines.append(current)
        while cell_width(word) > cols:
            lines.append(truncate(word, cols))
            word = word[cols:]
        current = word
    if current:
        lines.append(current)
    return lines


def fits(text: str, rows: int, cols: int) -> bool:
    """Whether *text* wraps into at most *rows* lines of *cols* cells."""
    return len(_wrap(sanitize(text), cols)) <= rows


def wrap_center(text: str, rows: int, cols: int, style: Style = FLAP) -> list[str]:
    """Wrap *text* and centre the block, returning exactly *rows* lines."""
    wrapped = _wrap(sanitize(text, style), cols)[:rows]
    top = (rows - len(wrapped)) // 2
    out = [""] * top + [line.center(cols).rstrip() for line in wrapped]
    return (out + [""] * rows)[:rows]
