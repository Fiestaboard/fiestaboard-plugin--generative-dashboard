"""Board geometry and character placement.

The plugin owns every character position. The model chooses what appears and
in what order; it never types spacing. That is what makes alignment identical
between cycles and width violations impossible.
"""

from dataclasses import dataclass

from .charset import cell_width, sanitize, truncate

# Minimum sensible width for a label/value pair. Below this a second column
# would leave no room for either half.
_MIN_TILE_WIDTH = 11

# Shortest label worth printing. Below this a label is a stub — "D" for DATE
# tells you nothing — so the tile is given the whole row instead.
_MIN_LABEL = 3

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


def _render_tile(tile: Tile, width: int, use_color: bool, reserve_dot: bool = False) -> str:
    """Render one tile into exactly *width* cells.

    Color is data, not label decoration: it renders as a status dot after the
    value, the way an indicator light sits beside a reading. When any tile in
    the grid is colored, *every* tile reserves the dot cell — presence of
    color must never change which column the numbers sit in.
    """
    dot = "{" + tile.color.lower() + "}" if (use_color and tile.color and reserve_dot) else ""

    inner = width - (1 if reserve_dot else 0)
    value = truncate(sanitize(tile.value), inner)
    # The label yields first: a shortened name beats a shortened number.
    label = truncate(sanitize(tile.label), max(0, inner - cell_width(value) - 1))
    gap = inner - cell_width(label) - cell_width(value)
    return label + (" " * max(0, gap)) + value + (dot or (" " if reserve_dot else ""))


def placed_count(
    tiles: list[Tile], geo: Geometry, banner: str = "", subtitle: str = ""
) -> int:
    """How many of *tiles* actually reach the board.

    Watching forty variables does not mean forty are shown; the board fits
    what it fits, and the count that matters is the one on screen.
    """
    rows = geo.rows - (1 if banner else 0) - (1 if banner and subtitle else 0)
    usable = [
        t for t in tiles
        if sanitize(t.value).strip() and fits_board(t.value, geo)
    ][: geo.tile_columns * rows]
    used = 0
    placed = 0
    for tile in usable:
        cost = geo.tile_columns if needs_full_row(tile.value, geo) else 1
        if used + cost > geo.tile_columns * rows:
            break
        used += cost
        placed += 1
    return placed


def render_banner(text: str, color: str | None, cols: int, weight: int = 2) -> str:
    """Centre a title, framed by color tiles when there is room for them.

    A double frame each side is what the best handmade pages use — it gives
    the title weight. Falls back to a single frame, then to plain text: if
    framing would cost a word, the words win.
    """
    body = truncate(sanitize(text), cols)
    if not body:
        return ""
    if color:
        marker = "{" + color.lower() + "}"
        for n in range(max(1, weight), 0, -1):
            framed = f"{marker * n} {body} {marker * n}"
            if cell_width(framed) <= cols:
                pad = (cols - cell_width(framed)) // 2
                return (" " * pad) + framed
    return body.center(cols).rstrip()


def render_grid(
    tiles: list[Tile],
    geo: Geometry,
    banner: str = "",
    use_color: bool = True,
    banner_color: str | None = None,
    subtitle: str = "",
) -> list[str]:
    """Place *tiles* into the board grid, returning exactly ``geo.rows`` lines."""
    lines: list[str] = []
    hue = banner_color if use_color else None
    banner_text = render_banner(banner, hue, geo.cols, weight=2)
    if banner_text:
        lines.append(banner_text)
        # A subtitle only makes sense beneath a title; framed lighter, the way
        # the reference page frames its date line under the city name.
        subtitle_text = render_banner(subtitle, hue, geo.cols, weight=1)
        if subtitle_text:
            lines.append(subtitle_text)

    grid_rows = geo.rows - len(lines)
    placed = list(tiles[: geo.tile_columns * grid_rows])

    # One rule for the whole board: a dot column exists for everyone or for
    # no one, so colored and plain rows keep their values aligned.
    reserve_dot = use_color and any(t.color for t in tiles)

    def flush(buffer: list[Tile]) -> None:
        # A row with one tile on a multi-column board spans the full width —
        # label left, value right — because a half-empty row in mid-board is
        # the single loudest tell that a layout was generated.
        if len(buffer) == 1 and geo.tile_columns > 1:
            lines.append(_render_tile(buffer[0], geo.cols, use_color, reserve_dot).rstrip())
            return
        cells = [
            # Every column but the last gives up one cell as a gutter.
            _render_tile(
                tile,
                geo.tile_width if column == geo.tile_columns - 1 else geo.tile_width - 1,
                use_color,
                reserve_dot,
            )
            for column, tile in enumerate(buffer)
        ]
        lines.append(" ".join(cells).rstrip())

    buffer: list[Tile] = []
    for tile in placed:
        if len(lines) >= geo.rows:
            break
        if not sanitize(tile.value).strip():
            # A label with nothing beside it is noise, not a stat.
            continue
        if not fits_board(tile.value, geo):
            continue
        # A value too long to leave room for a label gets the whole row, rather
        # than being labelled "D" and leaving the reader to guess.
        if needs_full_row(tile.value, geo):
            if buffer:
                flush(buffer)
                buffer = []
            if len(lines) < geo.rows:
                lines.append(_render_tile(tile, geo.cols, use_color, reserve_dot).rstrip())
            continue
        buffer.append(tile)
        if len(buffer) == geo.tile_columns:
            flush(buffer)
            buffer = []
    if buffer and len(lines) < geo.rows:
        flush(buffer)

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


def wrap_center(text: str, rows: int, cols: int) -> list[str]:
    """Wrap *text* and centre the block, returning exactly *rows* lines."""
    wrapped = _wrap(sanitize(text), cols)[:rows]
    top = (rows - len(wrapped)) // 2
    out = [""] * top + [line.center(cols).rstrip() for line in wrapped]
    return (out + [""] * rows)[:rows]
