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


def _render_tile(tile: Tile, width: int, use_color: bool) -> str:
    """Render one tile into exactly *width* cells."""
    prefix = "{" + tile.color.lower() + "}" if (use_color and tile.color) else ""

    inner = width - cell_width(prefix)
    value = truncate(sanitize(tile.value), inner)
    # The label yields first: a shortened name beats a shortened number.
    label = truncate(sanitize(tile.label), max(0, inner - cell_width(value) - 1))
    gap = inner - cell_width(label) - cell_width(value)
    return prefix + label + (" " * max(0, gap)) + value


def render_grid(
    tiles: list[Tile],
    geo: Geometry,
    banner: str = "",
    use_color: bool = True,
) -> list[str]:
    """Place *tiles* into the board grid, returning exactly ``geo.rows`` lines."""
    lines: list[str] = []
    banner_text = truncate(sanitize(banner), geo.cols) if banner else ""
    if banner_text:
        lines.append(banner_text.center(geo.cols).rstrip())

    grid_rows = geo.rows - len(lines)
    placed = list(tiles[: geo.tile_columns * grid_rows])

    for row in range(grid_rows):
        chunk = placed[row * geo.tile_columns : (row + 1) * geo.tile_columns]
        if not chunk:
            lines.append("")
            continue
        cells: list[str] = []
        for column, tile in enumerate(chunk):
            # Every column but the last gives up one cell as a gutter.
            is_last = column == geo.tile_columns - 1
            cells.append(
                _render_tile(tile, geo.tile_width if is_last else geo.tile_width - 1, use_color)
            )
        lines.append(" ".join(cells).rstrip())

    return lines[: geo.rows]


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
