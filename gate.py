"""Deciding whether anything actually happened.

This is the plugin's primary anti-jitter mechanism and its primary cost
control: when nothing has moved materially, no LLM call is made and the board
is left exactly as it is.
"""

# Guards the percentage division when the previous value was zero.
_EPSILON = 1e-9


def as_number(value: object) -> float | None:
    """Parse *value* as a number, tolerating grouping commas and a percent sign.

    Returns ``None`` when the value is not numeric, which the caller treats as
    "compare as text".
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None
    cleaned = value.strip().replace(",", "").rstrip("%").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_material(old: object, new: object, threshold_pct: float) -> bool:
    """Whether the move from *old* to *new* is worth redrawing the board for."""
    old_num = as_number(old)
    new_num = as_number(new)

    if old_num is None or new_num is None:
        return str(old) != str(new)

    delta = abs(new_num - old_num)
    if delta == 0:
        return False
    return (delta / max(abs(old_num), _EPSILON)) * 100 >= threshold_pct


def material_changes(
    old: dict[str, str],
    new: dict[str, str],
    thresholds: dict[str, float],
    default_pct: float,
) -> list[str]:
    """Return the variable refs that moved materially between two snapshots.

    A variable appearing or disappearing is always material — the board's
    contents change either way.
    """
    changed: list[str] = []
    for ref in sorted(set(old) | set(new)):
        if ref not in old or ref not in new:
            changed.append(ref)
            continue
        threshold = thresholds.get(ref, default_pct)
        if is_material(old[ref], new[ref], threshold):
            changed.append(ref)
    return changed
