"""Reading the variable catalog and live values from the core registry.

Deliberately never calls ``registry.build_template_context()``: that fans out
to every enabled plugin on a thread pool, this plugin included, so calling it
from a fetch would re-enter this plugin on another thread. Fetching only the
plugins the watchlist names avoids the problem by construction.
"""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariableChoice:
    """One selectable variable, as shown in the settings picker."""

    ref: str
    label: str
    description: str
    group: str
    preview: str
    disabled: bool = False
    disabled_reason: str = ""


def _registry() -> Any:
    """The core plugin registry. Indirected so tests can replace it."""
    from src.plugins import get_plugin_registry

    return get_plugin_registry()


def default_label(ref: str) -> str:
    """Board label for a ``plugin.var`` ref when the user has not set one.

    Underscores have no flap on the board, so they become spaces, and the
    board is uppercase-only.
    """
    return ref.split(".", 1)[-1].replace("_", " ").upper()


def readable_name(ref: str) -> str:
    """The variable's name in title case, for reading rather than for the board."""
    return ref.split(".", 1)[-1].replace("_", " ").title()


def plugin_display_name(plugin_id: str) -> str:
    """Display name for one plugin. Cheap: a manifest lookup, no data fetch."""
    return _plugin_name(_registry(), plugin_id)


def picker_label(ref: str, plugin_name: str, custom: str = "") -> str:
    """How a variable reads in the settings picker.

    The variable comes first because the chosen-row list truncates on overflow
    — leading with the plugin name would cut off the half that distinguishes
    one row from another.
    """
    return f"{custom or readable_name(ref)} ({plugin_name})"


def _plugin_name(registry: Any, plugin_id: str) -> str:
    """Display name of a plugin, falling back to its id."""
    try:
        manifest = registry.get_manifest(plugin_id)
    except Exception:
        return plugin_id
    return getattr(manifest, "name", None) or plugin_id


def variable_catalog(
    exclude_plugin_id: str,
    max_value_width: int,
    query: str = "",
) -> list[VariableChoice]:
    """Every variable the user could watch, including ones they should not.

    Unsuitable variables come back ``disabled`` with a reason rather than
    omitted, so the picker can explain itself instead of silently hiding them.
    """
    registry = _registry()
    metadata = registry.get_all_variables_with_metadata()
    needle = query.strip().lower()

    choices: list[VariableChoice] = []
    for plugin_id in sorted(metadata):
        group = _plugin_name(registry, plugin_id)
        for name in sorted(metadata[plugin_id]):
            meta = metadata[plugin_id][name] or {}
            ref = f"{plugin_id}.{name}"
            description = str(meta.get("description") or "")
            label = picker_label(ref, group)

            # Search the label too, so the plugin's display name is findable:
            # a user types "star trek", the ref only says "star_trek_quotes".
            if needle and needle not in f"{label} {ref} {description}".lower():
                continue

            disabled = False
            reason = ""
            if plugin_id == exclude_plugin_id:
                disabled = True
                reason = "A dashboard cannot watch itself."
            else:
                max_length = meta.get("max_length")
                if isinstance(max_length, int) and max_length > max_value_width:
                    disabled = True
                    reason = (
                        f"Up to {max_length} characters — too wide for a "
                        f"{max_value_width}-cell tile."
                    )

            preview = str(meta.get("preview") or "")
            # The settings widget renders `label` and nothing else — it ignores
            # `group` and `preview` entirely, and shows `description` only as a
            # hover tooltip. So attribution has to be in the label, or two
            # plugins that both expose "temp" become indistinguishable. The
            # search box filters on label + description, which is why the ref
            # goes in the description: it makes "temp_f" findable.
            detail = " · ".join(part for part in (preview, description, ref) if part)

            choices.append(
                VariableChoice(
                    ref=ref,
                    label=label,
                    description=detail,
                    group=group,
                    preview=preview,
                    disabled=disabled,
                    disabled_reason=reason,
                )
            )
    return choices


def read_values(
    refs: Sequence[str],
    board: Any,
    exclude_plugin_id: str,
) -> dict[str, str]:
    """Current values for *refs*, fetching each source plugin exactly once.

    Refs whose plugin is unavailable, or whose variable is absent, are omitted
    rather than blanked — the gate reads that absence as a real change.
    """
    wanted: dict[str, list[str]] = defaultdict(list)
    for ref in refs:
        plugin_id, _, name = ref.partition(".")
        if not plugin_id or not name or plugin_id == exclude_plugin_id:
            continue
        wanted[plugin_id].append(name)

    if not wanted:
        return {}

    registry = _registry()
    values: dict[str, str] = {}
    for plugin_id, names in wanted.items():
        try:
            result = registry.fetch_plugin_data(plugin_id, board)
        except Exception:
            logger.warning("Failed reading %s for the dashboard", plugin_id, exc_info=True)
            continue
        if not getattr(result, "available", False) or not getattr(result, "data", None):
            continue
        for name in names:
            raw = result.data.get(name)
            if raw is None or isinstance(raw, list | dict):
                continue
            values[f"{plugin_id}.{name}"] = str(raw)
    return values
