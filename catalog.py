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


def _build_choice(
    ref: str,
    group: str,
    description: str,
    preview: str,
    disabled: bool,
    reason: str,
) -> VariableChoice:
    """Assemble one picker row.

    The settings widget renders ``label`` and nothing else — it ignores
    ``group`` and ``preview`` entirely, and shows ``description`` only as a
    hover tooltip. So attribution has to be in the label, and the ref goes in
    the description because that is part of what the search box matches.
    """
    return VariableChoice(
        ref=ref,
        label=picker_label(ref, group),
        description=" · ".join(part for part in (preview, description, ref) if part),
        group=group,
        preview=preview,
        disabled=disabled,
        disabled_reason=reason,
    )


def _matches(choice: VariableChoice, needle: str) -> bool:
    """Whether *choice* satisfies the search box.

    Searches the label too, so a plugin's display name is findable: a user
    types "star trek", but the ref only ever says "star_trek_quotes".
    """
    if not needle:
        return True
    haystack = f"{choice.label} {choice.ref} {choice.description} {choice.disabled_reason}"
    return needle in haystack.lower()


def _unusable_reason(plugin_id: str, exclude_plugin_id: str, max_length: object, width: int) -> str:
    """Why this variable cannot be watched, or "" if it can."""
    if plugin_id == exclude_plugin_id:
        return "A dashboard cannot watch itself."
    if isinstance(max_length, int) and max_length > width:
        return f"Up to {max_length} characters — too wide for a {width}-cell tile."
    return ""


def variable_catalog(
    exclude_plugin_id: str,
    max_value_width: int,
    query: str = "",
) -> list[VariableChoice]:
    """Every variable the user could watch, including ones they cannot.

    Unsuitable variables come back ``disabled`` with a reason rather than
    omitted, so the picker can explain itself instead of silently hiding
    them. That includes variables belonging to plugins that are installed but
    switched off: core's catalog covers enabled plugins only, and you cannot
    decide whether to enable something if you cannot see what it offers.

    Usable options are listed first, so the ones you can actually pick are
    not buried behind an alphabetically luckier plugin you have turned off.
    """
    registry = _registry()
    metadata = registry.get_all_variables_with_metadata()
    needle = query.strip().lower()

    usable: list[VariableChoice] = []
    unusable: list[VariableChoice] = []

    for plugin_id in sorted(metadata):
        group = _plugin_name(registry, plugin_id)
        for name in sorted(metadata[plugin_id]):
            meta = metadata[plugin_id][name] or {}
            reason = _unusable_reason(
                plugin_id, exclude_plugin_id, meta.get("max_length"), max_value_width
            )
            choice = _build_choice(
                ref=f"{plugin_id}.{name}",
                group=group,
                description=str(meta.get("description") or ""),
                preview=str(meta.get("preview") or ""),
                disabled=bool(reason),
                reason=reason,
            )
            (unusable if reason else usable).append(choice)

    for plugin_id in sorted(_disabled_plugin_ids(registry, set(metadata))):
        manifest = registry.get_manifest(plugin_id)
        variables = getattr(manifest, "variables", None)
        if variables is None:
            continue
        group = getattr(manifest, "name", None) or plugin_id
        for name in sorted(getattr(variables, "simple", {}) or {}):
            meta = variables.get_variable_metadata(name)
            reason = _unusable_reason(
                plugin_id, exclude_plugin_id, getattr(meta, "max_length", None), max_value_width
            ) or f"Enable {group} to watch its variables."
            unusable.append(
                _build_choice(
                    ref=f"{plugin_id}.{name}",
                    group=group,
                    description=str(getattr(meta, "description", "") or ""),
                    preview="",
                    disabled=True,
                    reason=reason,
                )
            )

    return [c for c in usable + unusable if _matches(c, needle)]


def _disabled_plugin_ids(registry: Any, enabled: set[str]) -> set[str]:
    """Installed plugins that expose nothing because they are switched off."""
    try:
        installed = set(registry.plugins)
    except Exception:
        return set()
    return installed - enabled


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
