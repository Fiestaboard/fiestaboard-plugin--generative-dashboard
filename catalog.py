"""Reading the variable catalog and live values from the core registry.

Deliberately never calls ``registry.build_template_context()``: that fans out
to every enabled plugin on a thread pool, this plugin included, so calling it
from a fetch would re-enter this plugin on another thread. Fetching only the
plugins the watchlist names avoids the problem by construction.
"""

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
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


def _unusable_reason(
    plugin_id: str, exclude_plugin_id: str, max_length: object, width: int, name: str = ""
) -> str:
    """Why this variable cannot be watched, or "" if it can."""
    if plugin_id == exclude_plugin_id:
        return "A dashboard cannot watch itself."
    if name.endswith("_color"):
        # These hold a colour name for {{x_color}} tiles. As a stat they read
        # "FOG COL 66", which tells you nothing.
        return "A color code for template use, not a stat."
    if isinstance(max_length, int) and max_length > width:
        return f"Up to {max_length} characters — wider than the {width}-cell board."
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
                plugin_id, exclude_plugin_id, meta.get("max_length"), max_value_width, name
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
                plugin_id, exclude_plugin_id, getattr(meta, "max_length", None),
                max_value_width, name,
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


def eligible_refs(exclude_plugin_id: str, max_value_width: int) -> list[str]:
    """Every variable that could actually be put on a board right now.

    This is the default pool, and it runs on the render path, so it reads
    manifests only. It must never reach ``get_all_variables_with_metadata``:
    that calls ``build_template_context``, which dispatches to a thread pool
    that calls this plugin's own ``get_data`` again — an infinite recursion
    that exhausts the pool and hangs every API endpoint, not just this plugin.

    Eligibility needs no live values anyway. A variable's declared max_length
    is enough to know whether it could ever fit the board.
    """
    registry = _registry()
    try:
        enabled = list(registry.enabled_plugins)
    except Exception:
        return []

    refs: list[str] = []
    for plugin_id in sorted(enabled):
        if plugin_id == exclude_plugin_id:
            continue
        manifest = registry.get_manifest(plugin_id)
        variables = getattr(manifest, "variables", None)
        if variables is None:
            continue
        for name in sorted(getattr(variables, "simple", {}) or {}):
            meta = variables.get_variable_metadata(name)
            if _unusable_reason(
                plugin_id, exclude_plugin_id,
                getattr(meta, "max_length", None), max_value_width, name,
            ):
                continue
            refs.append(f"{plugin_id}.{name}")
    return refs


def prompt_groups(refs: "Sequence[str]") -> list[dict]:
    """Refs organized the way their plugin authors organized them.

    The model should see 'Air Quality & Fog', its one-line purpose, and the
    manifest's own variable groups ('Grass Pollen'), not a flat wall of
    cryptic ref prefixes — a reader cannot infer what muni.is_delayed means
    as fast as a section header can say it. Manifest-only, like everything on
    the generation path.

    Returns ``[{"plugin", "about", "vars": [(group_label, [refs])]}]`` with
    ungrouped refs first under an empty label, then groups in manifest order.
    """
    registry = _registry()
    by_plugin: dict[str, list[str]] = defaultdict(list)
    for ref in refs:
        plugin_id, _, name = ref.partition(".")
        if plugin_id and name:
            by_plugin[plugin_id].append(ref)

    sections: list[dict] = []
    for plugin_id in sorted(by_plugin):
        try:
            manifest = registry.get_manifest(plugin_id)
        except Exception:
            manifest = None
        variables = getattr(manifest, "variables", None)
        group_labels = {
            gid: getattr(group, "label", str(gid))
            for gid, group in (getattr(variables, "groups", None) or {}).items()
        }

        grouped: dict[str, list[str]] = defaultdict(list)
        for ref in by_plugin[plugin_id]:
            gid = ""
            if variables is not None:
                meta = variables.get_variable_metadata(ref.partition(".")[2])
                gid = str(getattr(meta, "group", "") or "")
            grouped[group_labels.get(gid, "") if gid else ""].append(ref)

        ordered: list[tuple[str, list[str]]] = []
        if grouped.get(""):
            ordered.append(("", grouped[""]))
        for gid in group_labels:
            label = group_labels[gid]
            if grouped.get(label):
                ordered.append((label, grouped[label]))

        sections.append({
            "plugin": getattr(manifest, "name", None) or plugin_id,
            "about": str(getattr(manifest, "description", "") or "")[:80],
            "vars": ordered or [("", by_plugin[plugin_id])],
        })
    return sections


def rotation_page_names(exclude_plugin_id: str) -> list[str]:
    """Names of the other pages this wall rotates through.

    Core keeps no history of what the board displayed, but the rotation's
    page names are stable and say what is already covered — a model that
    knows a dedicated weather page exists can stop composing one and surface
    what the rotation lacks. Pages built on this plugin are excluded so the
    dashboard never treats itself as competition.
    """
    try:
        from src.pages.storage import PageStorage

        pages = PageStorage().list_all()
    except Exception:
        return []

    names: list[str] = []
    for page in pages:
        try:
            blob = page.model_dump_json()
        except Exception:
            blob = ""
        if exclude_plugin_id in blob:
            continue
        name = str(getattr(page, "name", "") or "").strip()
        if name:
            names.append(name)
    return names[:30]


def variable_descriptions(refs: "Sequence[str]") -> dict[str, str]:
    """Manifest descriptions for *refs* — what each number means.

    'Wind speed in MPH' tells the model both the meaning and the unit, which
    is the difference between showing '7.2' and '7.2 MPH'. Manifest-only for
    the same reason as :func:`eligible_refs`: this runs on the generation
    path and must never fan out through the template context.
    """
    registry = _registry()
    by_plugin: dict[str, list[str]] = defaultdict(list)
    for ref in refs:
        plugin_id, _, name = ref.partition(".")
        if plugin_id and name:
            by_plugin[plugin_id].append(name)

    out: dict[str, str] = {}
    for plugin_id, names in by_plugin.items():
        try:
            manifest = registry.get_manifest(plugin_id)
        except Exception:
            continue
        variables = getattr(manifest, "variables", None)
        if variables is None:
            continue
        for name in names:
            meta = variables.get_variable_metadata(name)
            description = str(getattr(meta, "description", "") or "")
            if description:
                out[f"{plugin_id}.{name}"] = description
    return out


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
    timeout: float = 2.5,
    fallback: dict[str, str] | None = None,
) -> dict[str, str]:
    """Current values for *refs*, fetching each source plugin exactly once.

    Fetched concurrently under a deadline. With every plugin enabled this can
    be fifty-odd sources, several of them network-bound; done one after
    another that overruns the fifteen seconds core allows a plugin during
    ``build_template_context``, and an overrunning plugin is dropped from the
    render context entirely — every one of its variables then renders as
    "???" on the board.

    Whatever has not arrived by the deadline falls back to its last known
    value when one is supplied, which keeps the deadline short without losing
    a stat every time one source is slow. Refs whose plugin is unavailable, or whose
    variable is absent, are omitted rather than blanked — the gate reads that
    absence as a real change.
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
    executor = ThreadPoolExecutor(max_workers=min(8, len(wanted)))
    try:
        futures = {
            executor.submit(registry.fetch_plugin_data, plugin_id, board): plugin_id
            for plugin_id in wanted
        }
        done, not_done = futures_wait(futures, timeout=timeout)
        if not_done:
            logger.warning(
                "%d plugin(s) did not answer within %.1fs and were left out of "
                "the dashboard this cycle: %s",
                len(not_done), timeout, [futures[f] for f in not_done],
            )

        values: dict[str, str] = {}
        late = {futures[f] for f in not_done}
        if fallback:
            for ref, value in fallback.items():
                if ref.partition(".")[0] in late:
                    values[ref] = value
        for future in done:
            plugin_id = futures[future]
            try:
                result = future.result()
            except Exception:
                logger.warning("Failed reading %s for the dashboard", plugin_id, exc_info=True)
                continue
            if not getattr(result, "available", False) or not getattr(result, "data", None):
                continue
            for name in wanted[plugin_id]:
                raw = result.data.get(name)
                if raw is None or isinstance(raw, list | dict):
                    continue
                values[f"{plugin_id}.{name}"] = str(raw)
        return values
    finally:
        # Never block on stragglers; they finish into a discarded result.
        executor.shutdown(wait=False, cancel_futures=True)
