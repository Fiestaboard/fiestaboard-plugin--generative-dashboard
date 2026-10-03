"""Where the model comes from.

- FiestaBoard's AI providers (Settings -> AI Providers, pasted key or signed
  in) through ``PluginBase.ai_complete`` (FiestaBoard 9.11.0). ``ai_provider``
  picks one (blank = FiestaBot's default) and ``ai_model`` a model (blank =
  that provider's default).
- A separate API key: a saved ``api_key`` with ``api_base_url`` and
  ``model``, sent by the plugin's own client exactly as before. It wins
  whenever it is set, on any core.
"""

from __future__ import annotations

from typing import Any

from .llm import CoreLLM, DashboardLLM

UPDATE_MESSAGE = "Update FiestaBoard to use its AI providers, or paste an API key"


class ConnectionUnavailable(Exception):
    """No usable connection right now; the message says what to do."""


def uses_api_key(config: dict[str, Any]) -> bool:
    return bool(config.get("api_key"))


def core_complete(plugin: Any):
    """``plugin.ai_complete`` on FiestaBoard 9.11.0 or later, else None."""
    complete = getattr(plugin, "ai_complete", None)
    return complete if callable(complete) else None


def is_ready(config: dict[str, Any], plugin: Any) -> bool:
    """Cheap check for the render path: could a generation start now?"""
    return uses_api_key(config) or core_complete(plugin) is not None


def can_generate(config: dict[str, Any], plugin: Any) -> bool:
    """:func:`is_ready`, and FiestaBoard's AI has the provider to answer.

    The render path asks this before starting a worker. While AI is off, no
    provider is set up, or the chosen one was deleted, every attempt would
    fail at once, and a cold board renders constantly, so it would start a
    worker and log a warning on every render. Saving settings still only
    needs :func:`is_ready`: AI may be set up later.
    """
    if uses_api_key(config):
        return True
    if core_complete(plugin) is None:
        return False
    providers = getattr(plugin, "ai_providers", None)
    if not callable(providers):
        return True
    try:
        ids = {str(p.get("id")) for p in providers() or [] if isinstance(p, dict)}
    except Exception:  # noqa: BLE001 - unreadable settings: let ai_complete say why
        return True
    chosen = str(config.get("ai_provider") or "")
    return bool(ids) and (not chosen or chosen in ids)


def build_client(
    config: dict[str, Any], plugin: Any, *, temperature: float, max_tokens: int | None
):
    """The client to compose with. Raises :class:`ConnectionUnavailable`."""
    if uses_api_key(config):
        return DashboardLLM(
            base_url=str(config.get("api_base_url", "https://api.openai.com/v1")),
            api_key=str(config.get("api_key", "")),
            model=str(config.get("model", "gpt-4o-mini")),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    complete = core_complete(plugin)
    if complete is None:
        raise ConnectionUnavailable(UPDATE_MESSAGE)
    return CoreLLM(
        complete,
        provider_id=str(config.get("ai_provider") or "") or None,
        model=str(config.get("ai_model") or "").strip() or None,
        temperature=temperature,
        max_tokens=max_tokens,
    )
