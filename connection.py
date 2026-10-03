"""Where the model comes from.

- FiestaBoard's AI providers (Settings -> AI Providers, pasted key or signed
  in) through ``PluginBase.ai_complete`` (FiestaBoard 9.9.0). ``ai_provider``
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
    """``plugin.ai_complete`` on FiestaBoard 9.9.0 or later, else None."""
    complete = getattr(plugin, "ai_complete", None)
    return complete if callable(complete) else None


def is_ready(config: dict[str, Any], plugin: Any) -> bool:
    """Cheap check for the render path: could a generation start now?"""
    return uses_api_key(config) or core_complete(plugin) is not None


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
