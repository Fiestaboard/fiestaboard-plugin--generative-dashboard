"""Where the model connection comes from.

Three sources, picked by the ``llm_source`` setting:

- ``api_key`` (the default, and every config written before 1.27): the
  pasted ``api_key`` and ``api_base_url``, exactly as always.
- ``openrouter``: the key from this plugin's OpenRouter sign-in
  (``get_oauth_token()``, FiestaBoard 9.9.0 ``key_exchange`` flow).
- ``fiestabot``: one of the AI providers set up for FiestaBot in
  Settings -> AI, pasted key or signed in, as long as it speaks the
  OpenAI chat-completions protocol.

Core imports are lazy and guarded, so the module loads on any core.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

SOURCES = ("api_key", "openrouter", "fiestabot")
DEFAULT_SOURCE = "api_key"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_PROTOCOLS = ("", "openai")


class ConnectionUnavailable(Exception):
    """No usable connection right now; the message says what to do."""


@dataclass(frozen=True)
class Endpoint:
    source: str
    base_url: str
    api_key: str
    model: str
    # Called on a 401. Returns a fresh key to retry with once, or None.
    on_rejected: Callable[[], str | None] = field(default=lambda: None, compare=False)


def source_of(config: dict[str, Any]) -> str:
    value = config.get("llm_source")
    return value if value in SOURCES else DEFAULT_SOURCE


def _model(config: dict[str, Any]) -> str:
    return str(config.get("model") or DEFAULT_MODEL)


# -- core hooks (patched in tests) -------------------------------------------


def _ai_providers() -> dict[str, Any]:
    try:
        from src.config_manager import get_config_manager

        return get_config_manager().get_ai_providers() or {}
    except Exception:  # an older core, or no config yet
        logger.debug("FiestaBot providers unavailable", exc_info=True)
        return {}


def _resolve_provider_auth(provider: dict[str, Any]) -> dict[str, Any]:
    """Core's resolver (9.9.0); before that only pasted-key providers exist."""
    try:
        from src.ai.sign_in import resolve_provider_auth
    except ImportError:
        if provider.get("sign_in"):
            raise ConnectionUnavailable(
                "That FiestaBot provider uses sign-in, which needs FiestaBoard 9.9 or later."
            ) from None
        return provider
    return resolve_provider_auth(provider)


def _report_ai_rejected(provider_id: str) -> str | None:
    try:
        from src.ai.sign_in import connection_id_for
        from src.oauth.service import get_oauth_service

        return get_oauth_service().report_rejected(connection_id_for(provider_id))
    except Exception:
        logger.debug("Could not report a rejected FiestaBot sign-in", exc_info=True)
        return None


# -- FiestaBot providers ------------------------------------------------------


def fiestabot_providers() -> list[dict[str, Any]]:
    providers = _ai_providers().get("providers") or []
    return [p for p in providers if isinstance(p, dict) and p.get("id")]


def is_openai_compatible(provider: dict[str, Any]) -> bool:
    return str(provider.get("protocol") or "") in OPENAI_PROTOCOLS


def pick_provider(config: dict[str, Any]) -> dict[str, Any]:
    providers = fiestabot_providers()
    if not providers:
        raise ConnectionUnavailable(
            "No FiestaBot AI provider is set up. Add one in Settings → AI."
        )
    wanted = str(config.get("fiestabot_provider") or "")
    if wanted:
        for provider in providers:
            if provider["id"] == wanted:
                return provider
        raise ConnectionUnavailable(
            "The chosen FiestaBot AI provider is gone. Pick another in this plugin's settings."
        )
    default_id = _ai_providers().get("default_provider_id")
    for provider in providers:
        if provider["id"] == default_id:
            return provider
    return providers[0]


def _fiestabot(config: dict[str, Any]) -> Endpoint:
    provider = pick_provider(config)
    name = provider.get("name") or provider["id"]
    if not is_openai_compatible(provider):
        raise ConnectionUnavailable(
            f"FiestaBot provider {name!r} is not OpenAI-compatible. Pick another."
        )
    try:
        ready = _resolve_provider_auth(provider)
    except ConnectionUnavailable:
        raise
    except Exception as exc:  # core's AIGenerationError: signed out
        raise ConnectionUnavailable(str(exc)) from exc
    models = ready.get("models") or []
    model = ready.get("default_model") or (models[0] if models else "") or _model(config)
    signed_in = bool(provider.get("sign_in"))
    provider_id = str(provider["id"])
    return Endpoint(
        source="fiestabot",
        base_url=str(ready.get("base_url") or DEFAULT_BASE_URL),
        api_key=str(ready.get("api_key") or ""),
        model=str(model),
        on_rejected=(lambda: _report_ai_rejected(provider_id)) if signed_in else (lambda: None),
    )


# -- entry point ------------------------------------------------------------------


def resolve(config: dict[str, Any], plugin: Any) -> Endpoint:
    """The endpoint to call now. Raises :class:`ConnectionUnavailable`."""
    source = source_of(config)
    if source == "openrouter":
        get_token = getattr(plugin, "get_oauth_token", None)
        if get_token is None:
            raise ConnectionUnavailable("OpenRouter sign-in needs FiestaBoard 9.9 or later.")
        token = get_token()
        if not token:
            raise ConnectionUnavailable("Sign in to OpenRouter in this plugin's settings.")
        model = _model(config)
        report = getattr(plugin, "report_oauth_rejected", None)
        return Endpoint(
            source="openrouter",
            base_url=OPENROUTER_BASE_URL,
            api_key=str(token),
            # OpenRouter names models vendor/model; a bare name is OpenAI's.
            model=model if "/" in model else f"openai/{model}",
            on_rejected=report if report is not None else (lambda: None),
        )
    if source == "fiestabot":
        return _fiestabot(config)
    return Endpoint(
        source="api_key",
        base_url=str(config.get("api_base_url", DEFAULT_BASE_URL)),
        api_key=str(config.get("api_key", "")),
        model=str(config.get("model", DEFAULT_MODEL)),
    )


def is_ready(config: dict[str, Any], plugin: Any) -> bool:
    """Cheap check for the render path: could a generation start now?"""
    source = source_of(config)
    if source == "api_key":
        return bool(config.get("api_key"))
    if source == "openrouter":
        get_token = getattr(plugin, "get_oauth_token", None)
        return bool(get_token and get_token())
    try:
        provider = pick_provider(config)
    except ConnectionUnavailable:
        return False
    return is_openai_compatible(provider)
