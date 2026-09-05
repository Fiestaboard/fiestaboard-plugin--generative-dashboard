"""A durable record of every composition the model lands.

Prompt-tuning taste needs evidence: what the board actually said, hour by
hour, in the model's own choices — not a memory of glancing at it. Each
landed composition appends one JSON line beside the plugin (rotated so it
stays small), and the caller also emits a single tagged log line so recent
compositions can be queried remotely through the app's /logs endpoint.
"""

import json
import logging
import os
import pathlib

logger = logging.getLogger(__name__)

# Roughly a week of half-hourly compositions before rotation trims the head.
MAX_BYTES = 400_000
KEEP_LINES = 300


def _log_path() -> pathlib.Path:
    """Beside the plugin: survives updates (untracked) and reboots.

    The env override exists for tests: registry reloads can leave several
    copies of this module alive at once, and an environment variable is the
    one binding they all share.
    """
    override = os.environ.get("GENERATIVE_DASHBOARD_COMPLOG")
    if override:
        return pathlib.Path(override)
    return pathlib.Path(__file__).resolve().parent / "composition_log.jsonl"


def record(entry: dict) -> None:
    """Append *entry*; never let logging break a composition swap."""
    try:
        path = _log_path()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if path.stat().st_size > MAX_BYTES:
            lines = path.read_text(encoding="utf-8").splitlines()[-KEEP_LINES:]
            # Entries can be large; trim from the head until the tail fits.
            while lines and sum(len(l) + 1 for l in lines) > MAX_BYTES:
                lines.pop(0)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        logger.debug("Composition log write failed", exc_info=True)


def last_for(key: str) -> dict | None:
    """The most recent entry for one board key, or None."""
    try:
        lines = _log_path().read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get("key") == key:
            return entry
    return None


def recent_logs_for(key: str, limit: int = 6) -> list[tuple[str, str]]:
    """(time, log-line) pairs for *key*, oldest first, to reseed the journal."""
    try:
        lines = _log_path().read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out: list[tuple[str, str]] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get("key") == key and entry.get("log"):
            when = str(entry.get("at") or "")
            out.append((when[11:16] if len(when) >= 16 else when, str(entry["log"])))
    return out[-limit:]
