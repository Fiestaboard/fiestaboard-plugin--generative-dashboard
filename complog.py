"""A durable record of every composition the model lands.

Prompt-tuning taste needs evidence: what the board actually said, hour by
hour, in the model's own choices — not a memory of glancing at it. Each
landed composition appends one JSON line beside the plugin (rotated so it
stays small), and the caller also emits a single tagged log line so recent
compositions can be queried remotely through the app's /logs endpoint.
"""

import json
import logging
import pathlib

logger = logging.getLogger(__name__)

# Roughly a week of half-hourly compositions before rotation trims the head.
MAX_BYTES = 400_000
KEEP_LINES = 300


def _log_path() -> pathlib.Path:
    """Beside the plugin: survives updates (untracked) and reboots."""
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
