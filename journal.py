"""A short record of what has been happening.

A snapshot cannot tell the model that it has been sunny all day, or that the
fog was thick this morning and has since cleared. Values only say what is true
now. So the model writes one line about the current state each time it
composes, and those lines come back to it on the next composition — the board
gets a memory, written in the model's own words, without this plugin trying to
infer narrative from numbers.
"""

from collections import deque

# Enough to show a shape to the day without crowding the prompt.
DEFAULT_LIMIT = 6

# One line, not an essay. A rambling entry would push out the older ones.
MAX_ENTRY = 90


class Journal:
    """The last few observations, oldest first."""

    def __init__(self, limit: int = DEFAULT_LIMIT) -> None:
        self._entries: deque[tuple[str, str]] = deque(maxlen=limit)

    def add(self, when: str, text: str) -> None:
        """Record *text* unless it is blank or repeats the last entry."""
        cleaned = " ".join(str(text).split())[:MAX_ENTRY]
        if not cleaned:
            return
        if self._entries and self._entries[-1][1] == cleaned:
            return
        self._entries.append((when, cleaned))

    def render(self) -> str:
        """The record as prompt text, oldest first so it reads as a story."""
        return "\n".join(f"- {when} {text}" for when, text in self._entries)

    def __len__(self) -> int:
        return len(self._entries)
