"""What time it is, in terms a model can reason about.

The model already knows what Christmas Eve means, or that a Tuesday 8am is a
commute. It just has no idea when "now" is. Handing it the date, the clock,
and a few plain descriptions is enough for it to weight the board sensibly —
transit in the morning, air quality on a hot afternoon, trivia late at night —
without this plugin hard-coding any of those rules.
"""

from datetime import datetime

# Meteorological seasons, northern hemisphere. Close enough for "is it
# reasonable to care about the surf right now".
_SEASONS = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


def _part_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def _clock(now: datetime) -> str:
    """12-hour time without the platform-specific %-I padding flag."""
    hour = now.hour % 12 or 12
    suffix = "AM" if now.hour < 12 else "PM"
    return f"{hour}:{now.minute:02d} {suffix}"


def describe_now(now: datetime) -> str:
    """One line of temporal context for the prompt."""
    weekend = now.weekday() >= 5
    parts = [
        f"{now.strftime('%A')} {now.day} {now.strftime('%B %Y')}",
        _clock(now),
        _part_of_day(now.hour),
        "weekend" if weekend else "weekday",
        _SEASONS[now.month],
    ]
    if not weekend and (7 <= now.hour < 10 or 16 <= now.hour < 19):
        parts.append("commute hours")
    return ", ".join(p for p in parts if p)
