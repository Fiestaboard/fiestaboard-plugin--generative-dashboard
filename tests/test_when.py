"""Temporal context: what the model is told about when 'now' is."""

from datetime import datetime

from plugins.generative_dashboard.when import describe_now


def _at(y, m, d, hh, mm=0):
    return describe_now(datetime(y, m, d, hh, mm))


def test_it_names_the_day_and_date():
    text = _at(2026, 9, 3, 18, 28)
    assert "THURSDAY" in text.upper()
    assert "2026" in text


def test_it_gives_a_readable_clock_time():
    assert "6:28 PM" in _at(2026, 9, 3, 18, 28)


def test_morning_afternoon_evening_and_night_are_distinguished():
    assert "morning" in _at(2026, 9, 3, 8)
    assert "afternoon" in _at(2026, 9, 3, 14)
    assert "evening" in _at(2026, 9, 3, 19)
    assert "night" in _at(2026, 9, 3, 2)


def test_it_says_whether_it_is_a_weekday_or_the_weekend():
    assert "weekday" in _at(2026, 9, 3, 9)      # Thursday
    assert "weekend" in _at(2026, 9, 5, 9)      # Saturday


def test_it_names_the_season():
    assert "summer" in _at(2026, 7, 15, 12)
    assert "winter" in _at(2026, 1, 15, 12)
    assert "spring" in _at(2026, 4, 15, 12)
    assert "autumn" in _at(2026, 10, 15, 12)


def test_it_flags_a_commute_hour_on_a_weekday():
    assert "commute" in _at(2026, 9, 3, 8)
    assert "commute" in _at(2026, 9, 3, 17)


def test_a_weekend_morning_is_not_a_commute():
    assert "commute" not in _at(2026, 9, 5, 8)


def test_midnight_and_noon_read_correctly():
    assert "12:00 AM" in _at(2026, 9, 3, 0)
    assert "12:00 PM" in _at(2026, 9, 3, 12)
