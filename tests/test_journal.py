"""The board's memory: what has been happening, not just what is true now."""

from plugins.generative_dashboard.journal import Journal


def test_it_remembers_what_the_model_reported():
    j = Journal()
    j.add("14:02", "Fog thick, visibility 1.2mi")
    assert "Fog thick" in j.render()


def test_entries_carry_their_time():
    j = Journal()
    j.add("14:02", "Fog thick")
    assert "14:02" in j.render()


def test_it_keeps_the_most_recent_entries_and_drops_the_oldest():
    j = Journal(limit=3)
    for i in range(6):
        j.add(f"1{i}:00", f"entry {i}")
    text = j.render()
    assert "entry 5" in text and "entry 3" in text
    assert "entry 0" not in text and "entry 2" not in text


def test_it_reads_oldest_first_so_the_story_runs_forward():
    j = Journal()
    j.add("09:00", "morning fog")
    j.add("15:00", "cleared up")
    assert j.render().index("morning fog") < j.render().index("cleared up")


def test_an_empty_journal_renders_as_nothing():
    assert Journal().render() == ""


def test_a_repeated_observation_is_not_recorded_twice():
    # "still sunny" every five minutes would crowd out the actual story.
    j = Journal()
    j.add("14:00", "Sunny and calm")
    j.add("14:05", "Sunny and calm")
    assert j.render().count("Sunny and calm") == 1


def test_blank_entries_are_ignored():
    j = Journal()
    j.add("14:00", "   ")
    j.add("14:01", "")
    assert j.render() == ""


def test_entries_are_trimmed_so_one_rambling_line_cannot_dominate():
    j = Journal()
    j.add("14:00", "x" * 500)
    assert len(j.render()) < 260
