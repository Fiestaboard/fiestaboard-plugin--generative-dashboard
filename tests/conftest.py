"""Shared fixtures."""

import pytest


@pytest.fixture(autouse=True)
def isolated_composition_log(tmp_path, monkeypatch):
    """Every test gets an empty composition log.

    Hydration means a shared real log leaks one test's composition into the
    next test's 'cold' board. The env override reaches every copy of the
    module, however many the registry reloads have left alive.
    """
    path = tmp_path / "isolated-complog.jsonl"
    monkeypatch.setenv("GENERATIVE_DASHBOARD_COMPLOG", str(path))
    return path


@pytest.fixture
def manifest():
    return {
        "id": "generative_dashboard",
        "name": "Generative Dashboard",
        "version": "1.0.0",
        "description": "Test",
        "author": "FiestaBoard Team",
        "live_data": True,
        "settings_schema": {
            "type": "object",
            "properties": {
                "refresh_seconds": {"type": "integer", "default": 300, "minimum": 120}
            },
        },
    }


@pytest.fixture
def config():
    return {
        "enabled": True,
        "api_key": "sk-test",
        "api_base_url": "https://api.test/v1",
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "output_mode": "grid",
        "refresh_seconds": 300,
        "default_threshold_pct": 5,
        "use_color": True,
        "watchlist": ["air.aqi", "wx.temp"],
        "labels": {},
        "pinned": [],
        "notes": {},
        "thresholds": {},
    }
