"""Shared fixtures."""

import pytest


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
