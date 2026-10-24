"""Shared test fixtures: keep runs fully local unless a test opts into the LLM."""

import pytest


@pytest.fixture(autouse=True)
def _default_off_remote_llm(monkeypatch):
    """Remote-LLM escalation is user-configured in production; tests stay local."""
    monkeypatch.setenv("FBU_LLM_ESCALATION", "off")
