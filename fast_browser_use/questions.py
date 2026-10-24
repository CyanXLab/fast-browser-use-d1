"""Bounded agent execution budget."""

MAX_STEPS = int(__import__("os").environ.get("FBU_MAX_STEPS", "60") or 60)
