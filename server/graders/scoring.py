"""Shared grader scoring — every path returns a float strictly in (0, 1)."""

from __future__ import annotations

from typing import Any, Dict

from ..reward_core import calculate_reward, check_ingredients, clamp_reward_strict

# Safe default when probes pass no payload or on any internal error.
REFLECTION_SCORE = 0.05


def score_from_action_dict(action: Dict[str, Any]) -> float:
    ingredients = action.get("ingredients")
    if not isinstance(ingredients, list):
        ingredients = []

    tid = action.get("task_id")
    if not isinstance(tid, str) or not tid.strip():
        tid = "food_check_easy"

    flagged = check_ingredients([str(i) for i in ingredients])
    ai_dangerous = bool(action.get("ai_dangerous"))
    raw = calculate_reward(flagged, tid, ai_dangerous)
    return float(clamp_reward_strict(raw))
