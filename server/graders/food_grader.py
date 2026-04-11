"""Per-task grader entrypoint for openenv.yaml (scores must lie strictly in (0, 1))."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..reward_core import calculate_reward, check_ingredients, clamp_reward_strict

_DEFAULT_REFLECTION_SCORE = 0.05


def grade(action: Optional[Dict[str, Any]] = None, session: Any = None) -> float:
    """Mirror HTTP /step reward for static validation; safe for parameterless probes."""
    if action is None or not isinstance(action, dict):
        return _DEFAULT_REFLECTION_SCORE

    ingredients = action.get("ingredients")
    if not isinstance(ingredients, list):
        ingredients = []

    task_id = action.get("task_id") or "food_check_easy"
    if not isinstance(task_id, str):
        task_id = "food_check_easy"

    flagged = check_ingredients([str(i) for i in ingredients])
    ai_dangerous = bool(action.get("ai_dangerous"))

    raw = calculate_reward(flagged, task_id, ai_dangerous)
    return clamp_reward_strict(raw)
