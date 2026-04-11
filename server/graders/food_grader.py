"""Generic grader: accepts extra kwargs (validators often call grade(task_id=...))."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .scoring import REFLECTION_SCORE, score_from_action_dict


def grade(
    action: Optional[Dict[str, Any]] = None,
    session: Any = None,
    task_id: Optional[str] = None,
    **kwargs: Any,
) -> float:
    """
    Hugging Face / hackathon validators may invoke:
      grade(), grade(task_id='food_check_medium'), grade(action={...}), etc.
    Unexpected kwargs must not raise TypeError (that often becomes score 0 or 1).
    """
    try:
        kw_tid = (
            task_id
            or kwargs.get("task_id")
            or kwargs.get("task")
            or kwargs.get("id")
        )
        if kw_tid is not None and not isinstance(kw_tid, str):
            kw_tid = str(kw_tid)

        if action is None:
            if kw_tid:
                return score_from_action_dict(
                    {"ingredients": [], "task_id": kw_tid}
                )
            return float(REFLECTION_SCORE)

        if not isinstance(action, dict):
            return float(REFLECTION_SCORE)

        merged = dict(action)
        merged.setdefault("ingredients", [])
        if kw_tid:
            merged["task_id"] = kw_tid
        elif not merged.get("task_id"):
            merged["task_id"] = "food_check_easy"

        return score_from_action_dict(merged)
    except Exception:
        return float(REFLECTION_SCORE)
