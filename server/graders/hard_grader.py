"""Task food_check_hard — dedicated entry point for manifest validators."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .scoring import REFLECTION_SCORE, score_from_action_dict


def grade(
    action: Optional[Dict[str, Any]] = None,
    session: Any = None,
    task_id: Optional[str] = None,
    **kwargs: Any,
) -> float:
    try:
        base = action if isinstance(action, dict) else {}
        merged = {**base, "task_id": "food_check_hard"}
        merged.setdefault("ingredients", [])
        return score_from_action_dict(merged)
    except Exception:
        return float(REFLECTION_SCORE)
