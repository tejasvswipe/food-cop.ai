"""Shared reward logic for HTTP /step and static OpenEnv grader validation."""

from __future__ import annotations

import math
from typing import List

# Strictly inside (0, 1); margin avoids float edge cases vs 0.0 / 1.0 checks.
REWARD_MIN = 0.001
REWARD_MAX = 0.999

BANNED_INGREDIENTS = {
    "E128": "Red 2G - banned food dye (FSSAI+EFSA)",
    "E216": "Propyl p-hydroxybenzoate - banned (FSSAI)",
    "E217": "Sodium propyl p-hydroxybenzoate - banned (FSSAI)",
    "E240": "Formaldehyde - toxic (FSSAI+EFSA)",
    "TBHQ": "Tertiary butylhydroquinone - toxic (EFSA)",
    "E211": "Sodium benzoate - carcinogen (EFSA)",
    "E951": "Aspartame - banned in some countries (EFSA)",
    "E621": "MSG - causes reactions (FSSAI)",
    "partially hydrogenated oil": "Trans fat - banned by FDA",
    "potassium bromate": "Banned in EU and India (FSSAI+EFSA)",
    "sudan red": "Illegal dye (FSSAI+EFSA)",
    "rhodamine b": "Illegal synthetic dye (FSSAI)",
}


def clamp_reward_strict(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(v):
        return 0.5
    return min(REWARD_MAX, max(REWARD_MIN, v))


def check_ingredients(ingredients: List[str]) -> List[str]:
    flagged = []
    for ing in ingredients:
        ing_low = ing.lower()
        for banned_key, reason in BANNED_INGREDIENTS.items():
            if banned_key.lower() in ing_low:
                flagged.append(f"{ing}: {reason}")
                break
    return flagged


def calculate_reward(flagged, task_id, ai_dangerous: bool) -> float:
    base = 0.42

    if task_id not in ("food_check_easy", "food_check_medium", "food_check_hard"):
        return 0.50

    severity = len(flagged)

    if task_id == "food_check_easy":
        if severity >= 1:
            return 0.66
        return 0.44

    if task_id == "food_check_medium":
        score = base
        if severity >= 1:
            score += 0.10
        if severity >= 2:
            score += 0.07
        if ai_dangerous:
            score += 0.05
        return min(0.68, max(0.40, score))

    if task_id == "food_check_hard":
        score = 0.48
        if severity >= 1:
            score += 0.06
        if severity >= 2:
            score += 0.06
        if severity >= 3:
            score += 0.08
        if ai_dangerous:
            score += 0.04
        return min(0.72, max(0.50, score))

    return 0.50
