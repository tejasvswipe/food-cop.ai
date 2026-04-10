import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from dotenv import load_dotenv
from pathlib import Path

# =========================
# LOAD ENV
# =========================
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# =========================
# APP INIT (ONLY ONCE)
# =========================
app = FastAPI(
    title="Food Cop AI",
    description="Food safety inspection environment",
    version="1.0.0"
)

# =========================
# MODELS
# =========================
class FoodAction(BaseModel):
    product_name: str
    ingredients: List[str]
    task_id: Optional[str] = "task_easy"

class Observation(BaseModel):
    product_name: str
    ingredients: List[str]
    step: int
    verdict: Optional[str] = None
    flagged_ingredients: Optional[List[str]] = None
    ai_analysis: Optional[str] = None

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict

class ResetResponse(BaseModel):
    observation: Observation
    info: dict

# =========================
# DATA
# =========================
BANNED_INGREDIENTS = {
    "E128": "Banned food dye (carcinogenic)",
    "E216": "Banned preservative","E128": "Red 2G - banned food dye, linked to cancer (FSSAI+EFSA)",
    "E216": "Propyl p-hydroxybenzoate - banned preservative (FSSAI)",
    "E217": "Sodium propyl p-hydroxybenzoate - banned preservative (FSSAI)",
    "E240": "Formaldehyde - banned preservative, toxic (FSSAI+EFSA)",
    "TBHQ": "Tertiary butylhydroquinone - toxic at high doses (EFSA)",
    "E211": "Sodium benzoate - forms benzene carcinogen (EFSA)",
    "E951": "Aspartame - banned in some countries (EFSA)",
    "E621": "MSG - causes reactions in sensitive people (FSSAI)",
    "partially hydrogenated oil": "Trans fat - banned by FDA",
    "potassium bromate": "Banned in EU and India (FSSAI+EFSA)",
    "sudan red": "Illegal dye, toxic (FSSAI+EFSA)",
    "rhodamine b": "Illegal synthetic dye (FSSAI)",
    ,
}

state = {
    "step": 0,
    "task_id": "task_easy"
}

# =========================
# HELPERS
# =========================
def reset_state():
    state["step"] = 0

def check_ingredients(ingredients: List[str]):
    flagged = []
    for ing in ingredients:
        for banned in BANNED_INGREDIENTS:
            if banned.lower() in ing.lower():
                flagged.append(ing)
    return flagged

def calculate_reward(flagged: list, task_id: str) -> float:
    if task_id == "task_easy":
        return 1.0 if flagged else 0.0
    elif task_id == "task_medium":
        return min(len(flagged) * 0.5, 1.0)
    elif task_id == "task_hard":
        return min(len(flagged) * 0.3, 1.0)
    return 0.0

# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"status": "API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset", response_model=ResetResponse)
def reset(task_id: str = "task_easy"):
    reset_state()
    state["task_id"] = task_id

    return ResetResponse(
        observation=Observation(
            product_name="",
            ingredients=[],
            step=0,
            verdict="RESET",
            flagged_ingredients=[],
            ai_analysis="Ready"
        ),
        info={"task_id": task_id}
    )

@app.post("/step", response_model=StepResult)
def step(action: FoodAction):
    state["step"] += 1

    flagged = check_ingredients(action.ingredients)
    reward = calculate_reward(flagged, action.task_id)

    verdict = "UNSAFE" if flagged else "SAFE"

    return StepResult(
        observation=Observation(
            product_name=action.product_name,
            ingredients=action.ingredients,
            step=state["step"],
            verdict=verdict,
            flagged_ingredients=flagged,
            ai_analysis="rule-based"
        ),
        reward=reward,
        done=True,
        info={"task_id": action.task_id}
    )

# =========================
# ENTRY POINT (MANDATORY)
# =========================
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()