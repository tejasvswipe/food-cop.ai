from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from openai import OpenAI
import os
import uvicorn
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")

client = None
if API_BASE_URL and HF_TOKEN:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

app = FastAPI(title="Food Cop AI", version="1.0.1")

class FoodAction(BaseModel):
    product_name: str
    ingredients: List[str]
    task_id: Optional[str] = "food_check_easy"

class Observation(BaseModel):
    product_name: str
    ingredients: List[str]
    step: int
    verdict: Optional[str] = None
    flagged_ingredients: Optional[List[str]] = None
    ai_analysis: Optional[str] = None

class StepResult(BaseModel):
    observation: Observation
    reward: float = Field(..., gt=0.0, lt=1.0)
    done: bool
    info: Dict[str, Any]

class ResetResponse(BaseModel):
    observation: Observation
    info: Dict[str, Any]

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

DEFAULT_TASK_ID = "food_check_easy"

state = {
    "step": 0,
    "product_name": None,
    "ingredients": [],
    "done": False,
    "task_id": DEFAULT_TASK_ID,
}

def reset_state():
    state["step"] = 0
    state["product_name"] = None
    state["ingredients"] = []
    state["done"] = False
    state["task_id"] = DEFAULT_TASK_ID

def sanitize_score(x) -> float:
    try:
        if isinstance(x, bool):
            x = 1.0 if x else 0.0
        x = float(x)
    except Exception:
        x = 0.5
    return min(0.99, max(0.01, x))

def check_ingredients(ingredients: List[str]) -> List[str]:
    flagged = []
    for ing in ingredients:
        ing_low = ing.lower()
        for banned_key, reason in BANNED_INGREDIENTS.items():
            if banned_key.lower() in ing_low:
                flagged.append(f"{ing}: {reason}")
                break
    return flagged

def calculate_reward(flagged, task_id, ai_dangerous):

    base = 0.42

    # Always safe fallback (important for unseen tasks)
    if task_id not in ("food_check_easy", "food_check_medium", "food_check_hard"):
        return 0.50

    severity = len(flagged)

    # -------------------------
    # EASY: binary clarity task
    # -------------------------
    if task_id == "food_check_easy":
        if severity >= 1:
            return 0.66
        return 0.44

    # -------------------------
    # MEDIUM: graded detection
    # -------------------------
    if task_id == "food_check_medium":
        score = base

        if severity >= 1:
            score += 0.10
        if severity >= 2:
            score += 0.07

        if ai_dangerous:
            score += 0.05

        # keep clean mid-range separation
        return min(0.68, max(0.40, score))

    # -------------------------
    # HARD: multi-signal task
    # -------------------------
    if task_id == "food_check_hard":
        score = 0.48  # higher baseline for difficulty

        if severity >= 1:
            score += 0.06
        if severity >= 2:
            score += 0.06
        if severity >= 3:
            score += 0.08

        if ai_dangerous:
            score += 0.04

        # allow higher ceiling for hard
        return min(0.72, max(0.50, score))

    return 0.50

@app.get("/")
def home():
    return {"status": "Food Cop AI is running!", "model": MODEL_NAME}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/state")
def get_state():
    return state

@app.post("/reset", response_model=ResetResponse)
def reset(task_id: str = DEFAULT_TASK_ID):
    reset_state()
    state["task_id"] = task_id

    obs = Observation(
        product_name="",
        ingredients=[],
        step=0,
        verdict="RESET",
        flagged_ingredients=[],
        ai_analysis="Ready for inspection",
    )
    return ResetResponse(observation=obs, info={"task_id": task_id})

@app.post("/step", response_model=StepResult)
def step(action: FoodAction):
    state["step"] += 1
    state["product_name"] = action.product_name
    state["ingredients"] = action.ingredients

    task_id = action.task_id or state.get("task_id", DEFAULT_TASK_ID)
    state["task_id"] = task_id

    flagged = check_ingredients(action.ingredients)

    ai_response = "No AI analysis"
    ai_dangerous = False

    # SAFE LLM CALL (optional)
    if client is not None:
        try:
            prompt = f"""You are a strict Indian food safety expert using FSSAI and EFSA guidelines.
Product: {action.product_name}
Ingredients: {', '.join(action.ingredients)}
Flagged by database: {flagged if flagged else 'None'}
Answer SAFE or DANGEROUS."""

            chat = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )

            ai_response = chat.choices[0].message.content or "No AI analysis"
            ai_dangerous = ai_response.strip().upper().startswith("DANGEROUS")

        except Exception as e:
            ai_response = f"AI error: {str(e)}"
            ai_dangerous = False

    # IMPORTANT: compute reward FIRST
    raw_reward = calculate_reward(flagged, task_id, ai_dangerous)

    # SAFE CLAMP LAST (NEVER BEFORE ASSIGNMENT)
    reward = float(max(0.01, min(0.99, raw_reward)))

    verdict = "DANGEROUS" if (flagged or ai_dangerous) else "SAFE"

    obs = Observation(
        product_name=action.product_name,
        ingredients=action.ingredients,
        step=state["step"],
        verdict=verdict,
        flagged_ingredients=flagged,
        ai_analysis=ai_response,
    )

    return StepResult(
        observation=obs,
        reward=reward,
        done=True,
        info={
            "task_id": task_id,
            "flagged_count": len(flagged),
            "ai_dangerous": ai_dangerous,
        },
    )

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()