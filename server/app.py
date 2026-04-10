from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
import os
import uvicorn
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# =========================
# APP INIT
# =========================
app = FastAPI(
    title="Food Cop AI",
    description="Indian food safety inspector using FSSAI and EFSA rules",
    version="1.0.0",
    openapi_url="/openapi.json"
)

# =========================
# ENV SETUP
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

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
    "E128": "Red 2G - banned food dye, linked to cancer (FSSAI+EFSA)",
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
}

state = {
    "step": 0,
    "product_name": None,
    "ingredients": [],
    "done": False,
    "task_id": "task_easy"
}

# =========================
# HELPERS
# =========================
def reset_state():
    state["step"] = 0
    state["product_name"] = None
    state["ingredients"] = []
    state["done"] = False

def check_ingredients(ingredients: List[str]):
    flagged = []
    for ing in ingredients:
        for banned_key, reason in BANNED_INGREDIENTS.items():
            if banned_key.lower() in ing.lower():
                flagged.append(f"{ing}: {reason}")
    return flagged

def calculate_reward(flagged: list, task_id: str, ai_dangerous: bool) -> float:
    if task_id == "task_easy":
        if len(flagged) >= 1:
            return 1.0
        if ai_dangerous:
            return 0.5
        return 0.0
    elif task_id == "task_medium":
        score = 0.0
        if len(flagged) >= 1:
            score += 0.5
        if len(flagged) >= 2:
            score += 0.3
        if ai_dangerous:
            score += 0.2
        return min(score, 1.0)
    elif task_id == "task_hard":
        score = 0.0
        if len(flagged) >= 1:
            score += 0.3
        if len(flagged) >= 2:
            score += 0.3
        if len(flagged) >= 3:
            score += 0.2
        if ai_dangerous:
            score += 0.2
        return min(score, 1.0)
    return 0.0

# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except Exception:
        return "<h1>Food Cop AI is running!</h1>"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/state")
def get_state():
    return state

@app.post("/reset", response_model=ResetResponse)
def reset(task_id: str = "task_easy"):
    reset_state()
    state["task_id"] = task_id
    obs = Observation(
        product_name="",
        ingredients=[],
        step=0,
        verdict="RESET",
        flagged_ingredients=[],
        ai_analysis="Ready for inspection"
    )
    return ResetResponse(observation=obs, info={"task_id": task_id})

@app.post("/step", response_model=StepResult)
def step(action: FoodAction):
    state["step"] += 1
    task_id = action.task_id or state.get("task_id", "task_easy")
    flagged = check_ingredients(action.ingredients)

    ai_response = "No AI analysis (GROQ_API_KEY not set)"
    ai_dangerous = False

    if client:
        try:
            prompt = f"""You are a strict Indian food safety expert using FSSAI and EFSA guidelines.
Product: {action.product_name}
Ingredients: {', '.join(action.ingredients)}
Flagged by database: {flagged}
Analyze if this product is SAFE or UNSAFE. Start with YES if dangerous or NO if safe.
Give a clear explanation in 2-3 lines."""

            chat = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192"
            )
            ai_response = chat.choices[0].message.content
            ai_dangerous = "YES" in ai_response.upper()
        except Exception as e:
            ai_response = f"AI analysis failed: {str(e)}"

    reward = calculate_reward(flagged, task_id, ai_dangerous)
    verdict = "UNSAFE ❌" if (flagged or ai_dangerous) else "SAFE ✅"

    obs = Observation(
        product_name=action.product_name,
        ingredients=action.ingredients,
        step=state["step"],
        verdict=verdict,
        flagged_ingredients=flagged,
        ai_analysis=ai_response
    )

    return StepResult(
        observation=obs,
        reward=reward,
        done=True,
        info={"task_id": task_id}
    )

# =========================
# MAIN ENTRY
# =========================
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()