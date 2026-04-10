from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
import os
import uvicorn

# =========================
# APP INIT (ONLY ONCE)
# =========================
app = FastAPI(
    title="Food Cop AI",
    description="Indian food safety inspector using FSSAI and EFSA rules",
    version="1.0.0"
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
    "E128": "Red 2G banned dye",
    "E216": "Banned preservative",
    "E217": "Banned preservative",
    "E240": "Toxic formaldehyde"
}

state = {"step": 0, "task_id": "task_easy"}

# =========================
# HELPERS
# =========================
def check_ingredients(ingredients):
    flagged = []
    for ing in ingredients:
        for banned in BANNED_INGREDIENTS:
            if banned.lower() in ing.lower():
                flagged.append(ing)
    return flagged

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
    state["step"] = 0
    state["task_id"] = task_id

    obs = Observation(
        product_name="",
        ingredients=[],
        step=0,
        verdict="RESET"
    )
    return ResetResponse(observation=obs, info={"task_id": task_id})

@app.post("/step", response_model=StepResult)
def step(action: FoodAction):
    state["step"] += 1

    flagged = check_ingredients(action.ingredients)

    # AI optional (safe fallback)
    ai_response = "No AI analysis"
    ai_dangerous = False

    if client:
        try:
            chat = client.chat.completions.create(
                messages=[{"role": "user", "content": f"Is this dangerous? {action.ingredients}"}],
                model="llama3-8b-8192"
            )
            ai_response = chat.choices[0].message.content
            ai_dangerous = "YES" in ai_response.upper()
        except:
            pass

    if flagged or ai_dangerous:
        reward = 1.0
        verdict = "DANGEROUS"
    else:
        reward = 0.0
        verdict = "SAFE"

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
        info={}
    )

# =========================
# MAIN ENTRY (IMPORTANT)
# =========================
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()