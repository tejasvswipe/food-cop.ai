from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from openai import OpenAI
import os
import uvicorn
from dotenv import load_dotenv
from pathlib import Path

from server.reward_core import calculate_reward, check_ingredients, clamp_reward_strict

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")

client = None
if API_BASE_URL and HF_TOKEN:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

app = FastAPI(title="Food Cop AI", version="1.0.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.is_dir():
    app.mount(
        "/ui",
        StaticFiles(directory=str(_STATIC_DIR), html=True),
        name="ui",
    )

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

@app.get("/")
def home():
    return {
        "status": "Food Cop AI is running!",
        "model": MODEL_NAME,
        "ui": "/ui/",
    }

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

    raw_reward = calculate_reward(flagged, task_id, ai_dangerous)
    reward = clamp_reward_strict(raw_reward)

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
