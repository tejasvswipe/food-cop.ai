import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# Required ENV variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# OpenAI client (MANDATORY as per rules)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=GROQ_API_KEY
)

# Your environment base URL (HF Space or local)
BASE_URL = os.getenv("ENV_URL", "http://localhost:7860")


def run_task(task_name):
    rewards = []
    step_count = 0
    success = False

    print(f"[START] task={task_name} env=food_safety_env model={MODEL_NAME}")

    try:
        # RESET
        requests.post(f"{BASE_URL}/reset")

        # STEP INPUT (sample test case)
        action = "analyze_food"
        payload = {
            "product_name": "Test Snack",
            "ingredients": ["wheat", "E128", "salt"]
        }

        response = requests.post(f"{BASE_URL}/step", json=payload)
        result = response.json()

        step_count += 1

        # Reward logic (simple baseline)
        if "DANGEROUS" in str(result):
            reward = 1.0
            success = True
        else:
            reward = 0.0

        rewards.append(f"{reward:.2f}")

        print(f"[STEP] step={step_count} action={action} reward={reward:.2f} done=true error=null")

    except Exception as e:
        print(f"[STEP] step={step_count} action=error reward=0.00 done=true error={str(e)}")

    # END (MANDATORY)
    rewards_str = ",".join(rewards) if rewards else "0.00"
    print(f"[END] success={str(success).lower()} steps={step_count} rewards={rewards_str}")


if __name__ == "__main__":
    # Run all 3 tasks (required)
    run_task("task_easy")
    run_task("task_medium")
    run_task("task_hard")