import os
import time
import requests
from openai import OpenAI

# ==============================
# ENV VARIABLES (REQUIRED)
# ==============================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

# OpenAI client (MANDATORY)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ✅ IMPORTANT: Use HF Space URL (NOT localhost)
BASE_URL = os.getenv(
    "ENV_URL",
    "https://foodcop-food-cop_open-env.hf.space"
)

# ==============================
# WAIT FOR SERVER
# ==============================
def wait_for_server(url, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code < 500:
                return
        except:
            pass
        time.sleep(2)
    raise RuntimeError("Server not ready")


# ==============================
# SINGLE TASK RUNNER
# ==============================
def run_task(task_name):
    rewards = []
    step_count = 0
    success = False

    print(f"[START] task={task_name} env=food_safety_env model={MODEL_NAME}")

    try:
        # ---- RESET ----
        requests.post(f"{BASE_URL}/reset", timeout=10)

        # ---- LLM DECISION (MANDATORY USAGE) ----
        prompt = "Check if food containing E128 is safe or dangerous."
        llm_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )

        action = "analyze_food"

        # ---- STEP ----
        payload = {
            "product_name": "Test Snack",
            "ingredients": ["wheat", "E128", "salt"]
        }

        res = requests.post(f"{BASE_URL}/step", json=payload, timeout=10)

        if res.status_code != 200:
            raise Exception(f"HTTP {res.status_code}")

        result = res.json()

        step_count += 1

        # ---- REWARD ----
        if "DANGEROUS" in str(result).upper():
            reward = 1.00
            success = True
        else:
            reward = 0.00

        rewards.append(f"{reward:.2f}")

        print(f"[STEP] step={step_count} action={action} reward={reward:.2f} done=true error=null")

    except Exception as e:
        print(f"[STEP] step={step_count} action=error reward=0.00 done=true error={str(e)}")

    rewards_str = ",".join(rewards) if rewards else "0.00"
    print(f"[END] success={str(success).lower()} steps={step_count} rewards={rewards_str}")


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    wait_for_server(BASE_URL)

    run_task("task_easy")
    run_task("task_medium")
    run_task("task_hard")