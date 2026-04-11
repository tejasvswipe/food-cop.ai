import os
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
BASE_URL = os.getenv("ENV_URL", "http://localhost:7860")


def clamp_reward(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.5
    return min(0.94, max(0.05, x))


if not HF_TOKEN:
    print("[WARN] No API key found — AI calls will be skipped")
    client = None
else:
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )
    except Exception as e:
        print(f"[WARN] Client init failed: {e}")
        client = None


def wait_for_server(url: str, timeout: int = 60, interval: int = 3) -> None:
    print(f"[INIT] Waiting for server at {url}...")

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code < 500:
                print(f"[INIT] Server ready (status={r.status_code})")
                return
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"[INIT] Check error: {e}")

        time.sleep(interval)

    raise RuntimeError(f"Server at {url} not ready after {timeout}s")


def ask_llm(product_name: str, ingredients: list, verdict: str, flagged: list) -> str:
    if not client:
        return "AI skipped — no client"

    try:
        prompt = f"""You are a strict Indian food safety expert using FSSAI and EFSA guidelines.
Product: {product_name}
Ingredients: {', '.join(ingredients)}
Flagged by database: {flagged if flagged else 'None'}
Analyze if this product is SAFE or UNSAFE. Start with YES if dangerous or NO if safe.
Give a clear explanation in 2-3 lines."""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.choices[0].message.content or "No AI response"

    except Exception as e:
        return f"AI failed: {str(e)}"


def run_task(task_name: str) -> None:
    rewards = []
    step_count = 0
    success = False

    print(f"[START] task={task_name} env=food_safety_env model={MODEL_NAME}")

    try:
        # Reset environment
        reset_resp = requests.post(
            f"{BASE_URL}/reset?task_id={task_name}",
            timeout=15,
        )
        reset_resp.raise_for_status()

        # Action
        payload = {
            "product_name": "Test Snack",
            "ingredients": ["wheat", "E128", "salt"],
            "task_id": task_name,
        }

        response = requests.post(
            f"{BASE_URL}/step",
            json=payload,
            timeout=20,
        )
        response.raise_for_status()

        result = response.json()
        step_count += 1

        obs = result.get("observation", {}) or {}
        verdict = obs.get("verdict", "UNKNOWN")
        flagged = obs.get("flagged_ingredients", []) or []

        # LLM call (optional)
        llm_reply = ask_llm(
            product_name=payload["product_name"],
            ingredients=payload["ingredients"],
            verdict=verdict,
            flagged=flagged,
        )

        print(f"[LLM] {llm_reply[:120]}")

        # IMPORTANT: use ENV reward only
        reward = clamp_reward(result.get("reward", 0.5))
        success = reward > 0.5

        log_reward = f"{reward:.3f}"
        rewards.append(log_reward)

        print(
            f"[STEP] step={step_count} action=analyze_food "
            f"reward={log_reward} done=true error=null"
        )

    except Exception as e:
        reward = 0.40
        log_reward = f"{reward:.3f}"
        rewards.append(log_reward)

        print(
            f"[STEP] step={step_count} action=error "
            f"reward={log_reward} done=true error={str(e)}"
        )

        print(f"[END] success=false steps={step_count} rewards={','.join(rewards)}")
        return

    print(
        f"[END] success={str(success).lower()} "
        f"steps={step_count} rewards={','.join(rewards)}"
    )


if __name__ == "__main__":
    wait_for_server(BASE_URL)

    run_task("food_check_easy")
    run_task("food_check_medium")
    run_task("food_check_hard")