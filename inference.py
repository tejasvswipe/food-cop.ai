"""
inference.py — Food Cop AI
Starts the FastAPI server as a subprocess, waits for it to be ready,
then runs a full test suite (easy / medium / hard tasks).
"""

import subprocess
import sys
import time
import requests
import json

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_URL    = "http://0.0.0.0:7860"
STARTUP_TIMEOUT = 60   # seconds to wait for server boot
POLL_INTERVAL   = 2    # seconds between health-check polls

# ─────────────────────────────────────────────
# TEST CASES  (easy / medium / hard)
# ─────────────────────────────────────────────
TEST_CASES = [
    {
        "task_id": "task_easy",
        "label": "Easy — Candy bar with E128",
        "product_name": "Choco Candy Bar",
        "ingredients": ["sugar", "cocoa butter", "E128", "milk solids", "emulsifier"],
        "expect_dangerous": True,
    },
    {
        "task_id": "task_medium",
        "label": "Medium — Energy drink with E211 + Vitamin C",
        "product_name": "ZapEnergy Drink",
        "ingredients": ["carbonated water", "E211", "ascorbic acid", "caffeine", "sugar"],
        "expect_dangerous": True,
    },
    {
        "task_id": "task_hard",
        "label": "Hard — Processed snack with multiple banned additives",
        "product_name": "CrunchMaster Snack",
        "ingredients": [
            "refined flour", "E621", "E951", "TBHQ",
            "partially hydrogenated oil", "E128", "salt"
        ],
        "expect_dangerous": True,
    },
    {
        "task_id": "task_easy",
        "label": "Safe product — plain oats",
        "product_name": "Plain Oats",
        "ingredients": ["rolled oats", "water", "salt"],
        "expect_dangerous": False,
    },
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def start_server() -> subprocess.Popen:
    """Launch app.py as a background process."""
    print("[inference] Starting server …")
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def wait_for_server(base_url: str, timeout: int = STARTUP_TIMEOUT) -> None:
    """Poll /health until the server responds or timeout is reached."""
    deadline = time.time() + timeout
    attempt  = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = requests.get(f"{base_url}/health", timeout=3)
            if r.status_code == 200:
                print(f"[inference] Server ready after {attempt} poll(s) ✓")
                return
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.Timeout:
            pass
        print(f"[inference] Waiting for server … (attempt {attempt})")
        time.sleep(POLL_INTERVAL)
    raise RuntimeError(
        f"Server at {base_url} did not become ready within {timeout}s"
    )


def reset_env(task_id: str = "task_easy") -> dict:
    r = requests.post(
        f"{BASE_URL}/reset",
        params={"task_id": task_id},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def run_step(product_name: str, ingredients: list, task_id: str) -> dict:
    payload = {
        "product_name": product_name,
        "ingredients":  ingredients,
        "task_id":      task_id,
    }
    r = requests.post(
        f"{BASE_URL}/step",
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def print_result(label: str, result: dict, expect_dangerous: bool) -> bool:
    obs     = result.get("observation", {})
    verdict = obs.get("verdict", "UNKNOWN")
    reward  = result.get("reward", 0)
    flagged = obs.get("flagged_ingredients", [])
    ai_text = obs.get("ai_analysis", "")

    is_dangerous = "SAFE" not in verdict.upper()
    passed = is_dangerous == expect_dangerous

    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{'─'*60}")
    print(f"  {status}  |  {label}")
    print(f"  Verdict : {verdict}")
    print(f"  Reward  : {reward}")
    print(f"  Flagged : {flagged or 'none'}")
    print(f"  AI      : {ai_text[:200]}{'…' if len(ai_text) > 200 else ''}")
    return passed


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    server_proc = None
    passed_count = 0

    try:
        # 1. Boot server
        server_proc = start_server()

        # 2. Wait until healthy
        wait_for_server(BASE_URL)

        # 3. Run every test case
        print(f"\n[inference] Running {len(TEST_CASES)} test case(s) …")

        for tc in TEST_CASES:
            # Reset env for this task
            reset_env(tc["task_id"])

            # Submit step
            result = run_step(
                product_name=tc["product_name"],
                ingredients=tc["ingredients"],
                task_id=tc["task_id"],
            )

            # Evaluate
            ok = print_result(tc["label"], result, tc["expect_dangerous"])
            if ok:
                passed_count += 1

        # 4. Summary
        total = len(TEST_CASES)
        print(f"\n{'═'*60}")
        print(f"  Results : {passed_count}/{total} passed")
        print(f"{'═'*60}\n")

        if passed_count < total:
            sys.exit(1)   # non-zero → validator flags failure

    except Exception as exc:
        print(f"\n[inference] FATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    finally:
        # Always shut down the server subprocess cleanly
        if server_proc and server_proc.poll() is None:
            print("[inference] Shutting down server …")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()