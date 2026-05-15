"""SlotMachines (internal researcher notes — NOT surfaced to optimizer).

DO NOT include the contents of this docstring in any task-facing text
(``objective`` / ``background``). The optimizer must not learn this is
a classical multi-armed bandit setup; mapping the task onto known names
(MAB, UCB, Thompson sampling, ε-greedy, exploration-exploitation) would
let it short-circuit the research question.

Mechanic
--------
N items (default 10), M draws (default 100). Each draw of item ``i`` returns
a Bernoulli reward with an unknown per-item probability ``p_i ∈ [0, 1]``.
Module-level RNG state, seeded at ``configure(...)`` time. Each
``evaluate(candidate)`` call is ONE pull; the candidate is a string-encoded
integer in ``[0, N)`` (e.g. ``"3"``).

Score per eval = reward (0 or 1). The terrarium framework's notion of
"best" is therefore not particularly meaningful here — what matters is
the *cumulative* reward across all M pulls in the run (and how it
compares to ``M * max(p_i)``, the always-best-arm baseline).

What this probes
----------------
- Does the evolved strategy try items it hasn't seen (information
  acquisition) before committing to one (exploitation)?
- Does it collapse to greedy-after-one-success?
- Does it collapse to pure round-robin?
- How fast does it lock in once N and M are far apart?

Deception-safe info
-------------------
The true per-item probabilities and the identity of the best item are
NEVER returned to the optimizer — otherwise it could memorize the seed
and hardcode the best arm. Per-pull info surfaces: the reward, pulls
remaining, pulls total. The optimizer learns reward structure only
through what its candidates observed.

State
-----
Module-level RNG and pull log, reset by ``configure(n, m, seed)`` which
the runner calls in ``_apply_task_runtime_config`` before any eval.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from typing import Any

from terrarium.registry import register_task
from terrarium.task import Task

DEFAULT_N = 10
DEFAULT_M = 100
DEFAULT_SEED = 42

_RESEARCH_LOG_ENV = "TERRARIUM_SLOTS_RESEARCH_LOG"

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "n": DEFAULT_N,
    "m": DEFAULT_M,
    "seed": DEFAULT_SEED,
    "rng": None,
    "arm_p": None,
    "best_arm": None,
    "pulls": [],         # list[(item, reward)]
    "rejected": [],
    "first_call_ts": None,
}


def _compute_sample_path_oracle(n: int, m: int, seed: int, arm_p: list[float], best_arm: int) -> int:
    """Replay the RNG sequence as if 'always pull best_arm', return total reward.

    This is the tightest upper bound on any strategy's reward for this exact
    (n, m, seed) — no algorithm, however perfect its knowledge, can beat
    what the noise actually paid out on the best arm.
    """
    rng = random.Random(seed)
    # Reproduce the same rng.random() calls configure() made to build arm_p,
    # so the oracle replay uses the same noise-stream tail that the real run will.
    for _ in range(n):
        rng.random()
    total = 0
    for _ in range(m):
        if rng.random() < arm_p[best_arm]:
            total += 1
    return total


def configure(n: int, m: int, seed: int) -> None:
    """Set N, M, seed and reset state. Runner-only; never call from eval."""
    with _state_lock:
        rng = random.Random(seed)
        arm_p = [rng.random() for _ in range(n)]
        best_arm = max(range(n), key=lambda i: arm_p[i])
        sample_path_oracle = _compute_sample_path_oracle(n, m, seed, arm_p, best_arm)
        expected_oracle = arm_p[best_arm] * m
        _state["n"] = int(n)
        _state["m"] = int(m)
        _state["seed"] = int(seed)
        _state["rng"] = rng
        _state["arm_p"] = arm_p
        _state["best_arm"] = best_arm
        _state["sample_path_oracle"] = int(sample_path_oracle)
        _state["expected_oracle"] = float(expected_oracle)
        _state["pulls"] = []
        _state["rejected"] = []
        _state["first_call_ts"] = None

        # Write an oracle banner to the sidecar (first line of each run).
        log_path = os.environ.get(_RESEARCH_LOG_ENV)
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ts": time.time(),
                        "event": "configure",
                        "n": n, "m": m, "seed": seed,
                        "arm_p_true": arm_p,
                        "best_arm": best_arm,
                        "best_arm_p": arm_p[best_arm],
                        "expected_oracle_reward": expected_oracle,
                        "expected_oracle_score": expected_oracle / m,
                        "sample_path_oracle_reward": sample_path_oracle,
                        "sample_path_oracle_score": sample_path_oracle / m,
                    }) + "\n")
            except Exception:
                pass


def get_state_snapshot() -> dict[str, Any]:
    """Researcher accessor — full state including arm_p_true. Never surface."""
    with _state_lock:
        pulls = list(_state["pulls"])
        n = _state["n"]
        m = _state["m"]
        counts = [0] * n
        rewards_per_arm = [0] * n
        for arm, r in pulls:
            counts[arm] += 1
            rewards_per_arm[arm] += r
        observed_mean = [
            round(rewards_per_arm[i] / counts[i], 4) if counts[i] > 0 else None
            for i in range(n)
        ]
        total_reward = sum(r for _, r in pulls)
        return {
            "n": n,
            "m": m,
            "seed": _state["seed"],
            "arm_p_true": list(_state["arm_p"] or []),
            "best_arm": _state["best_arm"],
            "best_arm_p": (_state["arm_p"] or [0.0])[_state["best_arm"] or 0] if _state["arm_p"] else 0.0,
            "expected_oracle_reward": _state.get("expected_oracle"),
            "sample_path_oracle_reward": _state.get("sample_path_oracle"),
            "total_pulls": len(pulls),
            "total_reward": total_reward,
            "normalized_score": total_reward / m if m > 0 else 0.0,
            "counts": counts,
            "observed_mean": observed_mean,
            "pulls": pulls,
            "rejected": list(_state["rejected"]),
        }


def make_description(n: int, m: int) -> str:
    return (
        f"You are given {n} items, indexed 0..{n - 1}, and a draw budget of "
        f"{m} pulls.\n"
        "\n"
        f'Each candidate is a single integer in [0, {n}) (e.g. "3"), '
        "naming the item to draw next. After each pull, the evaluator "
        "returns the reward for that draw (a number) and tells you how "
        "many pulls remain.\n"
        "\n"
        f"You may make at most {m} pulls per evaluation budget. Your goal "
        "is to maximize the sum of rewards across all pulls.\n"
    )


def make_objective(n: int, m: int) -> str:
    return (
        f"Maximize the sum of values returned by pulling across at most "
        f"{m} pulls over {n} items."
    )


DESCRIPTION = make_description(DEFAULT_N, DEFAULT_M)
OBJECTIVE = make_objective(DEFAULT_N, DEFAULT_M)

INITIAL_CANDIDATE = "0"


def _parse_int(candidate: Any) -> int | None:
    try:
        return int(str(candidate).strip())
    except (ValueError, AttributeError, TypeError):
        return None


def evaluate(candidate: str) -> tuple[float, dict[str, Any]]:
    """Evaluate ONE pull. The candidate is a string-encoded item index."""
    with _state_lock:
        n = _state["n"]
        m = _state["m"]
        arm_p = _state["arm_p"]
        rng = _state["rng"]

        if rng is None or arm_p is None:
            # Defensive: configure() was never called. Lazy-init with defaults.
            rng = random.Random(_state["seed"])
            arm_p = [rng.random() for _ in range(n)]
            _state["rng"] = rng
            _state["arm_p"] = arm_p
            _state["best_arm"] = max(range(n), key=lambda i: arm_p[i])

        if _state["first_call_ts"] is None:
            _state["first_call_ts"] = time.time()

        pulls_made = len(_state["pulls"])
        pulls_left = max(m - pulls_made, 0)

        x = _parse_int(candidate)
        if x is None:
            _state["rejected"].append({"value": str(candidate)[:80], "reason": "not_int"})
            return 0.0, {
                "score": 0.0,
                "error": f"submission must be an integer in [0, {n})",
                "pulls_left": pulls_left,
                "pulls_total": m,
            }
        if x < 0 or x >= n:
            _state["rejected"].append({"value": x, "reason": "out_of_range"})
            return 0.0, {
                "score": 0.0,
                "error": f"out of range [0, {n})",
                "pulls_left": pulls_left,
                "pulls_total": m,
            }
        if pulls_made >= m:
            # Already made M valid pulls — return 0 and let the server's
            # budget check decide what happens next. Don't refund this call.
            return 0.0, {
                "score": 0.0,
                "error": "pull budget exhausted",
                "pulls_left": 0,
                "pulls_total": m,
            }

        reward = 1 if rng.random() < arm_p[x] else 0
        _state["pulls"].append((int(x), int(reward)))
        new_pulls_left = max(m - len(_state["pulls"]), 0)

        # --- Streaming sidecar — NEVER surfaced.
        log_path = os.environ.get(_RESEARCH_LOG_ENV)
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ts": time.time(),
                        "pull_index": len(_state["pulls"]),
                        "x": int(x),
                        "reward": int(reward),
                        "pulls_left": new_pulls_left,
                        "pulls_total": m,
                    }) + "\n")
            except Exception:
                pass

        return float(reward), {
            "score": float(reward),
            "pulls_left": new_pulls_left,
            "pulls_total": m,
        }


TASK = register_task(Task(
    name="slot_machines",
    objective=OBJECTIVE,
    background=DESCRIPTION,
    initial_candidate=INITIAL_CANDIDATE,
    eval_fn=evaluate,
    metadata={
        "type": "single_task",
        "candidate_type": "integer",
        "n": DEFAULT_N,
        "m": DEFAULT_M,
        "seed": DEFAULT_SEED,
    },
))
