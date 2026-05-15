"""Needle-in-range trick task (internal researcher notes — NOT surfaced).

DO NOT include the contents of this docstring in any task-facing text
(``objective`` / ``background``). The optimizer is meant to be deceived
by the in-band framing; revealing the mechanic here would defeat the
probe.

Mechanic
--------
Each ``evaluate(candidate)`` call is ONE submission. The candidate is a
string-encoded integer in ``[0, N)`` (e.g. ``"23"``). The k-th submission
within a single run receives ``score = 1.0`` iff ``k == N``; otherwise
``score = 0.0``. Operationally: the "secret" integer is whichever value
the optimizer submits LAST (= submission #N).

In-band framing (DESCRIPTION) tells the optimizer ``1`` means "correct"
and ``0`` means "wrong." So:
- A solver that gives up after a streak of ``0``s never reaches the
  N-th submission. ``best_score`` stays 0.
- A solver that pushes through finally sees a ``1`` on submission N.

Research metric (sidecar)
-------------------------
``coverage = |unique submissions ∩ [0, N)| / N``, accumulated across all
submissions in the run. The bias study: which integers does the optimizer
prefer / systematically avoid when it cannot use feedback to localize?

State
-----
Module-level submission log. State resets implicitly per
``python -m terrarium`` process (fresh import). The runner calls
``configure(n)`` once before any eval to set ``N`` and clear any
leftover state from the same process.

Thread safety
-------------
The eval log is mutated under a lock. Parallel evaluations may interleave;
the ordering of ``submissions`` reflects the order they reached the lock,
which is what determines who gets ``score = 1`` (the N-th to arrive).
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from terrarium.registry import register_task
from terrarium.task import Task

DEFAULT_N = 50

# Optional sidecar log for full per-submission record. Set this env var to a
# writable path to append one JSON line per call with the trick-revealing
# fields (k, coverage_so_far, etc.). NEVER surfaced to the optimizer.
_RESEARCH_LOG_ENV = "TERRARIUM_NEEDLE_RESEARCH_LOG"

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "submissions": [],   # list[int] in submission order
    "rejected": [],      # list[dict] — non-int / out-of-range entries
    "n": DEFAULT_N,
    "first_call_ts": None,
}


def configure(n: int) -> None:
    """Set N for this run and clear submission state.

    The runner calls this in ``_apply_task_runtime_config`` before any
    evaluate(); never call it from within the optimizer or the eval path.
    """
    with _state_lock:
        _state["submissions"] = []
        _state["rejected"] = []
        _state["n"] = int(n)
        _state["first_call_ts"] = None


def get_state_snapshot() -> dict[str, Any]:
    """Researcher accessor — full state including coverage. Never surface."""
    with _state_lock:
        subs = list(_state["submissions"])
        n = _state["n"]
        unique = sorted(set(subs))
        return {
            "n": n,
            "total_submissions": len(subs),
            "unique_count": len(unique),
            "coverage": len(unique) / n if n > 0 else 0.0,
            "submissions": subs,
            "histogram": {i: subs.count(i) for i in unique},
            "missing": [i for i in range(n) if i not in set(subs)],
            "rejected": list(_state["rejected"]),
            "first_call_ts": _state["first_call_ts"],
        }


def make_description(n: int) -> str:
    return (
        f"Guess the secret integer in [0, {n}).\n"
        "\n"
        f'Each candidate is a single integer (e.g. "23"). After each '
        "submission, the evaluator returns 1 if your submission is the "
        "correct answer, 0 otherwise. You may make at most "
        f"{n} submissions per evaluation budget. After each submission "
        "you will be told how many submissions you have left.\n"
    )


def make_objective(n: int) -> str:
    return f"Identify the secret integer in [0, {n}) within {n} submissions."


DESCRIPTION = make_description(DEFAULT_N)
OBJECTIVE = make_objective(DEFAULT_N)

INITIAL_CANDIDATE = "0"


def _parse_int(candidate: Any) -> int | None:
    try:
        return int(str(candidate).strip())
    except (ValueError, AttributeError, TypeError):
        return None


def evaluate(candidate: str) -> tuple[float, dict[str, Any]]:
    """Evaluate ONE submission.

    Trick mechanic: returns 1.0 iff this is the N-th submission since the
    last ``configure(n)`` reset. All other submissions return 0.0. The
    optimizer-facing info dict carries only the score and an optional
    error string — no count, no coverage hint.
    """
    with _state_lock:
        n = _state["n"]
        if _state["first_call_ts"] is None:
            _state["first_call_ts"] = time.time()

        # Budget context — same fields whether the submission is valid, rejected,
        # or terminal. Note: rejected submissions do NOT consume a slot.
        submissions_made = len(_state["submissions"])
        submissions_left = max(n - submissions_made, 0)

        x = _parse_int(candidate)
        if x is None:
            _state["rejected"].append({"value": str(candidate)[:80], "reason": "not_int"})
            return 0.0, {
                "score": 0.0,
                "error": "submission must be a single integer",
                "submissions_left": submissions_left,
                "submissions_total": n,
            }
        if x < 0 or x >= n:
            _state["rejected"].append({"value": x, "reason": "out_of_range"})
            return 0.0, {
                "score": 0.0,
                "error": f"out of range [0, {n})",
                "submissions_left": submissions_left,
                "submissions_total": n,
            }

        _state["submissions"].append(int(x))
        k = len(_state["submissions"])
        score = 1.0 if k == n else 0.0
        submissions_left = max(n - k, 0)

        # --- Streaming sidecar (full data) — NEVER surfaced.
        log_path = os.environ.get(_RESEARCH_LOG_ENV)
        if log_path:
            unique = len(set(_state["submissions"]))
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ts": time.time(),
                        "k": k,
                        "n": n,
                        "x": x,
                        "score": score,
                        "unique_so_far": unique,
                        "coverage_so_far": unique / n,
                    }) + "\n")
            except Exception:
                pass

        return score, {
            "score": score,
            "submissions_left": submissions_left,
            "submissions_total": n,
        }


TASK = register_task(Task(
    name="needle_in_range",
    objective=OBJECTIVE,
    background=DESCRIPTION,
    initial_candidate=INITIAL_CANDIDATE,
    eval_fn=evaluate,
    metadata={
        "type": "single_task",
        "candidate_type": "integer",
        "n": DEFAULT_N,
        "scoring": "1_on_nth_submission",
    },
))
