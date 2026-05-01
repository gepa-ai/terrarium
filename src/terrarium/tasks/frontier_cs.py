"""Frontier-CS algorithmic track: one terrarium task per problem.

Each problem in the Frontier-CS algorithmic benchmark (172 problems) is
registered as an independent single-task — because each problem has its own
statement, so a single candidate cannot generalize across problems.

Task naming:
    frontier_cs_algo_<problem_id>   e.g. frontier_cs_algo_1, frontier_cs_algo_107
    frontier_cs_algo_smoke          curated 3-problem dataset task for pipeline tests

Data source:
    Problem metadata (statements, configs, IDs) is pulled from the HuggingFace
    dataset ``FrontierCS/Frontier-CS`` (one-time ~378 MB download, cached by
    the ``datasets`` library thereafter).

Evaluation:
    Uses ``frontier_cs.SingleEvaluator`` with Docker backend. The evaluator
    needs the Frontier-CS repo cloned on disk (it reads ``algorithmic/problems/
    <id>/testdata/`` and ``chk.cc`` to run the judge). On first evaluation,
    terrarium auto-clones the repo to ``FRONTIER_CS_DIR`` (default:
    ``~/.cache/terrarium/Frontier-CS``) and ``SingleEvaluator`` itself spins
    up the judge via ``docker compose up -d``.

    Install prerequisites::

        pip install 'terrarium[frontier_cs]'
        # Docker must be installed and running. That's it — terrarium will
        # clone the repo on first use.

    Override the clone location::

        export FRONTIER_CS_DIR=/path/to/existing/clone

Usage::

    python -m terrarium task.name=frontier_cs_algo_107 adapter=gepa
    python -m terrarium task=frontier_cs_algo_smoke adapter=claude_code
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import subprocess
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

_FRONTIER_CS_REPO_URL = "https://github.com/FrontierCS/Frontier-CS"
_JUDGE_PORT = 8081
_JUDGE_URL = f"http://localhost:{_JUDGE_PORT}"

logger = logging.getLogger(__name__)

from terrarium.registry import register_task_factory
from terrarium.task import Example, Task

# Minimal compilable C++ seed. Starts from "the beginning" — the LM must
# implement the problem from scratch based on the statement embedded in the
# task description. Doing this (rather than seeding with examples/reference.cpp)
# both (a) guarantees we don't leak a known-good solution and (b) matches the
# HuggingFace dataset, which does not ship reference solutions.
_INITIAL_CANDIDATE = """\
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    // TODO: Read from stdin, compute the answer, write to stdout per the
    // problem statement. The judge will compile this file with g++ and run
    // it against the problem's testdata; a custom checker scores the output
    // in [0, 100] (higher is better; 100 = perfect). Optimization problems
    // interpolate between a baseline and best value.
    return 0;
}
"""

_OBJECTIVE = "Solve an algorithmic problem by writing a C++ program."

# Curated small subset for fast adapter smoke tests. Picks small numeric IDs
# (which tend to be earlier, shorter problems). Override by editing this list
# or by calling adapters directly with a specific frontier_cs_algo_<id>.
_SMOKE_IDS: tuple[str, ...] = ("0", "1", "2")


def _frontier_cs_dir() -> Path:
    """Return the Frontier-CS repo path (persistent cache location by default)."""
    env = os.environ.get("FRONTIER_CS_DIR")
    if env:
        return Path(env).expanduser()
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "terrarium" / "Frontier-CS"


def _ensure_repo(base: Path) -> None:
    """Clone the Frontier-CS repo into ``base`` if it isn't there yet.

    Idempotent — a directory with an ``algorithmic/`` subdirectory is treated
    as an existing clone. Raises RuntimeError with a hand-install hint if git
    isn't available or the clone fails.
    """
    if (base / "algorithmic").is_dir():
        return
    if shutil.which("git") is None:
        raise RuntimeError(
            "git is required to auto-clone Frontier-CS but was not found on PATH. "
            f"Install git, or manually clone {_FRONTIER_CS_REPO_URL} to {base} "
            "and set FRONTIER_CS_DIR."
        )
    base.parent.mkdir(parents=True, exist_ok=True)
    print(f"[terrarium] Cloning Frontier-CS into {base} (one-time)...", flush=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", _FRONTIER_CS_REPO_URL, str(base)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"git clone of {_FRONTIER_CS_REPO_URL} failed ({e}). "
            f"Clone manually to {base} and/or set FRONTIER_CS_DIR."
        ) from e


@lru_cache(maxsize=1)
def _algorithmic_rows() -> dict[str, dict[str, Any]]:
    """Load the Frontier-CS HF dataset; return algorithmic rows keyed by problem_id.

    Cached — the HF ``datasets`` library also caches the parquet on disk, so
    the first call downloads ~378 MB and subsequent calls are instant.
    """
    from datasets import load_dataset

    ds = load_dataset("FrontierCS/Frontier-CS", split="test")
    rows: dict[str, dict[str, Any]] = {}
    for row in ds:
        if row.get("category") != "algorithmic":
            continue
        rows[str(row["problem_id"])] = {
            "statement": row.get("statement", ""),
        }
    return rows


def _judge_is_alive() -> bool:
    """Check if the judge server is responding on the default port."""
    try:
        import requests
    except ImportError:
        return False

    try:
        r = requests.get(f"{_JUDGE_URL}/problems", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _ensure_shared_judge(base: Path) -> bool:
    """Start the judge server if not already running, using a cross-process file lock.

    Multiple terrarium processes can run in parallel. Without coordination they
    all race to ``docker compose up``, creating duplicate Docker networks and
    failing. This function serialises the startup: the first process to acquire
    the lock starts the judge; the rest wait for it to become healthy.
    """
    if _judge_is_alive():
        return True

    lock_path = base / "algorithmic" / ".terrarium_judge.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            # Another process may have started it while we waited for the lock.
            if _judge_is_alive():
                return True

            compose_dir = base / "algorithmic"
            logger.info("Starting shared judge server in %s", compose_dir)
            result = subprocess.run(
                ["docker", "compose", "up", "-d", "--wait"],
                cwd=compose_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                logger.error("docker compose failed: %s", result.stderr.strip())
                return False

            # Wait for the judge HTTP endpoint to become ready.
            deadline = time.time() + 60
            while time.time() < deadline:
                if _judge_is_alive():
                    logger.info("Judge server is ready")
                    return True
                time.sleep(2)

            logger.error("Judge server did not become ready within 60 s")
            return False
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def _evaluate(candidate: str, *, problem_id: str) -> tuple[float, dict[str, Any]]:
    """Evaluate a C++ candidate on one Frontier-CS algorithmic problem.

    Never raises — compile errors, timeouts, and evaluator exceptions all come
    back as ``(0.0, info)`` with diagnostic fields so reflection-based adapters
    (GEPA, Claude Code) can read the error and fix the code.
    """
    try:
        # NOTE: We construct AlgorithmicLocalRunner directly rather than going
        # through SingleEvaluator because SingleEvaluator.algorithmic_runner
        # (single_evaluator.py:113) doesn't forward base_dir to the runner \u2014
        # it only passes base_dir to the research/skypilot runners. Upstream
        # bug; bypass it by instantiating the runner ourselves.
        from frontier_cs.runner import AlgorithmicLocalRunner
    except ImportError as e:
        return 0.0, {
            "score": 0.0,
            "problem_id": problem_id,
            "status": "error",
            "message": (
                "frontier-cs package is not installed. Install with: "
                "pip install 'terrarium[frontier_cs]' "
                f"(underlying error: {e})"
            ),
            "logs": "",
        }

    base = _frontier_cs_dir()
    try:
        _ensure_repo(base)
    except RuntimeError as e:
        return 0.0, {
            "score": 0.0,
            "problem_id": problem_id,
            "status": "error",
            "message": str(e),
            "logs": "",
        }

    # Start the judge once across all parallel terrarium processes.
    if not _ensure_shared_judge(base):
        return 0.0, {
            "score": 0.0,
            "problem_id": problem_id,
            "status": "error",
            "message": "Could not start the Frontier-CS judge server (Docker).",
            "logs": "",
        }

    try:
        runner = AlgorithmicLocalRunner(base_dir=base, auto_start=False)
        result = runner.evaluate(str(problem_id), candidate)
    except Exception as e:
        return 0.0, {
            "score": 0.0,
            "problem_id": problem_id,
            "status": "error",
            "message": f"{type(e).__name__}: {e}",
            "logs": "",
        }

    score = float(result.score) if result.score is not None else 0.0
    return score, {
        "score": score,
        "problem_id": problem_id,
        "status": str(getattr(result, "status", "unknown")),
        "message": getattr(result, "message", None),
        "logs": getattr(result, "logs", None),
        "duration_seconds": getattr(result, "duration_seconds", None),
    }


def _make_problem_task(problem_id: str) -> Task:
    """Factory for a single-problem task. Runs on first get_task() access."""
    rows = _algorithmic_rows()
    if problem_id not in rows:
        raise KeyError(
            f"Frontier-CS algorithmic problem {problem_id!r} not found in HF dataset."
        )
    row = rows[problem_id]

    def eval_fn(candidate: str) -> tuple[float, dict[str, Any]]:
        return _evaluate(candidate, problem_id=problem_id)

    return Task(
        name=f"frontier_cs_algo_{problem_id}",
        objective=_OBJECTIVE,
        background=row["statement"],
        initial_candidate=_INITIAL_CANDIDATE,
        eval_fn=eval_fn,
        metadata={
            "type": "single_task",
            "candidate_type": "code",
            "language": "cpp",
            "frontier_cs_track": "algorithmic",
            "frontier_cs_problem_id": problem_id,
        },
    )


def _make_smoke_task() -> Task:
    """Curated subset as a dataset task — lets adapters smoke-test the full
    candidate→judge→score pipeline on a handful of problems before committing
    to a long run on a single one.

    Note: a single candidate won't generalize across problems (they have
    different statements). This task is for plumbing validation only.
    """
    rows = _algorithmic_rows()
    available = [pid for pid in _SMOKE_IDS if pid in rows]
    if not available:
        raise RuntimeError(
            f"None of the smoke IDs {_SMOKE_IDS} are in the Frontier-CS HF dataset. "
            f"Available IDs: {sorted(rows)[:10]}..."
        )

    examples = [
        Example(
            id=pid,
            inputs={"problem_id": pid, "statement": rows[pid]["statement"]},
            expected=None,
        )
        for pid in available
    ]
    def eval_fn(candidate: str, example: Example) -> tuple[float, dict[str, Any]]:
        return _evaluate(candidate, problem_id=example.inputs["problem_id"])

    statements = "\n\n---\n\n".join(
        f"### Problem {pid}\n\n{rows[pid]['statement']}" for pid in available
    )
    return Task(
        name="frontier_cs_algo_smoke",
        objective=_OBJECTIVE,
        background=statements,
        initial_candidate=_INITIAL_CANDIDATE,
        eval_fn=eval_fn,
        train_set=examples,
        metadata={
            "type": "multi_task",
            "candidate_type": "code",
            "language": "cpp",
            "frontier_cs_track": "algorithmic",
            "official": False,
            "purpose": "adapter pipeline smoke test",
        },
    )


def _register_all() -> None:
    """Register the smoke task + one factory per algorithmic problem.

    Registration is cheap: each factory captures just the problem_id. The HF
    dataset isn't loaded until a factory is actually invoked (via get_task()),
    and then it's loaded exactly once (cached).

    If the ``datasets`` package isn't installed, we skip silently — the
    Frontier-CS extra wasn't installed, and other terrarium tasks should still
    work.
    """
    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        return

    try:
        rows = _algorithmic_rows()
    except Exception as e:  # pragma: no cover - network/HF hub issues
        warnings.warn(
            f"Could not load Frontier-CS problem list from HuggingFace: {e}. "
            "Per-problem tasks (frontier_cs_algo_<id>) will not be registered. "
            "Other terrarium tasks are unaffected.",
            stacklevel=2,
        )
        return

    register_task_factory("frontier_cs_algo_smoke", _make_smoke_task)
    for pid in rows:
        # Default-arg binding pins pid into each lambda's closure.
        register_task_factory(
            f"frontier_cs_algo_{pid}",
            lambda p=pid: _make_problem_task(p),
        )


_register_all()
