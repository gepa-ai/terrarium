"""Run a reduced benchmark smoke matrix with real task evaluators.

This is intentionally small: one eval per task, reduced dataset sizes, and no
Circle Packing. It verifies that each benchmark mode produces truthful run
artifacts and that cheap tasks return meaningful evaluator output.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_ADAPTER = REPO_ROOT / "scripts" / "initial_candidate_adapter.py"


@dataclass(frozen=True)
class SmokeCase:
    name: str
    task: str
    mode: str
    overrides: tuple[str, ...]
    meaningful_check: str


CASES = (
    SmokeCase(
        name="aime_math_mini",
        task="aime_math_mini",
        mode="generalization",
        overrides=("task.train_limit=1", "task.val_limit=0", "task.test_limit=1"),
        meaningful_check="test_score_positive",
    ),
    SmokeCase(
        name="arc_agi",
        task="arc_agi",
        mode="generalization",
        overrides=("task.train_limit=1", "task.val_limit=0", "task.test_limit=1"),
        meaningful_check="best_score_positive",
    ),
    SmokeCase(
        name="cloudcast",
        task="cloudcast",
        mode="multi_task",
        overrides=("+task.train_limit=1", "+task.val_limit=1"),
        meaningful_check="best_score_positive",
    ),
    SmokeCase(
        name="frontier_cs_one_problem",
        task="frontier_cs_algo_smoke",
        mode="multi_task",
        overrides=("+task.train_limit=1",),
        meaningful_check="frontier_judge_success",
    ),
)


def _run_case(case: SmokeCase, output_root: Path, *, source_zshrc: bool) -> dict[str, Any]:
    run_dir = output_root / case.name
    command = [
        sys.executable,
        "-m",
        "terrarium",
        f"task={case.task}",
        "adapter=custom",
        f"adapter.path={CUSTOM_ADAPTER}",
        "budget.max_evals=1",
        f"hydra.run.dir={run_dir}",
        *case.overrides,
    ]
    prefix = "source ~/.zshrc >/dev/null 2>&1 || true; " if source_zshrc else ""
    shell_command = (
        f"{prefix}cd {shlex.quote(str(REPO_ROOT))}; "
        f"HYDRA_FULL_ERROR=1 PYTHONPATH=src {' '.join(shlex.quote(arg) for arg in command)}"
    )
    completed = subprocess.run(
        ["bash", "-lc", shell_command],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=900,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{case.name} failed with exit code {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    summary_path = run_dir / "summary.json"
    eval_path = run_dir / "evals" / "0.json"
    summary = json.loads(summary_path.read_text())
    eval_record = json.loads(eval_path.read_text())
    _validate_case(case, summary, eval_record)
    return {
        "case": case.name,
        "task": summary["task"],
        "mode": summary["benchmark"]["mode"],
        "best_score": summary["best_score"],
        "test_score": summary["test_score"],
        "budget_used": summary["budget"]["used"],
        "eval_status": eval_record.get("info", {}).get("status"),
        "output_dir": str(run_dir),
    }


def _validate_case(case: SmokeCase, summary: dict[str, Any], eval_record: dict[str, Any]) -> None:
    if summary["benchmark"]["mode"] != case.mode:
        raise AssertionError(f"{case.name}: expected mode {case.mode}, got {summary['benchmark']['mode']}")
    if summary["access_policy"]["execution"] != "unsandboxed":
        raise AssertionError(f"{case.name}: unexpected execution policy")
    if summary["access_policy"]["network"] != "model_api_and_eval_server_host_shared":
        raise AssertionError(f"{case.name}: unexpected network policy")
    if summary.get("sandbox_scope") is None:
        raise AssertionError(f"{case.name}: missing sandbox_scope")
    if summary["budget"]["used"] != 1:
        raise AssertionError(f"{case.name}: expected exactly one search eval")
    if case.mode == "generalization" and summary["test_score"] is None:
        raise AssertionError(f"{case.name}: generalization smoke missing hidden test score")
    if case.mode == "multi_task" and summary["test_score"] is not None:
        raise AssertionError(f"{case.name}: multi_task smoke should not have hidden test score")

    info = eval_record.get("info", {})
    if case.meaningful_check == "test_score_positive" and not summary["test_score"] > 0:
        raise AssertionError(f"{case.name}: expected positive hidden test score")
    if case.meaningful_check == "best_score_positive" and not summary["best_score"] > 0:
        error = info.get("error")
        raise AssertionError(f"{case.name}: expected positive best score; eval error={error!r}")
    if case.meaningful_check == "frontier_judge_success":
        if info.get("status") != "EvaluationStatus.SUCCESS":
            raise AssertionError(f"{case.name}: Frontier-CS judge did not return success: {info}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "smoke"
        / f"benchmark_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--no-source-zshrc",
        action="store_true",
        help="Do not source ~/.zshrc before each run. ARC needs this when API keys live there.",
    )
    args = parser.parse_args()

    if sys.version_info < (3, 11):
        raise SystemExit("Frontier-CS smoke requires Python >= 3.11 because frontier-cs requires it.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = [
        _run_case(case, args.output_root, source_zshrc=not args.no_source_zshrc)
        for case in CASES
    ]
    summary_path = args.output_root / "matrix_summary.json"
    summary_path.write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path), "results": results}, indent=2))


if __name__ == "__main__":
    main()
