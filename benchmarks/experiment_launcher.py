"""Launch Terrarium experiment matrices and aggregate run summaries.

The launcher intentionally shells out to ``python -m terrarium`` instead of
calling runner internals. That keeps every experiment on the same Hydra entry
point users run by hand, while this module only handles matrix expansion,
process scheduling, logs, and summary aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments"


@dataclass(frozen=True)
class ExperimentRun:
    name: str
    task: str
    algorithm: str
    seed: int | None
    output_dir: Path
    overrides: tuple[str, ...]
    command: tuple[str, ...]
    source_zshrc: bool


def load_config(path: Path) -> dict[str, Any]:
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


def expand_runs(config: dict[str, Any], *, config_path: Path | None = None) -> list[ExperimentRun]:
    matrix_name = str(config.get("name") or (config_path.stem if config_path else "experiment"))
    output_root = Path(str(config.get("output_root") or DEFAULT_OUTPUT_ROOT / matrix_name))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_root / timestamp
    defaults = _as_str_tuple(config.get("defaults", []))
    budgets = _expand_budgets(config.get("budgets", config.get("budget", {})))
    tasks = [_normalize_entry(item, "task") for item in _require_list(config, "tasks")]
    algorithms = [_normalize_entry(item, "algorithm") for item in _require_list(config, "algorithms")]
    seeds = config.get("seeds", [None])
    if seeds is None:
        seeds = [None]
    if not isinstance(seeds, list | tuple | ListConfig):
        seeds = [seeds]

    source_zshrc = bool(config.get("source_zshrc", True))
    runs: list[ExperimentRun] = []
    for task in tasks:
        for algorithm in algorithms:
            for budget_name, budget_overrides in budgets:
                for raw_seed in seeds:
                    seed = None if raw_seed is None else int(raw_seed)
                    name_parts = [task["name"], algorithm["name"]]
                    if budget_name:
                        name_parts.append(budget_name)
                    if seed is not None:
                        name_parts.append(f"seed{seed}")
                    run_name = "__".join(_slug(part) for part in name_parts)
                    output_dir = run_root / run_name
                    overrides = (
                        f"task={task['task']}",
                        f"adapter={algorithm['algorithm']}",
                        *defaults,
                        *task["overrides"],
                        *algorithm["overrides"],
                        *budget_overrides,
                        *_seed_overrides(algorithm, seed),
                        f"hydra.run.dir={output_dir}",
                    )
                    command = (sys.executable, "-m", "terrarium", *overrides)
                    runs.append(
                        ExperimentRun(
                            name=run_name,
                            task=task["name"],
                            algorithm=algorithm["name"],
                            seed=seed,
                            output_dir=output_dir,
                            overrides=tuple(overrides),
                            command=command,
                            source_zshrc=source_zshrc,
                        )
                    )
    return runs


def run_matrix(
    runs: list[ExperimentRun],
    *,
    max_workers: int,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    if dry_run:
        return [_dry_run_record(run) for run in runs]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        pending: dict[Future[dict[str, Any]], ExperimentRun] = {
            executor.submit(_run_one, run): run for run in runs
        }
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                run = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive aggregation
                    result = _base_record(run, "launcher_error", wall_time=None)
                    result["error"] = repr(exc)
                results.append(result)
                _print_status(result)
    return sorted(results, key=lambda row: row["name"])


def write_aggregate(results: list[dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "results.json"
    csv_path = output_root / "results.csv"
    json_path.write_text(json.dumps({"results": results}, indent=2, default=str) + "\n")

    fieldnames = [
        "name",
        "status",
        "task",
        "algorithm",
        "seed",
        "best_score",
        "test_score",
        "total_evals",
        "solver_cost",
        "solver_cost_search",
        "solver_cost_test",
        "optimizer_cost",
        "total_cost",
        "wall_time",
        "output_dir",
        "summary_path",
        "error",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"Aggregate JSON: {json_path}")
    print(f"Aggregate CSV:  {csv_path}")


def _run_one(run: ExperimentRun) -> dict[str, Any]:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run.output_dir / "launcher.stdout.log"
    stderr_path = run.output_dir / "launcher.stderr.log"
    command_path = run.output_dir / "launcher.command.txt"
    command_path.write_text(_display_command(run) + "\n")

    start = time.time()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        completed = subprocess.run(
            _subprocess_command(run),
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": _pythonpath()},
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    wall_time = time.time() - start

    if completed.returncode != 0:
        record = _base_record(run, "failed", wall_time=wall_time)
        record["returncode"] = completed.returncode
        record["error"] = f"terrarium exited with {completed.returncode}; see {stderr_path}"
        return record

    summary_path = run.output_dir / "summary.json"
    if not summary_path.exists():
        record = _base_record(run, "missing_summary", wall_time=wall_time)
        record["error"] = f"missing {summary_path}"
        return record

    summary = json.loads(summary_path.read_text())
    return _summary_record(run, summary, wall_time=wall_time, status="completed")


def _subprocess_command(run: ExperimentRun) -> list[str]:
    if not run.source_zshrc:
        return list(run.command)
    shell_command = (
        "source ~/.zshrc >/dev/null 2>&1 || true; "
        f"PYTHONPATH={shlex.quote(_pythonpath())} "
        + " ".join(shlex.quote(part) for part in run.command)
    )
    return ["bash", "-lc", shell_command]


def _display_command(run: ExperimentRun) -> str:
    return " ".join(shlex.quote(part) for part in _subprocess_command(run))


def _pythonpath() -> str:
    current = os.environ.get("PYTHONPATH")
    src = str(REPO_ROOT / "src")
    return src if not current else f"{src}{os.pathsep}{current}"


def _summary_record(run: ExperimentRun, summary: dict[str, Any], *, wall_time: float, status: str) -> dict[str, Any]:
    return {
        **_base_record(run, status, wall_time=wall_time),
        "summary_path": str(run.output_dir / "summary.json"),
        "best_score": summary.get("best_score"),
        "test_score": summary.get("test_score"),
        "total_evals": summary.get("total_evals"),
        "solver_cost": summary.get("solver_cost"),
        "solver_cost_search": summary.get("solver_cost_search"),
        "solver_cost_test": summary.get("solver_cost_test"),
        "optimizer_cost": summary.get("optimizer_cost"),
        "total_cost": summary.get("total_cost"),
        "adapter": summary.get("adapter"),
        "benchmark": summary.get("benchmark"),
        "budget": summary.get("budget"),
    }


def _base_record(run: ExperimentRun, status: str, *, wall_time: float | None) -> dict[str, Any]:
    return {
        "name": run.name,
        "status": status,
        "task": run.task,
        "algorithm": run.algorithm,
        "seed": run.seed,
        "wall_time": wall_time,
        "output_dir": str(run.output_dir),
        "summary_path": None,
        "best_score": None,
        "test_score": None,
        "total_evals": None,
        "solver_cost": None,
        "solver_cost_search": None,
        "solver_cost_test": None,
        "optimizer_cost": None,
        "total_cost": None,
        "error": None,
    }


def _dry_run_record(run: ExperimentRun) -> dict[str, Any]:
    record = _base_record(run, "dry_run", wall_time=None)
    record["command"] = _display_command(run)
    return record


def _print_status(result: dict[str, Any]) -> None:
    if result["status"] == "completed":
        print(
            f"[completed] {result['name']} "
            f"best={result.get('best_score')} test={result.get('test_score')} "
            f"cost={result.get('total_cost')} evals={result.get('total_evals')}"
        )
    else:
        print(f"[{result['status']}] {result['name']}: {result.get('error')}")


def _normalize_entry(item: Any, kind: str) -> dict[str, Any]:
    if isinstance(item, str):
        return {"name": item, kind: item, "overrides": (), "seed_overrides": ()}
    if not isinstance(item, dict | DictConfig):
        raise TypeError(f"{kind} entries must be strings or mappings")
    data = dict(item)
    raw_name = data.get("name") or data.get(kind)
    if raw_name is None:
        raise ValueError(f"{kind} entry missing name")
    hydra_group = data.get(kind) or raw_name
    return {
        "name": str(raw_name),
        kind: str(hydra_group),
        "overrides": _as_str_tuple(data.get("overrides", [])),
        "seed_overrides": _as_str_tuple(data.get("seed_overrides", [])),
    }


def _expand_budgets(raw: Any) -> list[tuple[str | None, tuple[str, ...]]]:
    if raw is None:
        return [(None, ())]
    if isinstance(raw, dict | DictConfig):
        return [(None, _budget_overrides(dict(raw)))]
    if isinstance(raw, list | tuple | ListConfig):
        budgets = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict | DictConfig):
                raise TypeError("budget list entries must be mappings")
            data = dict(item)
            name = str(data.pop("name", f"budget{idx}"))
            budgets.append((name, _budget_overrides(data)))
        return budgets
    raise TypeError("budgets must be a mapping or list of mappings")


def _budget_overrides(data: dict[str, Any]) -> tuple[str, ...]:
    overrides = []
    for key in ("max_evals", "max_token_cost"):
        if key in data:
            value = data[key]
            overrides.append(f"budget.{key}={_hydra_value(value)}")
    extra = data.get("overrides", [])
    return (*overrides, *_as_str_tuple(extra))


def _seed_overrides(algorithm: dict[str, Any], seed: int | None) -> tuple[str, ...]:
    if seed is None:
        return ()
    return tuple(template.format(seed=seed) for template in algorithm.get("seed_overrides", ()))


def _require_list(config: dict[str, Any], key: str) -> list[Any]:
    value = config.get(key)
    if not isinstance(value, list | tuple | ListConfig) or not value:
        raise ValueError(f"config must define non-empty '{key}' list")
    return list(value)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list | tuple | ListConfig):
        raise TypeError(f"expected string or list of strings, got {type(value).__name__}")
    return tuple(str(item) for item in value)


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug or "run"


def _max_parallel_runs(config: dict[str, Any], cli_override: int | None) -> int:
    """Return launcher-level process parallelism.

    ``max_parallel_runs`` is the preferred config key. ``max_workers`` remains
    accepted for older matrix files, but is ambiguous with GEPA engine workers.
    """
    if cli_override is not None:
        return cli_override
    return int(config.get("max_parallel_runs", config.get("max_workers", 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML matrix config")
    parser.add_argument("--max-workers", type=int, default=None, help="Override concurrent Terrarium run processes")
    parser.add_argument("--dry-run", action="store_true", help="Expand and print commands without launching")
    args = parser.parse_args()

    config = load_config(args.config)
    runs = expand_runs(config, config_path=args.config)
    max_workers = _max_parallel_runs(config, args.max_workers)
    output_root = runs[0].output_dir.parent if runs else DEFAULT_OUTPUT_ROOT

    print(f"Expanded {len(runs)} run(s); max_parallel_runs={max_workers}; output_root={output_root}")
    results = run_matrix(runs, max_workers=max_workers, dry_run=args.dry_run)
    write_aggregate(results, output_root)

    failures = [row for row in results if row["status"] not in {"completed", "dry_run"}]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
