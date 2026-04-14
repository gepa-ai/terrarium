"""Terrarium runner: library ``run()`` API + Hydra-managed ``main()`` entry point.

Two ways to drive a run:

1. **Library**::

       from terrarium import run
       result = run("circle_packing", adapter, max_evals=100)
       result = run("circle_packing", adapter, max_token_cost=5.0)
       result = run("circle_packing", adapter, max_evals=200, max_token_cost=10.0)

2. **CLI** (via ``python -m terrarium`` — see ``__main__.py``)::

       python -m terrarium task=aime_math adapter=claude_code budget.max_evals=200
       python -m terrarium adapter=gepa budget.max_token_cost=5.0
       python -m terrarium budget.max_evals=200 budget.max_token_cost=10.0

   Hydra composes the config from ``terrarium/conf/`` and calls :func:`main`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from terrarium.adapter import Adapter, Result
from terrarium.budget import BudgetExhausted, BudgetTracker
from terrarium.eval_server import EvalServer
from terrarium.registry import get_task
from terrarium.task import Task
from terrarium.tracking import TerrariumTracker, TrackingConfig

CONFIG_PATH = str(Path(__file__).parent / "conf")


def run(
    task: str | Task,
    adapter: str | Adapter,
    *,
    max_evals: int | None = 100,
    max_token_cost: float | None = None,
    max_concurrency: int = 8,
    tracking: TrackingConfig | None = None,
    output_dir: str | Path | None = None,
) -> Result:
    """Run an evolution system on a task.

    Loads task + adapter, creates the ``EvalServer`` (single budget choke point),
    starts the HTTP endpoint, and calls ``adapter.evolve(task, server)``.

    Args:
        task: Task name (e.g. "circle_packing") or a Task object.
        adapter: Path to an adapter .py file (must define ``create_adapter()``)
            or an Adapter object.
        max_evals: Maximum eval calls allowed. ``None`` for unlimited.
        max_token_cost: Maximum USD spend on adapter LLM tokens. ``None`` for
            unlimited. Enforcement is adapter-side (GEPA via
            ``max_reflection_cost``, Claude Code via ``--max-budget-usd``).
        max_concurrency: Max parallel evaluations in the server.
        tracking: Optional wandb/mlflow tracking config.
        output_dir: Directory where incremental log files (``eval_log.jsonl``,
            ``reflection_cost_log.jsonl``) are written during the run.

    Returns:
        :class:`Result` with ``best_score``, ``best_candidate``, ``total_evals``,
        ``eval_log``, and metadata.

    Example::

        result = run("circle_packing", "my_adapter.py", max_evals=100)
        result = run("circle_packing", "my_adapter.py", max_token_cost=5.0)
        result = run("circle_packing", "my_adapter.py", max_evals=200, max_token_cost=10.0)
    """
    if isinstance(task, str):
        task = get_task(task)

    if isinstance(adapter, str):
        adapter = load_adapter(adapter)

    tracker = TerrariumTracker(tracking) if tracking else None
    budget = BudgetTracker(max_evals=max_evals, max_token_cost=max_token_cost)
    server = EvalServer(task, budget, tracker=tracker, max_concurrency=max_concurrency, output_dir=output_dir)
    server.start()

    if tracker:
        tracker.start({"task": task.name, "max_evals": max_evals, "max_token_cost": max_token_cost})
        # Inject GEPA-level callback for iteration/valset metrics
        from terrarium.adapters.gepa import GEPAAdapter

        if isinstance(adapter, GEPAAdapter):
            adapter.callbacks.append(tracker.create_callback())

    start = time.time()

    try:
        result = adapter.evolve(task, server)
    except BudgetExhausted:
        result = Result(
            best_candidate=server.best_candidate,
            best_score=server.best_score,
            total_evals=budget.used,
            eval_log=server.eval_log,
        )
    finally:
        server.stop()

    result.total_evals = budget.used
    result.eval_log = server.eval_log
    result.metadata["wall_time"] = time.time() - start
    result.metadata["budget"] = budget.status()
    result.metadata["total_cost"] = server.total_cost
    result.metadata["progress_log"] = server.progress_log

    # Evaluate best candidate on held-out test set (outside budget).
    if isinstance(task, Task) and task.test_set and result.best_candidate:
        test_scores: dict[str, float] = {}
        for ex in task.test_set:
            try:
                score, _ = task.eval_fn(result.best_candidate, ex)
                test_scores[ex.id] = score
            except Exception:
                test_scores[ex.id] = 0.0
        result.metadata["test_scores"] = test_scores
        result.metadata["test_score"] = sum(test_scores.values()) / len(test_scores) if test_scores else 0.0

    if tracker:
        tracker.log_summary({
            "best_score": result.best_score,
            "total_evals": result.total_evals,
            "wall_time": result.metadata["wall_time"],
            "total_cost": server.total_cost,
        })
        tracker.end()

    return result


def load_adapter(path: str, **_: Any) -> Adapter:
    """Load an Adapter from a Python file.

    The file must define ``create_adapter() -> Adapter``. Extra kwargs are
    ignored (Hydra's ``_target_`` may pass internals like ``_partial_``).
    """
    path = str(Path(path).resolve())
    spec = importlib.util.spec_from_file_location("_terrarium_adapter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load adapter from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_terrarium_adapter"] = mod
    spec.loader.exec_module(mod)

    if not hasattr(mod, "create_adapter"):
        raise AttributeError(f"Adapter file {path} must define create_adapter() -> Adapter")

    return mod.create_adapter()


def _build_tracking_config(cfg: DictConfig) -> TrackingConfig | None:
    if not cfg.get("enabled", False):
        return None
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    raw.pop("enabled", None)
    return TrackingConfig(**raw)  # type: ignore[arg-type]



@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point — invoked by ``python -m terrarium``.

    Layout on disk (per run)::

        outputs/terrarium/<task>/<timestamp>/
          .hydra/                 # composed config, overrides, hydra log
          summary.json            # task, adapter, scores, eval_log, best_candidate
          <adapter_name>/         # whatever the adapter writes (GEPA run dir,
                                  # claude_code work files, etc.)
    """
    print(OmegaConf.to_yaml(cfg))

    hydra_cfg = HydraConfig.get()
    hydra_out = Path(hydra_cfg.runtime.output_dir)
    # Group name chosen from `defaults: - adapter: <name>` (or CLI override).
    adapter_name = hydra_cfg.runtime.choices.get("adapter", "adapter")
    adapter_dir = hydra_out / adapter_name

    adapter = instantiate(cfg.adapter)

    # Inject the per-run artifact dir into the adapter. Any adapter that
    # persists files can declare a ``run_dir`` attribute and the runner will
    # set it to <hydra_out>/<adapter_name>/, unless the adapter yaml already
    # set a non-null value (explicit user override wins).
    if hasattr(adapter, "run_dir") and not getattr(adapter, "run_dir", None):
        adapter.run_dir = str(adapter_dir)

    tracking = _build_tracking_config(cfg.tracking) if "tracking" in cfg else None

    # Tag wandb runs with the task name by default.
    if tracking and not tracking.wandb_tags:
        tracking.wandb_tags = [cfg.task.name]

    # Configure task-level solver LM (e.g. for aime_math's dspy evaluator).
    solver_lm = cfg.task.get('solver_lm')
    if solver_lm:
        import dspy
        lm_kwargs = {}
        if cfg.task.get('solver_temperature') is not None:
            lm_kwargs['temperature'] = cfg.task.solver_temperature
        if cfg.task.get('solver_max_tokens') is not None:
            lm_kwargs['max_tokens'] = cfg.task.solver_max_tokens
        dspy.configure(lm=dspy.LM(solver_lm, **lm_kwargs))

    result = run(
        cfg.task.name,
        adapter,
        max_evals=cfg.budget.get("max_evals"),
        max_token_cost=cfg.budget.get("max_token_cost"),
        max_concurrency=cfg.max_concurrency,
        tracking=tracking,
        output_dir=hydra_out,
    )

    summary: dict[str, Any] = {
        "task": cfg.task.name,
        "adapter": adapter_name,
        "adapter_target": cfg.adapter.get("_target_", "unknown"),
        "adapter_dir": str(adapter_dir),
        "best_score": result.best_score,
        "total_evals": result.total_evals,
        "total_cost": result.metadata.get("total_cost", 0.0),
        "reflection_cost_log": result.metadata.get("reflection_cost_log"),
        "wall_time": result.metadata.get("wall_time"),
        "budget": result.metadata.get("budget"),
        "test_score": result.metadata.get("test_score"),
        "test_scores": result.metadata.get("test_scores"),
        "progress_log": result.metadata.get("progress_log"),
        "eval_log": result.eval_log,
        "best_candidate": result.best_candidate,
    }

    summary_path = hydra_out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    # Compact stdout summary — drop eval_log/best_candidate which can be large.
    compact = {k: v for k, v in summary.items() if k not in ("eval_log", "best_candidate")}
    print(json.dumps(compact, indent=2, default=str))
    print(f"\nRun artifacts: {hydra_out}")
    print(f"Summary:       {summary_path}")

    if cfg.get("output"):
        out_path = Path(cfg.output)
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"Copy at:       {out_path}")


if __name__ == "__main__":
    main()
