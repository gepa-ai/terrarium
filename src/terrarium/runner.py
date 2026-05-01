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
import os
import sys
import time
from collections import Counter
from dataclasses import replace
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
from terrarium.task import Example, Task
from terrarium.tracking import TerrariumTracker, TrackingConfig

CONFIG_PATH = str(Path(__file__).parent / "conf")
BENCHMARK_MODES = {"single", "multi_task", "generalization"}
EXECUTION_POLICIES = {"unsandboxed", "sandboxed"}
HOST_SHARED_NETWORK_POLICIES = {"host_shared", "model_api_and_eval_server_host_shared"}
NETWORK_ISOLATED_POLICIES = {"network_namespace_isolated", "network_isolated", "none"}


def run(
    task: str | Task,
    adapter: str | Adapter,
    *,
    max_evals: int | None = 100,
    max_token_cost: float | None = None,
    max_concurrency: int = 8,
    benchmark: DictConfig | dict[str, Any] | None = None,
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
        benchmark: Optional benchmark contract config. If omitted, the task's
            metadata declares the mode, falling back to ``single``.
        tracking: Optional wandb/mlflow tracking config.
        output_dir: Directory where incremental log files (``evals/<i>.json``,
            ``summary.json``, ``reflection_cost_log.jsonl``) are written during
            the run.

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
    task = _prepare_task_for_benchmark(task, _benchmark_config(benchmark))
    official_task = task
    search_task = replace(task, test_set=None) if task.test_set else task

    if isinstance(adapter, str):
        adapter = load_adapter(adapter)

    tracker = TerrariumTracker(tracking) if tracking else None
    budget = BudgetTracker(max_evals=max_evals, max_token_cost=max_token_cost)
    server = EvalServer(search_task, budget, tracker=tracker, max_concurrency=max_concurrency, output_dir=output_dir)
    server.start()

    if tracker:
        tracker.start({"task": official_task.name, "max_evals": max_evals, "max_token_cost": max_token_cost})
        # Inject GEPA-level callback for iteration/valset metrics
        from terrarium.adapters.gepa import GEPAAdapter

        if isinstance(adapter, GEPAAdapter):
            adapter.callbacks.append(tracker.create_callback())

    start = time.time()

    try:
        result = adapter.evolve(search_task, server)
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
    adapter_cost = float(result.metadata.get("adapter_cost", 0.0))
    result.metadata["total_cost"] = server.total_cost + adapter_cost
    result.metadata["progress_log"] = server.progress_log

    # Hand the result back to the adapter for any artifact persistence
    # (logs, transcripts, workspace mirroring, tempdir cleanup, etc.).
    if output_dir is not None:
        adapter.process_result(result, Path(output_dir))

    # Evaluate best candidate on held-out test set (outside budget).
    if official_task.test_set and result.best_candidate:
        test_scores: dict[str, float] = {}
        for ex in official_task.test_set:
            try:
                score, _ = official_task.eval_fn(result.best_candidate, ex)
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
            "total_cost": result.metadata["total_cost"],
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


def _apply_perfect_score(adapter: Any, perfect_score: float | None) -> None:
    """Thread the top-level ``perfect_score`` into adapters that support early stopping.

    Sets ``adapter.stop_at_score`` when the adapter declares it and hasn't
    already been given a non-null value (user overrides win).
    """
    if perfect_score is None:
        return
    if hasattr(adapter, "stop_at_score") and getattr(adapter, "stop_at_score", None) is None:
        adapter.stop_at_score = float(perfect_score)


def _apply_effort(adapter: Any, effort: str | None) -> None:
    """Thread the top-level ``effort`` into whichever adapter knob exists.

    - ``adapter.effort`` (claude_code): set if currently ``None``.
    - ``adapter.reflection_lm_kwargs`` (gepa): ``setdefault`` the
      ``reasoning_effort`` key so user-supplied overrides still win.
    """
    if not effort:
        return
    if hasattr(adapter, "effort") and getattr(adapter, "effort", None) is None:
        adapter.effort = effort
    kwargs = getattr(adapter, "reflection_lm_kwargs", None)
    if isinstance(kwargs, dict):
        kwargs.setdefault("reasoning_effort", effort)


def _apply_sandbox(adapter: Any, sandbox: bool | None) -> None:
    """Thread the top-level ``sandbox`` flag into claude-code-based adapters.

    Sets ``adapter.sandbox`` when the adapter declares it and the caller hasn't
    already set a non-null value in the adapter yaml (user overrides win).
    """
    if sandbox is None:
        return
    if hasattr(adapter, "sandbox") and getattr(adapter, "sandbox", None) is None:
        adapter.sandbox = bool(sandbox)


def _apply_max_thinking_tokens(adapter: Any, max_thinking_tokens: int | None) -> None:
    """Thread the top-level ``max_thinking_tokens`` into adapters that support it.

    Sets ``adapter.max_thinking_tokens`` when the adapter declares it and
    hasn't already been given a non-null value (user overrides win).
    """
    if max_thinking_tokens is None:
        return
    if hasattr(adapter, "max_thinking_tokens") and getattr(adapter, "max_thinking_tokens", None) is None:
        adapter.max_thinking_tokens = int(max_thinking_tokens)


def _build_tracking_config(cfg: DictConfig) -> TrackingConfig | None:
    if not cfg.get("enabled", False):
        return None
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    raw.pop("enabled", None)
    return TrackingConfig(**raw)  # type: ignore[arg-type]


def _plain_config(cfg: Any) -> Any:
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg


def _benchmark_config(cfg: DictConfig | dict[str, Any] | None) -> DictConfig:
    if cfg is None:
        return OmegaConf.create({})
    if isinstance(cfg, DictConfig):
        return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    return OmegaConf.create(cfg)


def _effective_adapter_sandbox(adapter: Any, top_level_sandbox: bool | None) -> bool | None:
    effective = getattr(adapter, "effective_sandbox", None)
    if callable(effective):
        return effective(top_level_sandbox)
    if not hasattr(adapter, "sandbox"):
        return False
    value = getattr(adapter, "sandbox", None)
    if value is None:
        return None
    return bool(value)


def _sandbox_scope(adapter: Any, top_level_sandbox: bool | None) -> dict[str, bool]:
    scope = getattr(adapter, "sandbox_scope", None)
    if callable(scope):
        value = scope(top_level_sandbox)
        return {
            "optimizer_subprocess_sandbox": bool(value.get("optimizer_subprocess_sandbox", False)),
            "candidate_execution_sandbox": bool(value.get("candidate_execution_sandbox", False)),
            "network_namespace_isolated": bool(value.get("network_namespace_isolated", False)),
        }
    return {
        "optimizer_subprocess_sandbox": bool(_effective_adapter_sandbox(adapter, top_level_sandbox)),
        "candidate_execution_sandbox": False,
        "network_namespace_isolated": False,
    }


def _validate_access_policy(access_policy: DictConfig | None, adapter: Any, sandbox: bool | None) -> None:
    if not access_policy:
        return
    execution = access_policy.get("execution")
    network = access_policy.get("network")
    scope = _sandbox_scope(adapter, sandbox)
    if execution is not None and execution not in EXECUTION_POLICIES:
        raise ValueError("access_policy.execution must be one of: unsandboxed, sandboxed")
    if execution == "sandboxed" and not scope["candidate_execution_sandbox"]:
        raise ValueError(
            "access_policy.execution=sandboxed requires candidate/evaluator execution sandboxing. "
            "Use access_policy.execution=unsandboxed when only optimizer subprocesses are sandboxed."
        )
    if network in NETWORK_ISOLATED_POLICIES and not scope["network_namespace_isolated"]:
        raise ValueError(
            f"access_policy.network={network} requires network namespace isolation. "
            "Use network=model_api_and_eval_server_host_shared or network=host_shared for current sandboxes."
        )
    if network is not None and network not in HOST_SHARED_NETWORK_POLICIES | NETWORK_ISOLATED_POLICIES:
        raise ValueError(
            "access_policy.network must be one of: "
            "host_shared, model_api_and_eval_server_host_shared, "
            "network_namespace_isolated, network_isolated, none"
        )


def _split_limit(task_cfg: DictConfig | None, split_name: str) -> int | None:
    if task_cfg is None:
        return None
    value = task_cfg.get(f"{split_name}_limit")
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"task.{split_name}_limit must be >= 0")
    return value


def _limit_examples(examples: list[Example] | None, limit: int | None) -> list[Example] | None:
    if examples is None or limit is None:
        return examples
    return examples[:limit]


def _apply_task_split_limits(task: Task, task_cfg: DictConfig | None) -> Task:
    """Apply optional task.<split>_limit knobs for smoke/debug runs."""
    limits = {
        "train": _split_limit(task_cfg, "train"),
        "val": _split_limit(task_cfg, "val"),
        "test": _split_limit(task_cfg, "test"),
    }
    if all(limit is None for limit in limits.values()):
        return task

    metadata = dict(task.metadata)
    provenance = metadata.get("split_provenance")
    if isinstance(provenance, dict):
        provenance = dict(provenance)
        provenance["limited_for_run"] = {k: v for k, v in limits.items() if v is not None}
        provenance["run_split_sizes"] = {
            "train": len(_limit_examples(task.train_set, limits["train"]) or []),
            "val": len(_limit_examples(task.val_set, limits["val"]) or []),
            "test": len(_limit_examples(task.test_set, limits["test"]) or []),
        }
        metadata["split_provenance"] = provenance

    return replace(
        task,
        train_set=_limit_examples(task.train_set, limits["train"]),
        val_set=_limit_examples(task.val_set, limits["val"]),
        test_set=_limit_examples(task.test_set, limits["test"]),
        metadata=metadata,
    )


def _example_id_set(task: Task, split_name: str, split: list[Example] | None) -> set[str]:
    ids: list[str] = [ex.id for ex in split or []]
    duplicates = sorted(ex_id for ex_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"Task '{task.name}' has duplicate IDs in {split_name}_set: {preview}")
    return set(ids)


def _validate_generalization_splits(task: Task) -> None:
    if not task.train_set:
        raise ValueError(f"benchmark.mode=generalization requires task '{task.name}' to define train_set")
    if not task.test_set:
        raise ValueError(f"benchmark.mode=generalization requires task '{task.name}' to define test_set")

    split_ids = {
        "train": _example_id_set(task, "train", task.train_set),
        "val": _example_id_set(task, "val", task.val_set),
        "test": _example_id_set(task, "test", task.test_set),
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(split_ids[left] & split_ids[right])
        if overlap:
            preview = ", ".join(overlap[:5])
            raise ValueError(
                f"Task '{task.name}' has overlapping example IDs between "
                f"{left}_set and {right}_set: {preview}"
            )


def _normalize_benchmark_mode(mode: Any) -> str:
    if mode == "single_task":
        return "single"
    if mode in BENCHMARK_MODES:
        return str(mode)
    raise ValueError("benchmark.mode must be one of: single, multi_task, generalization")


def _prepare_task_for_benchmark(task: Task, benchmark_cfg: DictConfig | None) -> Task:
    """Apply the minimal benchmark contract before adapters see the task."""
    benchmark_cfg = benchmark_cfg or OmegaConf.create({})
    mode = _normalize_benchmark_mode(benchmark_cfg.get("mode") or task.metadata.get("type") or "single")
    use_val = bool(benchmark_cfg.get("use_val", True))

    metadata = dict(task.metadata)
    metadata_mode = metadata.get("type")

    if metadata_mode in BENCHMARK_MODES | {"single_task"} and _normalize_benchmark_mode(metadata_mode) != mode:
        raise ValueError(
            f"Task '{task.name}' declares metadata.type={metadata_mode!r}, "
            f"but benchmark.mode={mode!r}"
        )
    if "val_set" in metadata:
        raise ValueError(
            f"Task '{task.name}' stores validation examples in metadata['val_set']; "
            "use Task.val_set so all adapters get the same split contract."
        )

    if mode == "generalization":
        _validate_generalization_splits(task)
    elif mode == "multi_task" and not task.train_set:
        raise ValueError(f"benchmark.mode=multi_task requires task '{task.name}' to define train_set")

    val_set = task.val_set
    if not use_val:
        val_set = None

    test_set = task.test_set if mode == "generalization" else None

    metadata["type"] = mode
    prepared = replace(task, val_set=val_set, test_set=test_set, metadata=metadata)

    if mode == "single" and prepared.has_dataset:
        raise ValueError(
            f"benchmark.mode=single is inconsistent with dataset task '{task.name}'. "
            "Use benchmark.mode=multi_task or benchmark.mode=generalization."
        )

    return prepared


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

    try:
        hydra_out.mkdir(parents=True, exist_ok=True)
        (hydra_out / "run.pid").write_text(str(os.getpid()))
    except OSError:
        pass

    adapter = instantiate(cfg.adapter)

    # Inject the per-run artifact dir into the adapter. Any adapter that
    # persists files can declare a ``run_dir`` attribute and the runner will
    # set it to <hydra_out>/<adapter_name>/, unless the adapter yaml already
    # set a non-null value (explicit user override wins).
    if hasattr(adapter, "run_dir") and not getattr(adapter, "run_dir", None):
        adapter.run_dir = str(adapter_dir)

    # Thinking budget: ``max_thinking_tokens`` (fixed) and ``effort``
    # (adaptive) are mutually exclusive. When both are provided,
    # max_thinking_tokens wins and effort is ignored.
    max_thinking_tokens = cfg.get("max_thinking_tokens")
    effort = cfg.get("effort")
    if max_thinking_tokens is not None:
        if effort is not None:
            print(
                f"[terrarium] max_thinking_tokens={max_thinking_tokens} overrides "
                f"effort={effort} — using fixed thinking budget."
            )
        else:
            print(f"[terrarium] Fixed thinking budget: max_thinking_tokens={max_thinking_tokens}")
        _apply_max_thinking_tokens(adapter, max_thinking_tokens)
    elif effort is not None:
        _apply_effort(adapter, effort)

    _apply_perfect_score(adapter, cfg.get("perfect_score"))
    _apply_sandbox(adapter, cfg.get("sandbox"))
    _validate_access_policy(cfg.get("access_policy"), adapter, cfg.get("sandbox"))

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

    benchmark_cfg = OmegaConf.create(OmegaConf.to_container(cfg.get("benchmark", {}), resolve=True))
    if benchmark_cfg.get("mode") is None:
        benchmark_cfg.mode = cfg.task.get("mode")
    raw_task = _apply_task_split_limits(get_task(cfg.task.name), cfg.task)

    result = run(
        raw_task,
        adapter,
        max_evals=cfg.budget.get("max_evals"),
        max_token_cost=cfg.budget.get("max_token_cost"),
        max_concurrency=cfg.max_concurrency,
        benchmark=benchmark_cfg,
        tracking=tracking,
        output_dir=hydra_out,
    )

    summary: dict[str, Any] = {
        "task": cfg.task.name,
        "adapter": adapter_name,
        "adapter_target": cfg.adapter.get("_target_", "unknown"),
        "benchmark": _plain_config(benchmark_cfg),
        "access_policy": _plain_config(cfg.get("access_policy", {})),
        "sandbox_scope": _sandbox_scope(adapter, cfg.get("sandbox")),
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
