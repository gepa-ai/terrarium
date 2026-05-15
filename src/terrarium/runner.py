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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from terrarium.solver_lm import SolverBudgetExhausted
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
    solver_cost_tracker: Any | None = None,
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

    official_task = task
    _validate_task_contract(official_task)
    search_task = _task_for_search(official_task)

    if isinstance(adapter, str):
        adapter = load_adapter(adapter)

    tracker = TerrariumTracker(tracking) if tracking else None
    budget = BudgetTracker(max_evals=max_evals, max_token_cost=max_token_cost)
    server = EvalServer(search_task, budget, tracker=tracker, max_concurrency=max_concurrency, output_dir=output_dir)
    server.start()

    if tracker:
        tracker.start({"task": official_task.name, "max_evals": max_evals, "max_token_cost": max_token_cost})
        # Inject GEPA-level callback for iteration/valset metrics. Both the
        # native GEPAAdapter and the OmniAdapter (when ``backend=gepa``)
        # surface a ``.callbacks`` list that flows into the GEPA engine.
        from terrarium.adapters.gepa import GEPAAdapter
        from terrarium.adapters.omni import OmniAdapter

        if isinstance(adapter, GEPAAdapter | OmniAdapter):
            adapter.callbacks.append(tracker.create_callback())

    start = time.time()

    stop_reason: str | None = None
    try:
        result = adapter.evolve(search_task, server)
    except SolverBudgetExhausted as e:
        stop_reason = "solver_budget_exhausted"
        result = Result(
            best_candidate=server.best_candidate,
            best_score=server.best_score,
            total_evals=budget.used,
            eval_log=server.eval_log,
            metadata={"stop_error": str(e)},
        )
    except BudgetExhausted:
        stop_reason = "budget_exhausted"
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
    if stop_reason is not None:
        result.metadata.setdefault("stop_reason", stop_reason)
    solver_cost = _solver_cost(solver_cost_tracker)
    result.metadata["eval_cost"] = server.total_cost
    result.metadata["adapter_cost"] = adapter_cost
    result.metadata["solver_cost"] = solver_cost
    result.metadata["solver_cost_log"] = _solver_cost_log(solver_cost_tracker)
    result.metadata["total_cost"] = server.total_cost + adapter_cost + solver_cost
    if max_token_cost is not None:
        result.metadata["budget"]["optimizer_cost_used"] = adapter_cost
        result.metadata["budget"]["optimizer_budget_exhausted"] = adapter_cost >= float(max_token_cost)
    result.metadata["progress_log"] = server.progress_log
    result.metadata["best_visible_score"] = server.best_visible_score
    result.metadata["best_visible_source"] = server.best_visible_source
    result.metadata["best_validated_score"] = server.best_validated_score

    # Hand the result back to the adapter for any artifact persistence
    # (logs, transcripts, workspace mirroring, tempdir cleanup, etc.).
    if output_dir is not None:
        adapter.process_result(result, Path(output_dir))

    # Score the final submitted candidate on validation/test outside the search
    # eval budget. This makes dataset-task summaries report aggregate candidate
    # quality instead of the server's max individual-example score.
    if official_task.val_set and result.best_candidate:
        val_scores = _score_examples_unbudgeted(
            result,
            official_task,
            result.best_candidate,
            official_task.val_set,
            "val",
            max_concurrency,
        )
        result.metadata["final_val_scores"] = val_scores
        if not result.metadata.get("val_scoring_incomplete"):
            result.metadata["final_val_score"] = sum(val_scores.values()) / len(val_scores) if val_scores else 0.0
            result.best_score = result.metadata["final_val_score"]

    # Evaluate final submitted candidate on held-out test set (outside budget).
    if official_task.test_set and result.best_candidate:
        test_scores = _score_examples_unbudgeted(
            result,
            official_task,
            result.best_candidate,
            official_task.test_set,
            "test",
            max_concurrency,
        )
        result.metadata["test_scores"] = test_scores
        if not result.metadata.get("test_scoring_incomplete"):
            result.metadata["test_score"] = sum(test_scores.values()) / len(test_scores) if test_scores else 0.0

    solver_cost = _solver_cost(solver_cost_tracker)
    result.metadata["solver_cost"] = solver_cost
    result.metadata["solver_cost_log"] = _solver_cost_log(solver_cost_tracker)
    result.metadata["total_cost"] = server.total_cost + adapter_cost + solver_cost

    if tracker:
        tracker.log_summary({
            "best_score": result.best_score,
            "total_evals": result.total_evals,
            "wall_time": result.metadata["wall_time"],
            "total_cost": result.metadata["total_cost"],
        })
        tracker.end()

    return result


def _score_examples_unbudgeted(
    result: Result,
    task: Task,
    candidate: str,
    examples: list[Any],
    split: str,
    max_concurrency: int = 8,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not examples:
        return scores

    max_workers = max(1, min(max_concurrency, len(examples)))

    def score_example(ex: Any) -> tuple[str, float]:
        try:
            score, _ = task.eval_fn(candidate, ex)
            return ex.id, score
        except BudgetExhausted:
            raise
        except Exception:
            return ex.id, 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = iter(examples)
        futures = {executor.submit(score_example, ex): ex for ex in _take(pending, max_workers)}

        while futures:
            for future in as_completed(futures):
                break
            futures.pop(future)
            try:
                example_id, score = future.result()
            except BudgetExhausted as e:
                result.metadata[f"{split}_scoring_incomplete"] = True
                result.metadata[f"{split}_scoring_error"] = str(e)
                for pending_future in futures:
                    pending_future.cancel()
                break
            scores[example_id] = score

            try:
                ex = next(pending)
            except StopIteration:
                continue
            futures[executor.submit(score_example, ex)] = ex
    return scores


def _take(items: Any, limit: int) -> list[Any]:
    taken = []
    for _ in range(limit):
        try:
            taken.append(next(items))
        except StopIteration:
            break
    return taken


def _solver_cost(solver_cost_tracker: Any | None) -> float:
    return float(getattr(solver_cost_tracker, "total_cost", 0.0) or 0.0)


def _solver_cost_log(solver_cost_tracker: Any | None) -> list[dict[str, Any]] | None:
    cost_log = getattr(solver_cost_tracker, "cost_log", None)
    return list(cost_log) if cost_log is not None else None


def _task_for_search(task: Task) -> Task:
    """Return the adapter-visible task with held-out test examples removed."""
    if not task.test_set:
        return task

    metadata = dict(task.metadata)
    metadata["heldout_test_size"] = len(task.test_set)
    return replace(task, test_set=None, metadata=metadata)


def _validate_task_contract(task: Task) -> None:
    """Validate split ids before exposing a task to any adapter."""
    seen: dict[str, str] = {}
    for split_name, dataset in (
        ("train", task.train_set),
        ("val", task.val_set),
        ("test", task.test_set),
    ):
        if not dataset:
            continue
        split_ids: set[str] = set()
        for ex in dataset:
            if not ex.id:
                raise ValueError(f"{task.name}: {split_name} example has empty id")
            if ex.id in split_ids:
                raise ValueError(f"{task.name}: duplicate example id {ex.id!r} within {split_name}")
            if ex.id in seen:
                raise ValueError(
                    f"{task.name}: example id {ex.id!r} appears in both {seen[ex.id]} and {split_name}"
                )
            split_ids.add(ex.id)
            seen[ex.id] = split_name


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


def _merge_val_into_train(task: Task, metadata: dict[str, Any]) -> tuple[list[Example] | None, None]:
    """Use all visible development examples for search without a validation channel."""
    if not task.val_set:
        return task.train_set, None
    train_set = list(task.train_set or [])
    train_set.extend(task.val_set)
    provenance = metadata.get("split_provenance")
    if isinstance(provenance, dict):
        provenance = dict(provenance)
        provenance["validation_policy"] = "merged_into_train"
        provenance["run_split_sizes"] = {
            "train": len(train_set),
            "val": 0,
            "test": len(task.test_set or []),
        }
        metadata["split_provenance"] = provenance
    return train_set, None


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
    if "use_val" in benchmark_cfg:
        raise ValueError("benchmark.use_val was renamed to benchmark.split_train_val")
    split_train_val = bool(benchmark_cfg.get("split_train_val", True))

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

    train_set = task.train_set
    val_set = task.val_set
    # Omni composition can run heterogeneous backends in one adapter call.
    # Preserve the original visible split so each member can choose whether
    # it wants a real validation channel or a merged train+val view.
    metadata["_terrarium_source_train_set"] = list(task.train_set) if task.train_set is not None else None
    metadata["_terrarium_source_val_set"] = list(task.val_set) if task.val_set is not None else None
    metadata["_terrarium_split_train_val"] = split_train_val
    if not split_train_val:
        train_set, val_set = _merge_val_into_train(task, metadata)

    test_set = task.test_set if mode == "generalization" else None

    metadata["type"] = mode
    prepared = replace(task, train_set=train_set, val_set=val_set, test_set=test_set, metadata=metadata)

    if mode == "single" and prepared.has_dataset:
        raise ValueError(
            f"benchmark.mode=single is inconsistent with dataset task '{task.name}'. "
            "Use benchmark.mode=multi_task or benchmark.mode=generalization."
        )

    return prepared


def _apply_task_runtime_config(task: Task, task_cfg: DictConfig) -> Task:
    """Thread Hydra task runtime knobs into task evaluators that need them."""
    if task.name == "aime_math":
        from terrarium.tasks.aime_math import evaluate_with_solver

        model_id = task_cfg.get("solver_lm")
        temperature = task_cfg.get("solver_temperature")
        max_tokens = task_cfg.get("solver_max_tokens")
        timeout = task_cfg.get("solver_timeout")
        num_retries = task_cfg.get("solver_num_retries")

        def evaluate(candidate: str, example: Example) -> tuple[float, dict[str, Any]]:
            return evaluate_with_solver(
                candidate,
                example,
                solver_lm=str(model_id) if model_id is not None else None,
                solver_temperature=float(temperature) if temperature is not None else None,
                solver_max_tokens=int(max_tokens) if max_tokens is not None else None,
                solver_timeout=float(timeout) if timeout is not None else None,
                solver_num_retries=int(num_retries) if num_retries is not None else None,
            )

        metadata = dict(task.metadata)
        if model_id is not None:
            metadata["solver_lm"] = str(model_id)
        if temperature is not None:
            metadata["solver_temperature"] = float(temperature)
        if max_tokens is not None:
            metadata["solver_max_tokens"] = int(max_tokens)
        if timeout is not None:
            metadata["solver_timeout"] = float(timeout)
        if num_retries is not None:
            metadata["solver_num_retries"] = int(num_retries)
        return replace(task, eval_fn=evaluate, metadata=metadata)

    if task.name in ("finer", "formula"):
        from importlib import import_module

        _solve = import_module(f"terrarium.tasks.finance.{task.name}").evaluate_with_solver

        model_id = task_cfg.get("solver_lm")
        temperature = task_cfg.get("solver_temperature")
        max_tokens = task_cfg.get("solver_max_tokens")
        timeout = task_cfg.get("solver_timeout")
        num_retries = task_cfg.get("solver_num_retries")

        def evaluate(candidate: str, example: Example) -> tuple[float, dict[str, Any]]:
            return _solve(
                candidate,
                example,
                solver_lm=str(model_id) if model_id is not None else None,
                solver_temperature=float(temperature) if temperature is not None else None,
                solver_max_tokens=int(max_tokens) if max_tokens is not None else None,
                solver_timeout=float(timeout) if timeout is not None else None,
                solver_num_retries=int(num_retries) if num_retries is not None else None,
            )

        metadata = dict(task.metadata)
        if model_id is not None:
            metadata["solver_lm"] = str(model_id)
        if temperature is not None:
            metadata["solver_temperature"] = float(temperature)
        if max_tokens is not None:
            metadata["solver_max_tokens"] = int(max_tokens)
        if timeout is not None:
            metadata["solver_timeout"] = float(timeout)
        if num_retries is not None:
            metadata["solver_num_retries"] = int(num_retries)
        return replace(task, eval_fn=evaluate, metadata=metadata)

    if task.name == "needle_in_range":
        from terrarium.tasks.needle_in_range import (
            configure as _configure_nir,
            make_description as _nir_description,
            make_objective as _nir_objective,
        )

        n = int(task_cfg.get("n") or task.metadata.get("n") or 50)
        _configure_nir(n)

        metadata = dict(task.metadata)
        metadata["n"] = n
        return replace(
            task,
            objective=_nir_objective(n),
            background=_nir_description(n),
            metadata=metadata,
        )
    if task.name == "slot_machines":
        from terrarium.tasks.slot_machines import (
            configure as _configure_slots,
            make_description as _slots_description,
            make_objective as _slots_objective,
        )

        def _pick(key: str, default: int) -> int:
            v = task_cfg.get(key)
            if v is None:
                v = task.metadata.get(key, default)
            return int(v)

        n = _pick("n", 10)
        m = _pick("m", 100)
        seed = _pick("seed", 42)
        _configure_slots(n, m, seed)

        metadata = dict(task.metadata)
        metadata["n"] = n
        metadata["m"] = m
        metadata["seed"] = seed
        return replace(
            task,
            objective=_slots_objective(n, m),
            background=_slots_description(n, m),
            metadata=metadata,
        )

    if task.name != "arc_agi":
        return task

    from terrarium.tasks.arc_agi import evaluate as evaluate_arc_agi

    model_id = str(task_cfg.get("solver_lm") or "openrouter/google/gemini-3-flash-preview")
    max_llm_calls = int(task_cfg.get("max_llm_calls") or 10)

    def evaluate(candidate: str, example: Example) -> tuple[float, dict[str, Any]]:
        return evaluate_arc_agi(candidate, example, model_id=model_id, max_llm_calls=max_llm_calls)

    metadata = dict(task.metadata)
    metadata["solver_lm"] = model_id
    metadata["max_llm_calls"] = max_llm_calls
    return replace(task, eval_fn=evaluate, metadata=metadata)


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
    # (adaptive) are independent knobs. When both are set, the adapter
    # forwards both to the underlying runtime.
    max_thinking_tokens = cfg.get("max_thinking_tokens")
    effort = cfg.get("effort")
    if max_thinking_tokens is not None:
        print(f"[terrarium] Fixed thinking budget: max_thinking_tokens={max_thinking_tokens}")
        _apply_max_thinking_tokens(adapter, max_thinking_tokens)
    if effort is not None:
        _apply_effort(adapter, effort)

    _apply_perfect_score(adapter, cfg.get("perfect_score"))
    _apply_sandbox(adapter, cfg.get("sandbox"))
    _validate_access_policy(cfg.get("access_policy"), adapter, cfg.get("sandbox"))

    tracking = _build_tracking_config(cfg.tracking) if "tracking" in cfg else None

    # Tag wandb runs with the task name by default.
    if tracking and not tracking.wandb_tags:
        tracking.wandb_tags = [cfg.task.name]

    # Configure task-level solver LM (e.g. for aime_math's dspy evaluator).
    solver_cost_tracker = None
    solver_lm = cfg.task.get('solver_lm')
    if solver_lm:
        import dspy
        from terrarium.solver_lm import CostTrackedDSPyLM

        lm_kwargs = {}
        if cfg.task.get('solver_temperature') is not None:
            lm_kwargs['temperature'] = cfg.task.solver_temperature
        if cfg.task.get('solver_max_tokens') is not None:
            lm_kwargs['max_tokens'] = cfg.task.solver_max_tokens
        if cfg.task.get('solver_timeout') is not None:
            lm_kwargs['timeout'] = cfg.task.solver_timeout
        if cfg.task.get('solver_num_retries') is not None:
            lm_kwargs['num_retries'] = cfg.task.solver_num_retries
        solver_cost_tracker = CostTrackedDSPyLM(
            solver_lm,
            max_cost=cfg.budget.get("max_solver_cost"),
            **lm_kwargs,
        )
        dspy.configure(lm=solver_cost_tracker)

    benchmark_cfg = OmegaConf.create(OmegaConf.to_container(cfg.get("benchmark", {}), resolve=True))
    if benchmark_cfg.get("mode") is None:
        benchmark_cfg.mode = cfg.task.get("mode")
    raw_task = get_task(cfg.task.name)
    raw_task = _apply_task_runtime_config(raw_task, cfg.task)

    result = run(
        raw_task,
        adapter,
        max_evals=cfg.budget.get("max_evals"),
        max_token_cost=cfg.budget.get("max_token_cost"),
        max_concurrency=cfg.max_concurrency,
        benchmark=benchmark_cfg,
        tracking=tracking,
        output_dir=hydra_out,
        solver_cost_tracker=solver_cost_tracker,
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
        "eval_cost": result.metadata.get("eval_cost", 0.0),
        "adapter_cost": result.metadata.get("adapter_cost", 0.0),
        "solver_cost": result.metadata.get("solver_cost", 0.0),
        "solver_cost_log": result.metadata.get("solver_cost_log"),
        "reflection_cost_log": result.metadata.get("reflection_cost_log"),
        "wall_time": result.metadata.get("wall_time"),
        "budget": result.metadata.get("budget"),
        "final_val_score": result.metadata.get("final_val_score"),
        "final_val_scores": result.metadata.get("final_val_scores"),
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
