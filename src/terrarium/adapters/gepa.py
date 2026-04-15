"""GEPA adapter: runs optimize_anything as the evolution backend.

Config layout mirrors :class:`gepa.optimize_anything.GEPAConfig`:

- ``engine``: kwargs for :class:`EngineConfig` (budget, parallelism, caching, ...)
- ``reflection``: kwargs for :class:`ReflectionConfig` (reflection LM, minibatch, ...)
- ``merge``: kwargs for :class:`MergeConfig` (or ``None`` to disable)
- ``refiner``: kwargs for :class:`RefinerConfig` (or ``None`` to disable)

Plus top-level :func:`optimize_anything` args:

- ``objective``, ``background``: reflection prompt context.

The runner always sets ``engine.run_dir`` (from ``self.run_dir``, which the
runner normally injects = ``<hydra_run_dir>/<adapter_name>``) and
``engine.max_metric_calls`` (= ``max_evals``). Everything else is user-overridable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from terrarium.adapter import Result
from terrarium.budget import BudgetExhausted
from terrarium.task import Task

if TYPE_CHECKING:
    from terrarium.eval_server import EvalServer


class GEPAAdapter:
    """Adapter that runs GEPA's ``optimize_anything`` against a Terrarium task.

    All evaluations go through the terrarium ``EvalServer`` (direct Python API,
    no HTTP overhead). Budget is enforced by the server.

    Args:
        run_dir: Directory for GEPA run artifacts. The terrarium runner injects
            ``<hydra_run_dir>/<adapter_name>`` here; override in yaml if you
            want a different location.
        engine: Keyword args for :class:`EngineConfig`. The runner always
            overrides ``max_metric_calls`` (= ``max_evals``) and ``run_dir``
            (= ``self.run_dir``).
        reflection: Keyword args for :class:`ReflectionConfig`
            (e.g. ``reflection_lm``, ``reflection_minibatch_size``,
            ``module_selector``, ``batch_sampler``, ``reflection_prompt_template``).
        merge: Keyword args for :class:`MergeConfig`, or ``None`` to disable merging.
        refiner: Keyword args for :class:`RefinerConfig`, or ``None`` to disable
            the automatic refiner step.
        objective: Optimization goal passed to ``optimize_anything``.
        background: Domain context passed to ``optimize_anything``.
        callbacks: GEPA callbacks list (appended to by the runner's tracker).

    Example (Python)::

        adapter = GEPAAdapter(
            reflection={"reflection_lm": "openai/gpt-5.1", "reflection_minibatch_size": 3},
            engine={"max_workers": 32, "cache_evaluation": True},
            refiner={"max_refinements": 2},
            objective="Maximize sum of radii for 26 non-overlapping circles.",
        )

    Example (Hydra CLI)::

        python -m terrarium \\
            adapter.reflection.reflection_lm=openai/gpt-5.1 \\
            adapter.engine.max_workers=32 \\
            adapter.refiner='{max_refinements: 2}' \\
            adapter.objective='Maximize sum of radii...'
    """

    def __init__(
        self,
        run_dir: str = "outputs/terrarium",
        engine: dict[str, Any] | None = None,
        reflection: dict[str, Any] | None = None,
        merge: dict[str, Any] | None = None,
        refiner: dict[str, Any] | None = None,
        objective: str | None = None,
        background: str | None = None,
        callbacks: list[Any] | None = None,
        reflection_lm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.engine = dict(engine) if engine else {}
        self.reflection = dict(reflection) if reflection else {}
        self.merge = dict(merge) if merge else None
        self.refiner = dict(refiner) if refiner else None
        self.objective = objective
        self.background = background
        self.callbacks = callbacks or []
        # Extra kwargs passed to ``gepa.lm.LM(...)`` when wrapping a string
        # ``reflection.reflection_lm``. ``timeout`` is forwarded to litellm
        # (its httpx default ~600s otherwise cuts off long extended-thinking
        # responses); ``num_retries`` controls retry on transient failures.
        self.reflection_lm_kwargs = dict(reflection_lm_kwargs) if reflection_lm_kwargs else {}

    def evolve(self, task: Task, server: EvalServer) -> Result:
        from gepa.lm import LM
        from gepa.optimize_anything import GEPAConfig, optimize_anything

        budget = server.budget

        reflection_kwargs = dict(self.reflection)
        reflection_lm: LM | None = None
        if "reflection_lm" in reflection_kwargs:
            reflection_lm = LM(reflection_kwargs["reflection_lm"], **self.reflection_lm_kwargs)
            reflection_kwargs["reflection_lm"] = reflection_lm

        # Runner-controlled engine fields override whatever the user set.
        engine_kwargs: dict[str, Any] = {
            **self.engine,
            "run_dir": self.run_dir,
            "max_metric_calls": budget.max_evals,
        }
        if budget.max_token_cost is not None:
            engine_kwargs["max_reflection_cost"] = budget.max_token_cost

        cost_callback: _ReflectionCostCallback | None = None
        callbacks = list(self.callbacks)
        if reflection_lm is not None:
            cost_callback = _ReflectionCostCallback(reflection_lm, server.tracker, output_dir=self.run_dir)
            callbacks.append(cost_callback)

        if task.val_set:
            callbacks.append(_ProgressCallback(server, reflection_lm=reflection_lm))

        # GEPAConfig.__post_init__ converts dict -> nested config dataclass.
        config = GEPAConfig(
            engine=engine_kwargs,
            reflection=reflection_kwargs,
            merge=self.merge,
            refiner=self.refiner,
            callbacks=callbacks or None,
        )

        # Bridge: optimize_anything calls this evaluator, which goes through
        # the terrarium server (same budget counter as the HTTP endpoint).
        def evaluator(candidate, example=None, **kwargs):
            return server.evaluate(candidate, example)

        oa_kwargs: dict[str, Any] = {
            "seed_candidate": task.initial_candidate,
            "evaluator": evaluator,
            "config": config,
        }

        if task.has_dataset:
            if task.train_set:
                oa_kwargs["dataset"] = task.train_set
            if task.test_set:
                oa_kwargs["valset"] = task.test_set
            if "val_set" in task.metadata:
                oa_kwargs.setdefault("valset", task.metadata["val_set"])

        # Both adapters pass the same two task fields through to the
        # evolution system. yaml-level ``adapter.objective`` /
        # ``adapter.background`` overrides win.
        objective = self.objective or task.objective
        background = self.background or task.background
        if objective:
            oa_kwargs["objective"] = objective
        if background:
            oa_kwargs["background"] = background

        try:
            gepa_result = optimize_anything(**oa_kwargs)
        except BudgetExhausted:
            gepa_result = None

        reflection_meta: dict[str, Any] = {}
        if cost_callback is not None:
            reflection_meta = {"reflection_cost_log": cost_callback.cost_log}

        if gepa_result is not None:
            return Result(
                best_candidate=gepa_result.best_candidate,
                best_score=gepa_result.val_aggregate_scores[gepa_result.best_idx],
                total_evals=server.budget.used,
                eval_log=server.eval_log,
                metadata={"gepa_result": gepa_result, **reflection_meta},
            )

        return Result(
            best_candidate=server.best_candidate,
            best_score=server.best_score,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
            metadata=reflection_meta,
        )

    def process_result(self, result: Result, output_dir: Path) -> None:
        """No-op: GEPA writes its own artifacts via callbacks during ``evolve``."""
        return


class _ReflectionCostCallback:
    """GEPA callback that snapshots reflection LM cost at each iteration."""

    def __init__(self, lm: Any, tracker: Any | None, output_dir: str | Path | None = None) -> None:
        self._lm = lm
        self._tracker = tracker
        self._cost_log: list[dict[str, Any]] = []
        self._log_path: Path | None = None
        if output_dir is not None:
            self._log_path = Path(output_dir) / "reflection_cost_log.jsonl"
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def cost_log(self) -> list[dict[str, Any]]:
        return list(self._cost_log)

    def on_iteration_end(self, event: dict[str, Any]) -> None:
        entry = {
            "iteration": event["iteration"],
            "reflection_cost": self._lm.total_cost,
            "reflection_tokens_in": self._lm.total_tokens_in,
            "reflection_tokens_out": self._lm.total_tokens_out,
        }
        self._cost_log.append(entry)
        if self._log_path is not None:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        if self._tracker is not None:
            self._tracker.log_metrics(
                {
                    "reflection/cost": entry["reflection_cost"],
                    "reflection/tokens_in": entry["reflection_tokens_in"],
                    "reflection/tokens_out": entry["reflection_tokens_out"],
                },
                step=entry["iteration"],
            )



class _ProgressCallback:
    """GEPA callback that logs val_score to progress_log.jsonl via the server."""

    def __init__(self, server: Any, reflection_lm: Any = None) -> None:
        self._server = server
        self._reflection_lm = reflection_lm

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        candidate_dict = event.get("candidate", {})
        candidate_text = next(iter(candidate_dict.values()), None) if candidate_dict else None
        reflection_cost = self._reflection_lm.total_cost if self._reflection_lm else 0.0
        self._server.log_progress(event["average_score"], candidate=candidate_text, reflection_cost=reflection_cost)


def create_adapter(**kwargs: Any) -> GEPAAdapter:
    """Factory for adapter-from-file loading (``adapter=custom`` with this file)."""
    return GEPAAdapter(**kwargs)
