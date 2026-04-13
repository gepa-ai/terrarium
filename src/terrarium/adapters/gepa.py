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
    ) -> None:
        self.run_dir = run_dir
        self.engine = dict(engine) if engine else {}
        self.reflection = dict(reflection) if reflection else {}
        self.merge = dict(merge) if merge else None
        self.refiner = dict(refiner) if refiner else None
        self.objective = objective
        self.background = background
        self.callbacks = callbacks or []

    def evolve(self, task: Task, server: EvalServer, max_evals: int) -> Result:
        from gepa.optimize_anything import GEPAConfig, optimize_anything

        # Runner-controlled engine fields override whatever the user set.
        engine_kwargs: dict[str, Any] = {
            **self.engine,
            "run_dir": self.run_dir,
            "max_metric_calls": max_evals,
        }

        # GEPAConfig.__post_init__ converts dict -> nested config dataclass.
        config = GEPAConfig(
            engine=engine_kwargs,
            reflection=self.reflection,
            merge=self.merge,
            refiner=self.refiner,
            callbacks=self.callbacks if self.callbacks else None,
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

        # Prefer adapter-level objective/background; fall back to task metadata.
        objective = self.objective or task.metadata.get("objective")
        background = self.background or task.metadata.get("background")
        if objective is not None:
            oa_kwargs["objective"] = objective
        if background is not None:
            oa_kwargs["background"] = background

        try:
            gepa_result = optimize_anything(**oa_kwargs)
        except BudgetExhausted:
            gepa_result = None

        if gepa_result is not None:
            return Result(
                best_candidate=gepa_result.best_candidate,
                best_score=gepa_result.val_aggregate_scores[gepa_result.best_idx],
                total_evals=server.budget.used,
                eval_log=server.eval_log,
                metadata={"gepa_result": gepa_result},
            )

        return Result(
            best_candidate=server.best_candidate,
            best_score=server.best_score,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
        )


def create_adapter(**kwargs: Any) -> GEPAAdapter:
    """Factory for adapter-from-file loading (``adapter=custom`` with this file)."""
    return GEPAAdapter(**kwargs)
