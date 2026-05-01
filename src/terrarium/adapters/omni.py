"""Omni adapter: dispatches a terrarium task to ``gepa.omni``.

Picks any registered omni backend (``gepa`` / ``claude_code`` / ``meta_harness``)
via the ``backend`` field. terrarium's outer ``EvalServer`` stays the budget
choke point — every eval omni performs is routed through it, so the runner's
``eval_log``, ``progress_log``, and ``total_cost`` populate exactly like they
do for the native adapters.

Internally we still construct an omni ``EvalServer`` (omni's backend protocol
expects one), but with ``output_dir=None`` so it doesn't mirror per-eval JSON
into a second location — terrarium's outer server already owns that
persistence. Backend-specific artifacts (gepa run dir, claude session
transcripts, meta-harness state) still land under ``run_dir``.

Example (Hydra CLI)::

    python -m terrarium adapter=omni adapter.backend=gepa task=circle_packing
    python -m terrarium adapter=omni adapter.backend=meta_harness budget.max_evals=200
    python -m terrarium adapter=omni adapter.backend=claude_code \\
        adapter.config='{model: sonnet}'
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from terrarium.adapter import Result
from terrarium.task import Task

if TYPE_CHECKING:
    from terrarium.eval_server import EvalServer


class OmniAdapter:
    """Adapter that runs a ``gepa.omni`` backend against a terrarium task.

    Args:
        backend: Omni backend name — ``gepa``, ``claude_code``, or
            ``meta_harness`` — or a constructed ``gepa.omni.Backend`` instance.
        config: Free-form mapping forwarded to ``OmniConfig.config``. Each
            backend's factory pops the keys it understands; unknown keys
            produce warnings (e.g. ``{"reflection": {"reflection_lm": "openai/gpt-5"}}``
            for the gepa backend, ``{"model": "sonnet"}`` for claude_code).
        run_dir: Where the omni backend writes its workspace artifacts. Left
            null in yaml — the runner injects ``<hydra_run_dir>/<adapter_name>/``.
        max_token_cost: Override ``server.budget.max_token_cost`` (rare; the
            runner already threads ``budget.max_token_cost`` into the server).
        stop_at_score: Score threshold for early stop. Forwarded as
            ``OmniConfig.stop_at_score``.
        effort: ``claude --effort`` for backends that spawn Claude Code.
            Mutex with ``max_thinking_tokens``. Runner threads top-level
            ``effort`` here when null.
        max_thinking_tokens: Fixed thinking-token budget. Mutex with ``effort``.
            Runner threads top-level ``max_thinking_tokens`` here when null.
        sandbox: Whether to wrap subprocess backends in bwrap/Seatbelt.
            Runner threads top-level ``sandbox`` here when null.
        callbacks: Optional GEPA callback list. Only consumed by the ``gepa``
            backend; ignored (and not forwarded) for claude_code / meta_harness
            so omni's unknown-key warning stays quiet.
    """

    def __init__(
        self,
        backend: str | Any = "gepa",
        config: dict[str, Any] | None = None,
        run_dir: str | None = None,
        max_token_cost: float | None = None,
        stop_at_score: float | None = None,
        effort: str | None = None,
        max_thinking_tokens: int | None = None,
        sandbox: bool | None = None,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.backend = backend
        self.config = dict(config) if config else {}
        self.run_dir = run_dir
        self.max_token_cost = max_token_cost
        self.stop_at_score = stop_at_score
        self.effort = effort
        self.max_thinking_tokens = max_thinking_tokens
        self.sandbox = sandbox
        self.callbacks = list(callbacks) if callbacks else []

    def evolve(self, task: Task, server: EvalServer) -> Result:
        from gepa.omni import (
            BudgetTracker,
            OmniConfig,
            optimize_anything_with_server,
        )
        from gepa.omni import EvalServer as OmniEvalServer
        from gepa.omni import Task as OmniTask

        # Translate terrarium Task → omni Task. Drop terrarium.test_set; the
        # terrarium runner already runs a held-out test eval after evolve()
        # via task.eval_fn directly. Map terrarium.val_set onto omni.test_set
        # because omni's gepa backend currently reads task.test_set as gepa's
        # ``valset`` (the in-search candidate-selection split — what terrarium
        # calls val_set). When omni's gepa backend is updated to read val_set
        # directly, drop the test_set= line.
        omni_task = OmniTask(
            name=task.name,
            initial_candidate=task.initial_candidate,
            objective=task.objective,
            background=task.background,
            train_set=task.train_set,
            val_set=task.val_set,
            test_set=task.val_set,
            metadata=dict(task.metadata),
        )

        # Route inner evals through terrarium's outer EvalServer so the
        # runner's eval_log / progress_log / total_cost populate. The inner
        # omni server is required by the omni backend protocol, but we
        # disable its eval-mirroring (output_dir=None) so per-eval JSON only
        # lands in terrarium's tree.
        def evaluate(candidate: str, example: Any | None = None) -> tuple[float, dict[str, Any]]:
            return server.evaluate(candidate, example)

        budget = server.budget
        max_token_cost = self.max_token_cost if self.max_token_cost is not None else budget.max_token_cost
        if budget.max_evals is None and max_token_cost is None:
            raise ValueError("OmniAdapter requires at least one of budget.max_evals or budget.max_token_cost")

        inner_budget = BudgetTracker(max_evals=budget.max_evals, max_token_cost=max_token_cost)
        inner_server = OmniEvalServer(
            omni_task,
            evaluate,
            inner_budget,
            max_concurrency=server.max_concurrency,
            output_dir=None,  # outer terrarium server owns eval persistence
        )

        cfg = OmniConfig(
            backend=self.backend,
            run_dir=self.run_dir,
            stop_at_score=self.stop_at_score,
            effort=self.effort,
            max_thinking_tokens=self.max_thinking_tokens,
            sandbox=bool(self.sandbox) if self.sandbox is not None else False,
            config=self._build_backend_config(),
        )

        inner_server.start()
        try:
            omni_result = optimize_anything_with_server(inner_server, cfg)
        finally:
            inner_server.stop()

        return Result(
            best_candidate=omni_result.best_candidate,
            best_score=omni_result.best_score,
            metadata={**omni_result.metadata, "omni_backend": getattr(self.backend, "name", self.backend)},
        )

    def process_result(self, result: Result, output_dir: Path) -> None:
        """No-op — omni backends write their own artifacts under ``run_dir``
        (= ``<hydra_run_dir>/<adapter_name>/``), which is already inside the
        run's output tree."""
        return

    def _build_backend_config(self) -> dict[str, Any]:
        """Forward the user's free-form ``config`` plus terrarium-injected
        callbacks. Callbacks are only attached when ``backend == "gepa"`` —
        the other built-in backends don't read them and would emit an
        unknown-key warning."""
        merged = dict(self.config)
        if self.callbacks and isinstance(self.backend, str) and self.backend == "gepa":
            existing = list(merged.get("callbacks") or [])
            existing.extend(self.callbacks)
            merged["callbacks"] = existing
        return merged


def create_adapter(**kwargs: Any) -> OmniAdapter:
    """Factory for ``adapter=custom`` loading via ``terrarium.runner.load_adapter``."""
    return OmniAdapter(**kwargs)
