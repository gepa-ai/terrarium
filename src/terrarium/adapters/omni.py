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

Single-backend example::

    python -m terrarium adapter=omni adapter.backend=gepa task=circle_packing
    python -m terrarium adapter=omni adapter.backend=meta_harness budget.max_evals=200
    python -m terrarium adapter=omni adapter.backend=claude_code \\
        adapter.config='{model: sonnet}'

Ensemble example (run gepa + claude_code in parallel, take the best)::

    python -m terrarium adapter=omni adapter.strategy=best_of \\
        'adapter.configs=[{backend: gepa}, {backend: claude_code, config: {model: sonnet}}]' \\
        budget.max_evals=200

When ``strategy != "single"`` the budget is **pre-partitioned**: each of the
N inner backends gets ``max_evals // N`` (and ``max_token_cost / N`` if set).
The outer terrarium server's budget is the hard cap; the inner partitions are
soft caps that bound each backend's individual share.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from terrarium.adapter import Result
from terrarium.task import Task

if TYPE_CHECKING:
    from terrarium.eval_server import EvalServer

Strategy = Literal["single", "sequential", "parallel", "best_of", "vote"]


class OmniAdapter:
    """Adapter that runs one or more ``gepa.omni`` backends against a terrarium task.

    Args:
        backend: Omni backend name — ``gepa``, ``claude_code``, or
            ``meta_harness`` — or a constructed ``gepa.omni.Backend`` instance.
            Used as the single backend when ``strategy="single"``; ignored
            otherwise (each entry in ``configs`` carries its own ``backend``).
        config: Free-form mapping forwarded to ``OmniConfig.config`` for the
            single-backend path. Each backend's factory pops the keys it
            understands; unknown keys produce warnings (e.g.
            ``{"reflection": {"reflection_lm": "openai/gpt-5"}}`` for the gepa
            backend, ``{"model": "sonnet"}`` for claude_code). Ignored when
            ``strategy != "single"`` — use per-entry ``config`` in ``configs``.
        strategy: Composition strategy:

            - ``"single"`` (default) — run one backend (``self.backend``).
              Backward-compatible with the original adapter.
            - ``"sequential"`` — chain backends; each best becomes the next
              backend's seed. Budget is fair-shared: each stage gets its own
              ``total / N`` partition, same as the parallel-family strategies.
            - ``"parallel"`` — run N backends concurrently against N inner
              servers (budget pre-partitioned ``total / N``). Returns the
              first backend's result by convention; full per-backend results
              are in ``metadata["all_results"]``.
            - ``"best_of"`` — parallel + select the highest ``best_score``.
            - ``"vote"`` — parallel + a single re-score pass via terrarium's
              outer server, then select the highest re-score.
        configs: List of per-backend config dicts. Each entry may carry
            ``backend``, ``config``, ``effort``, ``max_thinking_tokens``,
            ``stop_at_score`` to override the adapter-level defaults for that
            ensemble member. Required when ``strategy != "single"``.
        run_dir: Where the omni backend writes its workspace artifacts. Left
            null in yaml — the runner injects ``<hydra_run_dir>/<adapter_name>/``.
            For ensembles each member gets ``<run_dir>/<idx>_<backend>/``.
        max_token_cost: Override ``server.budget.max_token_cost`` (rare; the
            runner already threads ``budget.max_token_cost`` into the server).
        stop_at_score: Score threshold for early stop. Forwarded as
            ``OmniConfig.stop_at_score``.
        effort: ``claude --effort`` for backends that spawn Claude Code.
            Independent of ``max_thinking_tokens`` — both can be set together.
            Runner threads top-level ``effort`` here when null.
        max_thinking_tokens: Fixed thinking-token budget. Independent of
            ``effort`` (both can be set together). Runner threads top-level
            ``max_thinking_tokens`` here when null.
        sandbox: Whether to wrap subprocess backends in bwrap/Seatbelt.
            Runner threads top-level ``sandbox`` here when null.
        callbacks: Optional GEPA callback list. Only consumed by ``gepa``
            backends; ignored (and not forwarded) for claude_code / meta_harness
            so omni's unknown-key warning stays quiet.
        max_workers: Override the parallel ensemble's thread-pool size.
            Defaults to ``len(configs)``.
    """

    def __init__(
        self,
        backend: str | Any = "gepa",
        config: dict[str, Any] | None = None,
        strategy: Strategy = "single",
        configs: list[dict[str, Any]] | None = None,
        run_dir: str | None = None,
        max_token_cost: float | None = None,
        stop_at_score: float | None = None,
        effort: str | None = None,
        max_thinking_tokens: int | None = None,
        sandbox: bool | None = None,
        callbacks: list[Any] | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.backend = backend
        self.config = dict(config) if config else {}
        self.strategy: Strategy = strategy
        self.configs = [dict(c) for c in configs] if configs else []
        self.run_dir = run_dir
        self.max_token_cost = max_token_cost
        self.stop_at_score = stop_at_score
        self.effort = effort
        self.max_thinking_tokens = max_thinking_tokens
        self.sandbox = sandbox
        self.callbacks = list(callbacks) if callbacks else []
        self.max_workers = max_workers

        if self.strategy != "single" and not self.configs:
            raise ValueError(f"OmniAdapter.strategy={self.strategy!r} requires a non-empty configs list")

    def evolve(self, task: Task, server: EvalServer) -> Result:
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

        budget = server.budget
        max_token_cost = self.max_token_cost if self.max_token_cost is not None else budget.max_token_cost
        if budget.max_evals is None and max_token_cost is None:
            raise ValueError("OmniAdapter requires at least one of budget.max_evals or budget.max_token_cost")

        # Route inner evals through terrarium's outer EvalServer so the
        # runner's eval_log / progress_log / total_cost populate. Inner
        # servers disable per-eval JSON mirroring (output_dir=None) — the
        # outer server already owns that persistence.
        def evaluate(candidate: str, example: Any | None = None) -> tuple[float, dict[str, Any]]:
            return server.evaluate(candidate, example)

        if self.strategy == "single":
            return self._run_single(omni_task, evaluate, server, budget.max_evals, max_token_cost)
        return self._run_ensemble(omni_task, evaluate, server, budget.max_evals, max_token_cost)

    def _run_single(
        self,
        omni_task: Any,
        evaluate: Any,
        server: EvalServer,
        max_evals: int | None,
        max_token_cost: float | None,
    ) -> Result:
        from gepa.omni import (
            BudgetTracker,
            OmniConfig,
            optimize_anything_with_server,
        )
        from gepa.omni import EvalServer as OmniEvalServer

        inner_budget = BudgetTracker(max_evals=max_evals, max_token_cost=max_token_cost)
        inner_server = OmniEvalServer(
            omni_task,
            evaluate,
            inner_budget,
            max_concurrency=server.max_concurrency,
            output_dir=None,
        )

        cfg = OmniConfig(
            backend=self.backend,
            run_dir=self.run_dir,
            stop_at_score=self.stop_at_score,
            effort=self.effort,
            max_thinking_tokens=self.max_thinking_tokens,
            sandbox=bool(self.sandbox) if self.sandbox is not None else False,
            config=self._build_backend_config(self.backend, self.config),
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

    def _run_ensemble(
        self,
        omni_task: Any,
        evaluate: Any,
        server: EvalServer,
        max_evals: int | None,
        max_token_cost: float | None,
    ) -> Result:
        from gepa.omni import (
            BudgetTracker,
            optimize_best_of_with_server,
            optimize_parallel_with_server,
            optimize_sequential_with_server,
            optimize_vote_with_server,
        )
        from gepa.omni import EvalServer as OmniEvalServer

        n = len(self.configs)
        omni_configs = [self._materialize_config(entry, idx) for idx, entry in enumerate(self.configs)]

        # All ensemble strategies (sequential / parallel / best_of / vote)
        # use the same fair-share partition: one inner server per config, each
        # carrying ``total / N`` of the outer budget. Sequential's stages run
        # one at a time but each stage still has its own pre-partitioned cap,
        # so a productive early stage can't monopolize the pool.
        per_evals = max_evals // n if max_evals is not None else None
        per_cost = (max_token_cost / n) if max_token_cost is not None else None
        inner_servers = [
            OmniEvalServer(
                omni_task,
                evaluate,
                BudgetTracker(max_evals=per_evals, max_token_cost=per_cost),
                max_concurrency=server.max_concurrency,
                output_dir=None,
            )
            for _ in range(n)
        ]

        for s in inner_servers:
            s.start()
        try:
            if self.strategy == "sequential":
                omni_result = optimize_sequential_with_server(inner_servers, omni_configs)
                return self._wrap_result(omni_result)
            if self.strategy == "parallel":
                results = optimize_parallel_with_server(inner_servers, omni_configs, max_workers=self.max_workers)
                # Convention: surface the first backend's result; full list
                # in metadata. Callers wanting the best should use best_of.
                primary = results[0]
                primary.metadata["all_results"] = results
                return self._wrap_result(primary)
            if self.strategy == "best_of":
                omni_result = optimize_best_of_with_server(inner_servers, omni_configs, max_workers=self.max_workers)
                return self._wrap_result(omni_result)
            if self.strategy == "vote":
                omni_result = optimize_vote_with_server(
                    inner_servers, omni_configs, evaluate, max_workers=self.max_workers
                )
                return self._wrap_result(omni_result)
            raise ValueError(f"Unknown strategy: {self.strategy!r}")
        finally:
            for s in inner_servers:
                s.stop()

    def _materialize_config(self, entry: dict[str, Any], idx: int) -> Any:
        """Build an :class:`OmniConfig` for one ensemble member.

        Per-entry keys (``backend``, ``config``, ``effort``,
        ``max_thinking_tokens``, ``stop_at_score``) override the adapter-level
        defaults. ``run_dir`` is auto-derived as ``<self.run_dir>/<idx>_<backend>/``
        unless the entry provides one explicitly.
        """
        from gepa.omni import OmniConfig

        backend = entry.get("backend", self.backend)
        backend_config = entry.get("config") or {}
        effort = entry.get("effort", self.effort)
        max_thinking_tokens = entry.get("max_thinking_tokens", self.max_thinking_tokens)
        stop_at_score = entry.get("stop_at_score", self.stop_at_score)
        sandbox = entry.get("sandbox", self.sandbox)

        if entry.get("run_dir"):
            run_dir = entry["run_dir"]
        elif self.run_dir:
            backend_name = getattr(backend, "name", backend) if not isinstance(backend, str) else backend
            run_dir = str(Path(self.run_dir) / f"{idx}_{backend_name}")
        else:
            run_dir = None

        return OmniConfig(
            backend=backend,
            run_dir=run_dir,
            stop_at_score=stop_at_score,
            effort=effort,
            max_thinking_tokens=max_thinking_tokens,
            sandbox=bool(sandbox) if sandbox is not None else False,
            config=self._build_backend_config(backend, backend_config),
        )

    def _wrap_result(self, omni_result: Any) -> Result:
        backend_name: Any
        if self.strategy == "single":
            backend_name = getattr(self.backend, "name", self.backend)
        else:
            backend_name = self.strategy
        return Result(
            best_candidate=omni_result.best_candidate,
            best_score=omni_result.best_score,
            metadata={**omni_result.metadata, "omni_backend": backend_name, "omni_strategy": self.strategy},
        )

    def process_result(self, result: Result, output_dir: Path) -> None:
        """No-op — omni backends write their own artifacts under ``run_dir``
        (= ``<hydra_run_dir>/<adapter_name>/``), which is already inside the
        run's output tree."""
        return

    def _build_backend_config(self, backend: Any, config: dict[str, Any]) -> dict[str, Any]:
        """Forward the per-backend ``config`` plus terrarium-injected callbacks.
        Callbacks are only attached when ``backend == "gepa"`` — the other
        built-in backends don't read them and would emit an unknown-key
        warning."""
        merged = dict(config)
        if self.callbacks and isinstance(backend, str) and backend == "gepa":
            existing = list(merged.get("callbacks") or [])
            existing.extend(self.callbacks)
            merged["callbacks"] = existing
        return merged


def create_adapter(**kwargs: Any) -> OmniAdapter:
    """Factory for ``adapter=custom`` loading via ``terrarium.runner.load_adapter``."""
    return OmniAdapter(**kwargs)
