"""optimize_anything adapter: dispatches a terrarium task to ``gepa.optimize_anything``.

Picks any registered optimize_anything engine (``gepa`` / ``autoresearch`` /
``meta_harness`` / ``best_of_n``) via the ``engine`` field. terrarium's outer
``EvalServer`` stays the budget choke point — every eval the engine performs is
routed through it, so the runner's ``eval_log``, ``progress_log``, and
``total_cost`` populate exactly like they do for the native adapters.

Internally we still construct a gepa ``EvalServer`` (the engine protocol expects
one) by handing ``gepa.optimize_anything`` our ``evaluate`` callback. Engine
artifacts (gepa run dir, autoresearch session transcripts, meta-harness state)
land under ``run_dir``.

This is the primary terrarium adapter: a single configurable surface over every
``gepa.optimize_anything`` engine, plus ensemble compositions. The native
``gepa`` / ``claude_code`` / ``meta_harness`` adapters are deprecated in favor of
``engine=gepa`` / ``engine=autoresearch`` / ``engine=meta_harness`` here.

Single-engine example::

    python -m terrarium adapter=optimize_anything adapter.engine=gepa task=circle_packing
    python -m terrarium adapter=optimize_anything adapter.engine=meta_harness budget.max_evals=200
    python -m terrarium adapter=optimize_anything adapter.engine=autoresearch \\
        adapter.engine_config='{model: sonnet}'

Ensemble example (run gepa + autoresearch in parallel, take the best)::

    python -m terrarium adapter=optimize_anything adapter.strategy=best_of \\
        'adapter.configs=[{engine: gepa}, {engine: autoresearch, engine_config: {model: sonnet}}]' \\
        budget.max_evals=200

When ``strategy != "single"`` the budget is **pre-partitioned**: each of the
N inner engines gets ``max_evals // N`` (and ``max_token_cost / N`` if set).
The outer terrarium server's budget is the hard cap; the inner partitions are
soft caps that bound each engine's individual share.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from terrarium.adapter import Result
from terrarium.adapters.optimize_anything_handoff import HandoffConfig, collect_stage_handoff
from terrarium.budget import BudgetExhausted as TerrariumBudgetExhausted
from terrarium.task import Task

if TYPE_CHECKING:
    from terrarium.eval_server import EvalServer

Strategy = Literal["single", "sequential", "adaptive_sequential", "parallel", "best_of", "vote"]


class OptimizeAnythingAdapter:
    """Adapter that runs one or more ``gepa.optimize_anything`` engines against a terrarium task.

    Args:
        engine: optimize_anything engine name — ``gepa``, ``autoresearch``,
            ``meta_harness``, or ``best_of_n`` — or a constructed
            ``gepa.optimize_anything.Engine`` instance. Used as the single
            engine when ``strategy="single"``; ignored otherwise (each entry in
            ``configs`` carries its own ``engine``).
        engine_config: Free-form mapping forwarded to
            ``OptimizeAnythingConfig.engine_config`` for the single-engine path.
            Each engine pops the keys it understands; unknown keys produce
            warnings (e.g. ``{"reflection": {"reflection_lm": "openai/gpt-5"}}``
            for the gepa engine, ``{"model": "sonnet"}`` for autoresearch).
            Ignored when ``strategy != "single"`` — use per-entry
            ``engine_config`` in ``configs``.
        strategy: Composition strategy:

            - ``"single"`` (default) — run one engine (``self.engine``).
            - ``"sequential"`` — chain engines; each best becomes the next
              engine's seed. Budget is fair-shared: each stage gets its own
              ``total / N`` partition, same as the parallel-family strategies.
            - ``"adaptive_sequential"`` — run bounded engine slices against
              one shared budget, switch to the next engine after a score
              plateau, and keep feeding the best aggregate engine result
              forward.
            - ``"parallel"`` — run N engines concurrently against N inner
              servers (budget pre-partitioned ``total / N``). Returns the
              first engine's result by convention; full per-engine results
              are in ``metadata["all_results"]``.
            - ``"best_of"`` — parallel + select the highest ``best_score``.
            - ``"vote"`` — parallel + a single re-score pass via terrarium's
              outer server, then select the highest re-score.
        configs: List of per-engine config dicts. Each entry may carry
            ``engine``, ``engine_config``, ``effort``, ``max_thinking_tokens``,
            ``stop_at_score``, ``split_train_val`` to override the
            adapter-level defaults for that ensemble member. Required when
            ``strategy != "single"``.
        run_dir: Where the engine writes its workspace artifacts. Left
            null in yaml — the runner injects ``<hydra_run_dir>/<adapter_name>/``.
            For ensembles each member gets ``<run_dir>/<idx>_<engine>/``.
        max_token_cost: Override ``server.budget.max_token_cost`` (rare; the
            runner already threads ``budget.max_token_cost`` into the server).
        stop_at_score: Score threshold for early stop. Forwarded as
            ``OptimizeAnythingConfig.stop_at_score``.
        effort: ``claude --effort`` for engines that spawn Claude Code.
            Independent of ``max_thinking_tokens`` — both can be set together.
            Runner threads top-level ``effort`` here when null.
        max_thinking_tokens: Fixed thinking-token budget. Independent of
            ``effort`` (both can be set together). Runner threads top-level
            ``max_thinking_tokens`` here when null.
        sandbox: Whether to wrap subprocess engines in bwrap/Seatbelt.
            Runner threads top-level ``sandbox`` here when null.
        callbacks: Optional GEPA callback list. Only consumed by ``gepa``
            engines; ignored (and not forwarded) for autoresearch / meta_harness
            so the unknown-key warning stays quiet.
        max_workers: Override the parallel ensemble's thread-pool size.
            Defaults to ``len(configs)``.
        handoff: Optional sequential-stage handoff config. ``{"mode": "rich"}``
            writes prior-stage eval records to disk and passes only manifest
            paths to the next stage via its ``engine_config["handoffs"]``.
        scheduler: Optional adaptive-sequential scheduler config. Supported
            keys: ``plateau_evals`` / ``slice_evals``, ``patience``,
            ``min_evals_per_stage``, ``improvement_epsilon``, ``cycle``, and
            ``max_switches``.
    """

    def __init__(
        self,
        engine: str | Any = "gepa",
        engine_config: dict[str, Any] | None = None,
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
        handoff: dict[str, Any] | str | None = None,
        scheduler: dict[str, Any] | None = None,
    ) -> None:
        self.engine = engine
        self.engine_config = dict(engine_config) if engine_config else {}
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
        self.handoff = HandoffConfig.from_value(handoff)
        self.scheduler = dict(scheduler) if scheduler else {}

        if self.strategy != "single" and not self.configs:
            raise ValueError(f"OptimizeAnythingAdapter.strategy={self.strategy!r} requires a non-empty configs list")

    def evolve(self, task: Task, server: EvalServer) -> Result:
        budget = server.budget
        max_token_cost = self.max_token_cost if self.max_token_cost is not None else budget.max_token_cost
        if budget.max_evals is None and max_token_cost is None:
            raise ValueError(
                "OptimizeAnythingAdapter requires at least one of budget.max_evals or budget.max_token_cost"
            )

        # Route inner evals through terrarium's outer EvalServer so the
        # runner's eval_log / progress_log / total_cost populate. The gepa
        # eval server wraps this callback; terrarium's outer server owns
        # per-eval persistence.
        def evaluate(candidate: str, example: Any | None = None) -> tuple[float, dict[str, Any]]:
            try:
                return server.evaluate(candidate, example)
            except TerrariumBudgetExhausted as exc:
                from gepa.optimize_anything import BudgetExhausted as GepaBudgetExhausted

                raise GepaBudgetExhausted(str(exc)) from exc

        if self.strategy == "single":
            oa_task = self._to_oa_task(task)
            return self._run_single(oa_task, evaluate, server, budget.max_evals, max_token_cost)
        return self._run_ensemble(task, evaluate, server, budget.max_evals, max_token_cost)

    def _to_oa_task(self, task: Task) -> Any:
        from gepa.optimize_anything import Task as OATask

        return OATask(
            name=task.name,
            initial_candidate=task.initial_candidate,
            objective=task.objective,
            background=task.background,
            train_set=task.train_set,
            val_set=task.val_set,
            # Terrarium owns held-out test reporting after search. gepa
            # optimize_anything owns engine-specific train/val visibility
            # policy internally.
            test_set=None,
        )

    def _run_single(
        self,
        oa_task: Any,
        evaluate: Any,
        server: EvalServer,
        max_evals: int | None,
        max_token_cost: float | None,
    ) -> Result:
        from gepa.optimize_anything import OptimizeAnythingConfig, optimize_anything_from_task

        cfg = OptimizeAnythingConfig(
            engine=self.engine,
            max_evals=max_evals,
            max_token_cost=max_token_cost,
            max_concurrency=server.max_concurrency,
            output_dir=self.run_dir,
            run_dir=self.run_dir,
            stop_at_score=self.stop_at_score,
            effort=self.effort,
            max_thinking_tokens=self.max_thinking_tokens,
            sandbox=bool(self.sandbox) if self.sandbox is not None else False,
            engine_config=self._build_engine_config(self.engine, self.engine_config),
        )

        oa_result = optimize_anything_from_task(oa_task, evaluate, cfg)

        return Result(
            best_candidate=oa_result.best_candidate,
            best_score=oa_result.best_score,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
            metadata={
                **oa_result.metadata,
                "optimize_anything_engine": getattr(self.engine, "name", self.engine),
            },
        )

    def _run_ensemble(
        self,
        task: Task,
        evaluate: Any,
        server: EvalServer,
        max_evals: int | None,
        max_token_cost: float | None,
    ) -> Result:
        from gepa.optimize_anything import optimize_anything_from_task

        n = len(self.configs)
        # Fixed ensemble strategies use the same fair-share partition.
        # Adaptive sequential uses bounded slices instead, so it can switch
        # between existing engines before one engine consumes its full share.
        use_fixed_partition = self.strategy != "adaptive_sequential"
        per_evals = max_evals // n if max_evals is not None and use_fixed_partition else None
        per_cost = (max_token_cost / n) if max_token_cost is not None and use_fixed_partition else None
        oa_configs = [
            self._materialize_config(
                entry,
                idx,
                max_evals=per_evals,
                max_token_cost=per_cost,
                max_concurrency=server.max_concurrency,
            )
            for idx, entry in enumerate(self.configs)
        ]

        if self.strategy == "adaptive_sequential":
            return self._run_adaptive_sequential(
                task,
                evaluate,
                server,
                max_evals=max_evals,
                max_token_cost=max_token_cost,
            )

        if self.strategy == "sequential":
            stage_results: list[Any] = []
            current_candidate = task.initial_candidate
            best_so_far: Any | None = None
            handoffs: list[dict[str, Any]] = []
            handoff_root = Path(self.run_dir) / "handoff" if self.run_dir else None
            evals_dir = server.output_dir / "evals" if server.output_dir is not None else None
            for stage_idx, (entry, cfg) in enumerate(zip(self.configs, oa_configs, strict=True)):
                if server.budget.exhausted:
                    break
                stage_task = self._to_oa_task(self._task_for_entry(task, entry))
                if handoffs:
                    cfg = replace(cfg, engine_config={**cfg.engine_config, "handoffs": list(handoffs)})
                stage_task = replace(stage_task, initial_candidate=current_candidate)
                eval_start = len(server.eval_log)
                result = optimize_anything_from_task(stage_task, evaluate, cfg)
                eval_end = len(server.eval_log)
                stage_results.append(result)
                if best_so_far is None or result.best_score > best_so_far.best_score:
                    best_so_far = result
                current_candidate = best_so_far.best_candidate
                if self.handoff.enabled and handoff_root is not None:
                    engine = str(entry.get("engine", self.engine))
                    handoffs.append(
                        collect_stage_handoff(
                            config=self.handoff,
                            handoff_root=handoff_root,
                            evals_dir=evals_dir,
                            stage_idx=stage_idx,
                            engine=engine,
                            eval_start=eval_start,
                            eval_end=eval_end,
                            best_candidate=result.best_candidate,
                            best_score=float(result.best_score),
                        )
                    )
            if not stage_results:
                return Result(
                    best_candidate=current_candidate,
                    best_score=server.best_score,
                    total_evals=server.budget.used,
                    eval_log=server.eval_log,
                    metadata={"adapter_cost": 0.0, "stage_results": []},
                )
            final = stage_results[-1]
            final.metadata["stage_results"] = stage_results
            self._attach_ensemble_costs(
                final,
                self.configs[: len(stage_results)],
                stage_results,
                key="stage_adapter_costs",
            )
            if handoffs:
                final.metadata["optimize_anything_handoffs"] = handoffs
            if best_so_far is not None:
                final.metadata["best_stage_score"] = best_so_far.best_score
                final.metadata["best_stage_candidate"] = best_so_far.best_candidate
            return self._wrap_result(final, server)

        # Parallel-family: one optimize_anything run per config, using the same
        # fair-share budget partition as sequential.
        oa_tasks = [self._to_oa_task(self._task_for_entry(task, entry)) for entry in self.configs]

        workers = self.max_workers if self.max_workers is not None else n
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(
                pool.map(
                    lambda pair: optimize_anything_from_task(pair[0], evaluate, pair[1]),
                    zip(oa_tasks, oa_configs),
                )
            )

        if self.strategy == "parallel":
            primary = results[0]
            primary.metadata["all_results"] = results
            self._attach_ensemble_costs(primary, self.configs, results, key="member_adapter_costs")
            return self._wrap_result(primary, server)
        if self.strategy == "best_of":
            winner = max(results, key=lambda r: r.best_score)
            winner.metadata["all_results"] = results
            self._attach_ensemble_costs(winner, self.configs, results, key="member_adapter_costs")
            return self._wrap_result(winner, server)
        if self.strategy == "vote":
            vote_scores: list[float] = []
            for r in results:
                try:
                    score, _ = evaluate(r.best_candidate)
                except Exception:
                    score = float("-inf")
                vote_scores.append(float(score))
            winner_idx = max(range(len(results)), key=lambda i: vote_scores[i])
            winner = results[winner_idx]
            winner.metadata["vote_scores"] = vote_scores
            winner.metadata["vote_candidates"] = [r.best_candidate for r in results]
            winner.metadata["vote_winner_idx"] = winner_idx
            winner.metadata["all_results"] = results
            self._attach_ensemble_costs(winner, self.configs, results, key="member_adapter_costs")
            return self._wrap_result(winner, server)
        raise ValueError(f"Unknown strategy: {self.strategy!r}")

    def _run_adaptive_sequential(
        self,
        task: Task,
        evaluate: Any,
        server: EvalServer,
        *,
        max_evals: int | None,
        max_token_cost: float | None,
    ) -> Result:
        from gepa.optimize_anything import optimize_anything_from_task

        n = len(self.configs)
        plateau_evals = int(
            self.scheduler.get(
                "plateau_evals",
                self.scheduler.get("slice_evals", max(1, (max_evals or 100) // max(1, n * 4))),
            )
        )
        patience = int(self.scheduler.get("patience", 1))
        min_evals_per_stage = int(self.scheduler.get("min_evals_per_stage", plateau_evals))
        improvement_epsilon = float(self.scheduler.get("improvement_epsilon", 0.0))
        cycle = _config_bool(self.scheduler.get("cycle", True))
        max_switches_raw = self.scheduler.get("max_switches")
        max_switches = int(max_switches_raw) if max_switches_raw is not None else None

        if plateau_evals <= 0:
            raise ValueError("adapter.scheduler.plateau_evals must be positive")
        if patience <= 0:
            raise ValueError("adapter.scheduler.patience must be positive")
        if min_evals_per_stage < 0:
            raise ValueError("adapter.scheduler.min_evals_per_stage cannot be negative")

        stage_results: list[Any] = []
        schedule: list[dict[str, Any]] = []
        current_idx = 0
        current_candidate = task.initial_candidate
        best_so_far: Any | None = None
        no_improvement_slices = 0
        current_stage_evals = 0
        switches = 0
        adapter_cost = 0.0
        idle_slices_in_round = 0
        handoffs: list[dict[str, Any]] = []
        handoff_root = Path(self.run_dir) / "handoff" if self.run_dir else None
        evals_dir = server.output_dir / "evals" if server.output_dir is not None else None

        while not server.budget.exhausted:
            if current_idx >= n:
                break
            outer_remaining = server.budget.remaining
            if outer_remaining is not None and outer_remaining <= 0:
                break
            if stage_results and outer_remaining is not None and outer_remaining < min_evals_per_stage:
                break
            slice_evals = plateau_evals if outer_remaining is None else min(plateau_evals, outer_remaining)
            if slice_evals <= 0:
                break

            remaining_cost = None
            if max_token_cost is not None:
                remaining_cost = max(0.0, max_token_cost - adapter_cost)
                if remaining_cost <= 0:
                    break

            entry = dict(self.configs[current_idx])
            if self.run_dir and "run_dir" not in entry:
                engine = str(entry.get("engine", self.engine))
                entry["run_dir"] = str(Path(self.run_dir) / f"{current_idx}_{engine}" / f"slice_{len(stage_results):04d}")
            cfg = self._materialize_config(
                entry,
                current_idx,
                max_evals=slice_evals,
                max_token_cost=remaining_cost,
                max_concurrency=server.max_concurrency,
            )

            stage_task = self._to_oa_task(self._task_for_entry(task, entry))
            if handoffs:
                cfg = replace(cfg, engine_config={**cfg.engine_config, "handoffs": list(handoffs)})
            stage_task = replace(stage_task, initial_candidate=current_candidate)

            eval_start = len(server.eval_log)
            budget_start = server.budget.used
            best_before = best_so_far.best_score if best_so_far is not None else float("-inf")
            result = optimize_anything_from_task(stage_task, evaluate, cfg)
            budget_end = server.budget.used
            eval_end = len(server.eval_log)
            eval_delta = budget_end - budget_start
            current_stage_evals += eval_delta
            stage_results.append(result)

            result_cost = float(getattr(result, "metadata", {}).get("adapter_cost", 0.0))
            adapter_cost += result_cost
            improved = float(result.best_score) > (float(best_before) + improvement_epsilon)
            if best_so_far is None or float(result.best_score) > (float(best_so_far.best_score) + improvement_epsilon):
                best_so_far = result
            if best_so_far is not None:
                current_candidate = best_so_far.best_candidate
            if improved:
                no_improvement_slices = 0
                idle_slices_in_round = 0
            else:
                no_improvement_slices += 1
                idle_slices_in_round = idle_slices_in_round + 1 if eval_delta == 0 else 0

            engine_name = str(entry.get("engine", self.engine))
            schedule.append(
                {
                    "engine_idx": current_idx,
                    "engine": engine_name,
                    "eval_start": budget_start,
                    "eval_end": budget_end,
                    "eval_delta": eval_delta,
                    "best_before": best_before,
                    "best_after": best_so_far.best_score if best_so_far is not None else float("-inf"),
                    "improved": improved,
                    "adapter_cost": result_cost,
                }
            )

            if self.handoff.enabled and handoff_root is not None:
                handoffs.append(
                    collect_stage_handoff(
                        config=self.handoff,
                        handoff_root=handoff_root,
                        evals_dir=evals_dir,
                        stage_idx=len(stage_results) - 1,
                        engine=engine_name,
                        eval_start=eval_start,
                        eval_end=eval_end,
                        best_candidate=result.best_candidate,
                        best_score=float(result.best_score),
                    )
                )

            if idle_slices_in_round >= n:
                break

            should_switch = (
                no_improvement_slices >= patience
                and current_stage_evals >= min_evals_per_stage
                and n > 1
            )
            if should_switch:
                if max_switches is not None and switches >= max_switches:
                    break
                next_idx = current_idx + 1
                if next_idx >= n:
                    if not cycle:
                        break
                    next_idx = 0
                switches += 1
                current_idx = next_idx
                current_stage_evals = 0
                no_improvement_slices = 0

        if not stage_results:
            return Result(
                best_candidate=current_candidate,
                best_score=server.best_score,
                total_evals=server.budget.used,
                eval_log=server.eval_log,
                metadata={
                    "adapter_cost": 0.0,
                    "stage_results": [],
                    "adaptive_schedule": schedule,
                    "adaptive_stop_reason": "budget_exhausted" if server.budget.exhausted else "no_slices_run",
                },
            )

        final = stage_results[-1]
        if best_so_far is not None:
            final.best_candidate = best_so_far.best_candidate
            final.best_score = best_so_far.best_score
            final.metadata["best_stage_score"] = best_so_far.best_score
            final.metadata["best_stage_candidate"] = best_so_far.best_candidate
        final.metadata["stage_results"] = stage_results
        final.metadata["adaptive_schedule"] = schedule
        final.metadata["adaptive_switches"] = switches
        final.metadata["adaptive_scheduler"] = {
            "plateau_evals": plateau_evals,
            "patience": patience,
            "min_evals_per_stage": min_evals_per_stage,
            "improvement_epsilon": improvement_epsilon,
            "cycle": cycle,
            "max_switches": max_switches,
        }
        final.metadata["adaptive_stop_reason"] = "budget_exhausted" if server.budget.exhausted else "scheduler_stopped"
        if handoffs:
            final.metadata["optimize_anything_handoffs"] = handoffs
        self._attach_ensemble_costs(final, [dict(row) for row in schedule], stage_results, key="stage_adapter_costs")
        return self._wrap_result(final, server)

    @staticmethod
    def _attach_ensemble_costs(
        result: Any,
        entries: list[dict[str, Any]],
        results: list[Any],
        *,
        key: str,
    ) -> None:
        cost_rows: list[dict[str, Any]] = []
        total = 0.0
        for idx, (entry, stage_result) in enumerate(zip(entries, results, strict=True)):
            cost = float(getattr(stage_result, "metadata", {}).get("adapter_cost", 0.0))
            total += cost
            cost_rows.append(
                {
                    "idx": idx,
                    "engine": str(entry.get("engine", "unknown")),
                    "adapter_cost": cost,
                    "best_score": float(getattr(stage_result, "best_score", float("nan"))),
                }
            )
        result.metadata[key] = cost_rows
        result.metadata["adapter_cost"] = total

    def _materialize_config(
        self,
        entry: dict[str, Any],
        idx: int,
        *,
        max_evals: int | None,
        max_token_cost: float | None,
        max_concurrency: int,
    ) -> Any:
        """Build an :class:`OptimizeAnythingConfig` for one ensemble member.

        Per-entry keys (``engine``, ``engine_config``, ``effort``,
        ``max_thinking_tokens``, ``stop_at_score``) override the adapter-level
        defaults. ``run_dir`` is auto-derived as ``<self.run_dir>/<idx>_<engine>/``
        unless the entry provides one explicitly.
        """
        from gepa.optimize_anything import OptimizeAnythingConfig

        engine = entry.get("engine", self.engine)
        engine_config = entry.get("engine_config") or {}
        effort = entry.get("effort", self.effort)
        max_thinking_tokens = entry.get("max_thinking_tokens", self.max_thinking_tokens)
        stop_at_score = entry.get("stop_at_score", self.stop_at_score)
        sandbox = entry.get("sandbox", self.sandbox)

        if entry.get("run_dir"):
            run_dir = entry["run_dir"]
        elif self.run_dir:
            engine_name = getattr(engine, "name", engine) if not isinstance(engine, str) else engine
            run_dir = str(Path(self.run_dir) / f"{idx}_{engine_name}")
        else:
            run_dir = None

        return OptimizeAnythingConfig(
            engine=engine,
            max_evals=max_evals,
            max_token_cost=max_token_cost,
            max_concurrency=max_concurrency,
            output_dir=run_dir,
            run_dir=run_dir,
            stop_at_score=stop_at_score,
            effort=effort,
            max_thinking_tokens=max_thinking_tokens,
            sandbox=bool(sandbox) if sandbox is not None else False,
            engine_config=self._build_engine_config(engine, engine_config),
        )

    def _wrap_result(self, oa_result: Any, server: EvalServer) -> Result:
        engine_name: Any
        if self.strategy == "single":
            engine_name = getattr(self.engine, "name", self.engine)
        else:
            engine_name = self.strategy
        return Result(
            best_candidate=oa_result.best_candidate,
            best_score=oa_result.best_score,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
            metadata={
                **oa_result.metadata,
                "optimize_anything_engine": engine_name,
                "optimize_anything_strategy": self.strategy,
            },
        )

    def process_result(self, result: Result, output_dir: Path) -> None:
        """No-op — engines write their own artifacts under ``run_dir``
        (= ``<hydra_run_dir>/<adapter_name>/``), which is already inside the
        run's output tree."""
        return

    def _build_engine_config(self, engine: Any, engine_config: dict[str, Any]) -> dict[str, Any]:
        """Forward the per-engine ``engine_config`` plus terrarium-injected
        callbacks. Callbacks are only attached when ``engine == "gepa"`` — the
        other built-in engines don't read them and would emit an unknown-key
        warning."""
        merged = dict(engine_config)
        if self.callbacks and isinstance(engine, str) and engine == "gepa":
            existing = list(merged.get("callbacks") or [])
            existing.extend(self.callbacks)
            merged["callbacks"] = existing
        return merged

    def _task_for_entry(self, task: Task, entry: dict[str, Any]) -> Task:
        """Return the task split view requested by one ensemble member.

        The runner applies ``benchmark.split_train_val`` before adapters run,
        but optimize_anything compositions may mix GEPA-like engines that need a
        real val set with agentic engines that should see train+val merged. The
        runner preserves the original visible split in private metadata so each
        sequential or parallel member can request its own view.
        """
        if "split_train_val" not in entry:
            return task

        split_train_val = _config_bool(entry.get("split_train_val"))
        metadata = dict(task.metadata)
        source_train = metadata.get("_terrarium_source_train_set", task.train_set)
        source_val = metadata.get("_terrarium_source_val_set", task.val_set)
        train_set = list(source_train) if source_train is not None else None
        val_set = list(source_val) if source_val is not None else None

        if not split_train_val and val_set:
            train_set = list(train_set or [])
            train_set.extend(val_set)
            val_set = None

        metadata["optimize_anything_split_train_val"] = split_train_val
        return replace(task, train_set=train_set, val_set=val_set, metadata=metadata)


def _config_bool(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "no", "off"}:
            return False
        if normalized in {"true", "1", "yes", "on"}:
            return True
    return bool(value)


def create_adapter(**kwargs: Any) -> OptimizeAnythingAdapter:
    """Factory for ``adapter=custom`` loading via ``terrarium.runner.load_adapter``."""
    return OptimizeAnythingAdapter(**kwargs)
