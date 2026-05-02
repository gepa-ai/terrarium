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
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from terrarium.adapter import Result
from terrarium.budget import BudgetExhausted
from terrarium.sandbox import bwrap_prefix, claude_settings_args, prepare_claude_home
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
        stop_at_score: float | None = None,
        max_thinking_tokens: int | None = None,
        sandbox: bool | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.engine = dict(engine) if engine else {}
        self.reflection = dict(reflection) if reflection else {}
        self.merge = dict(merge) if merge else None
        self.refiner = dict(refiner) if refiner else None
        self.objective = objective
        self.background = background
        self.callbacks = callbacks or []
        # Forwarded to ``gepa.lm.LM(...)`` (litellm path) or to the SDK LM —
        # e.g. ``timeout``, ``num_retries``.
        self.reflection_lm_kwargs = dict(reflection_lm_kwargs) if reflection_lm_kwargs else {}
        self.stop_at_score = stop_at_score
        self.max_thinking_tokens = max_thinking_tokens
        # None ⇒ inherit the runner's top-level ``sandbox:`` default.
        self.sandbox = sandbox

    def effective_sandbox(self, top_level_sandbox: bool | None = None) -> bool:
        """Whether this GEPA configuration actually creates a sandbox boundary."""
        sandbox = self.sandbox if self.sandbox is not None else top_level_sandbox
        if not sandbox:
            return False
        lm_name = self.reflection.get("reflection_lm")
        return isinstance(lm_name, str) and (
            lm_name.startswith("claude_code/") or lm_name.startswith("claude_code_agent/")
        )

    def sandbox_scope(self, top_level_sandbox: bool | None = None) -> dict[str, bool]:
        return {
            "optimizer_subprocess_sandbox": self.effective_sandbox(top_level_sandbox),
            "candidate_execution_sandbox": False,
            "network_namespace_isolated": False,
        }

    def evolve(self, task: Task, server: EvalServer) -> Result:
        from gepa.lm import LM
        from gepa.optimize_anything import GEPAConfig, optimize_anything

        budget = server.budget

        reflection_kwargs = dict(self.reflection)
        reflection_lm: Any | None = None
        agent_proposer: Any | None = None
        if "reflection_lm" in reflection_kwargs:
            lm_name = reflection_kwargs["reflection_lm"]
            if isinstance(lm_name, str) and lm_name.startswith("claude_code_agent/"):
                # File-based agent proposer: wired through GEPA's
                # ``custom_candidate_proposer`` hook, NOT ``reflection_lm``, so
                # claude receives the structured reflective_dataset (rather
                # than a pre-rendered prompt). Needs
                # ``engine.write_agent_state=True`` (set below) so the
                # run_dir contains agent-readable state files.
                from terrarium.adapters.gepa_cc_agent import ClaudeCodeAgentProposer

                effort = self.reflection_lm_kwargs.get("reasoning_effort")
                # Same resolution order other adapters use: adapter-level
                # override wins, else fall back to the task's fields. These
                # are what program.md-style proposers need to understand the
                # scoring rubric — bypassing the reflection_lm path means
                # we no longer get them for free via the GEPA prompt template.
                agent_objective = self.objective or task.objective
                agent_background = self.background or task.background
                agent_proposer = ClaudeCodeAgentProposer(
                    model=lm_name.split("/", 1)[1],
                    run_dir=self.run_dir,
                    objective=agent_objective,
                    background=agent_background,
                    max_budget_usd=budget.max_token_cost,
                    max_thinking_tokens=self.max_thinking_tokens,
                    effort=effort,
                    sandbox=bool(self.sandbox) if self.sandbox is not None else True,
                )
                # Strip reflection_lm — GEPA's validation rejects custom_candidate_proposer
                # when a reflection_lm is also set.
                reflection_kwargs.pop("reflection_lm", None)
                reflection_kwargs["custom_candidate_proposer"] = agent_proposer
            elif isinstance(lm_name, str) and lm_name.startswith("claude_code/"):
                # Text-only reflection LM: runs claude as a sandboxed subprocess
                # per reflection call, no filesystem I/O beyond a scratch dir.
                effort = self.reflection_lm_kwargs.get("reasoning_effort")
                reflection_lm = ClaudeCodeReflectionProposer(
                    model=lm_name.split("/", 1)[1],
                    max_budget_usd=budget.max_token_cost,
                    max_thinking_tokens=self.max_thinking_tokens,
                    effort=effort,
                    sandbox=bool(self.sandbox),
                )
                reflection_kwargs["reflection_lm"] = reflection_lm
            elif isinstance(lm_name, str) and lm_name.startswith("anthropic_sdk/"):
                # Bypass litellm via the official Anthropic SDK.
                from terrarium.adapters._anthropic_sdk_lm import AnthropicSdkLM
                lm_kwargs = dict(self.reflection_lm_kwargs)
                lm_kwargs.pop("reasoning_effort", None)
                if "num_retries" in lm_kwargs:  # litellm naming → SDK naming
                    lm_kwargs.setdefault("max_retries", lm_kwargs.pop("num_retries"))
                reflection_lm = AnthropicSdkLM(
                    model=lm_name.split("/", 1)[1],
                    max_thinking_tokens=self.max_thinking_tokens,
                    **lm_kwargs,
                )
                reflection_kwargs["reflection_lm"] = reflection_lm
            else:
                lm_kwargs = dict(self.reflection_lm_kwargs)
                if self.max_thinking_tokens is not None:
                    lm_kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.max_thinking_tokens}
                    lm_kwargs.pop("reasoning_effort", None)
                reflection_lm = LM(lm_name, **lm_kwargs)
                reflection_kwargs["reflection_lm"] = reflection_lm

        # Runner-controlled engine fields override whatever the user set.
        engine_kwargs: dict[str, Any] = {
            **self.engine,
            "run_dir": self.run_dir,
            "max_metric_calls": budget.max_evals,
        }
        # Agent proposer reads history from run_dir → needs the engine to write
        # its state to disk as agent-readable files. Respect user override.
        if agent_proposer is not None:
            engine_kwargs.setdefault("write_agent_state", True)

        cost_source = reflection_lm or agent_proposer
        # GEPA's ``engine.max_reflection_cost`` wires a ``MaxReflectionCostStopper``
        # against ``config.reflection.reflection_lm``. On the agent-proposer path
        # that field is ``None`` (we stripped it for the custom_candidate_proposer
        # contract), so the auto-wired stopper reads ``0.0`` forever and never
        # fires. Meanwhile the proposer's own ``raise BudgetExhausted`` is
        # swallowed by the blanket ``except Exception`` in
        # ``reflective_mutation.execute_proposal`` — resulting in an infinite
        # loop of no-op iterations. Wire the stopper ourselves, pointing at
        # whichever cost source is active.
        if budget.max_token_cost is not None and cost_source is None:
            engine_kwargs["max_reflection_cost"] = budget.max_token_cost

        cost_callback: _ReflectionCostCallback | None = None
        callbacks = list(self.callbacks)
        if cost_source is not None:
            cost_callback = _ReflectionCostCallback(cost_source, server.tracker, output_dir=self.run_dir)
            callbacks.append(cost_callback)

        if agent_proposer is not None:
            callbacks.append(_ReflectiveDatasetDumpCallback(self.run_dir))

        if task.val_set:
            callbacks.append(_ProgressCallback(server, reflection_lm=cost_source))

        stop_callbacks: list[Any] = []
        if self.stop_at_score is not None:
            from gepa.utils.stop_condition import ScoreThresholdStopper
            stop_callbacks.append(ScoreThresholdStopper(self.stop_at_score))
        if budget.max_token_cost is not None and cost_source is not None:
            from gepa.utils.stop_condition import MaxReflectionCostStopper
            stop_callbacks.append(
                MaxReflectionCostStopper(budget.max_token_cost, reflection_lm=cost_source)
            )

        # GEPAConfig.__post_init__ converts dict -> nested config dataclass.
        config = GEPAConfig(
            engine=engine_kwargs,
            reflection=reflection_kwargs,
            merge=self.merge,
            refiner=self.refiner,
            callbacks=callbacks or None,
            stop_callbacks=stop_callbacks or None,
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
            # val_set only — test_set is a held-out split reserved for
            # runner.py's post-run eval and must never leak into the
            # optimization loop.
            if task.val_set:
                oa_kwargs["valset"] = task.val_set
            if "val_set" in task.metadata:
                oa_kwargs.setdefault("valset", task.metadata["val_set"])

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
        adapter_cost = 0.0
        if cost_callback is not None:
            reflection_meta = {"reflection_cost_log": cost_callback.cost_log}
            if cost_callback.cost_log:
                adapter_cost = cost_callback.cost_log[-1]["reflection_cost"]

        if gepa_result is not None:
            return Result(
                best_candidate=gepa_result.best_candidate,
                best_score=gepa_result.val_aggregate_scores[gepa_result.best_idx],
                total_evals=server.budget.used,
                eval_log=server.eval_log,
                metadata={"gepa_result": gepa_result, "adapter_cost": adapter_cost, **reflection_meta},
            )

        return Result(
            best_candidate=server.best_candidate,
            best_score=server.best_score,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
            metadata={"adapter_cost": adapter_cost, **reflection_meta},
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


class _ReflectiveDatasetDumpCallback:
    """Write each iteration's reflective_dataset to disk under run_dir.

    GEPA core writes per-iteration meta / components / trace under
    ``iterations/NNNNN/`` (with ``NNNNN = state.i + 1`` — seed owns id 0),
    but the **adapter-curated** reflective_dataset (the thing the LM
    normally sees as ``<side_info>``) is never persisted by core. This
    callback closes that gap so the agent proposer can browse past
    iterations' structured feedback via
    ``iterations/NNNNN/reflective_dataset.json``.

    Only installed when the file-based agent proposer is selected.
    """

    def __init__(self, run_dir: str | Path | None) -> None:
        self._run_dir = Path(run_dir) if run_dir is not None else None

    def on_reflective_dataset_built(self, event: dict[str, Any]) -> None:
        if self._run_dir is None:
            return
        iteration = event.get("iteration")
        dataset = event.get("dataset") or event.get("reflective_dataset")
        if iteration is None or dataset is None:
            return
        # ``event["iteration"]`` is ``ctx.iteration`` — the 1-indexed
        # on-disk iteration id (seed owns 0, first loop proposal is 1).
        # No shifting needed.
        target = self._run_dir / "iterations" / f"{int(iteration):05d}" / "reflective_dataset.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(dataset, indent=2, default=str))



class ClaudeCodeReflectionProposer:
    """Drop-in replacement for ``gepa.lm.LM`` that routes reflection through
    the Claude Code CLI instead of litellm.

    Activated by setting ``reflection.reflection_lm=claude_code/<model>`` (e.g.
    ``claude_code/claude-sonnet-4-6``). Each ``__call__`` spawns one
    ``claude --print --output-format json`` subprocess with the reflection
    prompt as a positional argument. Nothing is written to disk — the new
    candidate text is read straight from the CLI's JSON ``result`` field.

    Conforms to GEPA's LanguageModel protocol ``(str | list[dict]) -> str``
    and exposes ``total_cost`` / ``total_tokens_in`` / ``total_tokens_out``
    so ``_ReflectionCostCallback`` and ``_ProgressCallback`` plug in unchanged.

    Budget stopper (same pattern as meta-harness):
        Before every call, if ``max_budget_usd`` is set and cumulative spend
        already meets/exceeds it, raise :class:`BudgetExhausted`. GEPA's
        ``optimize_anything`` catches this at the top level and returns the
        current best candidate; the terrarium runner re-catches it too.
    """

    def __init__(
        self,
        model: str,
        *,
        max_budget_usd: float | None = None,
        max_thinking_tokens: int | None = None,
        effort: str | None = None,
        sandbox: bool = True,
    ) -> None:
        self.model = model
        self.max_budget_usd = max_budget_usd
        self.max_thinking_tokens = max_thinking_tokens
        self.effort = effort
        self.sandbox = sandbox
        self._total_cost: float = 0.0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0
        self._lock = threading.Lock()

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return self._total_tokens_out

    @staticmethod
    def _flatten_prompt(prompt: str | list[dict[str, Any]]) -> str:
        """Collapse GEPA's optional message-list prompt to a single string that
        we can pass as a positional arg to ``claude --print``."""
        if isinstance(prompt, str):
            return prompt
        parts: list[str] = []
        for m in prompt:
            if not isinstance(m, dict):
                parts.append(str(m))
                continue
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            parts.append(f"{role}: {content}" if role else str(content))
        return "\n\n".join(parts)

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        # Stopper: refuse further calls once we've hit the token budget cap.
        if self.max_budget_usd is not None and self._total_cost >= self.max_budget_usd:
            raise BudgetExhausted(
                f"claude_code proposer spent ${self._total_cost:.2f} "
                f">= cap ${self.max_budget_usd:.2f}"
            )

        prompt_text = self._flatten_prompt(prompt)

        # Reflection is text-in / text-out — the subprocess has no reason to
        # touch the filesystem or the network. Always spawn inside a fresh
        # empty tempdir and sandbox it hard: no Bash, no file tools, no
        # network.
        cleanup = tempfile.TemporaryDirectory(prefix="terrarium_gepa_reflect_")
        work_dir = Path(cleanup.name)

        # Reflection is text-in / text-out: deny every tool so even if the
        # model tries to call Bash/Read/Edit it auto-fails. bwrap (when
        # sandboxed) is belt; the deny list is braces. Network stays
        # shared (claude needs api.anthropic.com); with no Bash there's
        # nothing to curl from inside the jail anyway.
        claude_home = prepare_claude_home(work_dir) if self.sandbox else None
        cmd: list[str] = bwrap_prefix(work_dir, claude_home=claude_home) if self.sandbox else []
        cmd += [
            "claude",
            "--print",
            prompt_text,
            "--output-format", "json",
            "--model", self.model,
            "--permission-mode", "bypassPermissions",
            "--disallowedTools=WebFetch,WebSearch,Bash,Read,Edit,Write,Glob,Grep,Task,NotebookEdit",
        ]
        if self.sandbox:
            cmd.extend(claude_settings_args(work_dir))  # macOS Seatbelt fallback
        # ``max_thinking_tokens`` takes precedence over ``--effort`` (same mutex
        # rule the runner enforces for the claude_code / meta_harness adapters).
        if self.max_thinking_tokens is None and self.effort is not None:
            cmd.extend(["--effort", self.effort])
        if self.max_budget_usd is not None:
            remaining = max(0.01, self.max_budget_usd - self._total_cost)
            cmd.extend(["--max-budget-usd", f"{remaining:.4f}"])

        env = {**os.environ}
        if claude_home is not None:
            env["HOME"] = str(claude_home)
        # Strip CLAUDECODE so nested claude-in-claude doesn't confuse the CLI
        # (same workaround meta-harness uses).
        env.pop("CLAUDECODE", None)
        # Raise per-response output cap for long C++ candidate bodies.
        env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")
        if self.max_thinking_tokens is not None:
            env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"
            env["MAX_THINKING_TOKENS"] = str(self.max_thinking_tokens)

        try:
            proc = subprocess.run(
                cmd, cwd=str(work_dir), env=env, capture_output=True, text=True
            )
        finally:
            cleanup.cleanup()

        payload: dict[str, Any] = {}
        stdout = (proc.stdout or "").strip()
        if stdout:
            try:
                payload = json.loads(stdout)
            except (json.JSONDecodeError, ValueError):
                payload = {}

        try:
            cost = float(payload.get("total_cost_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        usage = payload.get("usage") or {}
        try:
            tokens_in = int(usage.get("input_tokens", 0) or 0)
            tokens_out = int(usage.get("output_tokens", 0) or 0)
        except (TypeError, ValueError):
            tokens_in = tokens_out = 0
        result_text = payload.get("result", "") or ""

        with self._lock:
            self._total_cost += cost
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out

        return result_text

    def __repr__(self) -> str:
        return (
            f"ClaudeCodeProposer(model={self.model!r}, "
            f"cost=${self._total_cost:.2f})"
        )


def create_adapter(**kwargs: Any) -> GEPAAdapter:
    """Factory for adapter-from-file loading (``adapter=custom`` with this file)."""
    return GEPAAdapter(**kwargs)
