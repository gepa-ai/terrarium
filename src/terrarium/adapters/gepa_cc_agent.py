"""Claude Code agent proposer for GEPA (file-based, ``custom_candidate_proposer``).

Drops into GEPA via ``reflection.custom_candidate_proposer`` — **not**
``reflection_lm``. The reflection-LM hook is a text-in / text-out callable and
has no structured access to the per-iteration ``reflective_dataset``;
``custom_candidate_proposer`` does, with signature::

    (candidate, reflective_dataset, components_to_update) -> dict[str, str]

That's the contract this module implements. Each call:

1. Allocates a unique subdir under ``<run_dir>/proposals/`` (timestamp + pid +
   uuid) so concurrent proposers or multiple processes sharing the same
   ``run_dir`` can't clobber each other.
2. Writes the parent candidate components, the reflective dataset, and the
   components-to-update list as structured files in that subdir. Nothing goes
   into the prompt body — ``claude`` reads everything from disk.
3. Launches ``claude --print`` with ``cwd=run_dir`` and a short wrapper prompt
   pointing at the subdir. Sandbox scopes file tools to ``run_dir`` so Claude
   can also browse ``iterations/``, ``pareto/``, ``history.md`` for
   history.
4. Reads one file per updated component back from ``<subdir>/new/``. The file
   body is parsed for a fenced ```` ``` ```` block (same convention
   ``InstructionProposalSignature.output_extractor`` uses). Missing files fall
   back to the original candidate text so GEPA's acceptance criterion just
   rejects the no-op proposal.
5. Copies the agent's ``<subdir>/plan.md`` into
   ``iterations/<iter_id>/plan.md`` and rebuilds ``<run_dir>/history.md``
   from the iterations tree so the next proposer call sees a chronological
   log of every prior iteration's short plan + score outcome.

Needs ``engine.write_agent_state=True`` on the GEPA side so the run_dir
actually contains readable ``iterations/`` and ``pareto/``. The terrarium
``GEPAAdapter`` wiring sets that default when this proposer is selected.

Learns the current ``iteration`` (trace ``state.i``) and
``parent_iteration_id`` via the ``metadata`` kwarg on the ``ProposalFn``
contract — GEPA's ``reflective_mutation`` populates this dict on every
call.

Sandbox: OS jail (bwrap/Seatbelt) + Claude file-tool allow-list, both
scoped to ``run_dir``. Network is off. Bash is **on** — the OS jail
already confines shell commands to ``run_dir``, and grep/diff/jq/python
let the agent analyze past candidates and reflective datasets far more
effectively than the Claude file tools alone. Write isolation across
concurrent proposers is still convention-only (claude is *instructed* to
stay inside its own ``proposals/<subdir>/``); a stricter per-subdir write
jail or a git worktree per proposal would be needed to enforce this.

Budget + cost tracking mirrors :class:`ClaudeCodeReflectionProposer`: raises
:class:`BudgetExhausted` when cumulative spend hits ``max_budget_usd``,
exposes ``total_cost`` / ``total_tokens_in`` / ``total_tokens_out`` so the
existing reflection-cost GEPA callback plugs in unchanged (the callback only
needs those three attributes, not an LM protocol).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from terrarium.budget import BudgetExhausted
from terrarium.sandbox import sandbox_args

_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _extract_fenced(text: str) -> str:
    """Pull the last fenced ```` ``` ```` block out of ``text``.

    Matches the convention in :func:`gepa.strategies.instruction_proposal.
    InstructionProposalSignature.output_extractor`. If there's no complete
    fence, falls back to the raw text (stripped). If the content is preceded
    by an optional ``<language>`` spec, that line is dropped.
    """
    matches = _FENCE_RE.findall(text)
    if matches:
        return matches[-1].strip()
    # Handle partial/open fences: strip a leading ``` line if present.
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def _safe_component_filename(name: str, idx: int) -> str:
    """Derive a safe on-disk filename for a component name (mirrors
    :meth:`gepa.core.state.GEPAState._save_components`)."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return safe if safe else f"component_{idx:02d}"


class ClaudeCodeAgentProposer:
    """GEPA ``custom_candidate_proposer`` backed by the Claude Code CLI.

    Matches :class:`gepa.core.adapter.ProposalFn` — call signature
    ``(candidate, reflective_dataset, components_to_update) -> dict[str, str]``.

    Parallel-safe: each call writes under its own unique subdir of
    ``<run_dir>/proposals/``.
    """

    def __init__(
        self,
        model: str,
        run_dir: str | Path,
        *,
        objective: str | None = None,
        background: str | None = None,
        max_budget_usd: float | None = None,
        max_thinking_tokens: int | None = None,
        effort: str | None = None,
        sandbox: bool = True,
        subdir_prefix: str = "proposals",
    ) -> None:
        self.model = model
        self.run_dir = Path(run_dir)
        self.objective = objective
        self.background = background
        self.max_budget_usd = max_budget_usd
        self.max_thinking_tokens = max_thinking_tokens
        self.effort = effort
        self.sandbox = sandbox
        self.subdir_prefix = subdir_prefix
        self._total_cost: float = 0.0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0
        self._lock = threading.Lock()
        self._task_md_materialized = False

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return self._total_tokens_out

    def _ensure_task_md(self) -> None:
        """Write a single ``<run_dir>/task.md`` once per run.

        Mirrors the ``program.md`` pattern used by :class:`ClaudeCodeAdapter`:
        long task context is materialized on disk and referenced by path from
        the CLI prompt. One file (not a dir) so the agent incurs a single
        read per iteration instead of one-per-section.

        Idempotent and safe across parallel first-calls — file writes are
        last-writer-wins with identical content.
        """
        if self._task_md_materialized:
            return
        if self.objective is None and self.background is None:
            self._task_md_materialized = True
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        sections: list[str] = ["# Task"]
        if self.objective:
            sections.append("## Objective\n" + self.objective.strip())
        if self.background:
            sections.append("## Background\n" + self.background.strip())
        (self.run_dir / "task.md").write_text("\n\n".join(sections) + "\n")
        self._task_md_materialized = True

    def _allocate_subdir(self, iteration: int | None) -> Path:
        """Pick a collision-free subdir for this call under ``<run_dir>/<prefix>/``.

        Shape: ``iter_NNNNN-<uuid8>`` when the iteration id is known
        (standard path — supplied via ``metadata["iteration"]`` which is
        the 1-indexed on-disk iteration id); falls back to
        ``YYYYmmddTHHMMSS-<pid>-<uuid8>`` otherwise. The uuid suffix
        disambiguates parallel proposals that share the same iteration id
        (``num_parallel_proposals > 1``).
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        base = self.run_dir / self.subdir_prefix
        base.mkdir(parents=True, exist_ok=True)
        if iteration is not None:
            name = f"iter_{iteration:05d}-{uuid.uuid4().hex[:8]}"
        else:
            ts = time.strftime("%Y%m%dT%H%M%S")
            name = f"{ts}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        subdir = base / name
        subdir.mkdir(parents=True, exist_ok=False)
        return subdir

    def _copy_parent_iteration(self, parent_dir: Path, dest: Path) -> None:
        """Copy the slim parts of the parent iteration folder into ``dest``.

        Agent-locality convenience: copy ``meta.json``, ``val_scores.json``,
        ``plan.md`` and ``components/`` so the claude process can read the
        parent's program + reasoning next to its own working files without
        hopping across the run_dir. Intentionally excludes ``outputs/`` and
        ``trajectories/`` — those can be many MBs per iteration on adapters
        whose rollouts are big, and the agent can still reach them via
        ``iterations/<parent_id>/`` if it actually wants them.
        """
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("meta.json", "val_scores.json", "plan.md"):
            src = parent_dir / name
            if src.exists():
                (dest / name).write_text(src.read_text())
        comps_src = parent_dir / "components"
        if comps_src.is_dir():
            comps_dst = dest / "components"
            comps_dst.mkdir(parents=True, exist_ok=True)
            for f in comps_src.iterdir():
                if f.is_file():
                    (comps_dst / f.name).write_text(f.read_text())

    def _materialize(
        self,
        subdir: Path,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        parent_iteration_dir: Path | None,
    ) -> dict[str, str]:
        """Write reflection dataset + parent copy + index files into ``subdir``.

        Returns a map ``{component_name: safe_filename_stem}`` used to locate
        output files on disk. The parent candidate text is *not* re-written
        per-component any more — instead the whole parent iteration folder
        (slim parts) is copied into ``<subdir>/parent/`` via
        :meth:`_copy_parent_iteration`, so the agent has access to the
        parent's ``plan.md`` and scores as well as its component texts.
        """
        name_to_stem: dict[str, str] = {}
        used: set[str] = set()
        for i, name in enumerate(sorted(candidate.keys())):
            stem = _safe_component_filename(name, i)
            j = 0
            base_stem = stem
            while stem in used:
                j += 1
                stem = f"{base_stem}_{j}"
            used.add(stem)
            name_to_stem[name] = stem

        if parent_iteration_dir is not None:
            self._copy_parent_iteration(parent_iteration_dir, subdir / "parent")
        else:
            # First iteration or state unreadable: materialize parent texts
            # directly so the agent still has something to read, with no
            # ``plan.md`` / ``meta.json`` (absence is the signal).
            fallback_parent = subdir / "parent"
            fallback_parent.mkdir(parents=True, exist_ok=True)
            comps_dir = fallback_parent / "components"
            comps_dir.mkdir(parents=True, exist_ok=True)
            for name, text in candidate.items():
                (comps_dir / f"{name_to_stem[name]}.txt").write_text(text)

        reflection_dir = subdir / "reflection"
        reflection_dir.mkdir()
        for name in components_to_update:
            records = list(reflective_dataset.get(name, []))
            (reflection_dir / f"{name_to_stem[name]}.json").write_text(
                json.dumps(records, indent=2, default=str)
            )

        # ``components_to_update.json`` + ``_index.json`` are intentionally
        # *not* written: they're small and fully derivable from the wrapper
        # prompt (which lists every component's name + stem + input/output
        # paths). Skipping them saves the agent two reads per call.

        new_dir = subdir / "new"
        new_dir.mkdir()

        return name_to_stem

    def _wrapper_prompt(
        self,
        subdir: Path,
        components: list[str],
        stems: dict[str, str],
        *,
        first_iteration: bool,
        parent_iter_id: int | None,
    ) -> str:
        """Program.md-style proposal brief, iteration-folder-oriented.

        Long task context (objective, background — often thousands of chars,
        including the scoring rubric) lives in ``<run_dir>/task.md``; we
        reference it by path so every proposal call doesn't re-ship it
        through the CLI prompt. The agent is pointed at ``history.md`` (the
        rolling log of all prior plans + outcomes) and ``parent/`` (a local
        copy of its parent iteration) so it can plan with full context
        without loading the whole ``iterations/`` tree.
        """
        rel = subdir.relative_to(self.run_dir).as_posix()
        task_md_exists = (self.objective is not None) or (self.background is not None)
        # Listed in the Outputs section below — carries both the stem (for
        # reflection file lookup) and the write path. No sidecar
        # ``components_to_update.json`` / ``_index.json`` needed.
        files_to_produce = "\n".join(
            f"- `{rel}/new/{stems[name]}.md`  (component {name!r})"
            for name in components
        )
        task_section = (
            "## Task\nThe problem statement and **scoring rubric** are in "
            "`task.md` at the top of the working directory. Read it first — "
            "it defines what a higher score means and, for graded problems, "
            "what the achievable range actually is. Do not assume a score of "
            "1.0 is perfect unless `task.md` says so.\n\n"
            if task_md_exists
            else ""
        )
        if first_iteration:
            history_section = (
                "## History\nThis is the first proposal in this run — there "
                "is no `history.md` yet. Your `parent/` directory contains "
                "the **seed program** (no `plan.md`).\n\n"
            )
        else:
            history_section = (
                "## History\nRead `history.md` at the top of the working "
                "directory **first**. It's a chronological log of every "
                "prior iteration's short plan and score outcome (accepted "
                "or rejected). Use it to avoid repeating strategies that "
                "have already been tried and to build on what worked.\n\n"
            )
        # Enumerate concrete input paths per component, same pattern the
        # Outputs section uses. Passing literal ``<stem>`` placeholders
        # confused the agent — it Read the placeholder filename verbatim and
        # errored out. We have the real filenames in ``stems``; use them.
        parent_component_lines = "\n".join(
            f"   - `{rel}/parent/components/{stems[name]}.txt`  (component {name!r})"
            for name in sorted(stems)
        )
        reflection_lines = "\n".join(
            f"   - `{rel}/reflection/{stems[name]}.json`  (component {name!r})"
            for name in components
        )
        parent_section = (
            f"3. `{rel}/parent/` — parent iteration (`plan.md`, `meta.json`, "
            f"`val_scores.json`, plus component files):\n{parent_component_lines}"
        )
        if parent_iter_id is not None:
            parent_section += (
                f"\n   For full rollout outputs see "
                f"`iterations/{parent_iter_id:05d}/outputs/` + `trajectories/`."
            )
        return f"""\
You are proposing improved text for one or more components of a program.
Your goal is to raise the program's aggregate score on
future evaluations, based on the reflective feedback from past runs.

Your working directory is the shared run directory. All inputs and
outputs for **this** proposal call live under:
  `{rel}/`

{task_section}{history_section}## Inputs (read in this order)
1. `task.md` — problem statement + scoring rubric.
2. `history.md` — chronological log of prior iterations' plans and scores.
{parent_section}
4. Per-example feedback for each component to update:
{reflection_lines}

## Wider state (browse when useful)
- `iterations/NNNNN/` — every past iteration (accepted *and* rejected),
  with `meta.json`, `components/`, `plan.md`, `reflective_dataset.json`,
  `trace.json`. `00000/` is always the seed program.
- `pareto/` — current Pareto frontier(s) keyed by iteration id.

## Outputs you must produce
One improved-text file per component:
{files_to_produce}

Wrap the new component text in a single ```…``` fenced block. Anything
outside the block is treated as rationale and discarded by the engine.

**Also** write a short plan to `{rel}/plan.md` — **≤50 words**, plain prose.
Describe what you're changing and why. This file is concatenated with
every prior iteration's plan into `history.md` so future proposers can
see your reasoning; keep it tight.

## Rules
- Write only inside `{rel}/new/` and `{rel}/plan.md`.
- Do not modify other iterations, state files, `task.md`, `history.md`,
  or another proposal's directory.
- Do not attempt to run any evaluations or verifications.
  The evaluation is done after you finished (by the outer engine).
"""

    def _read_new_texts(
        self,
        subdir: Path,
        candidate: dict[str, str],
        components_to_update: list[str],
        stems: dict[str, str],
    ) -> dict[str, str]:
        """Read one proposed text per updated component from ``<subdir>/new/``.

        Missing or empty files fall back to the parent candidate text — GEPA's
        acceptance criterion then rejects the no-op proposal on its own.
        """
        out: dict[str, str] = {}
        new_dir = subdir / "new"
        for name in components_to_update:
            stem = stems[name]
            path = new_dir / f"{stem}.md"
            if not path.exists():
                out[name] = candidate[name]
                continue
            body = path.read_text()
            extracted = _extract_fenced(body)
            out[name] = extracted if extracted else candidate[name]
        return out

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, str]:
        """Propose new component texts via the Claude Code CLI.

        ``metadata`` is an open context dict supplied by GEPA's reflective
        mutation path. Keys we read (all optional):

        * ``"iteration"`` — the proposal's trace index (``state.i``). Used to
          name the proposals subdir, derive the on-disk iteration id for
          ``plan.md`` persistence, and decide whether ``history.md`` is
          worth rebuilding.
        * ``"parent_iteration_id"`` — on-disk iteration id of the parent
          candidate under ``iterations/NNNNN/``. Used to locate the parent
          iteration folder without scanning.

        When ``metadata`` is ``None`` or the keys are absent (direct
        non-reflective callers, tests, tooling) we fall back to UUID-only
        subdir naming and skip ``history.md`` + ``plan.md`` bookkeeping.
        """
        if self.max_budget_usd is not None and self._total_cost >= self.max_budget_usd:
            raise BudgetExhausted(
                f"claude_code_agent proposer spent ${self._total_cost:.2f} "
                f">= cap ${self.max_budget_usd:.2f}"
            )

        if not components_to_update:
            return {}

        meta = dict(metadata) if metadata else {}
        iteration = meta.get("iteration")
        parent_iter_id = meta.get("parent_iteration_id")
        # ``iteration`` from GEPA is already the 1-indexed on-disk iter id
        # (matches the ``Iteration N`` log line and ``iterations/NNNNN/``
        # subdir name); no further shifting needed.
        iteration = int(iteration) if iteration is not None else None
        parent_iter_id = int(parent_iter_id) if parent_iter_id is not None else None

        self._ensure_task_md()
        # Regenerate ``history.md`` from the iterations/ tree on disk. GEPA's
        # state.save() runs at the top of each outer-loop turn, so every
        # iteration strictly before this one already has
        # ``iterations/NNNNN/meta.json`` (+ optionally ``plan.md``) written.
        # Deriving history.md every call keeps the view in sync with disk
        # without any cross-thread bookkeeping.
        if iteration is not None:
            self._rebuild_history_md()

        subdir = self._allocate_subdir(iteration)
        parent_dir: Path | None = None
        if parent_iter_id is not None:
            iter_dir = self.run_dir / "iterations" / f"{parent_iter_id:05d}"
            if iter_dir.is_dir():
                parent_dir = iter_dir
        stems = self._materialize(
            subdir,
            candidate,
            reflective_dataset,
            components_to_update,
            parent_iteration_dir=parent_dir,
        )
        # Seed owns iteration id 0; the very first loop proposal runs at id 1.
        first_iteration = iteration == 1
        wrapper = self._wrapper_prompt(
            subdir,
            components_to_update,
            stems,
            first_iteration=first_iteration,
            parent_iter_id=parent_iter_id,
        )

        cmd: list[str] = [
            "claude",
            "--print",
            wrapper,
            "--output-format", "json",
            "--model", self.model,
        ]
        if self.sandbox:
            # File tools + Bash + OS bwrap jail, all scoped to run_dir.
            # Network stays off (no reason for reflection to leave the host).
            # Bash is left on because the bwrap jail already confines shell
            # commands to run_dir, and grep/diff/jq/python make the agent much
            # more effective at analyzing past candidates and reflective
            # datasets than the Claude file tools alone.
            # Caveat: bash bypasses Claude's per-tool allow-list, so a
            # misbehaving call can rm/mv anything in run_dir — write isolation
            # across concurrent proposers is still convention-only.
            cmd.extend(sandbox_args(
                self.run_dir,
                allow_network=False,
                allow_bash=True,
            ))
        else:
            cmd.extend(["--permission-mode", "bypassPermissions"])
        if self.max_thinking_tokens is None and self.effort is not None:
            cmd.extend(["--effort", self.effort])
        if self.max_budget_usd is not None:
            remaining = max(0.01, self.max_budget_usd - self._total_cost)
            cmd.extend(["--max-budget-usd", f"{remaining:.4f}"])

        env = {**os.environ}
        env.pop("CLAUDECODE", None)
        env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")
        if self.max_thinking_tokens is not None:
            env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"
            env["MAX_THINKING_TOKENS"] = str(self.max_thinking_tokens)

        proc = subprocess.run(
            cmd, cwd=str(self.run_dir), env=env, capture_output=True, text=True
        )

        payload: dict[str, Any] = {}
        stdout = (proc.stdout or "").strip()
        parse_error: str | None = None
        if stdout:
            try:
                payload = json.loads(stdout)
            except (json.JSONDecodeError, ValueError) as e:
                parse_error = f"{type(e).__name__}: {e}"
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

        with self._lock:
            self._total_cost += cost
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out

        # Surface claude failures. Without this, a non-zero exit (e.g. API
        # rate limit, sandbox setup failure, auth error) silently reports
        # zero cost/tokens and the proposer falls back to the parent
        # candidate — indistinguishable from a proposer that ran and
        # declined to change anything. Both a per-subdir file and a stderr
        # line go out so the failure shows up in the launcher log.
        is_error_payload = bool(payload.get("is_error"))
        empty_payload = not payload and not stdout
        if proc.returncode != 0 or parse_error or is_error_payload or empty_payload:
            stderr_tail = (proc.stderr or "")[-4000:]
            stdout_tail = (proc.stdout or "")[-4000:]
            reason = (
                f"returncode={proc.returncode}"
                + (f" parse_error={parse_error}" if parse_error else "")
                + (" is_error=true" if is_error_payload else "")
                + (" empty_output=true" if empty_payload else "")
            )
            try:
                (subdir / "claude_failure.json").write_text(
                    json.dumps(
                        {
                            "returncode": proc.returncode,
                            "parse_error": parse_error,
                            "is_error": is_error_payload,
                            "stderr_tail": stderr_tail,
                            "stdout_tail": stdout_tail,
                            "cost": cost,
                            "tokens_in": tokens_in,
                            "tokens_out": tokens_out,
                        },
                        indent=2,
                    )
                )
            except OSError:
                pass
            print(
                f"[ClaudeCodeAgentProposer] claude failure ({reason}) "
                f"subdir={subdir.name}; stderr: {stderr_tail.strip()[:400]}",
                file=sys.stderr,
                flush=True,
            )

        new_texts = self._read_new_texts(subdir, candidate, components_to_update, stems)

        # Copy the agent's plan.md into iterations/<iter_id>/plan.md so the
        # next proposer call — which regenerates history.md from disk —
        # sees it. ``iterations/<iter_id>/`` may not exist yet: GEPA's
        # state.save() writes it at the top of the next outer-loop turn,
        # but we create the dir here so the write is ordering-safe (state
        # save uses ``os.makedirs(..., exist_ok=True)``). Safe to skip when
        # iteration is unknown (legacy path).
        if iteration is not None:
            self._persist_plan_md(subdir, iteration)

        return new_texts

    def __repr__(self) -> str:
        return (
            f"ClaudeCodeAgentProposer(model={self.model!r}, "
            f"run_dir={str(self.run_dir)!r}, cost=${self._total_cost:.2f})"
        )

    # ------------------------------------------------------------------
    # Plan + history bookkeeping
    # ------------------------------------------------------------------

    def _persist_plan_md(self, subdir: Path, iteration: int) -> None:
        """Copy the agent's ``plan.md`` into ``iterations/<iter_id>/plan.md``.

        Best-effort — if the agent didn't write a plan, we silently skip.
        ``iterations/<iter_id>/`` is created if missing; GEPA's state.save
        populates the rest of that directory (meta.json, components/, ...)
        on the next outer-loop save pass without conflict.
        """
        plan_src = subdir / "plan.md"
        if not plan_src.exists():
            return
        # ``iteration`` is already the on-disk iter id (1-indexed).
        dest = self.run_dir / "iterations" / f"{iteration:05d}" / "plan.md"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(plan_src.read_text())
        except OSError as e:
            print(
                f"[ClaudeCodeAgentProposer] failed to persist plan.md for "
                f"iteration {iter_id}: {e}",
                file=sys.stderr,
                flush=True,
            )

    def _rebuild_history_md(self) -> None:
        """Rewrite ``<run_dir>/history.md`` from iterations/*/ on disk.

        Produces one chronological block per completed iteration::

            ### Iteration N — accepted (parent: M)
            score: 0.42 -> 0.51 (subsample)
            plan:
            <verbatim plan body>

        Seed (``iterations/00000/``) is skipped: it has no plan and no
        parent. Merge (multi-parent) iterations just show the first parent
        for now (``# TODO: multi-parent merges``).
        """
        iters_root = self.run_dir / "iterations"
        if not iters_root.exists():
            return
        blocks: list[str] = []
        for d in sorted(iters_root.iterdir()):
            if not d.is_dir():
                continue
            try:
                iter_id = int(d.name)
            except ValueError:
                continue
            if iter_id == 0:
                # Seed: no plan/parent to log.
                continue
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            accepted = meta.get("accepted")
            status = "accepted" if accepted else ("rejected" if accepted is False else "pending")
            parent_ids = meta.get("parent_iteration_ids") or []
            parent_str = (
                f"parent: {parent_ids[0]}" if parent_ids else "parent: ?"
            )
            before = meta.get("subsample_scores_before")
            after = meta.get("subsample_scores_after")
            if isinstance(before, list) and isinstance(after, list) and before and after:
                score_line = (
                    f"score: {sum(before):.4g} -> {sum(after):.4g} (subsample)"
                )
            elif isinstance(after, list) and after:
                score_line = f"score: -> {sum(after):.4g}"
            else:
                score_line = "score: (unknown)"
            plan_path = d / "plan.md"
            plan_body = None
            if plan_path.exists():
                try:
                    plan_body = plan_path.read_text().strip()
                except OSError:
                    plan_body = None
            header = f"### Iteration {iter_id} — {status} ({parent_str})"
            parts = [header, score_line]
            if plan_body:
                parts.append("plan:")
                parts.append(plan_body)
            else:
                parts.append("plan: (none)")
            blocks.append("\n".join(parts))
        history_path = self.run_dir / "history.md"
        try:
            if blocks:
                history_path.write_text("\n\n".join(blocks) + "\n")
            elif history_path.exists():
                # Degenerate case: all iterations disappeared (test setup?).
                history_path.unlink()
        except OSError as e:
            print(
                f"[ClaudeCodeAgentProposer] failed to rebuild history.md: {e}",
                file=sys.stderr,
                flush=True,
            )
