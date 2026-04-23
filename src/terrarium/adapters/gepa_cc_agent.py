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
   can also browse ``candidates/``, ``iterations/``, ``pareto/`` etc. for
   history.
4. Reads one file per updated component back from ``<subdir>/new/``. The file
   body is parsed for a fenced ```` ``` ```` block (same convention
   ``InstructionProposalSignature.output_extractor`` uses). Missing files fall
   back to the original candidate text so GEPA's acceptance criterion just
   rejects the no-op proposal.

Needs ``engine.write_agent_state=True`` on the GEPA side so the run_dir
actually contains readable ``candidates/``, ``iterations/``, ``pareto/``,
``rejected_proposals/``. The terrarium ``GEPAAdapter`` wiring sets that
default when this proposer is selected.

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

    def _allocate_subdir(self) -> Path:
        """Pick a collision-free subdir for this call under ``<run_dir>/<prefix>/``.

        Shape: ``YYYYmmddTHHMMSS-<pid>-<uuid8>``. Timestamp sorts in order of
        issue; pid + uuid disambiguate parallel workers and cross-process
        proposers sharing the same ``run_dir``.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        base = self.run_dir / self.subdir_prefix
        base.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        name = f"{ts}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        subdir = base / name
        subdir.mkdir(parents=True, exist_ok=False)
        return subdir

    def _materialize(
        self,
        subdir: Path,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Write parent candidate + reflective dataset + task files into ``subdir``.

        Returns a map ``{component_name: safe_filename_stem}`` used to locate
        parent/output files on disk.
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

        parent_dir = subdir / "parent"
        parent_dir.mkdir()
        for name, text in candidate.items():
            (parent_dir / f"{name_to_stem[name]}.txt").write_text(text)

        reflection_dir = subdir / "reflection"
        reflection_dir.mkdir()
        for name in components_to_update:
            records = list(reflective_dataset.get(name, []))
            (reflection_dir / f"{name_to_stem[name]}.json").write_text(
                json.dumps(records, indent=2, default=str)
            )

        (subdir / "components_to_update.json").write_text(
            json.dumps(
                [{"name": n, "stem": name_to_stem[n]} for n in components_to_update],
                indent=2,
            )
        )
        (subdir / "_index.json").write_text(json.dumps(name_to_stem, indent=2))

        new_dir = subdir / "new"
        new_dir.mkdir()

        return name_to_stem

    def _wrapper_prompt(self, subdir: Path, components: list[str], stems: dict[str, str]) -> str:
        """Program.md-style proposal brief.

        Structure mirrors ``terrarium.adapters.claude_code.build_program_md``
        (Objective / Background / Candidates / Strategy / Rules) so an agent
        trained by any of the other adapters (claude_code, meta_harness) sees
        a familiar shape. Long task context (objective, background — often
        thousands of chars, incl. the scoring rubric) lives in
        ``<run_dir>/task.md``; we reference it by path so every proposal call
        doesn't re-ship it through the CLI prompt.
        """
        rel = subdir.relative_to(self.run_dir).as_posix()
        task_md_exists = (self.objective is not None) or (self.background is not None)
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
        return f"""\
# GEPA agent proposer

You are proposing improved text for one or more components of a program
optimized by GEPA. Your goal is to raise the program's aggregate score on
future evaluations, based on the reflective feedback from past runs.

Your working directory is the shared GEPA run directory. All inputs and
outputs for **this** proposal call live under:
  `{rel}/`

{task_section}## Inputs (read first)
- `{rel}/parent/<stem>.txt` — current text of each component
- `{rel}/reflection/<stem>.json` — per-example records (inputs, model
  outputs, feedback, scores) for the components you must update
- `{rel}/components_to_update.json` — which components to propose new
  text for, with their filename stems
- `{rel}/_index.json` — full component-name → stem map

## Prior state (optional — absent on iteration 1)
- `candidates/NNNNN/components/*.txt` — past candidate component texts
- `candidates/NNNNN/meta.json` — scores, parents, discovery iteration
- `candidates/NNNNN/val_scores.json` — per-example val scores
- `pareto/` — current Pareto frontier(s)
- `iterations/NNNNN.json` — per-iteration trace
- `iterations/NNNNN/reflective_dataset.json` — past curated feedback
- `rejected_proposals/NNNNN.json` — proposals the engine already rejected
- `run_log.json` — the full program trace

## Strategy
1. Read `task.md` (if present) for the problem statement and scoring rules.
2. Read each `{rel}/parent/<stem>.txt` to see the current component text.
3. Read each `{rel}/reflection/<stem>.json` — focus on examples with low
   scores and the `feedback` / `message` / `logs` fields.
4. If useful, browse prior `candidates/` and `rejected_proposals/` to
   avoid re-trying things the engine has already seen.
5. Write your improved text to `{rel}/new/<stem>.md` for each component.

## Output format
One file per component in `components_to_update.json`:
{files_to_produce}

Wrap the new component text in a single ```…``` fenced block. Anything
outside the block is treated as rationale and discarded by the caller.
Use the same filename stem as the input.

## Tools
You have bash available; network is off. Use it for analysis when helpful
(grep across past candidates, diff two versions, python one-liners over
the reflective datasets). Do not run evaluation scripts: this proposer
only produces text; the engine evaluates and accepts/rejects the proposal.

## Rules
- Write only inside `{rel}/new/`.
- Do not modify other candidates, state files, `task.md`, or another
  proposal's directory.
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
    ) -> dict[str, str]:
        if self.max_budget_usd is not None and self._total_cost >= self.max_budget_usd:
            raise BudgetExhausted(
                f"claude_code_agent proposer spent ${self._total_cost:.2f} "
                f">= cap ${self.max_budget_usd:.2f}"
            )

        if not components_to_update:
            return {}

        self._ensure_task_md()
        subdir = self._allocate_subdir()
        stems = self._materialize(subdir, candidate, reflective_dataset, components_to_update)
        wrapper = self._wrapper_prompt(subdir, components_to_update, stems)

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

        return self._read_new_texts(subdir, candidate, components_to_update, stems)

    def __repr__(self) -> str:
        return (
            f"ClaudeCodeAgentProposer(model={self.model!r}, "
            f"run_dir={str(self.run_dir)!r}, cost=${self._total_cost:.2f})"
        )
