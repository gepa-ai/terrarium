"""Meta-Harness adapter: iterative candidate proposal via Claude Code.

Implements the Meta-Harness search loop (https://arxiv.org/abs/2603.28052)
against Terrarium's evaluation contract. Meta-Harness splits evolution into
two stages:

    Proposer (Claude subprocess)              Benchmark (this adapter)
    ┌────────────────────────────┐            ┌────────────────────────┐
    │ reads frontier + history,  │            │ reads pending_eval.json│
    │ writes 1+ candidates to    │──── files ─│ scores each candidate  │
    │ agents/, then pending_eval │            │ via server.evaluate    │
    │ .json listing them         │            │ updates state files    │
    └────────────────────────────┘            └────────────────────────┘

The adapter owns the outer loop: it launches the proposer, reads its pending
candidates, evaluates each one through the Terrarium ``EvalServer`` (same
budget counter as every other adapter), writes frontier and summary files
the next proposer iteration will read, and persists session transcripts.

Budget / effort / other top-level configs propagate the same way as the
``claude_code`` adapter:

- ``server.budget.max_evals`` is enforced by the ``EvalServer`` itself.
  When it trips mid-benchmark, we catch ``BudgetExhausted`` and return.
- ``server.budget.max_token_cost`` (adapter LLM spend) is enforced by passing
  ``--max-budget-usd <remaining>`` to each proposer session. We track
  cumulative proposer cost via the CLI's ``result`` stream event and refuse
  to spawn another session once the cap is reached.
- ``effort`` (top-level ``cfg.effort`` or ``adapter.effort``) is passed to
  ``claude --effort`` on every proposer session.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from terrarium.adapter import Result
from terrarium.budget import BudgetExhausted, BudgetTracker
from terrarium.sandbox import sandbox_args
from terrarium.task import Task

if TYPE_CHECKING:
    from terrarium.eval_server import EvalServer


# ── Embedded SKILL.md ───────────────────────────────────────────────
#
# Meta-Harness keeps its proposer instructions in a SKILL.md file loaded by
# claude_wrapper.  We embed a Terrarium-flavored version here and write it to
# ``<work_dir>/.claude/skills/terrarium-meta-harness/SKILL.md`` on every run so
# the proposer always sees an up-to-date copy.  The prose deliberately mirrors
# the reference-example skills (analyze → prototype → implement → pending) so
# the proposer behaves like the paper's runs.

SKILL_MD = """\
---
name: terrarium-meta-harness
description: Run one iteration of candidate evolution for a Terrarium task. Called by the meta-harness adapter.
---

# Meta-Harness (Terrarium Candidate Evolution)

Run ONE iteration of candidate evolution. Do all work in the main session \u2014 do NOT delegate to subagents. Constraints get lost when you delegate, leading to parameter-only changes and skipped prototyping.

**You do NOT run benchmarks.** You analyze results, prototype changes, and write new candidate files. The outer loop (the terrarium meta-harness adapter) handles benchmarking separately.

## CRITICAL CONSTRAINTS

- You MUST implement up to `max_candidates` new candidates every iteration (cap given in the task prompt; aim for 3 unless told otherwise).
- Do NOT write "the frontier is optimal" or "stop iterating", or abort early.
- ALWAYS complete all steps including prototyping.
- Design candidates as a mix of exploitation and exploration.

### Anti-parameter-tuning rules

The most common failure mode is creating candidates that are just parameter variants of existing ones. Check `evolution_summary.jsonl` for what's been tried \u2014 parameter sweeps (constants, thresholds, length caps, retry counts) almost always regress or tie.

**Good candidates change a fundamental mechanism:**

- A new algorithmic approach (e.g. a different solver structure, a different decomposition)
- A new prompt architecture (e.g. organize by failure clusters instead of listing rules sequentially)
- A new control-flow strategy (e.g. multi-pass refinement vs. single-shot, conditional branching on input shape)
- A new representation (e.g. tabular vs. narrative, normalized vs. raw)

**Bad candidates just tune numbers.** If the candidate text differs from a previous one only by changing constants, it's a parameter variant. Rewrite with a genuinely novel mechanism.

**Combining ideas is valid.** Take one mechanism from candidate A and another from candidate B, or draw on published approaches.

If the last 3 iterations explored the same axis (prompt wording, retrieval count, etc.), pick a different axis.

### Anti-overfitting rules

- **No example-specific hints.** Do not hardcode knowledge about specific test inputs you've seen. Candidates must be general-purpose.
- **Never echo example identifiers** in candidate code, prompts, or comments.
- **General patterns are OK.** Rules like "favor short outputs when ambiguous" or "validate the parse before submitting" are fine \u2014 they apply broadly.

## Budget

Token-cost and eval-count budgets are enforced by the adapter, not by you. Don't try to ration evals or refuse to propose because "budget might run out" \u2014 if the cap is reached, the adapter simply stops spawning new proposer sessions. Your only job is to produce strong candidates this iteration.

## WORKFLOW

**Do ALL steps yourself in the main session.**

### Step 0: Post-eval reports (write if missing)

Check the reports directory (path in the task prompt's "Run directories" section). For each past iteration that has results in `evolution_summary.jsonl` but NO report, write one. Each report should be **<=30 lines** covering: what changed, which candidates improved/regressed and why, and a takeaway for future iterations.

### Step 1: Analyze

1. **Read all state files:**
   - `task.md` \u2014 task objective + background + evaluation model
   - `evolution_summary.jsonl` \u2014 what's been tried (one JSON per candidate)
   - `frontier.json` \u2014 current best candidate and best score
   - `agents/baseline.txt` \u2014 the seed candidate
   - top-scoring `agents/iter*.txt` files
   - `state/eval_traces/<candidate_name>/*.json` \u2014 per-eval records from the
     terrarium eval server for every previously-scored candidate. **This is
     the most important source for failure analysis.** Each JSON contains
     the full `info` dict (compile errors, judge messages, per-example
     scores for dataset tasks, logs, status). Read the traces of the
     best-and-worst candidates to understand *why* they scored as they did
     before designing new candidates.

2. Formulate hypotheses \u2014 each must be falsifiable and target a different mechanism.

### Step 2: Prototype \u2014 MANDATORY

**You MUST prototype your mechanism before writing the final candidate.** Do NOT skip this step. Candidates that skip prototyping tend to have bugs or produce no improvement.

For each candidate:

1. Write a sketch in `/tmp/` that exercises the core idea in isolation (run code candidates manually; for prompt candidates, at least re-read and self-critique).
2. Try 2-3 variants and compare before picking the best one.
3. Delete sketches when done.

### Step 3: Implement

For each candidate:

1. Copy a top-performing existing candidate (or `agents/baseline.txt`) as a starting point, then make targeted modifications. Copy-then-edit ensures correct formatting and proven structure.
2. Implement the new mechanism according to your hypothesis.
3. **Self-critique (mandatory):** After writing, re-read the file and check: does this candidate introduce a genuinely NEW mechanism, or is it just a parameter variant? If only constants differ from the base, REWRITE with a truly novel mechanism.

The benchmark auto-discovers files in `agents/` \u2014 you don't need to register candidates anywhere else.

### Step 4: Write pending_eval.json

Write to the path specified in the task prompt (NOT hardcoded \u2014 it may be in a run-specific subdirectory):

```json
{
  "iteration": <N>,
  "candidates": [
    {
      "name": "<snake_case_name>",
      "file": "agents/iter<N>_<name>.txt",
      "hypothesis": "<falsifiable claim>",
      "axis": "exploitation|exploration",
      "base": "<what it builds on>",
      "components": ["tag1", "tag2", "..."]
    }
  ]
}
```

Output: `CANDIDATES: <name1>, <name2>, <name3>`

## Candidate format

A candidate is the **entire text content of a single file** at `agents/iter<N>_<name>.txt`. Whatever you write there \u2014 code, a prompt, JSON, whatever the task expects \u2014 is what the eval server scores. Read `task.md` to learn what shape the task expects.

- Use `Write` (not `Edit`) to create new candidate files.
- `agents/baseline.txt` is the seed; treat it as read-only.
- Don't write outside `agents/` or the pending_eval.json path.

## evolution_summary.jsonl Format

One JSON object per line, one line per evaluated candidate:

```json
{"iteration": 1, "name": "example_candidate", "score": 45.0, "axis": "exploitation", "hypothesis": "...", "delta": +2.1, "outcome": "45.00 (+2.1)", "components": ["tag1", "tag2"]}
```

## Component Analysis

Treat `evolution_summary.jsonl`, `frontier.json`, and any prior `agents/iter*.txt` files as the only shipped history sources.
"""


# ── Sandbox templates ───────────────────────────────────────────────

_TASK_MD = """\
# Task: {name}

{optional_sections}\
## Evaluation model

{eval_section}
"""

_EVAL_SINGLE = """\
This is a **single-task** optimization. The outer loop scores each candidate
once via the eval server; one scored candidate costs 1 unit of the
evaluation budget.
"""

_EVAL_DATASET = """\
This is a **dataset / generalization** task with {train_size} training
examples{val_note}. The outer loop scores each candidate on the full
training split; each example costs 1 unit of the evaluation budget, so one
fully-scored candidate costs {train_size} units. Your candidate must
generalize across *every* train example, not just one.
"""


def _build_task_md(task: Task) -> str:
    optional = ""
    if task.objective:
        optional += f"## Objective\n{task.objective}\n\n"
    if task.background:
        optional += f"## Background\n{task.background}\n\n"

    if task.has_dataset and task.train_set:
        val_note = (
            f" (plus a hidden val set of {len(task.val_set)} examples; not visible to the proposer)"
            if task.val_set else ""
        )
        eval_section = _EVAL_DATASET.format(train_size=len(task.train_set), val_note=val_note)
    else:
        eval_section = _EVAL_SINGLE

    return _TASK_MD.format(
        name=task.name,
        optional_sections=optional,
        eval_section=eval_section,
    )


# ── Sandbox materialization ────────────────────────────────────────


def _materialize_sandbox(work_dir: Path, task: Task, budget: BudgetTracker) -> None:
    """Lay out the proposer's read-only-ish context:

        <work_dir>/
            task.md                              (objective + background + eval model)
            .claude/skills/terrarium-meta-harness/SKILL.md
            agents/baseline.txt
            state/
                frontier.json                   (initial: no scores)
                evolution_summary.jsonl         (empty)
                reports/                        (empty)
                eval_traces/                    (per-candidate JSONs from eval server,
                                                 populated after each candidate scores)
            train/<id>.json                     (dataset tasks only)
    """
    del budget  # task.md is task-only; budget is dynamic and goes in the per-iteration prompt
    work_dir.mkdir(parents=True, exist_ok=True)

    (work_dir / "task.md").write_text(_build_task_md(task))

    skill_dir = work_dir / ".claude" / "skills" / "terrarium-meta-harness"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD)

    agents_dir = work_dir / "agents"
    agents_dir.mkdir(exist_ok=True)
    (agents_dir / "baseline.txt").write_text(task.initial_candidate)

    state_dir = work_dir / "state"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "reports").mkdir(exist_ok=True)
    (state_dir / "eval_traces").mkdir(exist_ok=True)

    frontier = state_dir / "frontier.json"
    if not frontier.exists():
        frontier.write_text(json.dumps({
            "best_name": "baseline",
            "best_score": None,
            "best_candidate_file": "agents/baseline.txt",
        }, indent=2))

    summary = state_dir / "evolution_summary.jsonl"
    if not summary.exists():
        summary.write_text("")

    # Dataset mode: materialize a *read-only preview* of the train examples.
    # We only include id + inputs (plus expected if the task already exposes
    # it) \u2014 matching what the claude_code adapter does.
    if task.train_set:
        train_dir = work_dir / "train"
        train_dir.mkdir(exist_ok=True)
        for ex in task.train_set:
            data: dict[str, Any] = {"id": ex.id, "inputs": ex.inputs}
            if ex.expected is not None:
                data["expected"] = ex.expected
            (train_dir / f"{ex.id}.json").write_text(json.dumps(data, indent=2))


# ── Proposer subprocess (claude --print stream-json) ────────────────


def _run_proposer(
    *,
    work_dir: Path,
    iteration: int,
    model: str,
    effort: str | None,
    max_candidates: int,
    max_budget_usd: float | None,
    pending_path: Path,
    log_dir: Path,
    max_thinking_tokens: int | None = None,
    sandbox: bool = True,
) -> tuple[int, float, str]:
    """Launch one proposer session. Returns (exit_code, cost_usd, session_id).

    Uses ``--output-format json`` so the subprocess emits a single, terminal
    JSON object on stdout with the fields we care about:

        {
          "type": "result",
          "session_id": "...",
          "total_cost_usd": 0.1234,
          "duration_ms": 12345,
          "usage": {"input_tokens": ..., "output_tokens": ...},
          "result": "<final assistant text>"
        }

    This is strictly more stable than parsing a stream-event schema.

    Session transcripts (the full per-event history) are written by the CLI
    to ``~/.claude/projects/<slug>/<session_id>.jsonl`` regardless of
    ``--output-format`` and are copied by ``process_result`` using the same
    pinned-session-id trick the ``claude_code`` adapter uses.
    """
    prompt = _render_task_prompt(work_dir, iteration, max_candidates, pending_path)

    session_id = str(uuid.uuid4())
    cmd: list[str] = [
        "claude",
        "--print",
        prompt,
        "--output-format", "json",
        "--model", model,
        "--session-id", session_id,
    ]
    # Sandbox whitelists file tools + Bash inside work_dir (+ /tmp for the
    # SKILL's prototype-sketch step). Network stays off: the proposer only
    # reads state files and writes new candidates — it never calls the eval
    # server. Under the sandbox we stay in default permission mode so
    # unlisted tool calls auto-deny in --print; when sandbox is off, fall
    # back to bypassPermissions so --print doesn't deadlock on prompts.
    if sandbox:
        cmd.extend(sandbox_args(work_dir, extra_dirs=["/tmp"], allow_network=False))
    else:
        cmd.extend(["--permission-mode", "bypassPermissions"])
    if max_thinking_tokens is None and effort is not None:
        cmd.extend(["--effort", effort])
    if max_budget_usd is not None:
        cmd.extend(["--max-budget-usd", f"{max_budget_usd:.4f}"])

    env = {**os.environ, "TERRARIUM_WORK_DIR": str(work_dir)}
    # Strip CLAUDECODE so a claude-running-claude situation doesn't confuse
    # the CLI (same workaround the reference proposer uses).
    env.pop("CLAUDECODE", None)
    # Raise the per-response output-token cap. Default 32000 bites on tasks
    # where the proposer emits long candidate bodies (frontier_cs \u2014 big
    # C++ files) and aborts the session with "Claude's response exceeded
    # the 32000 output token maximum". 64k keeps the proposer alive while
    # still bounded. Honor any user override already set in the environment.
    env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")
    if max_thinking_tokens is not None:
        env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"
        env["MAX_THINKING_TOKENS"] = str(max_thinking_tokens)

    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"iter{iteration}_stdout.json"
    stderr_path = log_dir / f"iter{iteration}_stderr.txt"

    exit_code = 0
    cost_usd = 0.0
    stderr_tail = ""
    result_payload: dict[str, Any] = {}
    proc = subprocess.run(
        cmd,
        cwd=str(work_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    exit_code = proc.returncode
    stdout_path.write_text(proc.stdout or "")
    stderr_path.write_text(proc.stderr or "")
    stderr_tail = (proc.stderr or "")[-2000:]
    cost_usd, result_payload = _parse_proposer_result(proc.stdout or "", work_dir, session_id)

    (log_dir / f"iter{iteration}_meta.json").write_text(json.dumps({
        "iteration": iteration,
        "session_id": session_id,
        "exit_code": exit_code,
        "cost_usd": cost_usd,
        "cmd": cmd,
        "stderr_tail": stderr_tail,
        # Echo a few top-level result fields so a reviewer doesn't have to
        # jump into stdout.json.
        "result_summary": {
            "duration_ms": result_payload.get("duration_ms"),
            "num_turns": result_payload.get("num_turns"),
            "usage": result_payload.get("usage"),
            "is_error": result_payload.get("is_error"),
            "subtype": result_payload.get("subtype"),
        },
    }, indent=2))

    return exit_code, cost_usd, session_id


def _parse_proposer_result(stdout: str, work_dir: Path, session_id: str) -> tuple[float, dict[str, Any]]:
    """Extract (cost_usd, result_payload) from ``claude --output-format json``.

    With ``--output-format json`` the CLI prints exactly one JSON object to
    stdout once the session ends:

        {
          "type": "result",
          "subtype": "success",
          "is_error": false,
          "session_id": "...",
          "total_cost_usd": 0.1234,
          "duration_ms": 12345,
          "num_turns": 7,
          "usage": {"input_tokens": ..., "output_tokens": ...},
          "result": "<final assistant text>"
        }

    A missing / malformed payload indicates the CLI crashed before the end of
    the session. We report cost=0 in that case; the caller already surfaces
    the subprocess exit code + stderr tail in the iteration meta file, so the
    user has enough to diagnose.
    """
    del work_dir, session_id  # reserved for a future transcript-based fallback
    payload: dict[str, Any] = {}
    stdout = (stdout or "").strip()
    if stdout:
        try:
            payload = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            payload = {}

    try:
        cost = float(payload.get("total_cost_usd", 0.0) or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    return cost, payload


def _render_task_prompt(
    work_dir: Path,
    iteration: int,
    max_candidates: int,
    pending_path: Path,
) -> str:
    """Per-iteration user prompt. Kept short \u2014 the bulk of guidance lives
    in `task.md` (task spec) and the SKILL.md (process). Budget is *not*
    surfaced here: enforcement is the adapter's job; the SKILL tells the
    proposer they don't need to ration evals."""
    state = work_dir / "state"
    return (
        f"Run iteration {iteration} of the Terrarium meta-harness evolution loop. "
        f"Produce up to {max_candidates} candidate(s).\n\n"
        f"## Run directories (absolute paths)\n"
        f"- task.md: `{work_dir / 'task.md'}`\n"
        f"- state/frontier.json: `{state / 'frontier.json'}`\n"
        f"- state/evolution_summary.jsonl: `{state / 'evolution_summary.jsonl'}`\n"
        f"- state/eval_traces/<candidate_name>/: `{state / 'eval_traces'}`  (per-eval JSONs from the eval server \u2014 read for failure detail)\n"
        f"- state/reports/: `{state / 'reports'}`\n"
        f"- agents/: `{work_dir / 'agents'}`\n"
        f"- Write pending_eval.json to: `{pending_path}`\n\n"
        f"Follow the terrarium-meta-harness skill in `.claude/skills/`."
    )


# ── Candidate bookkeeping ───────────────────────────────────────────


def _read_pending(pending_path: Path) -> list[dict[str, Any]]:
    if not pending_path.exists():
        return []
    try:
        data = json.loads(pending_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    candidates = data.get("candidates", [])
    if not isinstance(candidates, list):
        return []
    return [c for c in candidates if isinstance(c, dict) and "file" in c and "name" in c]


def _load_candidate(work_dir: Path, relpath: str) -> str | None:
    path = (work_dir / relpath).resolve()
    # Defensive: refuse to read anything outside work_dir.
    try:
        path.relative_to(work_dir.resolve())
    except ValueError:
        return None
    if not path.exists() or path.is_dir():
        return None
    try:
        return path.read_text()
    except OSError:
        return None


def _score_candidate(server: "EvalServer", task: Task, candidate: str) -> tuple[float, dict[str, Any]]:
    """Score a candidate through the Terrarium eval server.

    For dataset tasks, evaluate on the full training split; for single tasks,
    one eval is enough. Both paths go through the same budget counter, so
    ``BudgetExhausted`` propagates to the caller identically.
    """
    if task.has_dataset and task.train_set:
        return server.evaluate_examples(candidate, split="train")
    return server.evaluate(candidate)


def _capture_eval_traces(
    server: "EvalServer",
    candidate_name: str,
    eval_idx_range: tuple[int, int],
    work_dir: Path,
) -> int:
    """Mirror the eval server's per-eval JSON records for a candidate into
    `state/eval_traces/<candidate_name>/` so the next proposer iteration can
    open them for failure analysis.

    The eval server already writes one JSON per eval at
    ``server.output_dir / "evals" / <i>.json`` (with the full ``info`` dict
    \u2014 compile errors, status, logs, per-example detail, etc.). After
    each candidate scores, we copy the JSONs whose 0-indexed positions fall
    in ``eval_idx_range = (before, after)`` (where ``before`` and ``after``
    are ``len(server.eval_log)`` snapshots taken around the call). This is
    the terrarium equivalent of the reference meta-harness's per-prediction
    `logs/<dataset>/<memory>/<model>/log.jsonl` traces.

    Returns the count of trace files copied (0 when ``server.output_dir`` is
    unset or the eval JSONs aren't on disk \u2014 e.g. in unit tests).
    """
    output_dir = getattr(server, "output_dir", None)
    if output_dir is None:
        return 0
    src_dir = Path(output_dir) / "evals"
    if not src_dir.exists():
        return 0

    before, after = eval_idx_range
    if after <= before:
        return 0

    dst_dir = work_dir / "state" / "eval_traces" / candidate_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for idx in range(before, after):
        src = src_dir / f"{idx}.json"
        if not src.exists():
            continue
        try:
            shutil.copy2(src, dst_dir / f"{idx}.json")
            n += 1
        except OSError:
            # Trace capture is best-effort; never let it break the run.
            pass
    return n


def _update_frontier(frontier_path: Path, name: str, file: str, score: float) -> bool:
    frontier: dict[str, Any] = {}
    if frontier_path.exists():
        try:
            frontier = json.loads(frontier_path.read_text())
        except (json.JSONDecodeError, OSError):
            frontier = {}
    prev = frontier.get("best_score")
    improved = prev is None or score > float(prev)
    if improved:
        frontier["best_name"] = name
        frontier["best_candidate_file"] = file
        frontier["best_score"] = score
        frontier_path.write_text(json.dumps(frontier, indent=2))
    return improved


def _append_summary(
    summary_path: Path,
    *,
    iteration: int,
    name: str,
    file: str,
    score: float | None,
    best_score: float | None,
    hypothesis: str,
    axis: str,
    components: list[str],
    outcome: str,
    budget_used: int,
    propose_time: float | None = None,
    bench_time: float | None = None,
) -> None:
    row: dict[str, Any] = {
        "iteration": iteration,
        "name": name,
        "file": file,
        "score": score,
        "delta": (score - best_score) if (score is not None and best_score is not None) else None,
        "hypothesis": hypothesis,
        "axis": axis,
        "components": components,
        "outcome": outcome,
        "budget_used": budget_used,
    }
    if propose_time is not None:
        row["propose_s"] = round(propose_time, 1)
    if bench_time is not None:
        row["bench_s"] = round(bench_time, 1)
    with open(summary_path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ── Minimal ANSI coloring (same convention as the other adapters' stdout) ──

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t: str) -> str: return _c("1", t)
def _dim(t: str) -> str: return _c("2", t)
def _green(t: str) -> str: return _c("32", t)
def _red(t: str) -> str: return _c("31", t)
def _yellow(t: str) -> str: return _c("33", t)
def _cyan(t: str) -> str: return _c("36", t)


def _log(*parts: str) -> None:
    # Flush-every-line so subprocess-launched runs show progress live.
    print(" ".join(parts), flush=True)


# ── The adapter ─────────────────────────────────────────────────────


class MetaHarnessAdapter:
    """Meta-Harness evolution against a Terrarium task.

    Args:
        model: Proposer model (``claude --model``). Defaults to ``opus`` to
            match the paper runs. Override per experiment.
        effort: ``claude --effort`` flag (``low|medium|high|max``). The
            Terrarium runner injects the top-level ``effort`` here when the
            user hasn't set it explicitly.
        run_dir: Workspace directory. When left null, the runner injects
            ``<hydra_run_dir>/meta_harness/``.
        max_iterations: Hard cap on proposer sessions. ``None`` means "keep
            going until the eval or token-cost budget is exhausted".
        max_candidates_per_iter: Upper bound on candidates per iteration.
            The proposer may produce fewer.
    Budget enforcement:
        - ``server.budget.max_evals`` is enforced server-side (each
          ``server.evaluate`` increments the counter and raises
          ``BudgetExhausted``).
        - ``server.budget.max_token_cost`` is enforced by the outer loop
          here: after every proposer session we read its ``total_cost_usd``
          and refuse to spawn another session once the cumulative spend
          reaches the cap. Each session also gets ``--max-budget-usd
          <remaining>`` as a runaway brake (claude treats it as a soft cap
          \u2014 it checks between turns and tends to overshoot, so this is
          only a safety floor, not precise rationing).
    """

    def __init__(
        self,
        model: str = "opus",
        effort: str | None = None,
        run_dir: str | None = None,
        max_iterations: int | None = None,
        max_candidates_per_iter: int = 3,
        stop_at_score: float | None = None,
        max_thinking_tokens: int | None = None,
        sandbox: bool | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.run_dir = run_dir
        self.max_iterations = max_iterations
        self.max_candidates_per_iter = max(1, int(max_candidates_per_iter))
        self.stop_at_score = stop_at_score
        self.max_thinking_tokens = max_thinking_tokens
        # Wrap every proposer subprocess in ``terrarium.sandbox``. None =
        # take the top-level ``sandbox:`` default from the runner.
        self.sandbox = sandbox
        self._pending_tempdir: tempfile.TemporaryDirectory[str] | None = None

    # ---- main entry ----

    def evolve(self, task: Task, server: "EvalServer") -> Result:
        budget = server.budget

        if self.run_dir:
            work_dir = Path(self.run_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._pending_tempdir = tempfile.TemporaryDirectory(prefix="terrarium_mh_")
            work_dir = Path(self._pending_tempdir.name)

        _materialize_sandbox(work_dir, task, budget)

        state_dir = work_dir / "state"
        frontier_path = state_dir / "frontier.json"
        summary_path = state_dir / "evolution_summary.jsonl"
        sessions_dir = work_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        session_ids: list[str] = []
        total_proposer_cost = 0.0

        # Unlimited-iteration mode: use an effective cap derived from budget
        # so we never spin forever when the token budget is exhausted but
        # eval budget is not (or vice versa). Meta-Harness is meant to be a
        # bounded search, so a large-but-finite default is safe.
        if self.max_iterations is not None:
            cap = self.max_iterations
        else:
            cap = 50  # generous fallback; budget will stop us first

        run_start = time.time()
        iteration = 0
        stop_reason = "completed"
        while iteration < cap:
            iteration += 1

            if budget.exhausted:
                stop_reason = "eval_budget_exhausted"
                break

            # Outer-loop budget enforcement: stop when cumulative proposer
            # spend has reached the cap. The next session's --max-budget-usd
            # is just the remaining budget as a (soft) runaway brake \u2014
            # claude overshoots it, so it's a safety floor, not rationing.
            per_session_cap: float | None = None
            if budget.max_token_cost is not None:
                remaining = budget.max_token_cost - total_proposer_cost
                if remaining <= 0:
                    stop_reason = "token_budget_exhausted"
                    break
                per_session_cap = remaining

            _log(
                f"\n{_bold(f'[meta-harness] iteration {iteration}/{cap}')} "
                f"{_dim(f'(evals used={budget.used}, cost=${total_proposer_cost:.3f})')}"
            )

            pending_path = state_dir / f"pending_eval_iter{iteration}.json"
            if pending_path.exists():
                pending_path.unlink()

            # ── Propose ─────────────────────────────────────────
            propose_start = time.time()
            exit_code, cost, session_id = _run_proposer(
                work_dir=work_dir,
                iteration=iteration,
                model=self.model,
                effort=self.effort,
                max_candidates=self.max_candidates_per_iter,
                max_budget_usd=per_session_cap,
                pending_path=pending_path,
                log_dir=sessions_dir,
                max_thinking_tokens=self.max_thinking_tokens,
                sandbox=bool(self.sandbox),
            )
            propose_time = time.time() - propose_start
            total_proposer_cost += cost
            session_ids.append(session_id)

            _log(
                f"  {_cyan('proposer')} exit={exit_code} cost=${cost:.3f} "
                f"time={_elapsed(propose_time)} session={session_id[:8]}"
            )

            if exit_code != 0:
                _append_summary(
                    summary_path,
                    iteration=iteration,
                    name="(proposer_failed)",
                    file="",
                    score=None,
                    best_score=self._best_score(frontier_path),
                    hypothesis="",
                    axis="",
                    components=[],
                    outcome=f"proposer_exit_{exit_code}",
                    budget_used=budget.used,
                    propose_time=propose_time,
                )
                stop_reason = "proposer_failed"

            candidates = _read_pending(pending_path)
            if not candidates:
                _log(f"  {_yellow('no candidates')} in pending_eval.json; stopping")
                _append_summary(
                    summary_path,
                    iteration=iteration,
                    name="(no_candidates)",
                    file="",
                    score=None,
                    best_score=self._best_score(frontier_path),
                    hypothesis="",
                    axis="",
                    components=[],
                    outcome="no_pending_eval",
                    budget_used=budget.used,
                    propose_time=propose_time,
                )
                stop_reason = "no_candidates"
                break

            _log(f"  {_cyan('benchmarking')} {len(candidates)} candidate(s)...")

            # ── Benchmark ───────────────────────────────────────
            bench_start = time.time()
            iter_improved = False
            for ci, c in enumerate(candidates, 1):
                name = c["name"]
                file = c["file"]
                cand_text = _load_candidate(work_dir, file)
                if cand_text is None:
                    _log(f"    [{ci}/{len(candidates)}] {_red('missing')} {name} ({file})")
                    _append_summary(
                        summary_path,
                        iteration=iteration,
                        name=name,
                        file=file,
                        score=None,
                        best_score=self._best_score(frontier_path),
                        hypothesis=c.get("hypothesis", ""),
                        axis=c.get("axis", ""),
                        components=c.get("components", []),
                        outcome="file_missing",
                        budget_used=budget.used,
                    )
                    continue

                eval_idx_before = len(server.eval_log)
                try:
                    score, info = _score_candidate(server, task, cand_text)
                except BudgetExhausted:
                    # Capture any traces produced before budget tripped
                    _capture_eval_traces(
                        server, name, (eval_idx_before, len(server.eval_log)), work_dir,
                    )
                    _log(f"    [{ci}/{len(candidates)}] {_red('budget exhausted')} during {name}")
                    _append_summary(
                        summary_path,
                        iteration=iteration,
                        name=name,
                        file=file,
                        score=None,
                        best_score=self._best_score(frontier_path),
                        hypothesis=c.get("hypothesis", ""),
                        axis=c.get("axis", ""),
                        components=c.get("components", []),
                        outcome="budget_exhausted",
                        budget_used=budget.used,
                    )
                    stop_reason = "eval_budget_exhausted"
                    break
                except Exception as e:  # noqa: BLE001 - record eval errors, continue
                    _log(f"    [{ci}/{len(candidates)}] {_red('eval error')} {name}: {e}")
                    _append_summary(
                        summary_path,
                        iteration=iteration,
                        name=name,
                        file=file,
                        score=None,
                        best_score=self._best_score(frontier_path),
                        hypothesis=c.get("hypothesis", ""),
                        axis=c.get("axis", ""),
                        components=c.get("components", []),
                        outcome=f"eval_error: {type(e).__name__}",
                        budget_used=budget.used,
                    )
                    continue

                # Mirror this candidate's per-eval JSONs (compile errors,
                # logs, per-example scores...) into state/eval_traces/<name>/
                # so the next proposer iteration can read them directly.
                trace_count = _capture_eval_traces(
                    server, name, (eval_idx_before, len(server.eval_log)), work_dir,
                )

                prev_best = self._best_score(frontier_path)
                improved = _update_frontier(frontier_path, name=name, file=file, score=score)
                if improved:
                    iter_improved = True

                delta_str = "(new_best)" if improved else (
                    f"({score - prev_best:+.4f})" if prev_best is not None else ""
                )
                trace_str = f" traces={trace_count}" if trace_count else ""
                _log(
                    f"    [{ci}/{len(candidates)}] {_bold(name)}: "
                    f"{_colorize_score(score)} {delta_str} "
                    f"{_dim(f'(budget_used={budget.used}{trace_str})')}"
                )

                _append_summary(
                    summary_path,
                    iteration=iteration,
                    name=name,
                    file=file,
                    score=score,
                    best_score=prev_best,
                    hypothesis=c.get("hypothesis", ""),
                    axis=c.get("axis", ""),
                    components=c.get("components", []),
                    outcome=(
                        f"{score:.4f}" if prev_best is None
                        else f"{score:.4f} ({score - prev_best:+.4f})"
                    ),
                    budget_used=budget.used,
                    propose_time=propose_time if ci == 1 else None,
                )

                # Log a progress-checkpoint for dataset tasks so the server's
                # progress_log.jsonl tracks the meta-harness trajectory just
                # like the GEPA valset_evaluated path does.
                if task.has_dataset:
                    try:
                        server.log_progress(
                            score,
                            candidate=cand_text,
                            reflection_cost=total_proposer_cost,
                        )
                    except Exception:
                        pass

                if self.stop_at_score is not None and score >= self.stop_at_score:
                    _log(f"    {_green('perfect score reached')}: {score:.4f} >= {self.stop_at_score}")
                    stop_reason = "perfect_score"
                    break

                # Tracker integration (wandb/mlflow): piggy-back on the
                # server's tracker so meta-harness iterations appear alongside
                # other adapters' per-iteration metrics.
                tracker = getattr(server, "tracker", None)
                if tracker is not None:
                    try:
                        tracker.log_metrics(
                            {
                                "meta_harness/iteration": iteration,
                                "meta_harness/candidate_score": score,
                                "meta_harness/best_score": (
                                    self._best_score(frontier_path) or 0.0
                                ),
                                "meta_harness/proposer_cost": total_proposer_cost,
                            },
                            step=budget.used,
                        )
                    except Exception:
                        pass

            bench_time = time.time() - bench_start

            _log(
                f"  {_dim(f'iter {iteration} wall={_elapsed(time.time() - run_start)} '
                       f'propose={_elapsed(propose_time)} bench={_elapsed(bench_time)}')}"
                + (f" {_green('IMPROVED')}" if iter_improved else "")
            )

            if stop_reason in ("eval_budget_exhausted", "perfect_score", "proposer_failed"):
                break

        _log(
            f"\n{_bold('[meta-harness] done')} "
            f"reason={stop_reason} iters={iteration} "
            f"best={self._best_score(frontier_path)} "
            f"evals={budget.used} proposer_cost=${total_proposer_cost:.3f}"
        )

        # Prefer the frontier's (name, score, file) tuple because for dataset
        # tasks it holds the *train-average* score of the winning candidate,
        # whereas ``server.best_score`` is the max over individual-example
        # evals (misleading for generalization tasks). Fall back to the
        # server's bookkeeping when the proposer never produced anything.
        best_candidate = server.best_candidate
        best_score = self._best_score(frontier_path)
        frontier = self._read_frontier(frontier_path)
        best_file = frontier.get("best_candidate_file") if frontier else None
        if best_file:
            loaded = _load_candidate(work_dir, best_file)
            if loaded is not None:
                best_candidate = loaded
        if best_score is None:
            best_score = server.best_score
        return Result(
            best_candidate=best_candidate,
            best_score=best_score if best_score is not None else server.best_score,
            total_evals=budget.used,
            eval_log=server.eval_log,
            metadata={
                "adapter_cost": total_proposer_cost,
                "meta_harness": {
                    "iterations_run": iteration,
                    "stop_reason": stop_reason,
                    "proposer_cost": total_proposer_cost,
                    "session_ids": session_ids,
                    "work_dir": str(work_dir),
                    "frontier": self._read_frontier(frontier_path),
                },
                "work_dir": str(work_dir),
                "session_ids": session_ids,
            },
        )

    # ---- artifact persistence ----

    def process_result(self, result: Result, output_dir: Path) -> None:
        """Copy session transcripts and mirror the work dir into ``output_dir``.

        Mirrors the claude_code adapter's behaviour: session transcripts end up
        under ``output_dir/sessions/<session_id>.jsonl`` and the entire
        workspace is copied under ``output_dir/work/`` when the work dir lives
        outside ``output_dir`` (i.e. when we used a tempdir).
        """
        work_dir = Path(result.metadata.get("work_dir", ""))
        session_ids: list[str] = result.metadata.get("session_ids", []) or []

        transcripts_dir = output_dir / "sessions"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        for sid in session_ids:
            _copy_session_transcript(work_dir, sid, transcripts_dir)

        if work_dir.exists() and not _is_under(work_dir, output_dir):
            shutil.copytree(work_dir, output_dir / "work", dirs_exist_ok=True)

        if self._pending_tempdir is not None:
            self._pending_tempdir.cleanup()
            self._pending_tempdir = None

    # ---- helpers ----

    def _best_score(self, frontier_path: Path) -> float | None:
        frontier = self._read_frontier(frontier_path)
        val = frontier.get("best_score") if frontier else None
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _read_frontier(self, frontier_path: Path) -> dict[str, Any]:
        if not frontier_path.exists():
            return {}
        try:
            return json.loads(frontier_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}


# ── Formatting helpers ─────────────────────────────────────────────


def _elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _colorize_score(score: float) -> str:
    s = f"{score:.4f}"
    # Heuristic: unknown scale, so we only colorize strong signals.
    if score > 0:
        return _green(s)
    if score < 0:
        return _red(s)
    return s


# ── Session transcript discovery (same mechanism as claude_code) ────

_SLUG_RE = re.compile(r"[^A-Za-z0-9-]")


def _claude_project_slug(cwd: Path) -> str:
    return _SLUG_RE.sub("-", str(cwd.resolve()))


def _copy_session_transcript(cwd: Path, session_id: str, dst_dir: Path) -> None:
    src = Path.home() / ".claude" / "projects" / _claude_project_slug(cwd) / f"{session_id}.jsonl"
    if not src.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst_dir / src.name)
    except OSError:
        pass


def _is_under(child: Path, parent: Path) -> bool:
    try:
        return child.resolve().is_relative_to(parent.resolve())
    except OSError:
        return False


def create_adapter(**kwargs: Any) -> MetaHarnessAdapter:
    """Factory for CLI usage (``adapter=custom adapter.path=...``)."""
    return MetaHarnessAdapter(**kwargs)
