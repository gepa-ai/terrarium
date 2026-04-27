"""Claude Code adapter: runs Claude Code as a black-box evolution system.

Architecture:
    terrarium process                   Claude Code subprocess
    ┌─────────────────────┐             ┌──────────────────────┐
    │ EvalServer (HTTP)   │◄── POST ────│ calls eval.sh        │
    │ - enforces budget   │── score ──► │ - reads score        │
    │ - tracks best       │             │ - mutates candidate  │
    └─────────────────────┘             └──────────────────────┘

The runner creates the EvalServer. This adapter just uses its HTTP endpoint.
Budget is enforced server-side. Claude Code cannot modify it.
When budget is exhausted, the server returns HTTP 429.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from terrarium.adapter import Result
from terrarium.budget import BudgetTracker
from terrarium.sandbox import sandbox_args
from terrarium.task import Task

if TYPE_CHECKING:
    from terrarium.eval_server import EvalServer


def _abs_str(p: Path | str) -> str:
    """Format an absolute path as Claude's ``//<path>`` permission-rule form."""
    return f"/{Path(p).resolve()}"

# Single-task eval script: one POST per call, returns one score.
# Usage: ./eval.sh <candidate_file>
EVAL_SCRIPT_SINGLE = """\
#!/usr/bin/env bash
# Usage: ./eval.sh <candidate_file>
# Evaluates the candidate once. Returns {{score, info, budget}}.
# Exit code 1 if budget exhausted.
set -euo pipefail

CANDIDATE_FILE="$1"
SERVER_URL="{server_url}"

CANDIDATE=$(cat "$CANDIDATE_FILE")
BODY=$(jq -n --arg c "$CANDIDATE" '{{candidate: $c}}')

RESPONSE=$(curl -s -w "\\n%{{http_code}}" -X POST "$SERVER_URL/evaluate" \\
    -H "Content-Type: application/json" \\
    -d "$BODY")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "$BODY"

if [ "$HTTP_CODE" = "429" ]; then
    echo "BUDGET_EXHAUSTED" >&2
    exit 1
fi
"""

# Dataset eval script: full-split eval or specific example IDs.
# Usage:
#   ./eval.sh <candidate_file>                     → eval on train split (default)
#   ./eval.sh <candidate_file> test                → eval on test split
#   ./eval.sh <candidate_file> --ids id1,id2,id3   → eval on specific examples
EVAL_SCRIPT_DATASET = """\
#!/usr/bin/env bash
# Usage: ./eval.sh <candidate_file> [split]
#        ./eval.sh <candidate_file> --ids id1,id2,id3
# Evaluates a candidate on the terrarium eval server.
# Without --ids: evaluates across ALL examples in the split (default: train).
# With --ids: evaluates only the specified examples (comma-separated).
# Examples run in parallel server-side.
# Exit code 1 if budget exhausted.
set -euo pipefail

CANDIDATE_FILE="$1"
shift
SERVER_URL="{server_url}"

CANDIDATE=$(cat "$CANDIDATE_FILE")

# Parse args: either --ids or a split name
IDS=""
SPLIT="train"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ids) IDS="$2"; shift 2 ;;
        *) SPLIT="$1"; shift ;;
    esac
done

if [ -n "$IDS" ]; then
    # Convert comma-separated IDs to JSON array
    IDS_JSON=$(echo "$IDS" | jq -R 'split(",")')
    BODY=$(jq -n --arg c "$CANDIDATE" --argjson ids "$IDS_JSON" '{{candidate: $c, example_ids: $ids}}')
else
    BODY=$(jq -n --arg c "$CANDIDATE" --arg s "$SPLIT" '{{candidate: $c, split: $s}}')
fi

RESPONSE=$(curl -s -w "\\n%{{http_code}}" -X POST "$SERVER_URL/evaluate_examples" \\
    -H "Content-Type: application/json" \\
    -d "$BODY")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "$BODY"

if [ "$HTTP_CODE" = "429" ]; then
    echo "BUDGET_EXHAUSTED" >&2
    exit 1
fi
"""

VALIDATE_SCRIPT = """\
#!/usr/bin/env bash
# Usage: ./validate.sh <candidate_file>
# Evaluates the candidate on the held-out validation set.
# Returns aggregate val_score only (individual scores hidden).
# Exit code 1 if budget exhausted.
set -euo pipefail

CANDIDATE_FILE="$1"
SERVER_URL="{server_url}"

CANDIDATE=$(cat "$CANDIDATE_FILE")
BODY=$(jq -n --arg c "$CANDIDATE" '{{candidate: $c}}')

RESPONSE=$(curl -s -w "\\n%{{http_code}}" -X POST "$SERVER_URL/validate" \\
    -H "Content-Type: application/json" \\
    -d "$BODY")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "$BODY"

if [ "$HTTP_CODE" = "429" ]; then
    echo "BUDGET_EXHAUSTED" >&2
    exit 1
fi
"""

# ── program.md templates ────────────────────────────────────────────────

_PROGRAM_MD = """\
# Task: {name}

{optional_sections}\
## Candidate
Your task is to iteratively improve the candidate in `candidate.txt`.
The initial candidate ({candidate_len} chars) is provided as a starting point.

{eval_section}

## Budget
{budget_section}

## Strategy
{strategy_section}

## Rules
{rules_section}
"""

_EVAL_GENERALIZATION = """\
## Evaluation
This is a **generalization** task with {train_size} training examples.
Training examples are in `train/` as individual JSON files.

### Train evaluation
```bash
# Evaluate on all training examples
./eval.sh candidate.txt

# Evaluate on specific examples
./eval.sh candidate.txt --ids example_0,example_1,example_2
```{cost_line}{val_section}"""

_EVAL_SINGLE = """\
## Evaluation
This is a **single-task** optimization.
```bash
./eval.sh candidate.txt
```{cost_line}"""

_VAL_SECTION = """
### Validation
There is a hidden validation set ({val_size} examples).
You cannot see individual val examples or their scores.
```bash
./validate.sh candidate.txt
```
Returns only the aggregate val_score.{val_cost}"""


def _budget_section(budget: BudgetTracker) -> str:
    lines: list[str] = []
    if budget.max_evals is not None:
        lines.append(f"- **Evaluation cap:** {budget.max_evals} calls (each eval = 1 unit).")
    if budget.max_token_cost is not None:
        lines.append(
            f"- **LLM token budget:** ${budget.max_token_cost:.2f} "
            "(enforced by the Claude CLI; you'll stop automatically when exhausted)."
        )
    if not lines:
        lines.append("- No hard budget limit — iterate freely.")
    return "\n".join(lines)


def _strategy_section(task: Task) -> str:
    if task.has_dataset and task.train_set:
        val_step = "5. Validate periodically with `./validate.sh candidate.txt`.\n" if task.val_set else ""
        final_n = 6 if task.val_set else 5
        return (
            "1. Read the training examples in `train/` to understand the task.\n"
            "   Each `train/<id>.json` includes a `seed_preview` field showing\n"
            "   how the seed candidate behaves on that example (per-example\n"
            "   score + domain summary in `Input` + seed's `Output` trace).\n"
            "   Use these previews to find failure patterns to attack.\n"
            "2. Run `./eval.sh candidate.txt` for a baseline.\n"
            "3. Analyze failures — inspect examples that scored 0.\n"
            "4. Improve the candidate; spot-check with `--ids`.\n"
            f"{val_step}{final_n}. Write your best candidate to `best_candidate.txt`."
        )
    return (
        "1. Read the candidate (`candidate.txt`) and the problem description above.\n"
        "2. Run `./eval.sh candidate.txt` for a baseline score and any judge feedback.\n"
        "3. Revise the candidate based on what you learned.\n"
        "4. Iterate; write your best solution to `best_candidate.txt` as you improve."
    )


def _perfect_score_section(perfect_score: float | None) -> str:
    if perfect_score is None:
        return ""
    return (
        f"\n## Perfect Score\n"
        f"The maximum achievable score is **{perfect_score}**. "
        f"Once you reach this score, stop iterating — further improvements are impossible.\n"
    )


def _rules_section(task: Task, budget: BudgetTracker) -> str:
    scripts = "eval.sh, validate.sh" if task.val_set else "eval.sh"
    val_rule = "\n- You cannot see the validation examples." if task.val_set else ""
    exhaust_rule = (
        "\n- When the budget is exhausted, scripts return BUDGET_EXHAUSTED."
        if budget.max_evals is not None else ""
    )
    return (
        f"- You cannot modify {scripts} or the server.{val_rule}\n"
        f"- Focus on meaningful improvements each iteration.{exhaust_rule}"
    )


def build_program_md(task: Task, budget: BudgetTracker, *, perfect_score: float | None = None) -> str:
    """Build structured program.md from task metadata.

    Mirrors what GEPA receives (``task.objective``, ``task.background``,
    dataset info) but in a format an agent can read and act on.
    """
    optional = ""
    if task.objective:
        optional += f"## Objective\n{task.objective}\n\n"
    if task.background:
        optional += f"## Background\n{task.background}\n\n"

    has_eval_budget = budget.max_evals is not None

    if task.has_dataset and task.train_set:
        train_size = len(task.train_set)
        cost_line = (
            f"\nEach example costs 1 budget unit. A full train eval costs {train_size} units."
            if has_eval_budget else ""
        )
        if task.val_set:
            val_size = len(task.val_set)
            val_cost = f" Costs {val_size} budget units." if has_eval_budget else ""
            # Prepend a newline so the validation block is separated from whatever
            # came before (cost line in eval-budget mode, closing fence otherwise).
            val_section = "\n" + _VAL_SECTION.format(val_size=val_size, val_cost=val_cost)
        else:
            val_section = ""
        eval_section = _EVAL_GENERALIZATION.format(
            train_size=train_size, cost_line=cost_line, val_section=val_section,
        )
    else:
        cost_line = "\nEach eval costs 1 budget unit." if has_eval_budget else ""
        eval_section = _EVAL_SINGLE.format(cost_line=cost_line)

    perfect_score_md = _perfect_score_section(perfect_score)

    return _PROGRAM_MD.format(
        name=task.name,
        optional_sections=optional,
        candidate_len=len(task.initial_candidate),
        eval_section=eval_section,
        budget_section=_budget_section(budget) + perfect_score_md,
        strategy_section=_strategy_section(task),
        rules_section=_rules_section(task, budget),
    )


def materialize_sandbox(
    work_dir: Path,
    task: Task,
    server_url: str,
    budget: BudgetTracker,
    *,
    perfect_score: float | None = None,
    train_preview: bool = True,
    preview_max_examples: int | None = None,
) -> None:
    """Set up the agent's sandbox workspace.

    Creates:
        work_dir/
            program.md          — structured task instructions
            candidate.txt       — initial candidate
            best_candidate.txt  — agent writes best here
            eval.sh             — train evaluation script
            validate.sh         — val evaluation script (generalization only)
            train/              — materialized training examples (generalization only)

    Args:
        train_preview: If True (default), evaluate the seed candidate on each
            training example at materialize-time and inline the resulting
            ``info`` dict (sans heavy raw fields) into ``train/<id>.json``.
            This gives the agent the same per-example side-info that GEPA's
            ``reflective_dataset`` provides — e.g. for cant_be_late, the
            ``spot_availability`` summary string. Without it, the agent only
            sees a ``trace_file`` path it can't read inside the sandbox,
            which artificially handicaps optimization on tasks whose inputs
            point to external data files.
        preview_max_examples: Cap how many examples get a preview (use the
            first N). Useful when ``train_set`` is huge and a full pass would
            dominate startup. None = preview every example.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # program.md — structured instructions
    (work_dir / "program.md").write_text(build_program_md(task, budget, perfect_score=perfect_score))

    # Candidate files
    (work_dir / "candidate.txt").write_text(task.initial_candidate)
    (work_dir / "best_candidate.txt").write_text(task.initial_candidate)

    # eval.sh — minimal one-shot script for single-task; full split/--ids
    # script for dataset tasks.
    eval_template = EVAL_SCRIPT_DATASET if task.has_dataset else EVAL_SCRIPT_SINGLE
    eval_script = work_dir / "eval.sh"
    eval_script.write_text(eval_template.format(server_url=server_url))
    eval_script.chmod(0o755)

    # validate.sh (generalization mode only)
    if task.val_set:
        validate_script = work_dir / "validate.sh"
        validate_script.write_text(VALIDATE_SCRIPT.format(server_url=server_url))
        validate_script.chmod(0o755)

    # Materialize train examples (with optional seed-eval preview)
    if task.train_set:
        train_dir = work_dir / "train"
        train_dir.mkdir(exist_ok=True)
        previews = _build_train_previews(task, preview_max_examples) if train_preview else {}
        for ex in task.train_set:
            data: dict[str, Any] = {"id": ex.id, "inputs": ex.inputs}
            if ex.expected is not None:
                data["expected"] = ex.expected
            if ex.id in previews:
                data["seed_preview"] = previews[ex.id]
            (train_dir / f"{ex.id}.json").write_text(json.dumps(data, indent=2, default=str))


def _build_train_previews(task: Task, max_examples: int | None) -> dict[str, dict[str, Any]]:
    """Run the seed candidate on each train example to capture side-info.

    Mirrors what GEPA's ``reflective_dataset`` provides: per-example score,
    selected ``Input`` fields (which include problem-domain summaries like
    ``spot_availability``), and the seed's ``Output`` (timeline / cost
    breakdown). Heavy debug fields are dropped to keep ``train/<id>.json``
    small.

    Failures are silent — a missing ``seed_preview`` just means that
    example didn't get one, which is no worse than the pre-preview behavior.
    """
    from concurrent.futures import ThreadPoolExecutor

    examples = list(task.train_set or ())
    if max_examples is not None:
        examples = examples[:max_examples]
    if not examples:
        return {}

    seed = task.initial_candidate
    eval_fn = task.eval_fn

    def _one(ex):
        try:
            score, info = eval_fn(seed, ex)
        except Exception:
            return ex.id, None
        # Drop heavy / non-portable fields; keep what GEPA's reflective_dataset
        # would surface to the proposer.
        info = info or {}
        preview: dict[str, Any] = {"score": score}
        for k in ("Input", "Output", "Error", "scores"):
            if k in info:
                preview[k] = info[k]
        return ex.id, preview

    out: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for eid, preview in pool.map(_one, examples):
            if preview is not None:
                out[eid] = preview
    return out



class ClaudeCodeAdapter:
    """Adapter that runs Claude Code as a black-box evolution subprocess.

    Uses the same EvalServer that the runner creates — just its HTTP endpoint.

    ``evolve`` runs the subprocess and stashes ``session_id`` and ``work_dir``
    in ``result.metadata``; ``process_result`` reads them back and copies the
    session transcript (and the work dir, if it lives outside ``output_dir``)
    into the run's output directory.
    """

    def __init__(
        self,
        model: str = "sonnet",
        max_turns: int | None = None,
        run_dir: str | None = None,
        effort: str | None = None,
        stop_at_score: float | None = None,
        max_thinking_tokens: int | None = None,
        sandbox: bool | None = None,
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        self.stop_at_score = stop_at_score
        # When set, artifacts (candidate.txt, eval.sh, best_candidate.txt,
        # plus anything Claude writes) persist under this dir. Otherwise a
        # tempdir is used and cleaned up on exit. The terrarium runner
        # injects <hydra_run_dir>/<adapter_name> at run time.
        self.run_dir = run_dir
        # ``--effort low|medium|high|max`` passed to ``claude --print``.
        # Controls extended-thinking budget. ``None`` = CLI default.
        self.effort = effort
        self.max_thinking_tokens = max_thinking_tokens
        # Wrap the claude subprocess in ``terrarium.sandbox`` (Bash in
        # bubblewrap/Seatbelt + file-tool deny rules). None = take the
        # top-level ``sandbox:`` default from the runner.
        self.sandbox = sandbox
        # Tempdir whose lifetime spans evolve → process_result, so the
        # adapter's workspace is still readable when process_result runs.
        # Cleaned up by process_result; on an evolve error we let
        # TemporaryDirectory's finalizer reclaim it.
        self._pending_tempdir: tempfile.TemporaryDirectory[str] | None = None

    def evolve(self, task: Task, server: EvalServer) -> Result:
        budget = server.budget

        # Pick work_dir.
        #
        # Subtle Seatbelt bug on macOS: when ``filesystem.allowRead`` contains
        # any path under ``~/`` (the user's home), the matching ``denyRead:
        # ["~/"]`` rule becomes a no-op for the *entire* home directory — so
        # Bash and the Read tool can both reach ``~/.claude/projects/.../
        # memory/``, ``~/.ssh/``, the project source tree, and anything else
        # under home. Empirically verified: same sandbox config, work_dir under
        # ``/var/folders`` denies these reads correctly; work_dir under
        # ``/Users/...`` leaks the entire home dir.
        #
        # When ``sandbox=True`` we therefore force a tempdir work_dir under
        # the system TMPDIR (which lives under ``/private/var/folders/`` on
        # macOS, outside ``~/``). ``process_result`` later copies any
        # artifacts back to ``self.run_dir`` so the sandbox isolation doesn't
        # cost the user their persisted artifacts.
        if self.sandbox:
            self._pending_tempdir = tempfile.TemporaryDirectory(prefix="terrarium_cc_")
            work_dir = Path(self._pending_tempdir.name)
        elif self.run_dir:
            work_dir = Path(self.run_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._pending_tempdir = tempfile.TemporaryDirectory(prefix="terrarium_cc_")
            work_dir = Path(self._pending_tempdir.name)

        # Set up the sandbox workspace
        materialize_sandbox(work_dir, task, server.url, budget, perfect_score=self.stop_at_score)

        candidate_file = work_dir / "candidate.txt"
        best_file = work_dir / "best_candidate.txt"
        eval_script = work_dir / "eval.sh"
        validate_script = work_dir / "validate.sh"

        # Build the prompt — point at program.md for full context
        prompt = (
            f"Read program.md for full task instructions. "
            f"The candidate to improve is in {candidate_file}. "
            f"Write your best candidate to {best_file}. "
            f"Use {eval_script} to evaluate."
        )
        if task.val_set:
            prompt += f" Use {validate_script} to check validation score."

        # Launch Claude Code. Under the default permission mode, ``--print``
        # auto-denies Read/Bash tool calls (no human to approve), so Claude
        # can never actually run ``eval.sh`` and just burns budget retrying.
        # Use ``bypassPermissions``, and disallow WebSearch so Claude stays
        # focused on local evaluation instead of browsing the web.
        # ``--session-id`` pins the UUID of the transcript file Claude
        # writes (~/.claude/projects/<cwd-slug>/<session_id>.jsonl) so we
        # can locate it deterministically — no race, no directory diff.
        # ``--disallowedTools`` is variadic; pass with ``=`` so it doesn't
        # swallow the trailing positional prompt.
        session_id = str(uuid.uuid4())
        cmd = [
            "claude",
            "--print",
            "--output-format", "json",
            "--model", self.model,
            "--session-id", session_id,
        ]
        # Sandbox whitelists file tools + Bash inside work_dir; network stays
        # on because eval.sh / validate.sh curl the local eval server. Under
        # the sandbox we stay in default permission mode so anything not
        # explicitly allowed auto-denies in --print. When sandbox is off,
        # fall back to bypassPermissions so --print doesn't deadlock on
        # permission prompts.
        if self.sandbox:
            # Claude Code's macOS sandbox blocks /dev/tcp and external binaries
            # (cat, curl, jq), so eval.sh / validate.sh can't run inside it.
            # Fix per https://code.claude.com/docs/en/sandboxing — list those
            # scripts in ``excludedCommands`` so they run unsandboxed (with
            # full network + system-tool access). To keep the escape hatch
            # safe, simultaneously:
            #   - deny the project source dir at the OS layer so other
            #     sandboxed Bash calls can't ``cat`` it (anti-cheat),
            #   - deny Read/Edit/Write of eval.sh + validate.sh through the
            #     permission allow-list so claude can't rewrite the
            #     unsandboxed scripts to do something else.
            repo_src = Path(__file__).resolve().parents[3] / "src"
            eval_sh = work_dir / "eval.sh"
            validate_sh = work_dir / "validate.sh"
            cmd.extend(sandbox_args(
                work_dir,
                excluded_commands=[
                    "./eval.sh", "./eval.sh *",
                    "bash ./eval.sh", "bash ./eval.sh *",
                    "./validate.sh", "./validate.sh *",
                    "bash ./validate.sh", "bash ./validate.sh *",
                ],
                deny_paths=[repo_src],
                deny_tool_patterns=[
                    f"Edit({_abs_str(eval_sh)})",
                    f"Write({_abs_str(eval_sh)})",
                    f"Edit({_abs_str(validate_sh)})",
                    f"Write({_abs_str(validate_sh)})",
                ],
            ))
        else:
            cmd.extend(["--permission-mode", "bypassPermissions"])
        if self.max_thinking_tokens is None and self.effort is not None:
            cmd.extend(["--effort", self.effort])
        if budget.max_token_cost is not None:
            cmd.extend(["--max-budget-usd", str(budget.max_token_cost)])
        cmd.append(prompt)

        env = {**os.environ, "TERRARIUM_WORK_DIR": str(work_dir)}
        if self.max_thinking_tokens is not None:
            env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"
            env["MAX_THINKING_TOKENS"] = str(self.max_thinking_tokens)

        adapter_cost = 0.0
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        adapter_cost = _extract_claude_cost(proc.stdout)

        best_candidate = best_file.read_text() if best_file.exists() else task.initial_candidate

        return Result(
            best_candidate=best_candidate,
            best_score=server.best_score,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
            metadata={
                "adapter_cost": adapter_cost,
                "session_id": session_id,
                "work_dir": str(work_dir),
            },
        )

    def process_result(self, result: Result, output_dir: Path) -> None:
        """Copy the session transcript and (when work dir lives outside
        ``output_dir``) mirror the work dir into ``output_dir``, then release
        the workspace tempdir.
        """
        work_dir = Path(result.metadata["work_dir"])
        session_id = result.metadata["session_id"]
        _copy_session_transcript(work_dir, session_id, output_dir / "sessions")
        if not _is_under(work_dir, output_dir):
            shutil.copytree(work_dir, output_dir / "work", dirs_exist_ok=True)
        if self._pending_tempdir is not None:
            self._pending_tempdir.cleanup()
            self._pending_tempdir = None


_SLUG_RE = re.compile(r"[^A-Za-z0-9-]")


def _claude_project_slug(cwd: Path) -> str:
    """Compute the slug Claude Code uses for its per-project session dir.

    Claude writes ``~/.claude/projects/<slug>/<session_id>.jsonl`` where
    ``<slug>`` is the absolute cwd with every non-alphanumeric character
    (``/``, ``_``, ``.``, etc.) replaced by ``-``. A naive ``replace("/","-")``
    misses underscores — e.g. ``cc_run_1`` → ``cc_run_1`` (wrong) instead of
    ``cc-run-1`` (right) — which is why transcripts silently weren't being
    captured.
    """
    return _SLUG_RE.sub("-", str(cwd.resolve()))


def _copy_session_transcript(cwd: Path, session_id: str, dst_dir: Path) -> None:
    """Copy the transcript for ``session_id`` (passed to ``claude --session-id``)
    into ``dst_dir``. Because we pinned the UUID at launch, there is no
    ambiguity about which file belongs to this run — concurrent processes get
    distinct UUIDs.
    """
    src = Path.home() / ".claude" / "projects" / _claude_project_slug(cwd) / f"{session_id}.jsonl"
    if not src.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst_dir / src.name)
    except OSError:
        # Never let transcript capture break an otherwise-successful run.
        pass


def _is_under(child: Path, parent: Path) -> bool:
    """True if ``child`` is the same as or nested inside ``parent`` (after
    resolving symlinks)."""
    try:
        return child.resolve().is_relative_to(parent.resolve())
    except OSError:
        return False


def _extract_claude_cost(stdout: str) -> float:
    """Extract ``total_cost_usd`` from ``claude --output-format json`` stdout."""
    stdout = (stdout or "").strip()
    if not stdout:
        return 0.0
    try:
        payload = json.loads(stdout)
        return float(payload.get("total_cost_usd", 0.0) or 0.0)
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0


def create_adapter(**kwargs: Any) -> ClaudeCodeAdapter:
    """Factory for CLI usage."""
    return ClaudeCodeAdapter(**kwargs)
