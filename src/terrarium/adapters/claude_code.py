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
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from terrarium.adapter import Result
from terrarium.task import Task

if TYPE_CHECKING:
    from terrarium.eval_server import EvalServer

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
BODY=$(echo "$RESPONSE" | head -n -1)

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
BODY=$(echo "$RESPONSE" | head -n -1)

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
BODY=$(echo "$RESPONSE" | head -n -1)

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
You have **{max_evals}** total evaluation units.
{budget_details}

## Strategy
1. Read the training examples to understand the task.
2. Start with the initial candidate and evaluate it.
3. Analyze failures — read specific examples that scored 0.
4. Improve the candidate and spot-check with `--ids`.
5. Validate periodically to check generalization.
6. Write your best candidate to `best_candidate.txt`.

## Rules
- You cannot modify eval.sh, validate.sh, or the server.
- You cannot see the validation examples.
- Focus on meaningful improvements each iteration.
- When budget is exhausted, scripts return BUDGET_EXHAUSTED.
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
```
Each example costs 1 budget unit. A full train eval costs {train_size} units.
{val_section}"""

_EVAL_SINGLE = """\
## Evaluation
This is a **single-task** optimization.
```bash
./eval.sh candidate.txt
```
Each eval costs 1 budget unit."""

_VAL_SECTION = """
### Validation
There is a hidden validation set ({val_size} examples).
You cannot see individual val examples or their scores.
```bash
./validate.sh candidate.txt
```
Returns only the aggregate val_score. Costs {val_size} budget units."""


def build_program_md(task: Task, max_evals: int) -> str:
    """Build structured program.md from task metadata.

    Mirrors what GEPA receives (``task.objective``, ``task.background``,
    dataset info) but in a format an agent can read and act on.
    """
    # Same two fields the GEPA adapter forwards into reflection prompts.
    optional = ""
    if task.objective:
        optional += f"## Objective\n{task.objective}\n\n"
    if task.background:
        optional += f"## Background\n{task.background}\n\n"

    # Evaluation section
    if task.has_dataset and task.train_set:
        train_size = len(task.train_set)
        val_section = ""
        if task.val_set:
            val_section = _VAL_SECTION.format(val_size=len(task.val_set))
        eval_section = _EVAL_GENERALIZATION.format(
            train_size=train_size, val_section=val_section,
        )
    else:
        eval_section = _EVAL_SINGLE

    # Budget details
    budget_details = ""
    if task.val_set:
        budget_details = (
            f"- Full train eval = {len(task.train_set)} units\n"
            f"- Validation = {len(task.val_set)} units\n"
            f"- Use train evals to iterate cheaply, validate when confident."
        )

    return _PROGRAM_MD.format(
        name=task.name,
        optional_sections=optional,
        candidate_len=len(task.initial_candidate),
        eval_section=eval_section,
        max_evals=max_evals,
        budget_details=budget_details,
    )


def materialize_sandbox(work_dir: Path, task: Task, server_url: str, max_evals: int) -> None:
    """Set up the agent's sandbox workspace.

    Creates:
        work_dir/
            program.md          — structured task instructions
            candidate.txt       — initial candidate
            best_candidate.txt  — agent writes best here
            eval.sh             — train evaluation script
            validate.sh         — val evaluation script (generalization only)
            train/              — materialized training examples (generalization only)
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # program.md — structured instructions
    (work_dir / "program.md").write_text(build_program_md(task, max_evals))

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

    # Materialize train examples
    if task.train_set:
        train_dir = work_dir / "train"
        train_dir.mkdir(exist_ok=True)
        for ex in task.train_set:
            data = {"id": ex.id, "inputs": ex.inputs}
            if ex.expected is not None:
                data["expected"] = ex.expected
            (train_dir / f"{ex.id}.json").write_text(json.dumps(data, indent=2))



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
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        # When set, artifacts (candidate.txt, eval.sh, best_candidate.txt,
        # plus anything Claude writes) persist under this dir. Otherwise a
        # tempdir is used and cleaned up on exit. The terrarium runner
        # injects <hydra_run_dir>/<adapter_name> at run time.
        self.run_dir = run_dir
        # Tempdir whose lifetime spans evolve → process_result, so the
        # adapter's workspace is still readable when process_result runs.
        # Cleaned up by process_result; on an evolve error we let
        # TemporaryDirectory's finalizer reclaim it.
        self._pending_tempdir: tempfile.TemporaryDirectory[str] | None = None

    def evolve(self, task: Task, server: EvalServer) -> Result:
        budget = server.budget

        if self.run_dir:
            work_dir = Path(self.run_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._pending_tempdir = tempfile.TemporaryDirectory(prefix="terrarium_cc_")
            work_dir = Path(self._pending_tempdir.name)

        # Set up the sandbox workspace
        materialize_sandbox(work_dir, task, server.url, budget.max_evals or 0)

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
            "--model", self.model,
            "--permission-mode", "bypassPermissions",
            "--session-id", session_id,
            "--disallowedTools=WebSearch",
        ]
        if budget.max_token_cost is not None:
            cmd.extend(["--max-budget-usd", str(budget.max_token_cost)])
        cmd.append(prompt)

        env = {**os.environ, "TERRARIUM_WORK_DIR": str(work_dir)}

        try:
            subprocess.run(
                cmd,
                cwd=str(work_dir),
                env=env,
                timeout=3600,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            pass

        best_candidate = best_file.read_text() if best_file.exists() else task.initial_candidate

        return Result(
            best_candidate=best_candidate,
            best_score=server.best_score,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
            metadata={
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


def _copy_session_transcript(cwd: Path, session_id: str, dst_dir: Path) -> None:
    """Copy the transcript for ``session_id`` (passed to ``claude --session-id``)
    into ``dst_dir``.

    Claude writes ``~/.claude/projects/<cwd-slug>/<session_id>.jsonl``, where
    ``cwd-slug`` is the absolute subprocess cwd with every ``/`` replaced by
    ``-``. Because we pinned the UUID at launch, there is no ambiguity about
    which file belongs to this run — concurrent processes get distinct UUIDs.
    """
    project_slug = str(cwd.resolve()).replace("/", "-")
    src = Path.home() / ".claude" / "projects" / project_slug / f"{session_id}.jsonl"
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


def create_adapter(**kwargs: Any) -> ClaudeCodeAdapter:
    """Factory for CLI usage."""
    return ClaudeCodeAdapter(**kwargs)
