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

import contextlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import json

from terrarium.adapter import Result
from terrarium.task import Task

if TYPE_CHECKING:
    from terrarium.eval_server import EvalServer

# Unified eval script: supports full-split eval, specific example IDs, or single-task.
# Usage:
#   ./eval.sh <candidate_file>                     → eval on train split (default)
#   ./eval.sh <candidate_file> test                → eval on test split
#   ./eval.sh <candidate_file> --ids id1,id2,id3   → eval on specific examples
EVAL_SCRIPT = """\
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

# System prompt given to Claude Code describing the task and eval protocol.
CLAUDE_CODE_SYSTEM = """\
You are an AI research system tasked with evolving and improving a candidate solution.

## Task
{task_description}

## Initial Candidate
The initial candidate is in: {candidate_file}

## How to Evaluate

`{eval_script} <candidate_file> [split | --ids id1,id2,id3]`

One script, flexible usage:

- **Full split** (default): `{eval_script} candidate.txt` or `{eval_script} candidate.txt train`
  Evaluates across ALL examples in the split. Examples run in parallel server-side.
  Costs N budget units (one per example).

- **Specific examples**: `{eval_script} candidate.txt --ids example_1,example_5,example_12`
  Evaluates only the listed examples. Costs 1 budget unit per example listed.
  Useful for targeted debugging after seeing which examples score low.

- **Single-task** (no dataset): `{eval_script} candidate.txt`
  Just evaluates the candidate. Costs 1 budget unit.

Response fields:
- `average_score`: Mean score across evaluated examples.
- `scores`: Per-example scores (dict of example_id → score).
- `num_evaluated` / `num_total`: How many examples were evaluated vs total in split.
- `partial`: True if budget ran out mid-evaluation.
- `errors`: Per-example errors, if any.
- `budget`: Remaining eval budget.

{dataset_info}

## Budget
{budget_info}
If budget runs out mid-evaluation, you get partial results.

## Strategy Tips
- Start with a full eval to see overall performance and identify weak examples.
- Use `--ids` to cheaply iterate on specific failing examples.
- Once you've improved on the weak examples, do another full eval to confirm.
- For single-task problems, each eval call costs exactly 1 budget unit.

## Goal
Iteratively improve the candidate to maximize the score. When you're done
(or budget is exhausted), write your best candidate to: {best_file}

## Rules
- Each example evaluation counts as one budget unit.
- You cannot modify the eval script or the server.
- Focus on making meaningful improvements each iteration.
"""


def build_program_md(task: Task, max_evals: int) -> str:
    """Build structured program.md from task metadata.

    Mirrors what GEPA receives (objective, background, dataset info)
    but in a format an agent can read and act on.
    """
    sections = []

    # Header
    sections.append(f"# Task: {task.name}\n")

    # Objective — same as what GEPA's reflection prompt gets
    objective = task.metadata.get("objective", "")
    if objective:
        sections.append(f"## Objective\n{objective}\n")

    # Background — same as what GEPA's reflection prompt gets
    background = task.metadata.get("background", "")
    if background:
        sections.append(f"## Background\n{background}\n")

    # Description
    if task.description:
        sections.append(f"## Description\n{task.description}\n")

    # Candidate info
    sections.append("## Candidate")
    sections.append("Your task is to iteratively improve the candidate in `candidate.txt`.")
    sections.append(f"The initial candidate ({len(task.initial_candidate)} chars) is provided as a starting point.\n")

    # Evaluation protocol
    sections.append("## Evaluation")
    if task.has_dataset and task.train_set:
        sections.append(f"This is a **generalization** task with {len(task.train_set)} training examples.")
        sections.append("Training examples are in `train/` as individual JSON files.\n")
        sections.append("### Train evaluation")
        sections.append("```bash")
        sections.append("# Evaluate on all training examples")
        sections.append("./eval.sh candidate.txt")
        sections.append("")
        sections.append("# Evaluate on specific examples")
        sections.append("./eval.sh candidate.txt --ids example_0,example_1,example_2")
        sections.append("```")
        sections.append(f"Each example costs 1 budget unit. A full train eval costs {len(task.train_set)} units.\n")

        if task.val_set:
            sections.append("### Validation")
            sections.append(f"There is a hidden validation set ({len(task.val_set)} examples).")
            sections.append("You cannot see individual val examples or their scores.")
            sections.append("```bash")
            sections.append("./validate.sh candidate.txt")
            sections.append("```")
            sections.append(f"Returns only the aggregate val_score. Costs {len(task.val_set)} budget units.\n")
    else:
        sections.append("This is a **single-task** optimization.")
        sections.append("```bash")
        sections.append("./eval.sh candidate.txt")
        sections.append("```")
        sections.append("Each eval costs 1 budget unit.\n")

    # Budget
    sections.append("## Budget")
    sections.append(f"You have **{max_evals}** total evaluation units.")
    if task.val_set:
        sections.append(f"- Full train eval = {len(task.train_set)} units")
        sections.append(f"- Validation = {len(task.val_set)} units")
        sections.append("- Use train evals to iterate cheaply, validate when confident.\n")

    # Strategy
    sections.append("## Strategy")
    sections.append("1. Read the training examples to understand the task.")
    sections.append("2. Start with the initial candidate and evaluate it.")
    sections.append("3. Analyze failures — read specific examples that scored 0.")
    sections.append("4. Improve the candidate and spot-check with `--ids`.")
    sections.append("5. Validate periodically to check generalization.")
    sections.append("6. Write your best candidate to `best_candidate.txt`.\n")

    # Rules
    sections.append("## Rules")
    sections.append("- You cannot modify eval.sh, validate.sh, or the server.")
    sections.append("- You cannot see the validation examples.")
    sections.append("- Focus on meaningful improvements each iteration.")
    sections.append("- When budget is exhausted, scripts return BUDGET_EXHAUSTED.\n")

    return "\n".join(sections)


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

    # eval.sh
    eval_script = work_dir / "eval.sh"
    eval_script.write_text(EVAL_SCRIPT.format(server_url=server_url))
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

    def evolve(self, task: Task, server: EvalServer) -> Result:
        budget = server.budget

        if self.run_dir:
            work_ctx: Any = contextlib.nullcontext(self.run_dir)
            Path(self.run_dir).mkdir(parents=True, exist_ok=True)
        else:
            work_ctx = tempfile.TemporaryDirectory(prefix="terrarium_cc_")
        with work_ctx as work_dir:
            work = Path(work_dir)

            # Set up the sandbox workspace
            materialize_sandbox(work, task, server.url, budget.max_evals or 0)

            candidate_file = work / "candidate.txt"
            best_file = work / "best_candidate.txt"
            eval_script = work / "eval.sh"

            # Build the prompt — point at program.md for full context
            prompt = (
                f"Read program.md for full task instructions. "
                f"The candidate to improve is in {candidate_file}. "
                f"Write your best candidate to {best_file}. "
                f"Use {eval_script} to evaluate."
            
            )
            if task.val_set:
                prompt += f" Use {work / 'validate.sh'} to check validation score."

            # Launch Claude Code
            cmd = ["claude", "--print", "--model", self.model]
            if budget.max_token_cost is not None:
                cmd.extend(["--max-budget-usd", str(budget.max_token_cost)])
            cmd.append(prompt)

            env = {**os.environ, "TERRARIUM_WORK_DIR": str(work)}

            try:
                subprocess.run(
                    cmd,
                    cwd=str(work),
                    env=env,
                    timeout=3600,
                    capture_output=True,
                    text=True,
                )
            except subprocess.TimeoutExpired:
                pass

            # Read the best candidate
            best_candidate = best_file.read_text() if best_file.exists() else task.initial_candidate

            return Result(
                best_candidate=best_candidate,
                best_score=server.best_score,
                total_evals=server.budget.used,
                eval_log=server.eval_log,
            )


def create_adapter(**kwargs: Any) -> ClaudeCodeAdapter:
    """Factory for CLI usage."""
    return ClaudeCodeAdapter(**kwargs)
