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
from terrarium.sandbox import DENY_WEB_TOOLS, bwrap_prefix, claude_settings_args
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

    Layout: ``program.md``, ``candidate.txt``, ``best_candidate.txt``,
    ``eval.sh``, ``validate.sh`` (val_set only), ``train/<id>.json`` (dataset).

    Args:
        train_preview: Evaluate the seed candidate on each train example at
            materialize-time and inline per-example side-info into
            ``train/<id>.json`` — gives the agent the same context GEPA's
            ``reflective_dataset`` provides (e.g. cant_be_late's
            ``spot_availability``) so it isn't blind to inputs whose data
            lives in external trace files.
        preview_max_examples: Cap how many examples get a preview. ``None``
            = preview every example.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    (work_dir / "program.md").write_text(build_program_md(task, budget, perfect_score=perfect_score))
    (work_dir / "candidate.txt").write_text(task.initial_candidate)
    (work_dir / "best_candidate.txt").write_text(task.initial_candidate)

    eval_template = EVAL_SCRIPT_DATASET if task.has_dataset else EVAL_SCRIPT_SINGLE
    eval_script = work_dir / "eval.sh"
    eval_script.write_text(eval_template.format(server_url=server_url))
    eval_script.chmod(0o755)

    if task.val_set:
        validate_script = work_dir / "validate.sh"
        validate_script.write_text(VALIDATE_SCRIPT.format(server_url=server_url))
        validate_script.chmod(0o755)

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
    """Eval the seed on each train example; return per-id side-info dicts.
    Heavy debug fields are stripped. Failures are silent (no preview emitted)."""
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
        ralph: bool = False,
        ralph_max_iterations: int = 50,
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
        # Ralph loop: when claude exits before the LLM/eval budget is out,
        # resume the session with ``--resume <session_id>`` and keep
        # iterating. Off by default to preserve single-shot behavior.
        self.ralph = ralph
        self.ralph_max_iterations = ralph_max_iterations
        # Tempdir whose lifetime spans evolve → process_result, so the
        # adapter's workspace is still readable when process_result runs.
        # Cleaned up by process_result; on an evolve error we let
        # TemporaryDirectory's finalizer reclaim it.
        self._pending_tempdir: tempfile.TemporaryDirectory[str] | None = None

    def evolve(self, task: Task, server: EvalServer) -> Result:
        budget = server.budget

        # When sandbox=True, force work_dir under TMPDIR. macOS Seatbelt bug:
        # ``denyRead: ["~/"]`` no-ops if ``allowRead`` contains any path under
        # ``~/``. A TMPDIR-rooted work_dir keeps allowRead outside home, so
        # the deny rule actually blocks ~/.claude/projects, ~/.ssh, the
        # project source tree, etc. ``process_result`` copies artifacts back
        # to self.run_dir so the user doesn't lose them.
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

        env = {**os.environ, "TERRARIUM_WORK_DIR": str(work_dir)}
        if self.max_thinking_tokens is not None:
            env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"
            env["MAX_THINKING_TOKENS"] = str(self.max_thinking_tokens)

        adapter_cost = 0.0
        ralph_iterations = 1
        proc = self._run_claude(
            work_dir=work_dir,
            session_id=session_id,
            prompt=prompt,
            budget=budget,
            adapter_cost=adapter_cost,
            resume=False,
            env=env,
        )
        adapter_cost += _extract_claude_cost(proc.stdout)

        if self.ralph:
            # Resume the same session until the budget is out or claude
            # signals it's done by erroring. Each iteration's
            # --max-budget-usd is the *remaining* LLM budget so the
            # final iteration self-caps inside the CLI.
            continue_prompt = (
                "Continue iterating on the candidate. Re-read program.md if needed. "
                "Run ./eval.sh and ./validate.sh as appropriate. "
                "Keep refining best_candidate.txt until you exhaust the budget "
                "or genuinely cannot find another improvement."
            )
            for _ in range(self.ralph_max_iterations - 1):
                if not self._has_budget_headroom(server, adapter_cost):
                    break
                if proc.returncode != 0:
                    break  # claude errored — don't retry blindly
                proc = self._run_claude(
                    work_dir=work_dir,
                    session_id=session_id,
                    prompt=continue_prompt,
                    budget=budget,
                    adapter_cost=adapter_cost,
                    resume=True,
                    env=env,
                )
                iter_cost = _extract_claude_cost(proc.stdout)
                adapter_cost += iter_cost
                ralph_iterations += 1
                # Defensive: if claude returns success but spent ~$0, it's
                # likely declining to continue. Stop to avoid a busy loop.
                if proc.returncode == 0 and iter_cost < 0.001:
                    break

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
                "ralph_iterations": ralph_iterations,
            },
        )

    def _run_claude(
        self,
        *,
        work_dir: Path,
        session_id: str,
        prompt: str,
        budget: BudgetTracker,
        adapter_cost: float,
        resume: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        """Invoke ``claude --print`` once. ``resume=True`` continues the pinned
        session via ``--resume <session_id>`` instead of starting a new one.
        ``--max-budget-usd`` is set to the *remaining* LLM budget so successive
        iterations don't stack the cap.
        """
        # bwrap (when sandboxed) scopes writes to work_dir; network is
        # shared so eval.sh / validate.sh can curl the local eval server
        # (and claude can reach api.anthropic.com). WebFetch/WebSearch
        # denied at the tool layer; bypassPermissions gives full file/Bash
        # access inside the jail.
        cmd: list[str] = bwrap_prefix(work_dir) if self.sandbox else []
        cmd += [
            "claude",
            "--print",
            "--output-format", "json",
            "--model", self.model,
        ]
        if resume:
            cmd.extend(["--resume", session_id])
        else:
            cmd.extend(["--session-id", session_id])
        cmd.extend([
            "--permission-mode", "bypassPermissions",
            DENY_WEB_TOOLS,
        ])
        if self.sandbox:
            cmd.extend(claude_settings_args(work_dir))  # macOS Seatbelt fallback
        if self.max_thinking_tokens is None and self.effort is not None:
            cmd.extend(["--effort", self.effort])
        if budget.max_token_cost is not None:
            remaining = max(0.0, budget.max_token_cost - adapter_cost)
            cmd.extend(["--max-budget-usd", f"{remaining:.6f}"])
        cmd.append(prompt)

        return subprocess.run(
            cmd,
            cwd=str(work_dir),
            env=env,
            capture_output=True,
            text=True,
        )

    def _has_budget_headroom(self, server: EvalServer, adapter_cost: float) -> bool:
        """Whether starting another Ralph iteration is worthwhile. Stops if the
        eval-count budget is exhausted, or if the LLM budget has < $0.05 left
        (no point spawning a CLI for pennies).
        """
        budget = server.budget
        if budget.exhausted:
            return False
        if budget.max_token_cost is not None:
            if adapter_cost >= budget.max_token_cost - 0.05:
                return False
        return True

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
