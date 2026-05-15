"""FiNER: internal researcher notes — NOT surfaced to optimizer. DO NOT
include in any task-facing text.

ACE's FiNER finance benchmark (XBRL US-GAAP tagging), arXiv:2510.04618.
Each example: a financial sentence + a candidate list of US-GAAP tags;
the model must output the correct tag(s). Correctness = ALL comma-separated
tags match (ACE ``_finer_answer_is_correct``: per-sample all-or-nothing).
Splits vendored from ACE eval/finance/data (train 1000 / val 500 /
test 441). Dataset-task, generalization mode; candidate = an evolved
prompt; one eval = one example. Scoring ported verbatim in
tasks/_ace_scoring.py (see its docstring for the one safe deviation).
"""

from __future__ import annotations

from typing import Any

from terrarium.registry import register_task_factory
from terrarium.task import Example, Task
from terrarium.tasks._ace_scoring import finer_answer_is_correct
from terrarium.tasks._finance_common import evaluate_with_solver as _solve
from terrarium.tasks._finance_common import load_finance_dataset

DESCRIPTION = """\
The candidate is a prompt string. For each item the model is given a
financial statement together with a list of allowed category tags, and
must output the correct tag(s). The score is 1.0 only if every required
tag matches, 0.0 otherwise.
"""

INITIAL_CANDIDATE = (
    "Read the financial statement and the list of allowed tags. Choose the "
    "tag(s) that best apply. Respond with only the tag(s), and end your "
    "response with Finish[<tag>] containing the final answer."
)


def evaluate(candidate: str, example: Example) -> tuple[float, dict[str, Any]]:
    return evaluate_with_solver(candidate, example)


def evaluate_with_solver(candidate: str, example: Example, **solver_kwargs: Any) -> tuple[float, dict[str, Any]]:
    return _solve(
        candidate,
        example,
        task_name="finer",
        is_correct=finer_answer_is_correct,
        **solver_kwargs,
    )


def _make_task() -> Task:
    train_set, val_set, test_set = load_finance_dataset("finer")
    return Task(
        name="finer",
        objective="Optimize a prompt that generalizes across XBRL tagging items.",
        background=DESCRIPTION,
        initial_candidate=INITIAL_CANDIDATE,
        eval_fn=evaluate,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        metadata={
            "type": "generalization",
            "candidate_type": "prompt",
            "split_provenance": {
                "source_dataset": "ACE eval/finance FiNER (arXiv:2510.04618)",
                "split_method": "ACE-provided train/val/test jsonl, vendored verbatim",
                "split_sizes": {
                    "train": len(train_set),
                    "val": len(val_set),
                    "test": len(test_set),
                },
            },
        },
    )


register_task_factory("finer", _make_task)
