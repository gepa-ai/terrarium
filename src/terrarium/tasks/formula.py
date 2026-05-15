"""Formula: internal researcher notes — NOT surfaced to optimizer. DO NOT
include in any task-facing text.

ACE's Formula finance benchmark (financial formula numeric QA),
arXiv:2510.04618. Each example: a named financial formula + a word
problem with numbers; the model must output the numeric answer.
Correctness = float equality after stripping thousands commas (ACE
``_formula_answer_is_correct``). Splits vendored from ACE
eval/finance/data (train 500 / val 300 / test 200). Dataset-task,
generalization mode; candidate = an evolved prompt; one eval = one
example. Scoring ported verbatim in tasks/_ace_scoring.py.
"""

from __future__ import annotations

from typing import Any

from terrarium.registry import register_task_factory
from terrarium.task import Example, Task
from terrarium.tasks._ace_scoring import formula_answer_is_correct
from terrarium.tasks._finance_common import evaluate_with_solver as _solve
from terrarium.tasks._finance_common import load_finance_dataset

DESCRIPTION = """\
The candidate is a prompt string. For each item the model is given a
named formula and a word problem with numeric values, and must output the
numeric answer. The score is 1.0 if the numeric answer matches the
expected value, 0.0 otherwise.
"""

INITIAL_CANDIDATE = (
    "Apply the given formula to the numbers in the question. Compute "
    "carefully and end your response with Finish[<number>] containing the "
    "final numeric answer rounded to two decimal places."
)


def evaluate(candidate: str, example: Example) -> tuple[float, dict[str, Any]]:
    return evaluate_with_solver(candidate, example)


def evaluate_with_solver(candidate: str, example: Example, **solver_kwargs: Any) -> tuple[float, dict[str, Any]]:
    return _solve(
        candidate,
        example,
        task_name="formula",
        is_correct=formula_answer_is_correct,
        **solver_kwargs,
    )


def _make_task() -> Task:
    train_set, val_set, test_set = load_finance_dataset("formula")
    return Task(
        name="formula",
        objective="Optimize a prompt that generalizes across financial formula problems.",
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
                "source_dataset": "ACE eval/finance Formula (arXiv:2510.04618)",
                "split_method": "ACE-provided train/val/test jsonl, vendored verbatim",
                "split_sizes": {
                    "train": len(train_set),
                    "val": len(val_set),
                    "test": len(test_set),
                },
            },
        },
    )


register_task_factory("formula", _make_task)
