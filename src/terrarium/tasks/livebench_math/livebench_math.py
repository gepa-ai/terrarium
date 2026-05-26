"""LiveBench math: internal researcher notes — NOT surfaced to the
optimizer. DO NOT include this docstring in any task-facing text
(``objective`` / ``background``).

LiveBench's math benchmark (arXiv:2406.19314), HF dataset
``livebench/math`` — 368 rows, one ``test`` split. Three underlying
problem families (competition multiple-choice/numeric, multi-part
proof-step ordering, symbolic computation) are NOT named in any
task-facing text: the optimizer must discover structure from
per-example feedback, not be handed the taxonomy. Scoring is the
verbatim LiveBench port in ``_livebench_scoring`` (dispatched on the
hidden subtask label); one eval = one example. Dataset-task,
generalization mode; candidate = an evolved prompt. Splits are
deterministic + stratified-by-subtask with a frozen test
(``_livebench_common``); the runner draws per-seed train/val subsets.
"""

from __future__ import annotations

from typing import Any

from terrarium.registry import register_task_factory
from terrarium.task import Example, Task
from terrarium.tasks.livebench_math._livebench_common import (
    SPLIT_SEED,
    evaluate_with_solver,
    load_livebench_math_dataset,
    split_breakdown,
)

DESCRIPTION = """\
The candidate is a prompt string. For each item the model is given a
mathematics problem whose statement already specifies how its final
answer must be presented. The score is the official correctness for
that item: 1.0 or 0.0 for problems with a single required answer, and a
fraction in [0, 1] for problems composed of several ordered parts.
"""

INITIAL_CANDIDATE = (
    "Solve the problem carefully, reasoning step by step. Then give the final "
    "answer in exactly the format the problem asks for, placing it at the very "
    "end of your response."
)


def evaluate(candidate: str, example: Example) -> tuple[float, dict[str, Any]]:
    return evaluate_with_solver(candidate, example)


def _make_task() -> Task:
    train_set, val_set, test_set = load_livebench_math_dataset()
    return Task(
        name="livebench_math",
        objective="Optimize a prompt that generalizes across diverse mathematics problems.",
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
                "source_dataset": "livebench/math (LiveBench, arXiv:2406.19314)",
                "split_method": (
                    "single test split (368); deterministic, stratified by subtask; "
                    "exactly 100 train / 100 val / remainder test (frozen)"
                ),
                "split_seed": SPLIT_SEED,
                "split_sizes": {
                    "train": len(train_set),
                    "val": len(val_set),
                    "test": len(test_set),
                },
                "per_subtask": split_breakdown(train_set, val_set, test_set),
            },
            "scoring_provenance": "verbatim LiveBench port; see _lb_upstream/_PROVENANCE.txt",
        },
    )


register_task_factory("livebench_math", _make_task)
