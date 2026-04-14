"""AIME math mini task: small slice for quick validation.

10 train / 15 val / 15 test from the HF dataset.
"""

from __future__ import annotations

import random
from typing import Any

from terrarium.registry import register_task_factory
from terrarium.task import Example, Task
from terrarium.tasks.aime_math import DESCRIPTION, INITIAL_CANDIDATE, evaluate

N_TRAIN = 10
N_VAL = 15
N_TEST = 15


def _load_mini_dataset() -> tuple[list[Example], list[Example], list[Example]]:
    from datasets import load_dataset

    raw = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
    examples = [
        Example(
            id=f"aime_{i}",
            inputs={"input": item["problem"], "solution": item["solution"]},
            expected=item["answer"],
        )
        for i, item in enumerate(raw)
    ]
    random.Random(42).shuffle(examples)
    train = examples[:N_TRAIN]
    val = examples[N_TRAIN : N_TRAIN + N_VAL]
    test = examples[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]
    return train, val, test


def _make_task() -> Task:
    train_set, val_set, test_set = _load_mini_dataset()
    return Task(
        name="aime_math_mini",
        description=DESCRIPTION,
        initial_candidate=INITIAL_CANDIDATE,
        eval_fn=evaluate,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        metadata={
            "type": "generalization",
            "candidate_type": "prompt",
            "objective": "Optimize a math-solving prompt that generalizes across AIME problems.",
        },
    )


register_task_factory("aime_math_mini", _make_task)
