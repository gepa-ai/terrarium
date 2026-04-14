"""Core task and example definitions for Terrarium."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Example:
    """A single dataset example for train/test sets."""

    id: str
    inputs: dict[str, Any]
    expected: Any | None = None


# Eval function signature:
#   single-task:  (candidate: str) -> (score, info)
#   dataset-task: (candidate: str, example: Example) -> (score, info)
EvalFn = Callable[..., tuple[float, dict[str, Any]]]


@dataclass
class Task:
    """A Terrarium task: everything an evolution system needs to run.

    Attributes:
        name: Unique identifier (e.g. "circle_packing").
        description: What this task optimizes (shown to the evolution system).
        initial_candidate: Seed text to evolve from.
        eval_fn: Scoring function controlled by terrarium.
        train_set: Training examples (used for optimization).
        val_set: Validation examples (used for candidate selection during optimization).
        test_set: Held-out test examples (evaluated after optimization, never seen during search).
        metadata: Extra context (objective string, background, etc.).
    """

    name: str
    description: str
    initial_candidate: str
    eval_fn: EvalFn
    train_set: list[Example] | None = None
    val_set: list[Example] | None = None
    test_set: list[Example] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_dataset(self) -> bool:
        return self.train_set is not None or self.val_set is not None or self.test_set is not None
