"""AIME math task: optimize a prompt for solving competition math problems.

This is a dataset task (generalization mode). The candidate is a natural
language prompt that guides an LLM to solve AIME-style math problems.
"""

from __future__ import annotations

from typing import Any

from terrarium.registry import register_task_factory
from terrarium.task import Example, Task

DESCRIPTION = """\
Optimize a natural language prompt for solving AIME-style math competition problems.
The candidate is a prompt string. An LLM uses this prompt to solve each problem,
and the score is 1.0 if the answer matches the expected integer, 0.0 otherwise.
"""

INITIAL_CANDIDATE = (
    "Solve the math problem carefully. Break down the steps and provide the final answer as a single number."
)


def _load_dataset() -> tuple[list[Example], list[Example], list[Example]]:
    """Load AIME math dataset from HuggingFace, returning (train, val, test)."""
    import random

    from datasets import load_dataset

    train_split: list[Example] = []
    train_load = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
    for i, item in enumerate(train_load):
        train_split.append(Example(
            id=f"aime_train_{i}",
            inputs={"input": item["problem"], "solution": item["solution"]},
            expected=item["answer"],
        ))

    random.Random(0).shuffle(train_split)

    test_split: list[Example] = []
    test_load = load_dataset("MathArena/aime_2025", "default", split="train")
    for i, item in enumerate(test_load):
        test_split.append(Example(
            id=f"aime_test_{i}",
            inputs={"input": item["problem"]},
            expected=item["answer"],
        ))

    mid = len(train_split) // 2
    return train_split[:mid], train_split[mid:], test_split


def evaluate(candidate: str, example: Example) -> tuple[float, dict[str, Any]]:
    """Evaluate a prompt on a single AIME math example.

    Uses richer feedback including the full solution when available,
    matching the working examples/aime_math setup.
    """
    import dspy

    class MathSolverSignature(dspy.Signature):
        input = dspy.InputField(desc="The math problem to solve.")
        answer = dspy.OutputField(desc="The final numerical answer.")

    lm = getattr(dspy.settings, "lm", None)
    history_before = len(getattr(lm, "history", []) or []) if lm is not None else 0

    predictor = dspy.ChainOfThought(MathSolverSignature)
    predictor.predict.signature.instructions = candidate
    prediction = predictor(input=example.inputs["input"])
    history_after = getattr(lm, "history", []) or [] if lm is not None else []
    new_history = history_after[history_before:]
    solver_cost = sum(float(entry.get("cost", 0.0) or 0.0) for entry in new_history if isinstance(entry, dict))
    solver_model = None
    if new_history and isinstance(new_history[-1], dict):
        solver_model = new_history[-1].get("model") or new_history[-1].get("response_model")

    correct_answer = int(example.expected)
    written_solution = example.inputs.get("solution", "")
    solution_suffix = (
        f" Here\'s the full step-by-step solution:\n{written_solution}\n\n"
        "Think about what takeaways you can learn from this solution to improve "
        "your future answers and approach to similar problems"
        if written_solution
        else ""
    )

    try:
        llm_answer = int(prediction.answer)
    except (ValueError, TypeError):
        feedback = (
            f"The final answer must be a valid integer and nothing else. "
            f"You responded with \'{prediction.answer}\', which couldn\'t be parsed as a python integer. "
            f"Please ensure your answer is a valid integer without any additional text or formatting. "
            f"The correct answer is \'{correct_answer}\'.{solution_suffix}"
            f"{' and ensure your final answer is a valid integer.' if written_solution else ''}"
        )
        return 0.0, {
            "score": 0.0,
            "input": example.inputs["input"],
            "prompt": candidate,
            "output": prediction.answer,
            "feedback": feedback,
            "cost": solver_cost,
            "solver_model": solver_model,
        }

    score = float(correct_answer == llm_answer)
    status = "correct" if score == 1.0 else "incorrect"
    feedback = f"Your answer is {status}. The correct answer is \'{correct_answer}\'.{solution_suffix}"
    return score, {
        "score": score,
        "input": example.inputs["input"],
        "prompt": candidate,
        "output": prediction.answer,
        "reasoning": getattr(prediction, "reasoning", ""),
        "feedback": feedback,
        "cost": solver_cost,
        "solver_model": solver_model,
    }


def _make_task() -> Task:
    """Build the AIME math task (lazy dataset loading)."""
    train_set, val_set, test_set = _load_dataset()
    return Task(
        name="aime_math",
        objective="Optimize a math-solving prompt that generalizes across AIME problems.",
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
                "source_dataset": "AI-MO/aimo-validation-aime and MathArena/aime_2025",
                "split_method": "shuffle_aimo_validation_aime_seed_0_half_train_half_val; MathArena/aime_2025_as_test",
                "split_seed": 0,
                "split_sizes": {
                    "train": len(train_set),
                    "val": len(val_set),
                    "test": len(test_set),
                },
            },
        },
    )


register_task_factory("aime_math", _make_task)
