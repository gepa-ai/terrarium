"""AIME math task: optimize a prompt for solving competition math problems.

This is a dataset task (generalization mode). The candidate is a natural
language prompt that guides an LLM to solve AIME-style math problems.
"""

from __future__ import annotations

from contextlib import nullcontext
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
    """Evaluate using the currently configured DSPy LM."""
    return evaluate_with_solver(candidate, example)


def evaluate_with_solver(
    candidate: str,
    example: Example,
    *,
    solver_lm: str | None = None,
    solver_temperature: float | None = None,
    solver_max_tokens: int | None = None,
    solver_timeout: float | None = None,
    solver_num_retries: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Evaluate a prompt on a single AIME math example.

    Uses richer feedback including the full solution when available,
    matching the working examples/aime_math setup.
    """
    import dspy
    from dspy.utils.exceptions import AdapterParseError

    class MathSolverSignature(dspy.Signature):
        input = dspy.InputField(desc="The math problem to solve.")
        answer = dspy.OutputField(desc="The final numerical answer.")

    lm = _build_eval_lm(
        solver_lm=solver_lm,
        solver_temperature=solver_temperature,
        solver_max_tokens=solver_max_tokens,
        solver_timeout=solver_timeout,
        solver_num_retries=solver_num_retries,
    )

    def new_solver_history() -> list[Any]:
        return list(getattr(lm, "history", []) or []) if lm is not None else []

    def solver_cost_and_model(history: list[Any]) -> tuple[float, str | None]:
        solver_cost = sum(float(entry.get("cost", 0.0) or 0.0) for entry in history if isinstance(entry, dict))
        solver_model = None
        if history and isinstance(history[-1], dict):
            solver_model = history[-1].get("model") or history[-1].get("response_model")
        return solver_cost, solver_model

    predictor = dspy.ChainOfThought(MathSolverSignature)
    predictor.predict.signature.instructions = candidate
    correct_answer = int(example.expected)
    written_solution = example.inputs.get("solution", "")
    solution_suffix = (
        f" Here\'s the full step-by-step solution:\n{written_solution}\n\n"
        "Think about what takeaways you can learn from this solution to improve "
        "your future answers and approach to similar problems"
        if written_solution
        else ""
    )

    lm_context = dspy.context(lm=lm) if lm is not None else nullcontext()

    try:
        with lm_context:
            prediction = predictor(input=example.inputs["input"])
    except AdapterParseError as exc:
        new_history = new_solver_history()
        solver_cost, solver_model = solver_cost_and_model(new_history)
        raw_output = getattr(exc, "lm_response", None) or str(exc)
        feedback = (
            "The solver response could not be parsed into the required structured answer fields. "
            f"The correct answer is '{correct_answer}'.{solution_suffix}"
        )
        return 0.0, {
            "score": 0.0,
            "input": example.inputs["input"],
            "prompt": candidate,
            "output": raw_output,
            "feedback": feedback,
            "error": "solver_parse_error",
            "cost": solver_cost,
            "solver_model": solver_model,
        }
    except Exception as exc:
        new_history = new_solver_history()
        solver_cost, solver_model = solver_cost_and_model(new_history)
        feedback = (
            "The solver call failed before producing a valid structured answer. "
            f"The correct answer is '{correct_answer}'.{solution_suffix}"
        )
        return 0.0, {
            "score": 0.0,
            "input": example.inputs["input"],
            "prompt": candidate,
            "output": str(exc),
            "feedback": feedback,
            "error": type(exc).__name__,
            "cost": solver_cost,
            "solver_model": solver_model,
        }

    new_history = new_solver_history()
    solver_cost, solver_model = solver_cost_and_model(new_history)

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


def _build_eval_lm(
    *,
    solver_lm: str | None,
    solver_temperature: float | None,
    solver_max_tokens: int | None,
    solver_timeout: float | None,
    solver_num_retries: int | None,
) -> Any | None:
    """Build a per-evaluation DSPy LM so parallel evals do not share history."""
    import dspy

    if solver_lm is not None:
        kwargs: dict[str, Any] = {}
        if solver_temperature is not None:
            kwargs["temperature"] = solver_temperature
        if solver_max_tokens is not None:
            kwargs["max_tokens"] = solver_max_tokens
        if solver_timeout is not None:
            kwargs["timeout"] = solver_timeout
        if solver_num_retries is not None:
            kwargs["num_retries"] = solver_num_retries
        return dspy.LM(solver_lm, **kwargs)

    configured = getattr(dspy.settings, "lm", None)
    if configured is None:
        return None

    model = getattr(configured, "model", None)
    if not model:
        return configured

    kwargs = dict(getattr(configured, "kwargs", {}) or {})
    if solver_timeout is not None:
        kwargs["timeout"] = solver_timeout
    if solver_num_retries is not None:
        kwargs["num_retries"] = solver_num_retries
    return dspy.LM(
        model,
        model_type=getattr(configured, "model_type", "chat"),
        cache=bool(getattr(configured, "cache", True)),
        cache_in_memory=bool(getattr(configured, "cache_in_memory", True)),
        num_retries=int(getattr(configured, "num_retries", 3)),
        **kwargs,
    )


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
