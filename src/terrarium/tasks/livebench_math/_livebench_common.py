"""LiveBench math: shared harness — internal researcher notes, NOT
surfaced to the optimizer. DO NOT include any of this in task-facing text.

Terrarium-native PROMPT OPTIMIZATION framing (like tasks/aime_math.py and
tasks/finance/_finance_common.py):

  - The candidate is an evolved *prompt* string; it becomes the dspy
    signature instructions. One eval = one example.
  - The model input is the raw LiveBench ``turns[0]`` — which already
    carries that problem's own answer-format instructions. Expected =
    LiveBench ``ground_truth``.
  - The model's answer is run through the verbatim-ported LiveBench
    scorer (``_livebench_scoring.score_livebench_math``), dispatched on
    the row's subtask, so scores stay directly comparable to LiveBench.
    Evolving a prompt that makes the model emit a parseable, correct
    answer IS the task. Subtask/task labels are kept in ``inputs`` for
    scorer dispatch ONLY and are never sent to the model nor placed in
    surfaced feedback (two-channel discipline).

Split contract (single-source dataset):
  livebench/math ships ONE ``test`` split (368 rows).
  ``load_livebench_math_dataset`` builds a DETERMINISTIC split STRATIFIED
  BY SUBTASK with a fixed ``SPLIT_SEED``: exactly 100 train / 100 val /
  remainder (~168) test. The test split is frozen and shared across all
  runs/seeds.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from terrarium.budget import BudgetExhausted
from terrarium.task import Example
from terrarium.tasks.livebench_math._livebench_scoring import (
    is_olympiad,
    olympiad_position_report,
    score_livebench_math,
)

SPLIT_SEED = 0
# RQ5 full sweep: count-targeted split. livebench/math = 368 rows. Exactly
# 100 train / 100 val, stratified by subtask via largest-remainder; test
# takes the remainder (~168), frozen and shared across all runs.
TRAIN_N = 100
VAL_N = 100


def _load_raw() -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("livebench/math", split="test")
    examples: list[Example] = []
    for row in ds:
        turns = row["turns"]
        question = turns[0] if isinstance(turns, (list, tuple)) else str(turns)
        examples.append(
            Example(
                id=str(row["question_id"]),
                inputs={
                    "input": question,
                    # dispatch-only; never sent to the model, never surfaced
                    "subtask": str(row["subtask"]),
                    "task": str(row["task"]),
                },
                expected=str(row["ground_truth"]),
            )
        )
    return examples


def _largest_remainder(target: int, sizes: dict[str, int]) -> dict[str, int]:
    """Allocate ``target`` items across groups proportional to ``sizes``,
    rounding via the largest-remainder method so the allocations sum to
    exactly ``target``."""
    grand = sum(sizes.values())
    raw = {s: target * n / grand for s, n in sizes.items()}
    alloc = {s: int(raw[s]) for s in sizes}
    deficit = target - sum(alloc.values())
    for s in sorted(sizes, key=lambda s: raw[s] - alloc[s], reverse=True)[:deficit]:
        alloc[s] += 1
    return alloc


def load_livebench_math_dataset() -> tuple[list[Example], list[Example], list[Example]]:
    """Deterministic split, stratified by subtask: exactly 100 train /
    100 val / remainder test (frozen)."""
    import random
    from collections import defaultdict

    examples = _load_raw()
    by_sub: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_sub[ex.inputs["subtask"]].append(ex)

    sizes = {s: len(g) for s, g in by_sub.items()}
    val_alloc = _largest_remainder(VAL_N, sizes)
    train_alloc = _largest_remainder(TRAIN_N, sizes)

    train: list[Example] = []
    val: list[Example] = []
    test: list[Example] = []
    for subtask in sorted(by_sub):
        group = sorted(by_sub[subtask], key=lambda e: e.id)
        random.Random(SPLIT_SEED).shuffle(group)
        n_val = val_alloc[subtask]
        n_train = train_alloc[subtask]
        if n_val + n_train > len(group):
            raise ValueError(
                f"subtask {subtask!r}: val+train ({n_val}+{n_train}) "
                f"exceeds available rows ({len(group)})"
            )
        val += group[:n_val]
        train += group[n_val : n_val + n_train]
        test += group[n_val + n_train :]
    return train, val, test


def split_breakdown(
    train: list[Example], val: list[Example], test: list[Example]
) -> dict[str, dict[str, int]]:
    from collections import Counter

    out: dict[str, dict[str, int]] = {}
    for name, split in (("train", train), ("val", val), ("test", test)):
        for sub, c in Counter(e.inputs["subtask"] for e in split).items():
            out.setdefault(sub, {"train": 0, "val": 0, "test": 0})[name] = c
    return out


def _build_feedback(subtask: str, ground_truth: str, llm_answer: str, score: float) -> str:
    """Example-scoped reflection text. Never names the task taxonomy."""
    if is_olympiad(subtask):
        return (
            f"Partial score {score:.3f}. {olympiad_position_report(ground_truth, llm_answer)} "
            "Re-derive the dependency order so each step precedes the step that uses it."
        )
    status = "correct" if score >= 1.0 else "incorrect"
    return (
        f"Your answer is {status}. The expected answer is {ground_truth!r}. "
        "Make sure the final answer is emitted in exactly the format the problem requests."
    )


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
    """Run the evolved prompt on one LiveBench-math example and score it
    LiveBench-faithfully."""
    import dspy
    from dspy.utils.exceptions import AdapterParseError

    # Reuse aime_math's per-eval LM builder (generic; avoids duplication).
    from terrarium.tasks.aime_math import _build_eval_lm

    class LiveBenchMathSolverSignature(dspy.Signature):
        input = dspy.InputField(desc="The problem to solve, including all of its instructions.")
        answer = dspy.OutputField(desc="The final answer in the exact format the problem requires.")

    lm = _build_eval_lm(
        solver_lm=solver_lm,
        solver_temperature=solver_temperature,
        solver_max_tokens=solver_max_tokens,
        solver_timeout=solver_timeout,
        solver_num_retries=solver_num_retries,
    )

    def history() -> list[Any]:
        return list(getattr(lm, "history", []) or []) if lm is not None else []

    def cost_and_model(h: list[Any]) -> tuple[float, str | None]:
        cost = sum(float(e.get("cost", 0.0) or 0.0) for e in h if isinstance(e, dict))
        model = None
        if h and isinstance(h[-1], dict):
            model = h[-1].get("model") or h[-1].get("response_model")
        return cost, model

    predictor = dspy.ChainOfThought(LiveBenchMathSolverSignature)
    predictor.predict.signature.instructions = candidate

    subtask = example.inputs["subtask"]
    question_text = example.inputs["input"]
    ground_truth = str(example.expected)
    lm_context = dspy.context(lm=lm) if lm is not None else nullcontext()

    try:
        with lm_context:
            prediction = predictor(input=question_text)
    except BudgetExhausted:
        raise
    except AdapterParseError as exc:
        cost, model = cost_and_model(history())
        return 0.0, {
            "score": 0.0,
            "input": question_text,
            "prompt": candidate,
            "output": getattr(exc, "lm_response", None) or str(exc),
            "reasoning": "",
            "feedback": (
                "The solver response could not be parsed into the required answer field. "
                f"The expected answer is {ground_truth!r}. Emit it in the exact format the "
                "problem requests."
            ),
            "error": "solver_parse_error",
            "cost": cost,
            "solver_model": model,
        }
    except Exception as exc:
        cost, model = cost_and_model(history())
        return 0.0, {
            "score": 0.0,
            "input": question_text,
            "prompt": candidate,
            "output": str(exc),
            "reasoning": "",
            "feedback": (
                f"The solver call failed ({type(exc).__name__}). The expected answer is "
                f"{ground_truth!r}."
            ),
            "error": type(exc).__name__,
            "cost": cost,
            "solver_model": model,
        }

    cost, model = cost_and_model(history())
    llm_answer = str(getattr(prediction, "answer", "") or "")
    reasoning = str(getattr(prediction, "reasoning", "") or "")

    try:
        score = score_livebench_math(subtask, ground_truth, llm_answer, question_text)
    except Exception as exc:
        return 0.0, {
            "score": 0.0,
            "input": question_text,
            "prompt": candidate,
            "output": llm_answer,
            "reasoning": reasoning,
            "feedback": (
                f"The answer could not be scored ({type(exc).__name__}). The expected answer "
                f"is {ground_truth!r}."
            ),
            "error": f"scoring_error:{type(exc).__name__}",
            "cost": cost,
            "solver_model": model,
        }

    return score, {
        "score": score,
        "input": question_text,
        "prompt": candidate,
        "output": llm_answer,
        "reasoning": reasoning,
        "feedback": _build_feedback(subtask, ground_truth, llm_answer, score),
        "cost": cost,
        "solver_model": model,
    }
