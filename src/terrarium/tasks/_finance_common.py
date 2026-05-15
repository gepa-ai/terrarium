"""ACE finance benchmarks: shared harness — internal researcher notes,
NOT surfaced to the optimizer. DO NOT include in any task-facing text.

Full ACE replication (arXiv:2510.04618). Both finer/formula are
single-turn generalization dataset tasks where:

  - The evolved *candidate* IS ACE's playbook: it fills the ``{playbook}``
    slot of the verbatim ``GENERATOR_PROMPT``. One eval = one example.
  - At load time, each example's raw ``context`` is split into
    (input_text, question) by the task's ACE parse fn, exactly as ACE's
    ``DataProcessor.process_task_data`` does.
  - At eval time the model is sent ACE's exact prompt
    ``GENERATOR_PROMPT.format(candidate, "(empty)", question, context)``
    (reflection = "(empty)", matching ACE's test path), the raw response
    is run through the verbatim ``extract_answer``, then the task's
    correctness check. This reproduces ACE's generator -> extract ->
    answer_is_correct pipeline so numbers are comparable to the paper.

Data is vendored at terrarium/data/finance/{task}_{split}.jsonl.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from terrarium.budget import BudgetExhausted
from terrarium.task import Example
from terrarium.tasks._ace_prompts import GENERATOR_PROMPT, PARSE_FN
from terrarium.tasks._ace_scoring import extract_answer

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "finance"


def load_finance_dataset(task_name: str) -> tuple[list[Example], list[Example], list[Example]]:
    """Load vendored splits, applying ACE's per-task context parse."""
    parse_fn = PARSE_FN[task_name]
    splits: list[list[Example]] = []
    for split in ("train", "val", "test"):
        path = _DATA_DIR / f"{task_name}_{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing vendored dataset: {path}")
        examples: list[Example] = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            original_context = item["context"]
            input_text, question = parse_fn(original_context)
            examples.append(Example(
                id=f"{task_name}_{split}_{i}",
                inputs={
                    "question": question,
                    "context": input_text,
                    "original_context": original_context,
                },
                expected=str(item["target"]),
            ))
        splits.append(examples)
    return splits[0], splits[1], splits[2]


def evaluate_with_solver(
    candidate: str,
    example: Example,
    *,
    task_name: str,
    is_correct: Callable[[str, str], bool],
    solver_lm: str | None = None,
    solver_temperature: float | None = None,
    solver_max_tokens: int | None = None,
    solver_timeout: float | None = None,
    solver_num_retries: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Run ACE's generator pipeline for one example and score it."""
    import dspy

    # Reuse aime_math's per-eval LM builder (generic; avoids duplication).
    from terrarium.tasks.aime_math import _build_eval_lm

    lm = _build_eval_lm(
        solver_lm=solver_lm,
        solver_temperature=solver_temperature,
        solver_max_tokens=solver_max_tokens,
        solver_timeout=solver_timeout,
        solver_num_retries=solver_num_retries,
    )

    def solver_cost_and_model() -> tuple[float, str | None]:
        history = list(getattr(lm, "history", []) or []) if lm is not None else []
        cost = sum(float(e.get("cost", 0.0) or 0.0) for e in history if isinstance(e, dict))
        model = None
        if history and isinstance(history[-1], dict):
            model = history[-1].get("model") or history[-1].get("response_model")
        return cost, model

    expected = str(example.expected)
    # ACE's exact generator prompt; the candidate IS the playbook slot.
    prompt = GENERATOR_PROMPT.format(
        candidate,
        "(empty)",
        example.inputs["question"],
        example.inputs["context"],
    )
    lm_context = dspy.context(lm=lm) if lm is not None else nullcontext()

    try:
        with lm_context:
            if lm is None:
                raise RuntimeError("no solver LM configured (set task.solver_lm)")
            completions = lm(messages=[{"role": "user", "content": prompt}])
        response = completions[0] if completions else ""
        if not isinstance(response, str):
            response = str(response)
    except BudgetExhausted:
        raise
    except Exception as exc:
        cost, model = solver_cost_and_model()
        return 0.0, {
            "score": 0.0,
            "task": task_name,
            "input": example.inputs["original_context"],
            "prompt": candidate,
            "output": str(exc),
            "feedback": (
                "The solver call failed before producing an answer. "
                f"The correct answer is '{expected}'."
            ),
            "error": type(exc).__name__,
            "cost": cost,
            "solver_model": model,
        }

    cost, model = solver_cost_and_model()
    extracted = extract_answer(response)
    ok = bool(is_correct(extracted, expected))
    score = float(ok)
    status = "correct" if ok else "incorrect"
    return score, {
        "score": score,
        "task": task_name,
        "input": example.inputs["original_context"],
        "prompt": candidate,
        "output": response,
        "extracted": extracted,
        "feedback": f"Your answer is {status}. The correct answer is '{expected}'.",
        "cost": cost,
        "solver_model": model,
    }
