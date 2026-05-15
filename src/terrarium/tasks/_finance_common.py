"""ACE finance benchmarks: shared harness — internal researcher notes,
NOT surfaced to the optimizer. DO NOT include in any task-facing text.

Common dataset loader + dspy solver shared by tasks/finer.py and
tasks/formula.py. Both are single-turn ``context -> answer`` generalization
dataset tasks ported from ACE (arXiv:2510.04618):

  - The candidate is a *prompt* the optimizer evolves (dspy signature
    instructions), exactly like tasks/aime_math.py. One eval = one example.
  - The model input is ACE's full ``context`` string (instruction +
    question, ending in "Answer:"). The expected value is ACE's ``target``.
  - The raw model answer is run through the verbatim-ported ACE
    ``extract_answer`` and then the task-specific correctness check, so
    scores are comparable to the ACE paper. ACE relies on the
    prompt/playbook to make the generator emit a parseable final answer
    (JSON ``final_answer`` / ``Finish[...]`` / boxed); the optimizer's job
    here is to evolve a prompt that does the same — that *is* the benchmark.

Data is vendored at terrarium/data/finance/{task}_{split}.jsonl so the
tasks are portable for any user (no external ACE checkout required).
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from terrarium.budget import BudgetExhausted
from terrarium.task import Example

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "finance"


def load_finance_dataset(task_name: str) -> tuple[list[Example], list[Example], list[Example]]:
    """Load vendored (train, val, test) splits for ``finer`` or ``formula``."""
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
            examples.append(Example(
                id=f"{task_name}_{split}_{i}",
                inputs={"input": item["context"]},
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
    """Run the evolved prompt on one finance example and score it ACE-faithfully."""
    import dspy
    from dspy.utils.exceptions import AdapterParseError

    # Reuse aime_math's per-eval LM builder (generic; avoids duplication).
    from terrarium.tasks.aime_math import _build_eval_lm
    from terrarium.tasks._ace_scoring import extract_answer

    class FinanceSolverSignature(dspy.Signature):
        input = dspy.InputField(desc="The finance question, including all instructions.")
        answer = dspy.OutputField(desc="The final answer in the exact format the question requires.")

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

    predictor = dspy.ChainOfThought(FinanceSolverSignature)
    predictor.predict.signature.instructions = candidate
    expected = str(example.expected)
    lm_context = dspy.context(lm=lm) if lm is not None else nullcontext()

    try:
        with lm_context:
            prediction = predictor(input=example.inputs["input"])
    except BudgetExhausted:
        raise
    except AdapterParseError as exc:
        cost, model = solver_cost_and_model()
        return 0.0, {
            "score": 0.0,
            "task": task_name,
            "input": example.inputs["input"],
            "prompt": candidate,
            "output": getattr(exc, "lm_response", None) or str(exc),
            "feedback": (
                "The solver response could not be parsed into the required "
                f"answer field. The correct answer is '{expected}'."
            ),
            "error": "solver_parse_error",
            "cost": cost,
            "solver_model": model,
        }
    except Exception as exc:
        cost, model = solver_cost_and_model()
        return 0.0, {
            "score": 0.0,
            "task": task_name,
            "input": example.inputs["input"],
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
    raw_answer = str(getattr(prediction, "answer", ""))
    extracted = extract_answer(raw_answer)
    ok = bool(is_correct(extracted, expected))
    score = float(ok)
    status = "correct" if ok else "incorrect"
    return score, {
        "score": score,
        "task": task_name,
        "input": example.inputs["input"],
        "prompt": candidate,
        "output": raw_answer,
        "extracted": extracted,
        "reasoning": getattr(prediction, "reasoning", ""),
        "feedback": f"Your answer is {status}. The correct answer is '{expected}'.",
        "cost": cost,
        "solver_model": model,
    }
