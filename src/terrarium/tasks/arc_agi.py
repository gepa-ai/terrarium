"""ARC-AGI task: optimize agent code for abstract reasoning puzzles.

This is a dataset task (generalization mode). The candidate is Python code
that implements an ARC-AGI solving agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from terrarium.registry import register_task_factory
from terrarium.task import Example, Task

DESCRIPTION = """\
Optimize Python agent code for solving ARC-AGI abstract reasoning puzzles.
The candidate is Python code defining `solve(train_inputs, train_outputs, test_inputs, llm)`
that infers transformation patterns from training examples and predicts test outputs.
Score = test accuracy (fraction of correctly predicted test grids), with a cost penalty.
"""

INITIAL_CANDIDATE = '''\
import json

def solve(train_inputs, train_outputs, test_inputs, llm):
    training_examples = "\\n".join(f"Input: {i}\\nOutput: {o}" for i, o in zip(train_inputs, train_outputs))
    problem_inputs = "\\n".join(f"Input {i}: {x}" for i, x in enumerate(train_inputs + test_inputs))

    prompt = f"""Solve an ARC AGI puzzle. Training examples:
{training_examples}

Predict the output grid for EACH input below. Return only JSON: a list of grids,
where each grid is a 2D list of integers, in the same order as the inputs.
{problem_inputs}"""
    response = llm(prompt)

    def is_grid(value):
        return (
            isinstance(value, list)
            and value
            and all(isinstance(row, list) for row in value)
            and all(not isinstance(cell, list) for row in value for cell in row)
        )

    def add_grids(value, out):
        if is_grid(value):
            out.append(value)
        elif isinstance(value, list):
            for item in value:
                if is_grid(item):
                    out.append(item)

    def extract_json_grids(text):
        grids = []
        for start, ch in enumerate(text):
            if ch != "[":
                continue
            depth = 0
            for end in range(start, len(text)):
                if text[end] == "[":
                    depth += 1
                elif text[end] == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            add_grids(json.loads(text[start:end + 1]), grids)
                        except Exception:
                            pass
                        break
        return grids

    grids = extract_json_grids(response)
    n_train = len(train_inputs)
    return {
        "train": grids[:n_train],
        "test": grids[n_train:]
    }
'''


def _load_dataset() -> tuple[list[Example], list[Example], list[Example]]:
    """Load ARC-AGI dataset, returning (train, val, test) as Example lists."""
    import random

    from datasets import load_dataset

    ds = load_dataset("dataartist/arc-agi")

    examples: list[Example] = []
    for split_name in ds:
        for item in ds[split_name]:
            train_items = item.get("train")
            test_items = item.get("test")
            if train_items is not None and test_items is not None:
                train_in = [ex["input"] for ex in train_items]
                train_out = [ex["output"] for ex in train_items]
                test_in = [ex["input"] for ex in test_items]
                test_out = [ex.get("output") for ex in test_items]
            else:
                train_in = item["train_in"]
                train_out = item["train_out"]
                test_in = item["test_in"]
                test_out = item.get("test_out")
            examples.append(Example(
                id=item.get("id", f"arc_{len(examples)}"),
                inputs={
                    "train_in": train_in,
                    "train_out": train_out,
                    "test_in": test_in,
                },
                expected=test_out,
            ))

    random.Random(0).shuffle(examples)
    n = len(examples)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    return examples[:train_end], examples[train_end:val_end], examples[val_end:]


@dataclass
class _TrackedLLM:
    model_id: str
    max_llm_calls: int
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(float(call.get("cost", 0.0)) for call in self.calls)

    def __call__(self, prompt: str, temperature: float = 1.0) -> str:
        if len(self.calls) >= self.max_llm_calls:
            raise RuntimeError(f"LLM budget exhausted ({self.max_llm_calls} calls)")

        import litellm
        from litellm import completion

        start = time.time()
        resp = completion(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0
        self.calls.append({
            "prompt": prompt,
            "response": content,
            "cost": cost,
            "duration": time.time() - start,
        })
        return content


def _compare_grid(pred: Any, gold: Any) -> tuple[bool, str]:
    if not isinstance(pred, list) or not pred or not isinstance(pred[0], list):
        return False, f"Prediction must be a non-empty 2D list. Correct grid: {gold}"
    if not isinstance(gold, list) or not gold or not isinstance(gold[0], list):
        return False, "Gold grid is missing or malformed."
    if (len(pred), len(pred[0])) != (len(gold), len(gold[0])):
        return False, f"Shape {(len(pred), len(pred[0]))} != expected {(len(gold), len(gold[0]))}."
    for row in pred:
        if not isinstance(row, list) or len(row) != len(pred[0]):
            return False, "Prediction rows must all be lists with the same width."
    wrong = [
        (i, j)
        for i in range(len(gold))
        for j in range(len(gold[0]))
        if not isinstance(pred[i][j], (int, float)) or int(pred[i][j]) != gold[i][j]
    ]
    if not wrong:
        return True, "Correct."
    return False, f"Incorrect values at {wrong[:10]}."


def _is_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) for row in value)
        and all(not isinstance(cell, list) for row in value for cell in row)
    )


def _evaluate_predictions(preds: list[Any], golds: list[Any]) -> tuple[float, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    for i, gold in enumerate(golds):
        if i >= len(preds):
            results.append({"idx": i, "correct": False, "feedback": "No prediction."})
            continue
        correct, feedback = _compare_grid(preds[i], gold)
        results.append({"idx": i, "correct": correct, "feedback": feedback})
    score = sum(1 for result in results if result["correct"]) / len(results) if results else 0.0
    return score, results


def _evaluate_test(test_preds: list[Any], test_out: list[Any] | None) -> tuple[float, list[dict[str, Any]]]:
    if not test_out:
        return 0.0, []
    results: list[dict[str, Any]] = []
    for i, gold in enumerate(test_out):
        pred = test_preds[i] if i < len(test_preds) else None
        if pred is None:
            attempts = []
        elif _is_grid(pred):
            attempts = [pred]
        elif isinstance(pred, list):
            attempts = pred[:2]
        else:
            attempts = [pred]
        attempt_results = [_compare_grid(attempt, gold) for attempt in attempts]
        correct = any(item[0] for item in attempt_results)
        feedback = next((item[1] for item in attempt_results if item[0]), attempt_results[0][1] if attempt_results else "No prediction.")
        results.append({"idx": i, "correct": correct, "feedback": feedback})
    return (1.0 if all(result["correct"] for result in results) else 0.0), results


def _run_agent(
    agent_code: str,
    train_in: list[Any],
    train_out: list[Any],
    test_in: list[Any],
    test_out: list[Any] | None,
    model_id: str,
    max_llm_calls: int,
) -> dict[str, Any]:
    llms = _TrackedLLM(model_id=model_id, max_llm_calls=max_llm_calls)
    try:
        namespace: dict[str, Any] = {}
        exec(agent_code, namespace)
        result = namespace["solve"](train_in, train_out, test_in, llms)
        train_preds = result.get("train", [])
        test_preds = result.get("test", [])
    except Exception as e:
        return {
            "training_score": 0.0,
            "test_score": 0.0,
            "training_results": [],
            "test_results": [],
            "error": str(e),
            "llms": llms,
        }

    training_score, training_results = _evaluate_predictions(train_preds, train_out)
    test_score, test_results = _evaluate_test(test_preds, test_out)
    return {
        "training_score": training_score,
        "test_score": test_score,
        "training_results": training_results,
        "test_results": test_results,
        "error": None,
        "llms": llms,
    }


def evaluate(candidate: str, example: Example, model_id: str = "openrouter/google/gemini-3-flash-preview") -> tuple[float, dict[str, Any]]:
    """Evaluate an ARC-AGI agent on a single puzzle."""
    result = _run_agent(
        agent_code=candidate,
        train_in=example.inputs["train_in"],
        train_out=example.inputs["train_out"],
        test_in=example.inputs["test_in"],
        test_out=example.expected,
        model_id=model_id,
        max_llm_calls=10,
    )

    llms = result["llms"]
    score = result["test_score"] - 0.1 * (llms.total_cost > 1.0)

    return score, {
        "score": score,
        "problem_id": example.id,
        "training_score": result["training_score"],
        "test_score": result["test_score"],
        "cost": llms.total_cost,
        "error": result["error"],
    }


def _make_task() -> Task:
    """Build the ARC-AGI task (lazy dataset loading)."""
    train_set, val_set, test_set = _load_dataset()
    return Task(
        name="arc_agi",
        objective="Build an ARC-AGI agent that maximizes test accuracy.",
        background=DESCRIPTION,
        initial_candidate=INITIAL_CANDIDATE,
        eval_fn=evaluate,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        metadata={
            "type": "generalization",
            "candidate_type": "code",
            "split_provenance": {
                "source_dataset": "dataartist/arc-agi",
                "split_method": "shuffle_all_splits_then_60_20_20",
                "split_seed": 0,
                "split_sizes": {
                    "train": len(train_set),
                    "val": len(val_set),
                    "test": len(test_set),
                },
            },
        },
    )


register_task_factory("arc_agi", _make_task)
