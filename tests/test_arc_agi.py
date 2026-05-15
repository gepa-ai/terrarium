from __future__ import annotations

import unittest

from terrarium.tasks.arc_agi import (
    INITIAL_CANDIDATE,
    _TrackedLLM,
    _evaluate_test,
    _run_agent,
    _test_outputs_or_none,
    evaluate,
)
from terrarium.task import Example


class ArcAgiTests(unittest.TestCase):
    def test_agent_runner_is_self_contained(self) -> None:
        candidate = """
def solve(train_inputs, train_outputs, test_inputs, llm):
    return {"train": train_outputs, "test": [[[1]]]}
"""
        result = _run_agent(
            agent_code=candidate,
            train_in=[[[0]]],
            train_out=[[[1]]],
            test_in=[[[0]]],
            test_out=[[[1]]],
            model_id="unused",
            max_llm_calls=0,
        )

        self.assertEqual(result["training_score"], 1.0)
        self.assertEqual(result["test_score"], 1.0)
        self.assertIsNone(result["error"])
        self.assertEqual(result["train_examples"][0]["prediction"], [[1]])
        self.assertTrue(result["train_examples"][0]["correct"])
        self.assertEqual(result["test_examples"][0]["prediction"], [[1]])
        self.assertTrue(result["test_examples"][0]["correct"])

    def test_tracked_llm_exposes_full_traces(self) -> None:
        llm = _TrackedLLM(model_id="unused", max_llm_calls=10)
        llm.calls.append({
            "prompt": "solve this",
            "response": "answer",
            "cost": 0.25,
            "duration": 1.5,
            "reasoning": "because",
        })

        traces = llm.get_traces()

        self.assertEqual(traces["llm_calls"], 1)
        self.assertEqual(traces["llm_budget"], 10)
        self.assertEqual(traces["total_cost"], 0.25)
        self.assertEqual(traces["trajectory"][0]["prompt"], "solve this")
        self.assertEqual(traces["trajectory"][0]["response"], "answer")
        self.assertEqual(traces["trajectory"][0]["duration"], 1.5)
        self.assertEqual(traces["trajectory"][0]["reasoning"], "because")

    def test_evaluate_returns_rich_arc_side_info(self) -> None:
        candidate = """
def solve(train_inputs, train_outputs, test_inputs, llm):
    return {"train": train_outputs, "test": [[[1]]]}
"""
        example = Example(
            id="arc_example",
            inputs={
                "train_in": [[[0]]],
                "train_out": [[[1]]],
                "test_in": [[[0]]],
            },
            expected=[[[1]]],
        )

        score, info = evaluate(candidate, example, model_id="unused")

        self.assertEqual(score, 1.0)
        self.assertEqual(info["problem_id"], "arc_example")
        self.assertEqual(info["agent_code"], candidate)
        self.assertEqual(info["train_examples"][0]["gold"], [[1]])
        self.assertEqual(info["test_examples"][0]["prediction"], [[1]])
        self.assertEqual(info["training_results"][0]["feedback"], "Correct.")
        self.assertEqual(info["test_results"][0]["feedback"], "Correct.")
        self.assertEqual(info["llm_calls"], 0)
        self.assertEqual(info["llm_budget"], 10)
        self.assertEqual(info["trajectory"], [])

    def test_initial_candidate_parser_preserves_structured_train_test_outputs(self) -> None:
        namespace: dict[str, object] = {}
        exec(INITIAL_CANDIDATE, namespace)
        solve = namespace["solve"]

        def llm(_prompt: str) -> str:
            return '{"train": [[[9]], [[8]]], "test": [[[7]]]}'

        result = solve(
            train_inputs=[[[0]], [[1]]],
            train_outputs=[[[9]], [[8]]],
            test_inputs=[[[2]]],
            llm=llm,
        )

        self.assertEqual(result["train"], [[[9]], [[8]]])
        self.assertEqual(result["test"], [[[7]]])

    def test_arc_test_score_is_fractional_across_test_grids(self) -> None:
        score, results = _evaluate_test(
            test_preds=[[[1]], [[0]]],
            test_out=[[[1]], [[2]]],
        )

        self.assertEqual(score, 0.5)
        self.assertEqual([item["correct"] for item in results], [True, False])

    def test_missing_arc_test_outputs_remain_unscored(self) -> None:
        self.assertIsNone(_test_outputs_or_none([[[1]], None]))
        self.assertIsNone(_test_outputs_or_none([]))
        self.assertEqual(_test_outputs_or_none([[[1]]]), [[[1]]])


if __name__ == "__main__":
    unittest.main()
