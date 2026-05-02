from __future__ import annotations

import unittest
import sys

from terrarium.adapter import Adapter, Result
from terrarium.adapters.omni import OmniAdapter
from terrarium.budget import BudgetExhausted, BudgetTracker
from terrarium.eval_server import EvalServer
from terrarium.runner import run
from terrarium.solver_lm import CostTrackedDSPyLM, SolverBudgetExhausted
from terrarium.task import Example, Task
from terrarium.tasks import aime_math


def _dataset_task() -> Task:
    def eval_fn(candidate: str, example: Example) -> tuple[float, dict]:
        return (1.0 if candidate == example.expected else 0.0), {"example_id": example.id}

    return Task(
        name="contract_task",
        initial_candidate="seed",
        eval_fn=eval_fn,
        train_set=[Example("train-1", {}, "train-answer")],
        val_set=[Example("val-1", {}, "val-answer")],
        test_set=[Example("test-1", {}, "test-answer")],
    )


class _SearchOnlyAdapter(Adapter):
    def evolve(self, task: Task, server: EvalServer) -> Result:
        assert task.test_set is None
        assert server.task.test_set is None
        assert server.task.metadata["heldout_test_size"] == 1

        train_score, _ = server.evaluate_examples("train-answer", split="train")
        val_score, _ = server.evaluate_examples("val-answer", split="val")
        with self.assert_raises(ValueError):
            server.evaluate_examples("test-answer", split="test")

        return Result(
            best_candidate="test-answer",
            best_score=(train_score + val_score) / 2,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
        )

    @staticmethod
    def assert_raises(exc_type: type[BaseException]):
        return unittest.TestCase().assertRaises(exc_type)


class _LeakyDirectAdapter(Adapter):
    def __init__(self, leaked_example: Example) -> None:
        self.leaked_example = leaked_example

    def evolve(self, task: Task, server: EvalServer) -> Result:
        with unittest.TestCase().assertRaises(ValueError):
            server.evaluate("test-answer", self.leaked_example)
        return Result(
            best_candidate="test-answer",
            best_score=0.0,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
        )


class _CapturingOmniAdapter(OmniAdapter):
    def __init__(self) -> None:
        super().__init__(backend="gepa")
        self.captured_task = None

    def _run_single(self, omni_task, evaluate, server, max_evals, max_token_cost):  # type: ignore[no-untyped-def]
        self.captured_task = omni_task
        return Result(
            best_candidate="test-answer",
            best_score=0.0,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
        )


class EvalContractTests(unittest.TestCase):
    def test_runner_hides_test_set_during_search_but_scores_afterwards(self) -> None:
        result = run(_dataset_task(), _SearchOnlyAdapter(), max_evals=2)

        self.assertEqual(result.total_evals, 2)
        self.assertEqual(result.best_score, 0.0)
        self.assertEqual(result.metadata["final_val_scores"], {"val-1": 0.0})
        self.assertEqual(result.metadata["final_val_score"], 0.0)
        self.assertEqual(result.metadata["test_scores"], {"test-1": 1.0})
        self.assertEqual(result.metadata["test_score"], 1.0)

    def test_runner_reports_final_validation_score_not_server_example_best(self) -> None:
        class AdapterWithMisleadingServerBest(Adapter):
            def evolve(self, task: Task, server: EvalServer) -> Result:
                server.evaluate_examples("train-answer", split="train")
                return Result(
                    best_candidate="val-answer",
                    best_score=server.best_score,
                    total_evals=server.budget.used,
                    eval_log=server.eval_log,
                )

        result = run(_dataset_task(), AdapterWithMisleadingServerBest(), max_evals=1)

        self.assertEqual(result.best_score, 1.0)
        self.assertEqual(result.metadata["final_val_score"], 1.0)
        self.assertEqual(result.metadata["test_score"], 0.0)

    def test_incomplete_final_validation_does_not_overwrite_best_score(self) -> None:
        class SolverBudgetedTask:
            calls = 0

            @staticmethod
            def eval_fn(candidate: str, example: Example) -> tuple[float, dict]:
                if example.id == "val-1" and candidate == "candidate":
                    raise SolverBudgetExhausted("solver cap")
                SolverBudgetedTask.calls += 1
                return (1.0 if candidate == example.expected else 0.0), {}

        task = Task(
            name="solver_budgeted_final",
            initial_candidate="seed",
            eval_fn=SolverBudgetedTask.eval_fn,
            train_set=[Example("train-1", {}, "candidate")],
            val_set=[Example("val-1", {}, "candidate")],
        )

        class CandidateAdapter(Adapter):
            def evolve(self, task: Task, server: EvalServer) -> Result:
                return Result("candidate", 0.75, server.budget.used, server.eval_log)

        result = run(task, CandidateAdapter(), max_evals=1)

        self.assertEqual(result.best_score, 0.75)
        self.assertTrue(result.metadata["val_scoring_incomplete"])
        self.assertNotIn("final_val_score", result.metadata)

    def test_runner_includes_solver_cost_in_total_cost(self) -> None:
        class SolverCostTracker:
            total_cost = 0.25
            cost_log = [{"call": 1, "cost": 0.25, "cumulative_cost": 0.25}]

        result = run(_dataset_task(), _SearchOnlyAdapter(), max_evals=2, solver_cost_tracker=SolverCostTracker())

        self.assertEqual(result.metadata["solver_cost"], 0.25)
        self.assertEqual(result.metadata["total_cost"], 0.25)
        self.assertEqual(result.metadata["solver_cost_log"], [{"call": 1, "cost": 0.25, "cumulative_cost": 0.25}])

    def test_direct_evaluate_rejects_examples_outside_visible_task(self) -> None:
        task = _dataset_task()
        result = run(task, _LeakyDirectAdapter(task.test_set[0]), max_evals=1)

        self.assertEqual(result.total_evals, 0)
        self.assertEqual(result.metadata["test_score"], 1.0)

    def test_runner_rejects_duplicate_or_overlapping_split_ids(self) -> None:
        task = _dataset_task()
        task.train_set = [Example("dup", {}, "a"), Example("dup", {}, "b")]
        with self.assertRaisesRegex(ValueError, "duplicate example id"):
            run(task, _SearchOnlyAdapter(), max_evals=1)

        task = _dataset_task()
        task.val_set = [Example("train-1", {}, "val-answer")]
        with self.assertRaisesRegex(ValueError, "appears in both train and val"):
            run(task, _SearchOnlyAdapter(), max_evals=1)

    def test_omni_adapter_does_not_relabel_validation_as_test(self) -> None:
        adapter = _CapturingOmniAdapter()
        result = run(_dataset_task(), adapter, max_evals=1)

        self.assertEqual(result.metadata["test_score"], 1.0)
        self.assertIsNotNone(adapter.captured_task)
        self.assertEqual(adapter.captured_task.val_set[0].id, "val-1")
        self.assertIsNone(adapter.captured_task.test_set)

    def test_dataset_example_ids_must_be_nonempty_and_visible(self) -> None:
        server = EvalServer(_dataset_task(), BudgetTracker(max_evals=10))

        with self.assertRaises(ValueError):
            server.evaluate_examples("anything", example_ids=[])
        with self.assertRaises(ValueError):
            server.evaluate_examples("anything", example_ids=["missing"])
        with self.assertRaises(ValueError):
            server.evaluate_examples("anything", example_ids=["test-1"])

    def test_http_dataset_requests_are_train_only(self) -> None:
        server = EvalServer(_dataset_task(), BudgetTracker(max_evals=10))
        server.start()
        try:
            import json
            import urllib.error
            import urllib.request

            def post(body: dict) -> urllib.response.addinfourl:
                req = urllib.request.Request(
                    f"{server.url}/evaluate_examples",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                return urllib.request.urlopen(req, timeout=5)

            with self.assertRaises(urllib.error.HTTPError) as empty_ctx:
                post({"candidate": "anything", "example_ids": []})
            self.assertEqual(empty_ctx.exception.code, 400)

            with self.assertRaises(urllib.error.HTTPError) as val_ctx:
                post({"candidate": "anything", "example_ids": ["val-1"]})
            self.assertEqual(val_ctx.exception.code, 400)

            with self.assertRaises(urllib.error.HTTPError) as split_ctx:
                post({"candidate": "anything", "split": "val"})
            self.assertEqual(split_ctx.exception.code, 400)
        finally:
            server.stop()

    def test_dataset_split_test_is_not_available_during_search(self) -> None:
        server = EvalServer(_dataset_task(), BudgetTracker(max_evals=10))

        with self.assertRaises(ValueError):
            server.evaluate_examples("anything", split="test")

    def test_batch_eval_propagates_budget_exhaustion(self) -> None:
        def eval_fn(_candidate: str, _example: Example) -> tuple[float, dict]:
            raise SolverBudgetExhausted("solver cap")

        task = Task(
            name="budgeted_dataset",
            initial_candidate="seed",
            eval_fn=eval_fn,
            train_set=[Example("train-1", {}, "x")],
        )
        server = EvalServer(task, BudgetTracker(max_evals=10))

        with self.assertRaises(SolverBudgetExhausted):
            server.evaluate_examples("anything", split="train")

        self.assertEqual(server.budget.used, 0)

    def test_http_reports_solver_budget_exhaustion_type(self) -> None:
        def eval_fn(_candidate: str, _example: Example | None = None) -> tuple[float, dict]:
            raise SolverBudgetExhausted("solver cap")

        task = Task(name="http_solver_budget", initial_candidate="seed", eval_fn=eval_fn)
        server = EvalServer(task, BudgetTracker(max_evals=10))
        server.start()
        try:
            import json
            import urllib.error
            import urllib.request

            req = urllib.request.Request(
                f"{server.url}/evaluate",
                data=json.dumps({"candidate": "anything"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req, timeout=5)
            self.assertEqual(ctx.exception.code, 429)
            body = json.loads(ctx.exception.read().decode())
            self.assertEqual(body["error_type"], "solver_budget_exhausted")
        finally:
            server.stop()

    def test_solver_cost_tracker_ignores_cache_hits(self) -> None:
        lm = CostTrackedDSPyLM("openai/gpt-4.1-nano")
        response = type("Response", (), {"cache_hit": True, "usage": {"total_tokens": 10}})()

        self.assertEqual(lm._completion_cost(response), 0.0)

    def _with_fake_dspy(self, predictor_cls, fn):
        original_dspy = sys.modules.get("dspy")
        fake_dspy = type(
            "FakeDspy",
            (),
            {
                "Signature": object,
                "InputField": staticmethod(lambda **_kwargs: None),
                "OutputField": staticmethod(lambda **_kwargs: None),
                "ChainOfThought": predictor_cls,
            },
        )
        sys.modules["dspy"] = fake_dspy
        try:
            return fn()
        finally:
            if original_dspy is None:
                sys.modules.pop("dspy", None)
            else:
                sys.modules["dspy"] = original_dspy

    def test_aime_solver_parse_errors_score_zero(self) -> None:
        class FailingPredictor:
            def __init__(self, _signature):
                self.predict = type("Predict", (), {"signature": type("Sig", (), {"instructions": ""})()})()

            def __call__(self, **_kwargs):
                raise ValueError("bad solver output")

        score, info = self._with_fake_dspy(
            FailingPredictor,
            lambda: aime_math.evaluate(
                "prompt",
                Example("aime", {"input": "problem", "solution": "solution"}, 7),
            ),
        )

        self.assertEqual(score, 0.0)
        self.assertIn("bad solver output", info["error"])

    def test_aime_missing_answer_scores_zero(self) -> None:
        class MissingAnswerPredictor:
            def __init__(self, _signature):
                self.predict = type("Predict", (), {"signature": type("Sig", (), {"instructions": ""})()})()

            def __call__(self, **_kwargs):
                return object()

        score, info = self._with_fake_dspy(
            MissingAnswerPredictor,
            lambda: aime_math.evaluate(
                "prompt",
                Example("aime", {"input": "problem", "solution": "solution"}, 7),
            ),
        )

        self.assertEqual(score, 0.0)
        self.assertIn("AttributeError", info["error"])

    def test_aime_solver_budget_exhaustion_propagates(self) -> None:
        class BudgetedOutPredictor:
            def __init__(self, _signature):
                self.predict = type("Predict", (), {"signature": type("Sig", (), {"instructions": ""})()})()

            def __call__(self, **_kwargs):
                raise BudgetExhausted("solver budget exhausted")

        with self.assertRaises(BudgetExhausted):
            self._with_fake_dspy(
                BudgetedOutPredictor,
                lambda: aime_math.evaluate(
                    "prompt",
                    Example("aime", {"input": "problem", "solution": "solution"}, 7),
                ),
            )


if __name__ == "__main__":
    unittest.main()
