from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from terrarium.adapters.claude_code import materialize_sandbox as materialize_claude_sandbox
from terrarium.adapters.meta_harness import _materialize_sandbox as materialize_meta_sandbox
from terrarium.budget import BudgetTracker
from terrarium.eval_server import EvalServer
from terrarium.adapter import Result
from terrarium.runner import _prepare_task_for_benchmark, run
from terrarium.task import Example, Task
from terrarium.tasks.arc_agi import _evaluate_test


def _eval(candidate: str, example: Example | None = None) -> tuple[float, dict]:
    del candidate
    if example is None:
        return 1.0, {"score": 1.0}
    return float(example.expected or 0.0), {"example_id": example.id}


def _task(
    *,
    name: str = "task",
    train: list[Example] | None = None,
    val: list[Example] | None = None,
    test: list[Example] | None = None,
    mode: str = "generalization",
) -> Task:
    return Task(
        name=name,
        initial_candidate="seed",
        eval_fn=_eval,
        train_set=train,
        val_set=val,
        test_set=test,
        metadata={"type": mode},
    )


class BenchmarkContractTests(unittest.TestCase):
    def test_generalization_requires_hidden_test(self) -> None:
        task = _task(train=[Example("train", {}, 1.0)], val=[Example("val", {}, 1.0)])

        with self.assertRaisesRegex(ValueError, "requires task 'task' to define test_set"):
            _prepare_task_for_benchmark(
                task,
                OmegaConf.create({"mode": "generalization", "use_val": True}),
            )

    def test_generalization_rejects_split_id_overlap_even_when_val_hidden(self) -> None:
        task = _task(
            train=[Example("same", {}, 1.0)],
            val=[Example("same", {}, 1.0)],
            test=[Example("test", {}, 1.0)],
        )

        with self.assertRaisesRegex(ValueError, "overlapping example IDs"):
            _prepare_task_for_benchmark(
                task,
                OmegaConf.create({"mode": "generalization", "use_val": False}),
            )

    def test_use_val_false_removes_val_from_adapter_view(self) -> None:
        task = _task(
            train=[Example("train", {}, 1.0)],
            val=[Example("val", {}, 1.0)],
            test=[Example("test", {}, 1.0)],
        )

        prepared = _prepare_task_for_benchmark(
            task,
            OmegaConf.create({"mode": "generalization", "use_val": False}),
        )

        self.assertIsNone(prepared.val_set)
        self.assertIsNotNone(prepared.test_set)

    def test_run_hides_test_set_from_adapter_and_server(self) -> None:
        seen: dict[str, object] = {}

        class CaptureAdapter:
            def evolve(self, task: Task, server: EvalServer) -> Result:
                seen["adapter_test_set"] = task.test_set
                seen["server_test_set"] = server.task.test_set
                return Result(best_candidate="seed", best_score=0.0)

        task = _task(
            train=[Example("train", {}, 1.0)],
            test=[Example("test", {}, 3.0)],
        )

        result = run(task, CaptureAdapter(), max_evals=10, max_concurrency=1)

        self.assertIsNone(seen["adapter_test_set"])
        self.assertIsNone(seen["server_test_set"])
        self.assertEqual(result.metadata["test_scores"], {"test": 3.0})
        self.assertEqual(result.metadata["test_score"], 3.0)

    def test_multi_task_allows_shared_train_and_val(self) -> None:
        examples = [Example("visible", {}, 1.0)]
        task = _task(
            train=examples,
            val=examples,
            test=[Example("not_official", {}, 0.0)],
            mode="multi_task",
        )

        prepared = _prepare_task_for_benchmark(
            task,
            OmegaConf.create({"mode": "multi_task", "use_val": True}),
        )

        self.assertEqual(prepared.metadata["type"], "multi_task")
        self.assertIs(prepared.train_set, prepared.val_set)
        self.assertIsNone(prepared.test_set)

    def test_legacy_single_task_mode_normalizes_to_single(self) -> None:
        task = _task(mode="single_task")

        prepared = _prepare_task_for_benchmark(
            task,
            OmegaConf.create({"mode": None, "use_val": True}),
        )

        self.assertEqual(prepared.metadata["type"], "single")

    def test_eval_server_never_indexes_or_evaluates_test_split(self) -> None:
        task = _task(
            train=[Example("train", {}, 1.0)],
            val=[Example("val", {}, 2.0)],
            test=[Example("test", {}, 3.0)],
        )
        server = EvalServer(task, BudgetTracker(max_evals=10), max_concurrency=1)

        self.assertNotIn("test", server._examples)
        with self.assertRaisesRegex(ValueError, "test split is hidden"):
            server.evaluate_examples("candidate", split="test")

    def test_eval_server_rejects_unknown_or_hidden_example_ids(self) -> None:
        task = _task(
            train=[Example("train", {}, 1.0)],
            test=[Example("test", {}, 3.0)],
        )
        server = EvalServer(task, BudgetTracker(max_evals=10), max_concurrency=1)

        with self.assertRaisesRegex(ValueError, "Unknown or hidden example IDs"):
            server.evaluate_examples("candidate", example_ids=["test"])

        with self.assertRaisesRegex(ValueError, "Unknown or hidden example IDs"):
            server.evaluate_examples("candidate", example_ids=["missing"])

    def test_eval_server_rejects_invalid_split_names(self) -> None:
        task = _task(train=[Example("train", {}, 1.0)])
        server = EvalServer(task, BudgetTracker(max_evals=10), max_concurrency=1)

        with self.assertRaisesRegex(ValueError, "split must be one of"):
            server.evaluate_examples("candidate", split="bogus")

    def test_claude_materialization_writes_visible_splits_not_test(self) -> None:
        task = _task(
            train=[Example("train", {"x": 1}, 1.0)],
            val=[Example("val", {"x": 2}, 2.0)],
            test=[Example("test", {"x": 3}, 3.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            materialize_claude_sandbox(
                work_dir,
                task,
                "http://127.0.0.1:1",
                BudgetTracker(max_evals=10),
            )

            self.assertTrue((work_dir / "train" / "train.json").exists())
            self.assertTrue((work_dir / "val" / "val.json").exists())
            self.assertFalse((work_dir / "test").exists())

    def test_meta_harness_materialization_writes_visible_splits_not_test(self) -> None:
        task = _task(
            train=[Example("train", {"x": 1}, 1.0)],
            val=[Example("val", {"x": 2}, 2.0)],
            test=[Example("test", {"x": 3}, 3.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            materialize_meta_sandbox(work_dir, task, BudgetTracker(max_evals=10))

            self.assertTrue((work_dir / "train" / "train.json").exists())
            self.assertTrue((work_dir / "val" / "val.json").exists())
            self.assertFalse((work_dir / "test").exists())

    def test_arc_test_scoring_accepts_grid_predictions_and_attempt_lists(self) -> None:
        gold = [[[1, 2], [3, 4]]]

        score, results = _evaluate_test([[[1, 2], [3, 4]]], gold)
        self.assertEqual(score, 1.0)
        self.assertTrue(results[0]["correct"])

        score, results = _evaluate_test([[[[9]], [[1, 2], [3, 4]]]], gold)
        self.assertEqual(score, 1.0)
        self.assertTrue(results[0]["correct"])


if __name__ == "__main__":
    unittest.main()
