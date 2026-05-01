from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from omegaconf import OmegaConf

from terrarium.adapters.claude_code import ClaudeCodeAdapter, materialize_sandbox as materialize_claude_sandbox
from terrarium.adapters.meta_harness import (
    MetaHarnessAdapter,
    _claude_project_slug as meta_claude_project_slug,
    _copy_session_transcript as copy_meta_session_transcript,
    _materialize_sandbox as materialize_meta_sandbox,
)
from terrarium.adapters.gepa import GEPAAdapter
from terrarium.budget import BudgetTracker
from terrarium.eval_server import EvalServer
from terrarium.adapter import Result
from terrarium.runner import (
    _effective_adapter_sandbox,
    _prepare_task_for_benchmark,
    _sandbox_scope,
    _validate_access_policy,
    run,
)
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

    def test_run_applies_benchmark_contract_to_programmatic_tasks(self) -> None:
        task = _task(
            train=[Example("same", {}, 1.0)],
            val=[Example("same", {}, 1.0)],
            test=[Example("test", {}, 1.0)],
        )

        class CaptureAdapter:
            def evolve(self, task: Task, server: EvalServer) -> Result:
                del task, server
                return Result(best_candidate="seed", best_score=0.0)

        with self.assertRaisesRegex(ValueError, "overlapping example IDs"):
            run(task, CaptureAdapter(), max_evals=10, max_concurrency=1)

    def test_run_respects_benchmark_use_val_false(self) -> None:
        seen: dict[str, object] = {}

        class CaptureAdapter:
            def evolve(self, task: Task, server: EvalServer) -> Result:
                seen["adapter_val_set"] = task.val_set
                seen["server_val_set"] = server.task.val_set
                return Result(best_candidate="seed", best_score=0.0)

        task = _task(
            train=[Example("train", {}, 1.0)],
            val=[Example("val", {}, 2.0)],
            test=[Example("test", {}, 3.0)],
        )

        run(
            task,
            CaptureAdapter(),
            max_evals=10,
            max_concurrency=1,
            benchmark={"mode": "generalization", "use_val": False},
        )

        self.assertIsNone(seen["adapter_val_set"])
        self.assertIsNone(seen["server_val_set"])

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

    def test_eval_server_rejects_unavailable_dataset_splits_without_single_fallback(self) -> None:
        def eval_with_single_fallback(candidate: str, example: Example | None = None) -> tuple[float, dict]:
            del candidate
            if example is None:
                return 99.0, {"path": "single_fallback"}
            return float(example.expected or 0.0), {"example_id": example.id}

        task = Task(
            name="task",
            initial_candidate="seed",
            eval_fn=eval_with_single_fallback,
            train_set=[Example("train", {}, 1.0)],
            val_set=None,
        )
        server = EvalServer(task, BudgetTracker(max_evals=10), max_concurrency=1)

        with self.assertRaisesRegex(ValueError, "val split is not available"):
            server.evaluate_examples("candidate", split="val")
        with self.assertRaisesRegex(ValueError, "example_ids must not be empty"):
            server.evaluate_examples("candidate", example_ids=[])
        self.assertEqual(server.budget.used, 0)

    def test_eval_server_http_evaluate_examples_contract_errors_are_400(self) -> None:
        task = _task(train=[Example("train", {}, 1.0)], test=[Example("test", {}, 3.0)])
        server = EvalServer(task, BudgetTracker(max_evals=10), max_concurrency=1)
        server.start()
        try:
            for body in (
                {"candidate": "candidate", "split": "test"},
                {"candidate": "candidate", "split": "bogus"},
                {"candidate": "candidate", "example_ids": ["missing"]},
            ):
                req = urllib.request.Request(
                    server.url + "/evaluate_examples",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(req, timeout=5)
                self.assertEqual(raised.exception.code, 400)
        finally:
            server.stop()

    def test_eval_server_http_rejects_hidden_val_after_use_val_false(self) -> None:
        task = _task(
            train=[Example("train", {}, 1.0)],
            val=[Example("val", {}, 2.0)],
            test=[Example("test", {}, 3.0)],
        )
        prepared = _prepare_task_for_benchmark(
            task,
            OmegaConf.create({"mode": "generalization", "use_val": False}),
        )
        server = EvalServer(prepared, BudgetTracker(max_evals=10), max_concurrency=1)
        server.start()
        try:
            req = urllib.request.Request(
                server.url + "/evaluate_examples",
                data=json.dumps({"candidate": "candidate", "split": "val"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(req, timeout=5)
            self.assertEqual(raised.exception.code, 400)
            self.assertEqual(server.budget.used, 0)
        finally:
            server.stop()

    def test_sandboxed_access_policy_requires_effective_sandbox(self) -> None:
        class AdapterWithSandbox:
            sandbox = False

        with self.assertRaisesRegex(ValueError, "requires candidate/evaluator execution sandboxing"):
            _validate_access_policy(
                OmegaConf.create({"execution": "sandboxed", "network": "host_shared"}),
                AdapterWithSandbox(),
                sandbox=False,
            )

    def test_sandboxed_access_policy_rejects_adapters_without_sandbox_capability(self) -> None:
        class CustomNoSandbox:
            pass

        with self.assertRaisesRegex(ValueError, "requires candidate/evaluator execution sandboxing"):
            _validate_access_policy(
                OmegaConf.create({"execution": "sandboxed", "network": "host_shared"}),
                CustomNoSandbox(),
                sandbox=True,
            )

    def test_sandboxed_access_policy_rejects_default_gepa_litellm_path(self) -> None:
        adapter = GEPAAdapter(reflection={"reflection_lm": "openai/gpt-5"}, sandbox=True)

        self.assertFalse(_effective_adapter_sandbox(adapter, True))
        with self.assertRaisesRegex(ValueError, "requires candidate/evaluator execution sandboxing"):
            _validate_access_policy(
                OmegaConf.create({"execution": "sandboxed", "network": "host_shared"}),
                adapter,
                sandbox=True,
            )

    def test_gepa_claude_code_path_reports_scoped_subprocess_sandbox(self) -> None:
        adapter = GEPAAdapter(reflection={"reflection_lm": "claude_code/sonnet"}, sandbox=True)

        self.assertTrue(_effective_adapter_sandbox(adapter, True))
        self.assertEqual(
            _sandbox_scope(adapter, True),
            {
                "optimizer_subprocess_sandbox": True,
                "candidate_execution_sandbox": False,
                "network_namespace_isolated": False,
            },
        )
        with self.assertRaisesRegex(ValueError, "requires candidate/evaluator execution sandboxing"):
            _validate_access_policy(
                OmegaConf.create({"execution": "sandboxed", "network": "host_shared"}),
                adapter,
                sandbox=True,
            )

    def test_strict_network_policy_requires_network_namespace_isolation(self) -> None:
        adapters = [
            ClaudeCodeAdapter(sandbox=True),
            MetaHarnessAdapter(sandbox=True),
            GEPAAdapter(reflection={"reflection_lm": "claude_code/sonnet"}, sandbox=True),
        ]

        for adapter in adapters:
            with self.subTest(adapter=type(adapter).__name__):
                self.assertFalse(_sandbox_scope(adapter, True)["network_namespace_isolated"])
                for network in ("network_isolated", "network_namespace_isolated", "none"):
                    with self.assertRaisesRegex(ValueError, "requires network namespace isolation"):
                        _validate_access_policy(
                            OmegaConf.create({"execution": "unsandboxed", "network": network}),
                            adapter,
                            sandbox=True,
                        )
                _validate_access_policy(
                    OmegaConf.create({
                        "execution": "unsandboxed",
                        "network": "model_api_and_eval_server_host_shared",
                    }),
                    adapter,
                    sandbox=True,
                )

    def test_unknown_network_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "access_policy.network must be one of"):
            _validate_access_policy(
                OmegaConf.create({"execution": "unsandboxed", "network": "model_api_and_eval_server"}),
                ClaudeCodeAdapter(sandbox=True),
                sandbox=True,
            )

    def test_unknown_execution_policy_is_rejected(self) -> None:
        for execution in ("no_execution", "candidate_sandboxed", "filesystem_sandboxed"):
            with self.subTest(execution=execution):
                with self.assertRaisesRegex(ValueError, "access_policy.execution must be one of"):
                    _validate_access_policy(
                        OmegaConf.create({"execution": execution, "network": "host_shared"}),
                        ClaudeCodeAdapter(sandbox=False),
                        sandbox=False,
                    )

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

    def test_meta_harness_transcript_copy_uses_isolated_claude_home(self) -> None:
        session_id = "session-123"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work_dir = root / "work"
            claude_home = root / "home"
            output_dir = root / "out"
            src_dir = claude_home / ".claude" / "projects" / meta_claude_project_slug(work_dir)
            src_dir.mkdir(parents=True)
            (src_dir / f"{session_id}.jsonl").write_text('{"ok": true}\n')

            copy_meta_session_transcript(
                work_dir,
                session_id,
                output_dir,
                claude_home=claude_home,
            )

            self.assertEqual((output_dir / f"{session_id}.jsonl").read_text(), '{"ok": true}\n')

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
