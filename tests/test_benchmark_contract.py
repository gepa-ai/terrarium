from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

from terrarium.adapters.claude_code import ClaudeCodeAdapter, materialize_sandbox as materialize_claude_sandbox
from terrarium.adapters.meta_harness import (
    MetaHarnessAdapter,
    _claude_project_slug as meta_claude_project_slug,
    _copy_session_transcript as copy_meta_session_transcript,
    _materialize_sandbox as materialize_meta_sandbox,
)
from terrarium.adapters.gepa import GEPAAdapter
from terrarium.adapters.optimize_anything_adapter import OptimizeAnythingAdapter
from terrarium.budget import BudgetTracker
from terrarium.eval_server import EvalServer
from terrarium.adapter import Result
from terrarium.registry import get_task, list_tasks
from terrarium.runner import (
    _effective_adapter_sandbox,
    _apply_task_runtime_config,
    _evaluate_heldout_test,
    _prepare_task_for_benchmark,
    _sandbox_scope,
    _validate_access_policy,
    run,
)
from terrarium.task import Example, Task
from terrarium.tasks.arc_agi import _evaluate_test
from terrarium.tasks import aime_math
from terrarium.tasks import frontier_cs


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
                OmegaConf.create({"mode": "generalization", "split_train_val": True}),
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
                OmegaConf.create({"mode": "generalization", "split_train_val": False}),
            )

    def test_split_train_val_false_merges_val_into_train(self) -> None:
        task = _task(
            train=[Example("train", {}, 1.0)],
            val=[Example("val", {}, 1.0)],
            test=[Example("test", {}, 1.0)],
        )

        prepared = _prepare_task_for_benchmark(
            task,
            OmegaConf.create({"mode": "generalization", "split_train_val": False}),
        )

        self.assertIsNone(prepared.val_set)
        self.assertEqual([ex.id for ex in prepared.train_set or []], ["train", "val"])
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

    def test_run_respects_benchmark_split_train_val_false(self) -> None:
        seen: dict[str, object] = {}

        class CaptureAdapter:
            def evolve(self, task: Task, server: EvalServer) -> Result:
                seen["adapter_train_ids"] = [ex.id for ex in task.train_set or []]
                seen["adapter_val_set"] = task.val_set
                seen["server_train_ids"] = [ex.id for ex in server.task.train_set or []]
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
            benchmark={"mode": "generalization", "split_train_val": False},
        )

        self.assertEqual(seen["adapter_train_ids"], ["train", "val"])
        self.assertIsNone(seen["adapter_val_set"])
        self.assertEqual(seen["server_train_ids"], ["train", "val"])
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

    def test_heldout_test_eval_uses_separate_parallel_budget(self) -> None:
        calls: list[str] = []

        def eval_fn(candidate: str, example: Example) -> tuple[float, dict]:
            del candidate
            calls.append(example.id)
            return float(example.expected or 0.0), {"cost": 0.25}

        task = Task(
            name="heldout",
            initial_candidate="seed",
            eval_fn=eval_fn,
            test_set=[Example("a", {}, 1.0), Example("b", {}, 0.0)],
            metadata={"type": "generalization"},
        )

        scores, cost, errors = _evaluate_heldout_test(task, "candidate", max_concurrency=2)

        self.assertEqual(scores, {"a": 1.0, "b": 0.0})
        self.assertEqual(cost, 0.5)
        self.assertEqual(errors, {})
        self.assertCountEqual(calls, ["a", "b"])

    def test_run_reports_optimizer_budget_status(self) -> None:
        class CostAdapter:
            def evolve(self, task: Task, server: EvalServer) -> Result:
                del task, server
                return Result(
                    best_candidate="seed",
                    best_score=0.0,
                    metadata={"adapter_cost": 0.75},
                )

        task = _task(train=[Example("train", {}, 1.0)], test=[Example("test", {}, 1.0)])

        result = run(task, CostAdapter(), max_evals=10, max_token_cost=0.5, max_concurrency=1)

        self.assertEqual(result.metadata["budget"]["optimizer_cost_used"], 0.75)
        self.assertTrue(result.metadata["budget"]["optimizer_budget_exhausted"])

    def test_optimize_anything_adapter_delegates_train_val_policy_to_gepa(self) -> None:
        task = _task(
            train=[Example("train", {}, 1.0)],
            val=[Example("val", {}, 2.0)],
            test=[Example("test", {}, 3.0)],
        )

        oa_task = OptimizeAnythingAdapter(engine="meta_harness")._to_oa_task(task)

        self.assertIs(oa_task.train_set, task.train_set)
        self.assertIs(oa_task.val_set, task.val_set)
        # Held-out test set is sealed off from the optimize_anything Task; the
        # terrarium runner owns post-search test reporting.
        self.assertIsNone(oa_task.test_set)

    def test_optimize_anything_adapter_translates_budget_exhaustion(self) -> None:
        from gepa.oa.budget import BudgetExhausted as GepaBudgetExhausted

        task = _task(mode="single")
        server = EvalServer(task, BudgetTracker(max_evals=1), max_concurrency=1)
        adapter = OptimizeAnythingAdapter(engine="gepa")

        def fake_optimize(oa_task, evaluate, cfg):
            del oa_task, cfg
            evaluate("first")
            with self.assertRaises(GepaBudgetExhausted):
                evaluate("second")
            return SimpleNamespace(best_candidate="first", best_score=1.0, metadata={})

        with patch("gepa.optimize_anything.optimize_anything_from_task", side_effect=fake_optimize):
            result = adapter.evolve(task, server)

        self.assertEqual(result.best_candidate, "first")

    def test_gepa_engine_recovers_saved_result_on_budget_exhaustion(self) -> None:
        from gepa.optimize_anything import OptimizeAnythingConfig, Task as OATask
        from gepa.oa.engines.gepa import GepaEngine
        from gepa.oa.budget import BudgetExhausted as GepaBudgetExhausted

        class FakeReflectionLM:
            total_cost = 1.25
            total_tokens_in = 100
            total_tokens_out = 20

        with tempfile.TemporaryDirectory() as tmp:
            engine = GepaEngine(
                OptimizeAnythingConfig(
                    engine="gepa",
                    run_dir=tmp,
                    engine_config={"reflection": {"reflection_lm": FakeReflectionLM()}},
                )
            )
            task = OATask(
                name="unit",
                seed_candidate="seed",
                objective="objective",
                train_set=[Example("train", {}, 1.0)],
                val_set=[Example("val", {}, 1.0)],
            )
            server = SimpleNamespace(
                budget=SimpleNamespace(max_evals=10, max_token_cost=None, used=10),
                tracker=None,
                eval_log=[],
                best_candidate="seed",
                best_score=0.0,
            )
            recovered = SimpleNamespace(
                best_candidate="validated-best",
                best_idx=0,
                val_aggregate_scores=[0.75],
            )

            with (
                patch("gepa.gepa_launcher.optimize_anything", side_effect=GepaBudgetExhausted("done")),
                patch.object(engine, "_load_result_from_state", return_value=recovered),
            ):
                result = engine.run(task, server)

        self.assertEqual(result.best_candidate, "validated-best")
        self.assertEqual(result.best_score, 0.75)
        # adapter_cost is the cost source's total_cost (treated as fresh).
        self.assertEqual(result.metadata["adapter_cost"], 1.25)

    def test_gepa_claude_code_agent_defaults_to_token_budget(self) -> None:
        # gepa's GepaEngine no longer owns a claude_code_agent convenience key
        # (proposer construction is the caller's job). The terrarium adapter
        # translates the claude_code_agent engine_config block into a
        # ClaudeCodeAgentProposer, defaulting its budget to max_token_cost.
        from terrarium.adapters.optimize_anything_adapter import OptimizeAnythingAdapter

        adapter = OptimizeAnythingAdapter(engine="gepa", run_dir="/tmp/gepa-cc-agent")
        merged = adapter._build_engine_config(
            "gepa",
            {"claude_code_agent": {"model": "sonnet"}},
            run_dir="/tmp/gepa-cc-agent",
            objective="objective",
            background="background",
            max_token_cost=0.5,
            effort=None,
            max_thinking_tokens=None,
            sandbox=None,
        )
        proposer = merged["reflection"]["custom_candidate_proposer"]
        self.assertEqual(proposer.max_budget_usd, 0.5)
        self.assertEqual(proposer.model, "sonnet")

    def test_meta_harness_frontier_has_terminal_bench_best_alias(self) -> None:
        from gepa.oa.engines.meta_harness import _update_frontier

        with tempfile.TemporaryDirectory() as tmp:
            frontier = Path(tmp) / "frontier.json"
            improved = _update_frontier(
                frontier,
                name="candidate",
                file="agents/iter1_candidate.txt",
                score=0.75,
                per_example_scores={"ex1": 1.0, "ex2": 0.0},
            )
            data = json.loads(frontier.read_text())

        self.assertTrue(improved)
        self.assertEqual(data["best_name"], "candidate")
        self.assertEqual(data["_best"]["agent"], "candidate")
        self.assertEqual(data["_best"]["avg_pass_rate"], 0.75)
        self.assertEqual(data["_best"]["candidate_file"], "agents/iter1_candidate.txt")
        self.assertEqual(data["per_example"]["ex1"]["best_name"], "candidate")

    def test_optimize_anything_sequential_rich_handoff_passes_manifest_paths_only(self) -> None:
        task = _task(mode="single")
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            server = EvalServer(task, BudgetTracker(max_evals=10), max_concurrency=1, output_dir=output_dir)
            adapter = OptimizeAnythingAdapter(
                strategy="sequential",
                configs=[{"engine": "gepa"}, {"engine": "autoresearch"}],
                run_dir=str(output_dir / "optimize_anything"),
                handoff={"mode": "rich", "max_evals": 5},
            )
            seen: dict[str, object] = {}

            def fake_optimize(oa_task, evaluate, cfg):
                if oa_task.initial_candidate == "seed":
                    evaluate("stage0-candidate")
                    return SimpleNamespace(best_candidate="stage0-best", best_score=0.4, metadata={})
                seen["initial_candidate"] = oa_task.initial_candidate
                seen["handoffs"] = cfg.engine_config.get("handoffs")
                evaluate("stage1-candidate")
                return SimpleNamespace(best_candidate="stage1-best", best_score=0.5, metadata={})

            with patch("gepa.optimize_anything.optimize_anything_from_task", side_effect=fake_optimize):
                result = adapter.evolve(task, server)

            self.assertEqual(seen["initial_candidate"], "stage0-best")
            handoffs = seen["handoffs"]  # type: ignore[index]
            self.assertEqual(len(handoffs), 1)
            manifest = handoffs[0]
            self.assertNotIn("evals", manifest)
            self.assertTrue(Path(manifest["summary_path"]).exists())
            self.assertTrue(Path(manifest["best_candidate_path"]).exists())
            eval_trace_dir = Path(manifest["eval_trace_dir"])
            self.assertTrue((eval_trace_dir / "0.json").exists())
            self.assertEqual(json.loads((eval_trace_dir / "0.json").read_text())["candidate"], "stage0-candidate")
            self.assertEqual(result.metadata["optimize_anything_handoffs"][0]["engine"], "gepa")

    def test_optimize_anything_sequential_entries_can_choose_split_policy(self) -> None:
        raw_task = _task(
            train=[Example("train", {}, 1.0)],
            val=[Example("val", {}, 1.0)],
            test=[Example("test", {}, 1.0)],
        )
        task = _prepare_task_for_benchmark(
            raw_task,
            OmegaConf.create({"mode": "generalization", "split_train_val": False}),
        )
        adapter = OptimizeAnythingAdapter(
            strategy="sequential",
            configs=[
                {"engine": "gepa", "split_train_val": True},
                {"engine": "autoresearch", "split_train_val": False},
            ],
        )
        server = EvalServer(task, BudgetTracker(max_evals=10), max_concurrency=1)
        seen: list[tuple[int, int]] = []

        def fake_optimize(oa_task, evaluate, cfg):
            del evaluate, cfg
            seen.append((len(oa_task.train_set or []), len(oa_task.val_set or [])))
            return SimpleNamespace(best_candidate=f"stage-{len(seen)}", best_score=float(len(seen)), metadata={})

        with patch("gepa.optimize_anything.optimize_anything_from_task", side_effect=fake_optimize):
            adapter.evolve(task, server)

        self.assertEqual(seen, [(1, 1), (2, 0)])

    def test_optimize_anything_sequential_stops_before_next_stage_when_budget_exhausted(self) -> None:
        task = _task(mode="single")
        adapter = OptimizeAnythingAdapter(
            strategy="sequential",
            configs=[{"engine": "gepa"}, {"engine": "autoresearch"}],
        )
        server = EvalServer(task, BudgetTracker(max_evals=1), max_concurrency=1)
        calls: list[str] = []

        def fake_optimize(oa_task, evaluate, cfg):
            del cfg
            calls.append(str(oa_task.initial_candidate))
            evaluate("stage0")
            return SimpleNamespace(best_candidate="stage0-best", best_score=1.0, metadata={})

        with patch("gepa.optimize_anything.optimize_anything_from_task", side_effect=fake_optimize):
            result = adapter.evolve(task, server)

        self.assertEqual(calls, ["seed"])
        self.assertEqual(result.best_candidate, "stage0-best")
        self.assertEqual(result.total_evals, 1)

    def test_optimize_anything_sequential_sums_stage_adapter_costs(self) -> None:
        task = _task(mode="single")
        adapter = OptimizeAnythingAdapter(
            strategy="sequential",
            configs=[{"engine": "autoresearch"}, {"engine": "gepa"}],
        )
        server = EvalServer(task, BudgetTracker(max_evals=10), max_concurrency=1)

        def fake_optimize(oa_task, evaluate, cfg):
            del evaluate, cfg
            if oa_task.initial_candidate == "seed":
                return SimpleNamespace(
                    best_candidate="stage0",
                    best_score=0.4,
                    metadata={"adapter_cost": 1.25},
                )
            return SimpleNamespace(
                best_candidate="stage1",
                best_score=0.7,
                metadata={"adapter_cost": 2.5},
            )

        with patch("gepa.optimize_anything.optimize_anything_from_task", side_effect=fake_optimize):
            result = adapter.evolve(task, server)

        self.assertEqual(result.metadata["adapter_cost"], 3.75)
        self.assertEqual(
            result.metadata["stage_adapter_costs"],
            [
                {"idx": 0, "engine": "autoresearch", "adapter_cost": 1.25, "best_score": 0.4},
                {"idx": 1, "engine": "gepa", "adapter_cost": 2.5, "best_score": 0.7},
            ],
        )

    def test_optimize_anything_adaptive_sequential_switches_on_plateau_and_returns_global_best(self) -> None:
        task = _task(mode="single")
        adapter = OptimizeAnythingAdapter(
            strategy="adaptive_sequential",
            configs=[{"engine": "gepa"}, {"engine": "autoresearch"}],
            scheduler={"plateau_evals": 1, "patience": 1, "min_evals_per_stage": 1, "cycle": False},
        )
        server = EvalServer(task, BudgetTracker(max_evals=3), max_concurrency=1)
        calls: list[tuple[str, str]] = []

        def fake_optimize(oa_task, evaluate, cfg):
            engine = cfg.engine
            calls.append((str(engine), oa_task.initial_candidate))
            evaluate(f"{engine}-candidate")
            if len(calls) <= 2:
                return SimpleNamespace(best_candidate="gepa-best", best_score=0.8, metadata={"adapter_cost": 1.0})
            return SimpleNamespace(best_candidate="regressed", best_score=0.2, metadata={"adapter_cost": 2.0})

        with patch("gepa.optimize_anything.optimize_anything_from_task", side_effect=fake_optimize):
            result = adapter.evolve(task, server)

        self.assertEqual(calls, [("gepa", "seed"), ("gepa", "gepa-best"), ("autoresearch", "gepa-best")])
        self.assertEqual(result.best_candidate, "gepa-best")
        self.assertEqual(result.best_score, 0.8)
        self.assertEqual(result.metadata["adapter_cost"], 4.0)
        self.assertEqual(result.metadata["adaptive_switches"], 1)
        self.assertEqual(
            [(row["engine"], row["improved"]) for row in result.metadata["adaptive_schedule"]],
            [("gepa", True), ("gepa", False), ("autoresearch", False)],
        )

    def test_optimize_anything_adaptive_sequential_reuses_budget_without_equal_prepartition(self) -> None:
        task = _task(mode="single")
        adapter = OptimizeAnythingAdapter(
            strategy="adaptive_sequential",
            configs=[{"engine": "gepa"}, {"engine": "autoresearch"}],
            scheduler={"plateau_evals": 1, "patience": 1, "min_evals_per_stage": 1, "cycle": False},
        )
        server = EvalServer(task, BudgetTracker(max_evals=5), max_concurrency=1)
        calls: list[str] = []

        def fake_optimize(oa_task, evaluate, cfg):
            del oa_task
            calls.append(str(cfg.engine))
            evaluate(f"{cfg.engine}-{len(calls)}")
            if len(calls) <= 2:
                return SimpleNamespace(best_candidate="gepa-best", best_score=0.4, metadata={})
            return SimpleNamespace(best_candidate="cc-best", best_score=0.6, metadata={})

        with patch("gepa.optimize_anything.optimize_anything_from_task", side_effect=fake_optimize):
            result = adapter.evolve(task, server)

        self.assertEqual(calls, ["gepa", "gepa", "autoresearch", "autoresearch"])
        self.assertEqual(result.total_evals, 4)
        self.assertEqual(result.best_candidate, "cc-best")
        self.assertEqual(result.best_score, 0.6)

    def test_optimize_anything_adaptive_sequential_does_not_start_low_budget_tail_slice(self) -> None:
        task = _task(mode="single")
        adapter = OptimizeAnythingAdapter(
            strategy="adaptive_sequential",
            configs=[{"engine": "gepa"}, {"engine": "autoresearch"}],
            scheduler={"plateau_evals": 2, "patience": 1, "min_evals_per_stage": 2, "cycle": True},
        )
        server = EvalServer(task, BudgetTracker(max_evals=3), max_concurrency=1)
        calls: list[str] = []

        def fake_optimize(oa_task, evaluate, cfg):
            del oa_task
            calls.append(str(cfg.engine))
            evaluate("candidate-a")
            evaluate("candidate-b")
            return SimpleNamespace(best_candidate="best", best_score=0.1, metadata={})

        with patch("gepa.optimize_anything.optimize_anything_from_task", side_effect=fake_optimize):
            result = adapter.evolve(task, server)

        self.assertEqual(calls, ["gepa"])
        self.assertEqual(result.total_evals, 2)
        self.assertEqual(result.metadata["adaptive_stop_reason"], "scheduler_stopped")

    def test_optimize_anything_adaptive_sequential_stops_on_token_cost_cap(self) -> None:
        task = _task(mode="single")
        adapter = OptimizeAnythingAdapter(
            strategy="adaptive_sequential",
            configs=[{"engine": "gepa"}, {"engine": "autoresearch"}],
            scheduler={"plateau_evals": 1, "patience": 1, "min_evals_per_stage": 1, "cycle": True},
        )
        server = EvalServer(task, BudgetTracker(max_evals=10, max_token_cost=1.0), max_concurrency=1)
        calls: list[str] = []

        def fake_optimize(oa_task, evaluate, cfg):
            del oa_task
            calls.append(str(cfg.engine))
            evaluate(f"{cfg.engine}-candidate")
            return SimpleNamespace(best_candidate="best", best_score=0.1, metadata={"adapter_cost": 1.5})

        with patch("gepa.optimize_anything.optimize_anything_from_task", side_effect=fake_optimize):
            result = adapter.evolve(task, server)

        self.assertEqual(calls, ["gepa"])
        self.assertEqual(result.metadata["adapter_cost"], 1.5)
        self.assertEqual(result.metadata["adaptive_schedule"][0]["engine"], "gepa")

    def test_optimize_anything_adaptive_sequential_uses_engine_aggregate_not_per_example_server_best(self) -> None:
        examples = [Example("a", {}, 0.0), Example("b", {}, 0.0)]
        task = Task(
            name="dataset",
            initial_candidate="seed",
            eval_fn=lambda candidate, example: (1.0 if candidate == "spiky" and example.id == "a" else 0.0, {}),
            train_set=examples,
            metadata={"type": "generalization"},
        )
        adapter = OptimizeAnythingAdapter(
            strategy="adaptive_sequential",
            configs=[{"engine": "gepa"}, {"engine": "autoresearch"}],
            scheduler={"plateau_evals": 2, "patience": 1, "min_evals_per_stage": 2, "cycle": False},
        )
        server = EvalServer(task, BudgetTracker(max_evals=4), max_concurrency=1)

        def fake_optimize(oa_task, evaluate, cfg):
            if cfg.engine == "gepa":
                evaluate("spiky", examples[0])
                evaluate("spiky", examples[1])
                return SimpleNamespace(best_candidate="aggregate-best", best_score=0.5, metadata={})
            self.assertEqual(oa_task.initial_candidate, "aggregate-best")
            evaluate("next", examples[0])
            return SimpleNamespace(best_candidate="next", best_score=0.1, metadata={})

        with patch("gepa.optimize_anything.optimize_anything_from_task", side_effect=fake_optimize):
            result = adapter.evolve(task, server)

        self.assertEqual(server.best_candidate, "spiky")
        self.assertEqual(server.best_score, 1.0)
        self.assertEqual(result.best_candidate, "aggregate-best")
        self.assertEqual(result.best_score, 0.5)

    def test_optimize_anything_parallel_entries_can_choose_split_policy(self) -> None:
        raw_task = _task(
            train=[Example("train", {}, 1.0)],
            val=[Example("val", {}, 1.0)],
            test=[Example("test", {}, 1.0)],
        )
        task = _prepare_task_for_benchmark(
            raw_task,
            OmegaConf.create({"mode": "generalization", "split_train_val": False}),
        )
        adapter = OptimizeAnythingAdapter(
            strategy="best_of",
            configs=[
                {"engine": "gepa", "split_train_val": True},
                {"engine": "meta_harness", "split_train_val": False},
            ],
            max_workers=1,
        )
        server = EvalServer(task, BudgetTracker(max_evals=10), max_concurrency=1)
        seen: list[tuple[int, int]] = []

        def fake_optimize(oa_task, evaluate, cfg):
            del evaluate, cfg
            seen.append((len(oa_task.train_set or []), len(oa_task.val_set or [])))
            return SimpleNamespace(best_candidate=f"stage-{len(seen)}", best_score=float(len(seen)), metadata={})

        with patch("gepa.optimize_anything.optimize_anything_from_task", side_effect=fake_optimize):
            adapter.evolve(task, server)

        self.assertEqual(seen, [(1, 1), (2, 0)])

    def test_optimize_anything_best_of_sums_member_adapter_costs(self) -> None:
        task = _task(mode="single")
        adapter = OptimizeAnythingAdapter(
            strategy="best_of",
            configs=[{"engine": "autoresearch"}, {"engine": "gepa"}],
            max_workers=1,
        )
        server = EvalServer(task, BudgetTracker(max_evals=10), max_concurrency=1)
        calls = 0

        def fake_optimize(oa_task, evaluate, cfg):
            del oa_task, evaluate, cfg
            nonlocal calls
            calls += 1
            if calls == 1:
                return SimpleNamespace(
                    best_candidate="member0",
                    best_score=0.4,
                    metadata={"adapter_cost": 1.25},
                )
            return SimpleNamespace(
                best_candidate="member1",
                best_score=0.7,
                metadata={"adapter_cost": 2.5},
            )

        with patch("gepa.optimize_anything.optimize_anything_from_task", side_effect=fake_optimize):
            result = adapter.evolve(task, server)

        self.assertEqual(result.best_candidate, "member1")
        self.assertEqual(result.metadata["adapter_cost"], 3.75)
        self.assertEqual(
            result.metadata["member_adapter_costs"],
            [
                {"idx": 0, "engine": "autoresearch", "adapter_cost": 1.25, "best_score": 0.4},
                {"idx": 1, "engine": "gepa", "adapter_cost": 2.5, "best_score": 0.7},
            ],
        )

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
            OmegaConf.create({"mode": "multi_task", "split_train_val": True}),
        )

        self.assertEqual(prepared.metadata["type"], "multi_task")
        self.assertIs(prepared.train_set, prepared.val_set)
        self.assertIsNone(prepared.test_set)

    def test_legacy_single_task_mode_normalizes_to_single(self) -> None:
        task = _task(mode="single_task")

        prepared = _prepare_task_for_benchmark(
            task,
            OmegaConf.create({"mode": None, "split_train_val": True}),
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

    def test_eval_server_http_rejects_val_split_after_split_train_val_false(self) -> None:
        task = _task(
            train=[Example("train", {}, 1.0)],
            val=[Example("val", {}, 2.0)],
            test=[Example("test", {}, 3.0)],
        )
        prepared = _prepare_task_for_benchmark(
            task,
            OmegaConf.create({"mode": "generalization", "split_train_val": False}),
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
            train=[Example("train", {"x": 1, "solution": "train-sol"}, 1.0)],
            val=[Example("val", {"x": 2, "solution": "val-sol"}, 2.0)],
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
            train_payload = json.loads((work_dir / "train" / "train.json").read_text())
            val_payload = json.loads((work_dir / "val" / "val.json").read_text())
            self.assertEqual(train_payload["expected"], 1.0)
            self.assertEqual(train_payload["inputs"]["solution"], "train-sol")
            self.assertEqual(val_payload["expected"], 2.0)
            self.assertEqual(val_payload["inputs"]["solution"], "val-sol")

    def test_meta_harness_materialization_writes_visible_splits_not_test(self) -> None:
        task = _task(
            train=[Example("train", {"x": 1, "solution": "train-sol"}, 1.0)],
            val=[Example("val", {"x": 2, "solution": "val-sol"}, 2.0)],
            test=[Example("test", {"x": 3}, 3.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            materialize_meta_sandbox(work_dir, task, BudgetTracker(max_evals=10))

            self.assertTrue((work_dir / "train" / "train.json").exists())
            self.assertTrue((work_dir / "val" / "val.json").exists())
            self.assertFalse((work_dir / "test").exists())
            train_payload = json.loads((work_dir / "train" / "train.json").read_text())
            val_payload = json.loads((work_dir / "val" / "val.json").read_text())
            self.assertEqual(train_payload["expected"], 1.0)
            self.assertEqual(train_payload["inputs"]["solution"], "train-sol")
            self.assertEqual(val_payload["expected"], 2.0)
            self.assertEqual(val_payload["inputs"]["solution"], "val-sol")

    def test_eval_server_tracks_aggregate_visible_selection(self) -> None:
        task = _task(
            train=[Example("train", {}, 1.0)],
            val=[Example("val", {}, 0.5)],
            test=[Example("test", {}, 1.0)],
        )
        server = EvalServer(task, BudgetTracker(max_evals=3))
        train_score, _ = server.evaluate_examples("train-candidate", split="train")
        val_result = server.validate("val-candidate")

        self.assertEqual(train_score, 1.0)
        self.assertEqual(val_result["val_score"], 0.5)
        self.assertEqual(server.best_visible_candidate, "train-candidate")
        self.assertEqual(server.best_visible_source, "train_aggregate")
        self.assertEqual(server.best_validated_candidate, "val-candidate")
        self.assertEqual(server.best_validated_score, 0.5)

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

    def test_claude_code_subprocess_failure_raises(self) -> None:
        task = _task(train=[Example("train", {}, 1.0)], test=[Example("test", {}, 1.0)])
        server = EvalServer(task, BudgetTracker(max_evals=1))
        server.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                adapter = ClaudeCodeAdapter(run_dir=tmp, sandbox=False)
                failed = subprocess.CompletedProcess(
                    args=["claude"],
                    returncode=1,
                    stdout='{"error":"Not logged in"}',
                    stderr="",
                )
                with patch("terrarium.adapters.claude_code.subprocess.run", return_value=failed):
                    with self.assertRaisesRegex(RuntimeError, "Claude Code subprocess failed"):
                        adapter.evolve(task, server)
        finally:
            server.stop()

    def test_claude_code_rejects_unsupported_max_turns(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_turns is not supported"):
            ClaudeCodeAdapter(max_turns=1)

    def test_claude_code_zero_eval_success_raises(self) -> None:
        task = _task(train=[Example("train", {}, 1.0)], test=[Example("test", {}, 1.0)])
        server = EvalServer(task, BudgetTracker(max_evals=1))
        server.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                adapter = ClaudeCodeAdapter(run_dir=tmp, sandbox=False)
                completed = subprocess.CompletedProcess(
                    args=["claude"],
                    returncode=0,
                    stdout='{"total_cost_usd":0.01}',
                    stderr="",
                )
                with patch("terrarium.adapters.claude_code.subprocess.run", return_value=completed):
                    with self.assertRaisesRegex(RuntimeError, "without calling"):
                        adapter.evolve(task, server)
        finally:
            server.stop()

    def test_claude_code_submits_best_validated_candidate_not_last_file(self) -> None:
        def eval_candidate(candidate: str, example: Example | None = None) -> tuple[float, dict]:
            del example
            return (1.0 if candidate == "validated" else 0.0), {}

        task = Task(
            name="task",
            initial_candidate="seed",
            eval_fn=eval_candidate,
            train_set=[Example("train", {}, 1.0)],
            val_set=[Example("val", {}, 1.0)],
            test_set=[Example("test", {}, 1.0)],
            metadata={"type": "generalization"},
        )
        server = EvalServer(task, BudgetTracker(max_evals=3))
        server.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                adapter = ClaudeCodeAdapter(run_dir=tmp, sandbox=False)

                def fake_run(cmd, **kwargs):
                    work_dir = Path(kwargs["cwd"])
                    (work_dir / "best_candidate.txt").write_text("unvalidated-last-edit")
                    server.validate("validated")
                    return subprocess.CompletedProcess(
                        args=cmd,
                        returncode=0,
                        stdout='{"total_cost_usd":0.01}',
                        stderr="",
                    )

                with patch("terrarium.adapters.claude_code.subprocess.run", side_effect=fake_run):
                    result = adapter.evolve(task, server)
        finally:
            server.stop()

        self.assertEqual(result.best_candidate, "validated")
        self.assertEqual(result.best_score, 1.0)
        self.assertEqual(result.metadata["submitted_candidate_source"], "best_validated_candidate")
        self.assertEqual(result.metadata["best_candidate_txt"], "unvalidated-last-edit")

    def test_claude_code_ralph_continue_prompt_omits_validate_without_val_set(self) -> None:
        task = _task(train=[Example("train", {}, 1.0)], test=[Example("test", {}, 1.0)])
        server = EvalServer(task, BudgetTracker(max_evals=2, max_token_cost=1.0))
        prompts: list[str] = []
        server.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                adapter = ClaudeCodeAdapter(
                    run_dir=tmp,
                    sandbox=False,
                )

                def fake_run(cmd, **kwargs):
                    del kwargs
                    prompts.append(cmd[-1])
                    server.budget.record(1.0)
                    return subprocess.CompletedProcess(
                        args=cmd,
                        returncode=0,
                        stdout='{"total_cost_usd":0.01}',
                        stderr="",
                    )

                with patch("terrarium.adapters.claude_code.subprocess.run", side_effect=fake_run):
                    adapter.evolve(task, server)
        finally:
            server.stop()

        self.assertEqual(len(prompts), 2)
        self.assertIn("Run ./eval.sh as appropriate.", prompts[1])
        self.assertNotIn("validate.sh", prompts[1])

    def test_claude_code_ralph_continue_prompt_mentions_validate_with_val_set(self) -> None:
        task = _task(
            train=[Example("train", {}, 1.0)],
            val=[Example("val", {}, 1.0)],
            test=[Example("test", {}, 1.0)],
        )
        server = EvalServer(task, BudgetTracker(max_evals=2, max_token_cost=1.0))
        prompts: list[str] = []
        server.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                adapter = ClaudeCodeAdapter(
                    run_dir=tmp,
                    sandbox=False,
                )

                def fake_run(cmd, **kwargs):
                    del kwargs
                    prompts.append(cmd[-1])
                    server.budget.record(1.0)
                    return subprocess.CompletedProcess(
                        args=cmd,
                        returncode=0,
                        stdout='{"total_cost_usd":0.01}',
                        stderr="",
                    )

                with patch("terrarium.adapters.claude_code.subprocess.run", side_effect=fake_run):
                    adapter.evolve(task, server)
        finally:
            server.stop()

        self.assertEqual(len(prompts), 2)
        self.assertIn("Run ./eval.sh and ./validate.sh as appropriate.", prompts[1])

    def test_meta_harness_proposer_failure_raises(self) -> None:
        task = _task(train=[Example("train", {}, 1.0)], test=[Example("test", {}, 1.0)])
        server = EvalServer(task, BudgetTracker(max_evals=1))

        with tempfile.TemporaryDirectory() as tmp:
            adapter = MetaHarnessAdapter(
                run_dir=tmp,
                max_iterations=1,
                max_candidates_per_iter=1,
                sandbox=False,
            )
            with patch(
                "terrarium.adapters.meta_harness._run_proposer",
                return_value=(1, 0.0, "session-123", None),
            ):
                with self.assertRaisesRegex(RuntimeError, "MetaHarness proposer failed"):
                    adapter.evolve(task, server)

    def test_meta_harness_no_candidates_raises(self) -> None:
        task = _task(train=[Example("train", {}, 1.0)], test=[Example("test", {}, 1.0)])
        server = EvalServer(task, BudgetTracker(max_evals=1))

        with tempfile.TemporaryDirectory() as tmp:
            adapter = MetaHarnessAdapter(
                run_dir=tmp,
                max_iterations=1,
                max_candidates_per_iter=1,
                sandbox=False,
            )
            with patch(
                "terrarium.adapters.meta_harness._run_proposer",
                return_value=(0, 0.0, "session-123", None),
            ):
                with self.assertRaisesRegex(RuntimeError, "produced no candidates"):
                    adapter.evolve(task, server)

    def test_meta_harness_uses_validation_as_search_signal_with_val_traces(self) -> None:
        def eval_candidate(candidate: str, example: Example | None = None) -> tuple[float, dict]:
            assert example is not None
            if example.id == "train":
                return (0.25 if candidate == "candidate" else 0.0), {"split": "train"}
            return (1.0 if candidate == "candidate" else 0.0), {"split": "val"}

        task = Task(
            name="task",
            initial_candidate="seed",
            eval_fn=eval_candidate,
            train_set=[Example("train", {}, 0.25)],
            val_set=[Example("val", {}, 1.0)],
            test_set=[Example("test", {}, 1.0)],
            metadata={"type": "generalization"},
        )
        with tempfile.TemporaryDirectory() as out_tmp, tempfile.TemporaryDirectory() as tmp:
            server = EvalServer(task, BudgetTracker(max_evals=4), output_dir=out_tmp)
            try:
                work_dir = Path(tmp)
                adapter = MetaHarnessAdapter(
                    run_dir=str(work_dir),
                    max_iterations=1,
                    max_candidates_per_iter=1,
                    sandbox=False,
                )

                def fake_proposer(**kwargs):
                    wd = kwargs["work_dir"]
                    (wd / "agents" / "iter1_candidate.txt").write_text("candidate")
                    kwargs["pending_path"].write_text(json.dumps({
                        "iteration": 1,
                        "candidates": [{
                            "name": "candidate",
                            "file": "agents/iter1_candidate.txt",
                            "hypothesis": "test",
                            "axis": "test",
                            "components": [],
                        }],
                    }))
                    return (0, 0.0, "session-123", None)

                with patch("terrarium.adapters.meta_harness._run_proposer", side_effect=fake_proposer):
                    result = adapter.evolve(task, server)

                trace_files = list((work_dir / "state" / "eval_traces" / "candidate").glob("*.json"))
                trace_payload = json.loads(trace_files[0].read_text())
            finally:
                server.stop()

        self.assertEqual(result.best_candidate, "candidate")
        self.assertEqual(result.best_score, 1.0)
        self.assertEqual(result.metadata["submitted_candidate_source"], "frontier_val")
        self.assertEqual(len(trace_files), 1)
        self.assertEqual(trace_payload["info"]["split"], "val")

    def test_arc_test_scoring_accepts_grid_predictions_and_attempt_lists(self) -> None:
        gold = [[[1, 2], [3, 4]]]

        score, results = _evaluate_test([[[1, 2], [3, 4]]], gold)
        self.assertEqual(score, 1.0)
        self.assertTrue(results[0]["correct"])

        score, results = _evaluate_test([[[[9]], [[1, 2], [3, 4]]]], gold)
        self.assertEqual(score, 1.0)
        self.assertTrue(results[0]["correct"])

    def test_aime_solver_parse_error_scores_zero(self) -> None:
        from dspy.utils.exceptions import AdapterParseError

        class FakeSignature:
            output_fields = {"reasoning": object(), "answer": object()}

        class FakePredict:
            def __init__(self) -> None:
                self.signature = type("SignatureConfig", (), {"instructions": ""})()

        class FakeChainOfThought:
            def __init__(self, _signature) -> None:
                self.predict = FakePredict()

            def __call__(self, **_kwargs):
                raise AdapterParseError("JSONAdapter", FakeSignature(), "{0, 1, 2}")

        example = Example("aime", {"input": "What is 1+1?", "solution": "It is 2."}, "2")

        with patch("dspy.ChainOfThought", FakeChainOfThought):
            score, info = aime_math.evaluate("answer with an integer", example)

        self.assertEqual(score, 0.0)
        self.assertEqual(info["score"], 0.0)
        self.assertEqual(info["error"], "solver_parse_error")
        self.assertIn("{0, 1, 2}", info["output"])

    def test_aime_runtime_config_threads_isolated_solver_settings(self) -> None:
        calls: dict[str, object] = {}

        def fake_eval(
            candidate: str,
            example: Example,
            *,
            solver_lm: str | None,
            solver_temperature: float | None,
            solver_max_tokens: int | None,
            solver_timeout: float | None,
            solver_num_retries: int | None,
        ) -> tuple[float, dict]:
            calls["candidate"] = candidate
            calls["example"] = example
            calls["solver_lm"] = solver_lm
            calls["solver_temperature"] = solver_temperature
            calls["solver_max_tokens"] = solver_max_tokens
            calls["solver_timeout"] = solver_timeout
            calls["solver_num_retries"] = solver_num_retries
            return 1.0, {"score": 1.0, "cost": 0.5}

        task = Task(
            name="aime_math",
            initial_candidate="seed",
            eval_fn=lambda candidate, example: (0.0, {}),
            train_set=[Example("train", {}, 1.0)],
            test_set=[Example("test", {}, 1.0)],
            metadata={"type": "generalization"},
        )

        with patch("terrarium.tasks.aime_math.evaluate_with_solver", side_effect=fake_eval):
            configured = _apply_task_runtime_config(
                task,
                OmegaConf.create({
                    "solver_lm": "anthropic/claude-haiku-4-5",
                    "solver_temperature": 0.0,
                    "solver_max_tokens": 32000,
                    "solver_timeout": 180,
                    "solver_num_retries": 0,
                }),
            )
            score, info = configured.eval_fn("candidate", Example("ex", {}, 1.0))

        self.assertEqual(score, 1.0)
        self.assertEqual(info["cost"], 0.5)
        self.assertEqual(calls["solver_lm"], "anthropic/claude-haiku-4-5")
        self.assertEqual(calls["solver_temperature"], 0.0)
        self.assertEqual(calls["solver_max_tokens"], 32000)
        self.assertEqual(calls["solver_timeout"], 180.0)
        self.assertEqual(calls["solver_num_retries"], 0)
        self.assertEqual(configured.metadata["solver_temperature"], 0.0)
        self.assertEqual(configured.metadata["solver_timeout"], 180.0)

    def test_aime_solver_runtime_error_scores_zero(self) -> None:
        class FakePredict:
            def __init__(self) -> None:
                self.signature = type("SignatureConfig", (), {"instructions": ""})()

        class FakeChainOfThought:
            def __init__(self, _signature) -> None:
                self.predict = FakePredict()

            def __call__(self, **_kwargs):
                raise TimeoutError("solver timed out")

        example = Example("aime", {"input": "What is 1+1?", "solution": "It is 2."}, "2")

        with patch("dspy.ChainOfThought", FakeChainOfThought):
            score, info = aime_math.evaluate("answer with an integer", example)

        self.assertEqual(score, 0.0)
        self.assertEqual(info["score"], 0.0)
        self.assertEqual(info["error"], "TimeoutError")
        self.assertIn("solver timed out", info["output"])

    def test_arc_runtime_config_threads_solver_settings(self) -> None:
        calls: dict[str, object] = {}

        def fake_eval(candidate: str, example: Example, *, model_id: str, max_llm_calls: int):
            calls["candidate"] = candidate
            calls["example"] = example
            calls["model_id"] = model_id
            calls["max_llm_calls"] = max_llm_calls
            return 0.5, {"score": 0.5}

        task = Task(
            name="arc_agi",
            initial_candidate="seed",
            eval_fn=lambda candidate, example: (0.0, {}),
            train_set=[Example("train", {}, 1.0)],
            test_set=[Example("test", {}, 1.0)],
            metadata={"type": "generalization"},
        )
        with patch("terrarium.tasks.arc_agi.evaluate", side_effect=fake_eval):
            configured = _apply_task_runtime_config(
                task,
                OmegaConf.create({"solver_lm": "test/model", "max_llm_calls": 3}),
            )
            score, _ = configured.eval_fn("candidate", Example("ex", {}, 1.0))

        self.assertEqual(score, 0.5)
        self.assertEqual(calls["model_id"], "test/model")
        self.assertEqual(calls["max_llm_calls"], 3)
        self.assertEqual(configured.metadata["solver_lm"], "test/model")
        self.assertEqual(configured.metadata["max_llm_calls"], 3)

    def test_listing_tasks_does_not_load_frontier_cs_dataset(self) -> None:
        with patch.object(
            frontier_cs,
            "_algorithmic_rows",
            side_effect=AssertionError("Frontier-CS rows should be loaded lazily"),
        ):
            names = list_tasks()

        self.assertIn("frontier_cs_algo_smoke", names)

    def test_frontier_cs_problem_task_resolves_lazily_by_name(self) -> None:
        frontier_cs._algorithmic_rows.cache_clear()
        rows = {"lazy_test": {"statement": "Return 0."}}

        with patch.object(frontier_cs, "_algorithmic_rows", return_value=rows) as load_rows:
            task = get_task("frontier_cs_algo_lazy_test")

        self.assertEqual(task.name, "frontier_cs_algo_lazy_test")
        self.assertEqual(task.background, "Return 0.")
        load_rows.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
