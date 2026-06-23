from __future__ import annotations

import unittest
from pathlib import Path

from benchmarks.experiment_launcher import _max_parallel_runs, expand_runs


class ExperimentLauncherTest(unittest.TestCase):
    def test_expands_task_algorithm_budget_seed_matrix(self) -> None:
        runs = expand_runs(
            {
                "name": "unit",
                "output_root": "/tmp/terrarium-launcher-test",
                "source_zshrc": False,
                "tasks": [{"name": "aime", "task": "aime_math", "overrides": ["max_concurrency=2"]}],
                "algorithms": [
                    {
                        "name": "gepa",
                        "algorithm": "optimize_anything",
                        "overrides": ["adapter.engine=gepa", "adapter.engine_config.engine.seed=0"],
                        "seed_overrides": ["adapter.engine_config.engine.seed={seed}"],
                    }
                ],
                "budgets": [{"name": "cheap", "max_evals": 3, "max_token_cost": 0.25}],
                "seeds": [7],
            },
            config_path=Path("unit.yaml"),
        )

        self.assertEqual(len(runs), 1)
        run = runs[0]
        self.assertEqual(run.task, "aime")
        self.assertEqual(run.algorithm, "gepa")
        self.assertIn("task=aime_math", run.overrides)
        self.assertIn("adapter=optimize_anything", run.overrides)
        self.assertIn("budget.max_evals=3", run.overrides)
        self.assertIn("budget.max_token_cost=0.25", run.overrides)
        self.assertIn("adapter.engine_config.engine.seed=7", run.overrides)
        self.assertTrue(str(run.output_dir).endswith("aime__gepa__cheap__seed7"))

    def test_sequential_composition_override_is_preserved_as_one_arg(self) -> None:
        runs = expand_runs(
            {
                "name": "unit",
                "tasks": ["aime_math_mini"],
                "algorithms": [
                    {
                        "name": "seq",
                        "algorithm": "optimize_anything",
                        "overrides": [
                            "adapter.strategy=sequential",
                            "adapter.configs=[{engine: gepa}, {engine: autoresearch}]",
                        ],
                    }
                ],
                "budget": {"max_evals": 2},
            }
        )

        self.assertIn("adapter.configs=[{engine: gepa}, {engine: autoresearch}]", runs[0].overrides)

    def test_max_parallel_runs_preferred_over_legacy_max_workers(self) -> None:
        self.assertEqual(_max_parallel_runs({"max_parallel_runs": 8, "max_workers": 2}, None), 8)
        self.assertEqual(_max_parallel_runs({"max_workers": 2}, None), 2)
        self.assertEqual(_max_parallel_runs({"max_parallel_runs": 8}, 3), 3)


if __name__ == "__main__":
    unittest.main()
