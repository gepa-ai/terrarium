"""Can't Be Late: optimize a cloud spot/on-demand scheduling strategy.

The candidate is a Python program defining ``EvolveSingleRegionStrategy._step``;
the evaluator runs the vendored sky-spot simulator on AWS spot-availability
traces and scores by negative cost (higher is better, ``-100_000`` on failure).

Simulator code is vendored under ``cant_be_late_lib/`` (originally from
``gepa/examples/adrs/can_be_late``). The trace dataset is **not bundled** —
download it once::

    mkdir -p ~/.cache/terrarium/cant_be_late
    cd ~/.cache/terrarium/cant_be_late
    curl -L -o real_traces.tar.gz https://github.com/UCB-ADRS/ADRS/raw/main/openevolve/examples/ADRS/cant-be-late/simulator/real_traces.tar.gz
    tar -xzf real_traces.tar.gz   # creates ./real/

Trace location resolution order:
  1. ``CANT_BE_LATE_DATA_DIR`` env var (path to the ``real/`` directory).
  2. ``~/.cache/terrarium/cant_be_late/real/`` (default cache).
  3. ``<vendored simulator>/real/`` (next to the bundled simulator code).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from terrarium.registry import register_task_factory
from terrarium.task import Example, Task

from terrarium.tasks.cant_be_late_lib import dataset as _dataset
from terrarium.tasks.cant_be_late_lib import simulation as _simulation

_TASK_NAME = "cant_be_late"

_OBJECTIVE = (
    'Optimize a cloud scheduling strategy for the "Can\'t Be Late" problem.'
)

_BACKGROUND = """Optimize a cloud scheduling strategy for the "Can't Be Late" problem.

The strategy decides when to use SPOT instances (cheap but can be preempted) vs ON_DEMAND
instances (expensive but reliable) to complete a task before its deadline. The goal is to
minimize cost while ensuring the task completes on time.

Key information about the problem domain:

- ClusterType.SPOT: Use spot instances (cheap, ~$0.3/hour, but can be preempted at any time)
- ClusterType.ON_DEMAND: Use on-demand instances (expensive, ~$1/hour, but guaranteed availability)
- ClusterType.NONE: Wait without using any instances (no cost, but no progress)
- restart_overhead: Time penalty incurred when switching from one instance type to another
- The strategy MUST ensure task completion before the deadline (hard constraint)
- Lower cost is better (scores are negative, representing cost in dollars)

Evaluation feedback format:
- Timeline format: start-end:TYPE@REGION[progress%] (e.g., "0.0-5.0:S@R0[50%]" means SPOT from hour 0-5 reaching 50% progress)
- Spot availability: S=available, X=unavailable (e.g., "0.0-10.0:S | 10.0-15.0:X" means spot available first 10h, then unavailable)

Optimization targets:
1. Reduce overall cost while maintaining deadline guarantees
2. Make better decisions about when to use SPOT vs ON_DEMAND
3. Handle spot unavailability more intelligently
4. Consider the trade-offs between waiting for spot and using on-demand"""

_INITIAL_CANDIDATE = """import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class EvolveSingleRegionStrategy(Strategy):
    NAME = 'evolve_single_region'

    def __init__(self, args):
        super().__init__(args)

    def reset(self, env, task):
        super().reset(env, task)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env

        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE

        remaining_time = self.deadline - env.elapsed_seconds

        if remaining_task_time + self.restart_overhead >= remaining_time:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
"""

_VENDORED_SIMULATOR_DIR = Path(__file__).resolve().parent / "cant_be_late_lib" / "simulator"


def _resolve_trace_root() -> Path | None:
    """Find the ``real/`` traces dir; return ``None`` if not present anywhere."""
    env = os.environ.get("CANT_BE_LATE_DATA_DIR")
    if env:
        path = Path(env).expanduser()
        if path.is_dir():
            return path
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    candidates = [
        cache_root / "terrarium" / "cant_be_late" / "real",
        _VENDORED_SIMULATOR_DIR / "real",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def _example_id(sample: dict[str, Any]) -> str:
    cfg = sample["config"]
    env_name = Path(sample["trace_file"]).parent.parent.parent.name
    trace = Path(sample["trace_file"]).stem
    return f"{env_name}__{trace}__d{cfg['duration']}_dl{cfg['deadline']}_o{cfg['overhead']}"


def _samples_to_examples(samples: list[dict[str, Any]]) -> list[Example]:
    return [Example(id=_example_id(s), inputs=s, expected=None) for s in samples]


def _make_task() -> Task:
    """Build the cant_be_late Task — invoked lazily on first ``get_task()``."""
    trace_root = _resolve_trace_root()
    if trace_root is None:
        cache = Path("~/.cache/terrarium/cant_be_late").expanduser()
        raise FileNotFoundError(
            "Can't Be Late trace data not found. Download it once:\n"
            f"  mkdir -p {cache}\n"
            f"  cd {cache}\n"
            "  curl -L -o real_traces.tar.gz https://github.com/UCB-ADRS/ADRS/"
            "raw/main/openevolve/examples/ADRS/cant-be-late/simulator/real_traces.tar.gz\n"
            "  tar -xzf real_traces.tar.gz\n"
            "Or set CANT_BE_LATE_DATA_DIR to an existing real/ directory."
        )

    splits = _dataset.load_trace_dataset(dataset_root=str(trace_root))
    train_set = _samples_to_examples(splits["train"])
    val_set = _samples_to_examples(splits["val"])
    test_set = _samples_to_examples(splits["test"])

    failed_score = _simulation.FAILED_SCORE

    def eval_fn(candidate: str, example: Example) -> tuple[float, dict[str, Any]]:
        sample = example.inputs
        program_path = _simulation.get_program_path(candidate)

        if not _simulation.syntax_is_valid(program_path):
            return failed_score, _simulation.syntax_failure_info(sample)

        success, cost, error, details = _simulation.run_simulation(
            program_path, sample["trace_file"], sample["config"]
        )
        if not success:
            return failed_score, _simulation.simulation_failure_info(error, sample)

        score = -cost
        return score, _simulation.simulation_success_info(score, sample, details)

    return Task(
        name=_TASK_NAME,
        objective=_OBJECTIVE,
        background=_BACKGROUND,
        initial_candidate=_INITIAL_CANDIDATE,
        eval_fn=eval_fn,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        metadata={
            "type": "generalization",
            "candidate_type": "code",
            "language": "python",
            "source": "vendored from gepa/examples/adrs/can_be_late",
            "split_provenance": {
                "source_dataset": str(trace_root),
                "split_method": "vendored_loader_train_val_test",
                "split_seed": None,
                "split_sizes": {
                    "train": len(train_set),
                    "val": len(val_set),
                    "test": len(test_set),
                },
            },
        },
    )


register_task_factory(_TASK_NAME, _make_task)
