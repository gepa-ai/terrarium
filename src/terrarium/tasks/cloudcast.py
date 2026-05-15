"""Cloudcast: optimize a multi-cloud broadcast routing algorithm.

The candidate is a Python program defining ``search_algorithm(src, dsts, G,
num_partitions)``; the evaluator simulates the resulting broadcast topology
over a NetworkX graph of AWS/GCP/Azure regions and scores ``1 / (1 + cost)``
(higher is better, ``-100_000`` on failure). This is a visible multi-task
search benchmark: the dataset is intentionally reused as the validation set.

All code, configs, and pricing/throughput profiles are vendored under
``cloudcast_lib/`` (originally from ``gepa/examples/adrs/cloudcast``). The
task works out of the box — only ``networkx``, ``pandas``, ``numpy`` are
required at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from terrarium.registry import register_task_factory
from terrarium.task import Example, Task

from terrarium.tasks.cloudcast_lib import dataset as _dataset
from terrarium.tasks.cloudcast_lib import simulation as _simulation

_TASK_NAME = "cloudcast"

_OBJECTIVE = (
    "Optimize a broadcast routing algorithm for multi-cloud data transfer."
)

_BACKGROUND = """Optimize a broadcast routing algorithm for multi-cloud data transfer.

The algorithm decides how to route data from a single source to multiple destinations
across cloud providers (AWS, GCP, Azure). The goal is to minimize total cost
(egress fees + instance costs) while maintaining good transfer times.

Key information about the problem domain:

- The network is represented as a directed graph where:
  - Nodes are cloud regions (e.g., "aws:us-east-1", "gcp:europe-west1-a", "azure:eastus")
  - Edges have 'cost' ($/GB for egress) and 'throughput' (Gbps bandwidth) attributes

- Data is partitioned into num_partitions chunks that can be routed independently
- Each partition can take a different path to reach each destination
- Total cost = egress costs (data_vol × edge_cost) + instance costs (runtime × cost_per_hour)

- The algorithm must return a BroadCastTopology object containing:
  - paths[dst][partition] = list of edges [[src, dst, edge_data], ...]
  - Each destination must have at least one valid path for each partition

Evaluation feedback format:
- Cost: Total transfer cost in dollars
- Transfer time: Maximum time for all destinations to receive data (seconds)

Optimization targets:
1. Reduce total cost (egress + instance costs)
2. Find paths that balance cost and throughput
3. Consider multipath routing for better bandwidth utilization
4. Exploit cloud provider pricing differences (e.g., intra-provider is cheaper)"""

_INITIAL_CANDIDATE = """import networkx as nx
import pandas as pd
import os
from typing import Dict, List


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, 'SingleDstPath'] = None):
        self.src = src
        self.dsts = dsts
        self.num_partitions = num_partitions
        if paths is not None:
            self.paths = paths
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)


def search_algorithm(src, dsts, G, num_partitions):
    \"\"\"
    Find broadcast paths from source to all destinations.

    Uses Dijkstra's shortest path algorithm based on cost as the edge weight.

    Args:
        src: Source node identifier (e.g., "aws:ap-northeast-1")
        dsts: List of destination node identifiers
        G: NetworkX DiGraph with cost and throughput edge attributes
        num_partitions: Number of data partitions

    Returns:
        BroadCastTopology object with paths for all destinations and partitions
    \"\"\"
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        path = nx.dijkstra_path(h, src, dst, weight="cost")
        for i in range(0, len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])

    return bc_topology
"""

_CONFIG_DIR = Path(__file__).resolve().parent / "cloudcast_lib" / "core" / "config"


def _samples_to_examples(samples: list[dict[str, Any]]) -> list[Example]:
    return [
        Example(
            id=Path(s["config_file"]).stem,
            inputs=s,
            expected=None,
        )
        for s in samples
    ]


def _make_task() -> Task:
    """Build the cloudcast Task — invoked lazily on first ``get_task()``."""
    samples = _dataset.load_config_dataset(config_dir=str(_CONFIG_DIR))
    if not samples:
        raise FileNotFoundError(
            f"No cloudcast configuration files found in {_CONFIG_DIR}. "
            "The vendored cloudcast_lib/core/config directory may be missing."
        )

    examples = _samples_to_examples(samples)
    failed_score = _simulation.FAILED_SCORE

    def eval_fn(candidate: str, example: Example) -> tuple[float, dict[str, Any]]:
        sample = example.inputs
        program_path = _simulation.get_program_path(candidate)

        if not _simulation.syntax_is_valid(program_path):
            return failed_score, _simulation.syntax_failure_info(sample)

        success, cost, transfer_time, error, details = _simulation.run_evaluation(
            program_path, sample["config_file"], sample["num_vms"]
        )
        if not success:
            return failed_score, _simulation.evaluation_failure_info(error, sample)

        score = 1.0 / (1.0 + cost)
        return score, _simulation.evaluation_success_info(
            score, cost, transfer_time, sample, details
        )

    return Task(
        name=_TASK_NAME,
        objective=_OBJECTIVE,
        background=_BACKGROUND,
        initial_candidate=_INITIAL_CANDIDATE,
        eval_fn=eval_fn,
        train_set=examples,
        val_set=examples,
        metadata={
            "type": "multi_task",
            "candidate_type": "code",
            "language": "python",
            "source": "vendored from gepa/examples/adrs/cloudcast",
            "split_provenance": {
                "source_dataset": str(_CONFIG_DIR),
                "split_method": "visible_dataset_reused_as_validation",
                "split_seed": None,
                "split_sizes": {
                    "train": len(examples),
                    "val": len(examples),
                    "test": 0,
                },
            },
        },
    )


register_task_factory(_TASK_NAME, _make_task)
