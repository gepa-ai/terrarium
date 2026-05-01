"""Minimal real-run adapter for Terrarium smoke tests.

This adapter does not evolve. It evaluates the task's initial candidate through
the real EvalServer and returns it as the best candidate so the runner still
exercises budget accounting, artifacts, and hidden post-search test evaluation.
"""

from __future__ import annotations

from pathlib import Path

from terrarium.adapter import Result


class InitialCandidateAdapter:
    def evolve(self, task, server):
        candidate = task.initial_candidate
        if task.has_dataset:
            score, info = server.evaluate_examples(candidate, split="train")
        else:
            score, info = server.evaluate(candidate)
        return Result(
            best_candidate=candidate,
            best_score=score,
            candidates=[{"candidate": candidate, "score": score}],
            metadata={"smoke_info": info},
        )

    def process_result(self, result: Result, output_dir: Path) -> None:
        return


def create_adapter():
    return InitialCandidateAdapter()
