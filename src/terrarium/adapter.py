"""Adapter protocol: the interface every evolution system implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from terrarium.task import Task

if TYPE_CHECKING:
    from terrarium.eval_server import EvalServer


class Adapter(Protocol):
    """Protocol that every evolution system adapter must implement.

    The adapter receives an ``EvalServer`` — the single choke point for all
    evaluations. In-process adapters call ``server.evaluate(candidate)``
    directly. External/black-box adapters point their subprocess at
    ``server.url`` for HTTP-based eval.

    Budget limits are available via ``server.budget`` (a :class:`BudgetTracker`).
    ``max_evals`` is enforced centrally by the server. ``max_token_cost`` must
    be enforced by each adapter in its own way (e.g. GEPA via
    ``max_reflection_cost``, Claude Code via ``--max-budget-usd``).

    The runner calls :meth:`evolve`, then :meth:`process_result`. Adapters that
    need to write artifacts to a known on-disk location (logs, transcripts,
    workspace contents) should stash any required state in ``result.metadata``
    during ``evolve`` and do the actual file IO in ``process_result`` — that
    way ``evolve`` stays focused on running the evolution.

    Example::

        class MyEvolver:
            def evolve(self, task, server):
                budget = server.budget
                candidate = task.initial_candidate
                best, best_score = candidate, -float("inf")
                while not budget.exhausted:
                    score, info = server.evaluate(candidate)
                    if score > best_score:
                        best_score, best = score, candidate
                    candidate = self.mutate(candidate, score, info)
                return Result(best_candidate=best, best_score=best_score)
    """

    def evolve(
        self,
        task: Task,
        server: EvalServer,
    ) -> Result:
        """Run evolution and return the best candidate found.

        Args:
            task: Task definition (description, initial_candidate, datasets).
            server: The eval server. Call ``server.evaluate(candidate)`` for
                    in-process eval, or use ``server.url`` for HTTP-based eval.
                    Raises ``BudgetExhausted`` when the eval budget is used up.
                    Access ``server.budget`` for budget limits (``max_evals``,
                    ``max_token_cost``) and status.

        Returns:
            Result containing the best candidate and metadata.
        """
        ...

    def process_result(self, result: Result, output_dir: Path) -> None:
        """Persist adapter-specific artifacts to ``output_dir`` after evolution.

        Called by the runner immediately after :meth:`evolve` completes (and
        after the runner has filled in ``result.eval_log`` and standard
        metadata) — but only when an output directory is configured for the
        run. Adapters that produced files, transcripts, or workspaces during
        ``evolve`` should override this to copy/write them under ``output_dir``.

        Args:
            result: The :class:`Result` returned by ``evolve`` — read
                ``result.metadata`` for any handoff state stashed there.
            output_dir: Run output directory.
        """
        return


@dataclass
class Result:
    """What an adapter returns after evolution."""

    best_candidate: str
    best_score: float
    total_evals: int = 0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    eval_log: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
