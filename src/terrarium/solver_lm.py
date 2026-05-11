"""Cost-tracked DSPy solver LM utilities."""

from __future__ import annotations

import threading
from typing import Any

import dspy
import litellm

from terrarium.budget import BudgetExhausted


class SolverBudgetExhausted(BudgetExhausted):
    """Raised when task-level solver LLM spend reaches its cap."""


class CostTrackedDSPyLM(dspy.LM):
    """DSPy LM that records and optionally caps solver-token cost."""

    def __init__(self, *args: Any, max_cost: float | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_cost = max_cost
        self.total_cost = 0.0
        self.total_calls = 0
        self.cost_log: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def forward(self, prompt=None, messages=None, **kwargs):  # type: ignore[no-untyped-def]
        if self.max_cost is not None:
            with self._lock:
                self._check_budget_locked()
                response = super().forward(prompt=prompt, messages=messages, **kwargs)
                cost = self._completion_cost(response)
                self._record_cost_locked(response, cost)
            return response

        with self._lock:
            self._check_budget_locked()

        response = super().forward(prompt=prompt, messages=messages, **kwargs)
        cost = self._completion_cost(response)

        with self._lock:
            self._record_cost_locked(response, cost)
            self._check_budget_locked()
        return response

    def _record_cost_locked(self, response: Any, cost: float) -> None:
        self.total_cost += cost
        self.total_calls += 1
        self.cost_log.append({
            "call": self.total_calls,
            "model": self.model,
            "cost": cost,
            "cumulative_cost": self.total_cost,
            "cache_hit": bool(getattr(response, "cache_hit", False)),
            "usage": dict(getattr(response, "usage", {}) or {}),
        })

    def _check_budget(self) -> None:
        with self._lock:
            self._check_budget_locked()

    def _check_budget_locked(self) -> None:
        if self.max_cost is not None and self.total_cost >= self.max_cost:
            raise SolverBudgetExhausted(
                f"Solver LLM budget exhausted: ${self.total_cost:.6f}/${self.max_cost:.6f} spent"
            )

    def _completion_cost(self, response: Any) -> float:
        if getattr(response, "cache_hit", False):
            return 0.0
        try:
            return float(litellm.completion_cost(completion_response=response, model=self.model) or 0.0)
        except Exception:
            return 0.0
