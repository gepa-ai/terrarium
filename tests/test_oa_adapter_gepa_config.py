"""Tests for OptimizeAnythingAdapter's gepa engine_config translation.

Guards the terrarium side of the gepa 1-to-1 GEPAConfig pass-through: gepa's
``GepaEngine`` no longer accepts a ``claude_code_agent`` convenience key, so the
adapter (the caller) translates it into the standard
``reflection.custom_candidate_proposer`` slot plus ``engine.write_agent_state``.
"""

import tempfile

from terrarium.adapters.gepa_cc_agent import ClaudeCodeAgentProposer
from terrarium.adapters.optimize_anything_adapter import OptimizeAnythingAdapter


def test_claude_code_agent_translates_to_custom_proposer():
    adapter = OptimizeAnythingAdapter(engine="gepa", run_dir="/tmp/tt-cca")
    merged = adapter._build_engine_config(
        "gepa",
        {
            "claude_code_agent": {"model": "sonnet"},
            "reflection": {"reflection_lm": None},
            "engine": {"parallel": True},
        },
        run_dir="/tmp/tt-cca",
        objective="maximize length",
        background="bg",
        max_token_cost=5.0,
        effort="high",
        max_thinking_tokens=None,
        sandbox=True,
    )

    # The convenience key is consumed, not forwarded to gepa.
    assert "claude_code_agent" not in merged
    proposer = merged["reflection"]["custom_candidate_proposer"]
    assert isinstance(proposer, ClaudeCodeAgentProposer)
    assert proposer.model == "sonnet"
    assert proposer.objective == "maximize length"
    assert proposer.background == "bg"
    assert proposer.effort == "high"
    assert proposer.max_budget_usd == 5.0
    # reflection_lm stays unset — the agent proposer replaces it.
    assert merged["reflection"]["reflection_lm"] is None
    # write_agent_state is auto-enabled so the proposer can read the run-dir tree.
    assert merged["engine"]["write_agent_state"] is True
    # Pre-existing engine keys survive.
    assert merged["engine"]["parallel"] is True


def test_claude_code_agent_per_key_overrides_win():
    adapter = OptimizeAnythingAdapter(engine="gepa", run_dir="/tmp/tt-cca")
    merged = adapter._build_engine_config(
        "gepa",
        {"claude_code_agent": {"model": "opus", "max_budget_usd": 2.0, "effort": "low"}},
        run_dir="/tmp/tt-cca",
        objective="obj",
        background=None,
        max_token_cost=5.0,
        effort="high",
        max_thinking_tokens=None,
        sandbox=False,
    )
    proposer = merged["reflection"]["custom_candidate_proposer"]
    assert proposer.model == "opus"
    assert proposer.max_budget_usd == 2.0  # explicit key beats max_token_cost
    assert proposer.effort == "low"  # explicit key beats adapter effort


def test_translated_config_is_accepted_by_gepa_engine():
    """The translated engine_config builds a GepaEngine (GEPAConfig-valid)."""
    from gepa.oa.config import OptimizeAnythingConfig
    from gepa.oa.engines.gepa import GepaEngine

    adapter = OptimizeAnythingAdapter(engine="gepa", run_dir="/tmp/tt-cca")
    merged = adapter._build_engine_config(
        "gepa",
        {"claude_code_agent": {"model": "sonnet"}},
        run_dir="/tmp/tt-cca",
        objective="obj",
        background=None,
        max_token_cost=None,
        effort=None,
        max_thinking_tokens=None,
        sandbox=None,
    )
    # No TypeError => every translated key is a real GEPAConfig field.
    GepaEngine(OptimizeAnythingConfig(engine="gepa", max_evals=4, engine_config=merged))


def test_effort_routes_to_reflection_lm_kwargs_for_gepa():
    """Adapter-level effort/thinking thread into gepa's reflection_lm_kwargs."""
    adapter = OptimizeAnythingAdapter(engine="gepa", run_dir="/tmp/tt-eff")
    merged = adapter._build_engine_config(
        "gepa",
        {},
        run_dir="/tmp/tt-eff",
        objective="obj",
        background=None,
        max_token_cost=None,
        effort="high",
        max_thinking_tokens=2048,
        sandbox=None,
    )
    lm_kwargs = merged["reflection"]["reflection_lm_kwargs"]
    assert lm_kwargs["reasoning_effort"] == "high"
    assert lm_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_effort_routes_to_engine_config_key_for_agent_engines():
    """Adapter-level effort/thinking become plain engine_config keys for agents."""
    adapter = OptimizeAnythingAdapter(engine="autoresearch")
    merged = adapter._build_engine_config(
        "autoresearch",
        {"model": "sonnet"},
        run_dir=None,
        objective="obj",
        background=None,
        max_token_cost=None,
        effort="high",
        max_thinking_tokens=2048,
        sandbox=None,
    )
    assert merged["effort"] == "high"
    assert merged["max_thinking_tokens"] == 2048
    # And such a config is accepted by the autoresearch engine.
    from gepa.oa.config import OptimizeAnythingConfig
    from gepa.oa.engines.autoresearch import AutoResearchEngine

    AutoResearchEngine(OptimizeAnythingConfig(engine="autoresearch", max_evals=4, engine_config=merged))


def test_gepa_engine_end_to_end_via_adapter():
    """The adapter drives gepa with a plain reflection_lm and returns a result."""
    from terrarium.budget import BudgetTracker
    from terrarium.eval_server import EvalServer
    from terrarium.task import Task

    class FakeLM:
        total_cost = 0.0
        total_tokens_in = 0
        total_tokens_out = 0

        def __call__(self, *args, **kwargs):
            return "IMPROVED CANDIDATE: a longer and better answer than the seed"

    def eval_fn(candidate, example=None):
        return float(len(candidate)) / 100.0, {"feedback": "ok"}

    with tempfile.TemporaryDirectory() as d:
        task = Task(
            name="smoke",
            initial_candidate="short",
            eval_fn=eval_fn,
            objective="maximize length",
            background="",
            train_set=None,
            val_set=None,
            test_set=None,
        )
        budget = BudgetTracker(max_evals=6)
        server = EvalServer(task, budget, max_concurrency=2, output_dir=d)
        server.start()
        try:
            adapter = OptimizeAnythingAdapter(
                engine="gepa",
                run_dir=d,
                engine_config={"reflection": {"reflection_lm": FakeLM()}},
            )
            result = adapter.evolve(task, server)
        finally:
            server.stop()

    # The FakeLM's longer candidate must actually be applied — a score above the
    # seed's (len("short")/100 = 0.05) proves the adapter-level engine_config
    # reached gepa (guards the single-path engine_config fallback).
    assert result.best_score > 0.05
    assert "IMPROVED CANDIDATE" in result.best_candidate
