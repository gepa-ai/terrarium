"""LiveBench math scoring — internal researcher notes, NOT surfaced to the
optimizer. DO NOT include any of this in task-facing text.

Verbatim port of the LiveBench math scorers + their dispatch logic
(LiveBench: A Challenging, Contamination-Free LLM Benchmark,
arXiv:2406.19314, github.com/LiveBench/LiveBench).

The four upstream files are vendored BYTE-FOR-BYTE under ``_lb_upstream/``
(SHA256s + fetch date recorded in ``_lb_upstream/_PROVENANCE.txt``). The
ONLY edit applied to them is rewriting the internal
``from livebench.process_results.util import ...`` line to the vendored
path — no scoring logic was touched.

Dispatch below replicates ``livebench/gen_ground_truth_judgment.py``'s
math branch exactly: the subtask name is split on ``_`` and

  - ``splits[0] in {amc, smc}`` or ``splits[1] == amc``  (covers
    ``amc_12a_2023`` and ``updated_amc_12a_2023``)  -> mathcontest, 0/1
  - ``splits[0] == aime``                               -> aime, 0/1
  - ``splits[0] in {imo, usamo}``                       -> proof
    rearrangement with ``edit_distance=True`` (Levenshtein), fractional
  - ``amps_hard`` in subtask                            -> amps_hard, 0/1

ONE deliberate, documented deviation (mirrors the safety-deviation
discipline used by tasks/finance/_ace_scoring.py):

  ``amps_hard_process_results`` has an OPTIONAL OpenAI-o3 LLM fallback
  (``is_equiv_llm``) that only fires when sympy equivalence already
  failed AND ``OPENAI_API_KEY`` is set. That path is non-deterministic,
  paid, network-dependent, and contradicts LiveBench's own
  "objective / contamination-free" design and Terrarium's deterministic
  budgeted-eval contract (additionally the upstream ``is_equiv_llm``
  references an undefined name in its ``except`` and would raise). We
  guarantee it is never taken by clearing ``OPENAI_API_KEY`` for the
  duration of the amps_hard scoring call only. The upstream file is left
  100% unmodified; this changes no score (the fallback only ran after a
  failed sympy check and is itself unreliable).
"""

from __future__ import annotations

import contextlib
import os

from terrarium.tasks.livebench_math._lb_upstream.amps_hard import (
    amps_hard_process_results,
)
from terrarium.tasks.livebench_math._lb_upstream.math_competitions import (
    aime_process_results,
    mathcontest_process_results,
)
from terrarium.tasks.livebench_math._lb_upstream.olympiad import (
    extract_expression_completions_from_generation,
    proof_rearrangement_process_results,
)


@contextlib.contextmanager
def _openai_key_cleared():
    """Guarantee the optional o3 fallback in amps_hard is unreachable."""
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved


def score_livebench_math(
    subtask: str,
    ground_truth: str,
    llm_answer: str,
    question_text: str,
) -> float:
    """Faithful per-row LiveBench math score. 1.0/0.0 for math_comp and
    AMPS_Hard; fractional in [0, 1] for olympiad (edit-distance match)."""
    splits = subtask.split("_")

    if splits[0] in ("amc", "smc") or (len(splits) > 1 and splits[1] == "amc"):
        return float(mathcontest_process_results(ground_truth, llm_answer, question_text))
    if splits[0] == "aime":
        return float(aime_process_results(ground_truth, llm_answer))
    if splits[0] in ("imo", "usamo"):
        return float(
            proof_rearrangement_process_results(ground_truth, llm_answer, edit_distance=True)
        )
    if "amps_hard" in subtask:
        with _openai_key_cleared():
            return float(amps_hard_process_results(ground_truth, llm_answer))

    raise ValueError(f"livebench_math: unroutable subtask {subtask!r}")


def is_olympiad(subtask: str) -> bool:
    return subtask.split("_")[0] in ("imo", "usamo")


def olympiad_position_report(ground_truth: str, llm_answer: str) -> str:
    """Example-scoped, derived purely from the same upstream extractor —
    no extra oracle. Used to build granular reflection feedback. Never
    mentions the task taxonomy."""
    try:
        gt = [int(n) for n in ground_truth.split(",")]
    except Exception:
        return ""
    try:
        completions = extract_expression_completions_from_generation(llm_answer, False)
    except Exception:
        completions = []
    matched = [
        i for i in range(len(gt)) if i < len(completions) and completions[i] == gt[i]
    ]
    wrong = [i for i in range(len(gt)) if i not in matched]
    return (
        f"You matched {len(matched)} of {len(gt)} positions. "
        f"Expected ordering: {gt}. Parsed from your answer: {completions}. "
        f"Positions still wrong (0-indexed): {wrong}."
    )
