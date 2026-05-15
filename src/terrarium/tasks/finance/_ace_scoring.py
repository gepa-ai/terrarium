"""ACE finance benchmarks: scoring port — internal researcher notes, NOT
surfaced to the optimizer. DO NOT include any of this in task-facing text.

Verbatim port of the answer extraction + correctness checks from the ACE
repo (Agentic Context Engineering, arXiv:2510.04618):
  - ace/utils.py::extract_boxed_content
  - ace/utils.py::extract_answer
  - ace/eval/finance/data_processor.py::_finer_answer_is_correct
  - ace/eval/finance/data_processor.py::_formula_answer_is_correct

Kept byte-faithful so terrarium scores are directly comparable to the
numbers reported by the ACE paper, with ONE deliberate, documented
deviation for safety:

  ACE's _finer_answer_is_correct calls the builtin ``eval()`` on both the
  model prediction and the ground-truth string to coerce numerics. That is
  arbitrary code execution on model output. We substitute
  ``ast.literal_eval``, which is outcome-equivalent for every realistic
  input (numeric literals coerce identically; bare tag names like
  ``Revenues`` raise and fall through the bare-except exactly as they do
  under ``eval``) but cannot execute arbitrary code. This changes no
  scoring outcome on the FiNER/Formula data; it only removes the RCE.
"""

from __future__ import annotations

import ast
import json
import re


def extract_boxed_content(text: str):
    """Helper: extract content from a \\boxed{...} span (brace-balanced)."""
    pattern = r"\\boxed\{"
    match = re.search(pattern, text)
    if not match:
        return None

    start = match.end() - 1  # position of opening brace
    brace_count = 0
    i = start
    while i < len(text):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1:i]
        i += 1
    return None


def extract_answer(response: str) -> str:
    """Extract the final answer from a model response (ACE-faithful)."""
    try:
        parsed = json.loads(response)
        answer = str(parsed.get("final_answer", "No final answer found"))
        return answer
    except (json.JSONDecodeError, KeyError, AttributeError):
        matches = re.findall(r"Finish\[(.*?)\]", response)
        if matches:
            return matches[-1]

        matches = re.findall(r'"final_answer"\s*:\s*"([^"]*)"', response)
        if matches:
            return matches[-1]

        matches = re.findall(r"'final_answer'\s*:\s*'([^']*)'", response)
        if matches:
            return matches[-1]

        matches = re.findall(r'[\'"]final_answer[\'"]\s*:\s*([^,}]+)', response)
        if matches:
            answer = matches[-1].strip()
            answer = re.sub(r"[,}]*$", "", answer)
            return answer

        final_answer_pattern = r"[Tt]he final answer is:?\s*\$?\\boxed\{"
        match = re.search(final_answer_pattern, response)
        if match:
            remaining_text = response[match.start():]
            boxed_content = extract_boxed_content(remaining_text)
            if boxed_content:
                return boxed_content

        matches = re.findall(r"[Tt]he final answer is:?\s*([^\n.]+)", response)
        if matches:
            answer = matches[-1].strip()
            answer = re.sub(r"^\$?\\boxed\{([^}]+)\}\$?$", r"\1", answer)
            answer = answer.replace("$", "").strip()
            if answer:
                return answer

        return "No final answer found"


def _coerce(value: str):
    """ACE's ``eval``-based numeric coercion, made safe via literal_eval."""
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def finer_answer_is_correct(predicted: str, ground_truth: str, return_counts: bool = False):
    """XBRL (FiNER) correctness — all comma-separated tags must match."""
    pred = [val.lower().strip() for val in predicted.split(",")]
    label = [val.lower().strip() for val in ground_truth.split(",")]
    count = 0

    if len(pred) != len(label):
        if len(pred) > len(label):
            pred = pred[:len(label)]
        else:
            pred += [""] * (len(label) - len(pred))

    for prediction, gt in zip(pred, label):
        gt_c = _coerce(gt)
        pred_c = _coerce(prediction.replace(",", "").replace("$", ""))
        if pred_c == gt_c:
            count += 1

    score = count / len(pred) if pred else 0
    if return_counts:
        return count, len(pred)
    return score == 1


def formula_answer_is_correct(predicted: str, ground_truth: str) -> bool:
    """Formula correctness — float equality after stripping thousands commas."""
    try:
        p = predicted.replace(",", "")
        g = ground_truth.replace(",", "")
        return float(p) == float(g)
    except Exception:
        return predicted == ground_truth
