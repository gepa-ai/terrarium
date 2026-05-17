"""ACE finance benchmarks: data-parse port — internal researcher notes,
NOT surfaced to the optimizer. DO NOT include in any task-facing text.

Verbatim port of ACE's per-task context parsing (arXiv:2510.04618):
  - ace/eval/finance/data_processor.py::parse_instruction_and_input  (finer)
  - ace/eval/finance/data_processor.py::parse_context_and_question_formula (formula)

Applied at dataset load, mirroring ``DataProcessor.process_task_data``,
so the model input is byte-faithful to ACE — in particular the formula
parse appends ACE's numeric-normalization instruction. The optimization
framing itself is terrarium-native prompt optimization (the candidate is
an evolved prompt, not ACE's playbook); ACE's GENERATOR_PROMPT is
deliberately NOT used here.
"""

from __future__ import annotations


def parse_instruction_and_input(all_context):
    """ACE finer parse (verbatim). Returns (input_text, instruction)."""
    if "Input: " in all_context and "Instruction: " in all_context:
        instruction_part = all_context.split("Input: ")[0].strip()
        instruction_part = instruction_part.split("Instruction: ")[1].strip()
        remaining = all_context.split("Input: ")[1]
        input_text = remaining.split("Answer: ")[0].strip()
        return input_text, instruction_part
    return "", all_context


def parse_context_and_question_formula(all_context):
    """ACE formula parse (verbatim). Returns (input_text, question)."""
    if "Question: " in all_context and ". Answer:" in all_context:
        parts = all_context.split("Question: ", 1)
        instruction_part = parts[0].strip()  # noqa: F841 (kept for parity)
        question_part = parts[1]
        question_text = question_part.split(". Answer:")[0].strip()
        if question_text.startswith('"') and question_text.endswith('"'):
            question_text = question_text[1:-1]
        question_text += (
            " Your answer should be a plain floating point number, round to "
            "the nearest hundredth if necessary. Do the necessary conversions, "
            "for example 5 million should be 5000000.0. "
        )
        return "", question_text
    return "", all_context


# Task -> parse fn, mirroring DataProcessor.process_task_data.
PARSE_FN = {
    "finer": parse_instruction_and_input,
    "formula": parse_context_and_question_formula,
}
