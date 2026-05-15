"""ACE finance benchmarks: prompt + data-parse port — internal researcher
notes, NOT surfaced to the optimizer. DO NOT include in task-facing text.

Verbatim port of ACE (arXiv:2510.04618):
  - ace/prompts/generator.py::GENERATOR_PROMPT
  - ace/eval/finance/data_processor.py::parse_instruction_and_input  (finer)
  - ace/eval/finance/data_processor.py::parse_context_and_question_formula (formula)

ACE's generator prompt is a 4-slot ``str.format`` template:
``GENERATOR_PROMPT.format(playbook, reflection, question, context)``.
In the terrarium integration the evolved *candidate* fills the
``{playbook}`` slot (that is exactly ACE's playbook role); ``reflection``
is "(empty)" at eval time (matching ACE's test path); ``question`` and
``context`` come from the per-task parse fn below, identical to ACE's
``DataProcessor.process_task_data``.
"""

from __future__ import annotations

# Verbatim from ace/prompts/generator.py
GENERATOR_PROMPT = """You are an analysis expert tasked with answering questions using your knowledge, a curated playbook of strategies and insights and a reflection that goes over the diagnosis of all previous mistakes made while answering the question.

**Instructions:**
- Read the playbook carefully and apply relevant strategies, formulas, and insights
- Pay attention to common mistakes listed in the playbook and avoid them
- Show your reasoning step-by-step
- Be concise but thorough in your analysis
- If the playbook contains relevant code snippets or formulas, use them appropriately
- Double-check your calculations and logic before providing the final answer

Your output should be a json object, which contains the following fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- bullet_ids: each line in the playbook has a bullet_id. all bulletpoints in the playbook that's relevant, helpful for you to answer this question, you should include their bullet_id in this list
- final_answer: your concise final answer


**Playbook:**
{}

**Reflection:**
{}

**Question:**
{}

**Context:**
{}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "bullet_ids": ["calc-00001", "fin-00002"],
  "final_answer": "[Your concise final answer here]"
}}

---
"""


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
