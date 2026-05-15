"""ACE finance benchmark tasks (FiNER + Formula).

Importing this package registers the finance tasks with the registry.
Internal helpers live alongside as private modules:
``_ace_scoring`` (verbatim ACE scoring port), ``_ace_prompts`` (verbatim
ACE context-parse port), ``_finance_common`` (shared dspy solver).
"""

from terrarium.tasks.finance import finer, formula  # noqa: F401

__all__ = ["finer", "formula"]
