"""Built-in Terrarium tasks.

Importing this package registers all bundled tasks with the registry.
"""

from terrarium.tasks import aime_math, aime_math_mini, arc_agi, circle_packing, frontier_cs, gso, optuna_blackbox

__all__ = ["circle_packing", "optuna_blackbox", "aime_math", "aime_math_mini", "arc_agi", "frontier_cs", "gso"]
