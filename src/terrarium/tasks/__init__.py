"""Built-in Terrarium tasks.

Importing this package registers all bundled tasks with the registry.
"""

from terrarium.tasks import (  # noqa: F401
    aime_math,
    aime_math_mini,
    arc_agi,
    cant_be_late,
    circle_packing,
    cloudcast,
    frontier_cs,
    optuna_blackbox,
)

__all__ = [
    "aime_math",
    "aime_math_mini",
    "arc_agi",
    "cant_be_late",
    "circle_packing",
    "cloudcast",
    "frontier_cs",
    "optuna_blackbox",
]
