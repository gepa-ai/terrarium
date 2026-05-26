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
    finance,
    frontier_cs,
    livebench_math,
    needle_in_range,
    optuna_blackbox,
    slot_machines,
)

__all__ = [
    "aime_math",
    "aime_math_mini",
    "arc_agi",
    "cant_be_late",
    "circle_packing",
    "cloudcast",
    "finance",
    "frontier_cs",
    "livebench_math",
    "needle_in_range",
    "optuna_blackbox",
    "slot_machines",
]
