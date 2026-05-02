"""Task registry for discovering and loading Terrarium tasks by name."""

from __future__ import annotations

from collections.abc import Callable

from terrarium.task import Task

_REGISTRY: dict[str, Task] = {}
_FACTORIES: dict[str, Callable[[], Task]] = {}


def register_task(task: Task) -> Task:
    """Register a task. Raises ValueError if a task with the same name exists."""
    if task.name in _REGISTRY or task.name in _FACTORIES:
        raise ValueError(f"Task '{task.name}' is already registered")
    _REGISTRY[task.name] = task
    return task


def register_task_factory(name: str, factory: Callable[[], Task]) -> None:
    """Register a lazy task factory. The factory is called on first access.

    Use this for dataset tasks where loading is expensive — the dataset
    won't be fetched until someone actually requests the task.
    """
    if name in _REGISTRY or name in _FACTORIES:
        raise ValueError(f"Task '{name}' is already registered")
    _FACTORIES[name] = factory


def get_task(name: str) -> Task:
    """Retrieve a registered task by name.

    If the built-in tasks haven't been loaded yet, this triggers a lazy import
    of ``terrarium.tasks`` so that all bundled tasks are available.
    """
    if not _REGISTRY and not _FACTORIES:
        _load_builtin_tasks()
    if name not in _REGISTRY and name not in _FACTORIES:
        _maybe_register_dynamic_task(name)
    if name in _FACTORIES:
        task = _FACTORIES.pop(name)()
        _REGISTRY[name] = task
        return task
    if name not in _REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(_REGISTRY | _FACTORIES)}")
    return _REGISTRY[name]


def list_tasks() -> list[str]:
    """Return sorted names of all registered tasks."""
    if not _REGISTRY and not _FACTORIES:
        _load_builtin_tasks()
    return sorted(set(_REGISTRY) | set(_FACTORIES))


def _load_builtin_tasks() -> None:
    """Import the tasks subpackage to trigger registration of built-in tasks."""
    import terrarium.tasks  # noqa: F401


def _maybe_register_dynamic_task(name: str) -> None:
    """Register known patterned tasks without expensive import-time discovery."""
    if name.startswith("frontier_cs_algo_") and name != "frontier_cs_algo_smoke":
        from terrarium.tasks.frontier_cs import _make_problem_task

        problem_id = name.removeprefix("frontier_cs_algo_")
        register_task_factory(name, lambda p=problem_id: _make_problem_task(p))
