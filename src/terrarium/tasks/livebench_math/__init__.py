"""LiveBench math benchmark task (generalization-mode prompt optimization).

Importing this package registers ``livebench_math`` with the registry.
Internal helpers live alongside as private modules: ``_livebench_scoring``
(thin dispatcher over the upstream ``livebench`` package's math scorers,
pinned via the ``[livebench_math]`` extra in ``pyproject.toml``) and
``_livebench_common`` (dataset loader + dspy solver).
"""

from terrarium.tasks.livebench_math import livebench_math  # noqa: F401

__all__ = ["livebench_math"]
