"""Built-in adapters for common evolution systems.

Adapters are the bridge between Terrarium and an evolution/search system.

The primary adapter is ``optimize_anything``: a single configurable surface over
every ``gepa.optimize_anything`` engine (``gepa`` / ``autoresearch`` /
``meta_harness`` / ``best_of_n``), plus ensemble compositions.

- ``optimize_anything`` — dispatcher for ``gepa.optimize_anything`` engines
  under a single ``engine`` knob (and ensemble ``strategy`` + ``configs``).

The remaining native adapters are **deprecated** (still callable, but they emit a
``DeprecationWarning``); prefer the equivalent ``optimize_anything`` engine:

- ``gepa`` — in-process GEPA. Use ``adapter.engine=gepa`` instead.
- ``claude_code`` — single Claude Code subprocess. Use
  ``adapter.engine=autoresearch`` instead.
- ``meta_harness`` — Meta-Harness loop (https://arxiv.org/abs/2603.28052).
  Use ``adapter.engine=meta_harness`` instead.
"""
