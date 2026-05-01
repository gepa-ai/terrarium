"""Built-in adapters for common evolution systems.

Adapters are the bridge between Terrarium and an evolution/search system:

- ``gepa`` \u2014 in-process GEPA via ``optimize_anything``.
- ``claude_code`` \u2014 launches a single Claude Code subprocess that evolves.
- ``meta_harness`` \u2014 Meta-Harness loop (https://arxiv.org/abs/2603.28052):
  iterative Claude-based proposer + Terrarium-side benchmarking.
- ``omni`` \u2014 dispatcher for ``gepa.omni`` backends (gepa / claude_code /
  meta_harness) under a single configurable surface.
"""
