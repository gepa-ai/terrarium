"""Allow running as ``python -m terrarium [hydra overrides...]``.

Hydra-managed entry. Override anything in ``terrarium/conf/`` from the CLI::

    python -m terrarium                              # defaults (adapter=optimize_anything)
    python -m terrarium task=aime_math max_evals=200
    python -m terrarium adapter.engine=autoresearch adapter.engine_config.model=opus
    python -m terrarium adapter=custom adapter.path=./my_adapter.py
    python -m terrarium adapter.engine_config.engine.max_workers=32 adapter.engine_config.reflection.reflection_minibatch_size=5
    python -m terrarium tracking=wandb tracking.wandb_project=foo
    python -m terrarium --help                       # show config groups and defaults
"""

from terrarium.runner import main

main()
