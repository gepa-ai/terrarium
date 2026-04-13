"""Allow running as ``python -m terrarium [hydra overrides...]``.

Hydra-managed entry. Override anything in ``terrarium/conf/`` from the CLI::

    python -m terrarium                              # defaults
    python -m terrarium task=aime_math max_evals=200
    python -m terrarium adapter=claude_code adapter.model=opus
    python -m terrarium adapter=custom adapter.path=./my_adapter.py
    python -m terrarium adapter.engine.max_workers=32 adapter.reflection.reflection_minibatch_size=5
    python -m terrarium tracking=wandb tracking.wandb_project=foo
    python -m terrarium --help                       # show config groups and defaults
"""

from terrarium.runner import main

main()
