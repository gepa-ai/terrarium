# Terrarium TODO

[Make sure you update this TODO.md whenever you finish a task.]

## Current Priorities

1. Done: write `docs/benchmark_spec.md`.
   - Define `single`, `multi_task`, and `generalization` mode.
   - Define `benchmark.use_val`.
   - State that `test_set` is always hidden during search.
   - Define budget accounting and final hidden-test reporting.
   - Document that task YAML declares `task.mode`.

2. Done: make dataset splits fair and auditable.
   - Moved `arc_agi` validation examples from `metadata["val_set"]` to `Task.val_set`.
   - Added startup validation for generalization tasks:
     - require `train_set`
     - require `test_set` for official hidden-test reporting
     - fail if train/val/test example IDs overlap
   - Added split provenance to task metadata:
     - source dataset
     - split method
     - split seed
     - split sizes
   - Reviewed `cloudcast` and classified it as `multi_task`, where shared train/val examples are intentional.
   - Added reduced-run split limits (`task.train_limit`, `task.val_limit`, `task.test_limit`) for smoke/debug runs.
   - Ran actual Hydra smoke runs for `aime_math_mini` and `arc_agi` with one visible train eval and one hidden post-search test eval.
   - Confirmed `summary.json` reports resolved `benchmark.mode`, `access_policy`, budget use, and hidden-test scores.
   - Done after self-audit: `run()` now removes `test_set` before handing the task to adapters or the eval server, while retaining it only for runner-owned post-search hidden scoring.
   - Done after self-audit: eval APIs reject hidden/unknown example IDs and invalid split names instead of silently falling back.
   - Done after self-audit: ARC-AGI seed parsing and test-grid scoring now handle normal grid outputs and two-attempt output lists correctly.
   - Re-ran reduced real smokes after audit fixes:
     - `outputs/smoke/aime_math_mini_post_audit`
     - `outputs/smoke/arc_agi_gemini_post_parser_fix`

3. Done: harden Claude Code sandboxing for official runs.
   - Done: use a per-run/per-agent `HOME`.
   - Done: copy only top-level Claude auth/config needed by the CLI.
   - Done: start with an empty `.claude/projects` directory.
   - Done: do not bind shared `~/.cache` by default.
   - Done: add `scripts/sandbox_probe.py` for the isolated-home/cache/sibling-path checks.
   - Done: expand `scripts/sandbox_probe.py` into a live Linux bwrap/Bash probe covering repo-root, sibling-run, prior-transcript, cache, hidden-data, workdir read/write, and localhost eval-server access.
   - Done: make cache access an explicit benchmark policy field.
   - Done: add tests confirming Claude Code and MetaHarness materialize visible train/val splits but never hidden test splits.
   - Done: document the Linux network policy. Current official policy keeps the host network namespace shared so local eval-server and provider API access work; arbitrary outbound Bash network is part of this access policy and results should not be pooled with stricter network-isolated runs.

## Sandbox Probe Tests

Done: `scripts/sandbox_probe.py` now proves that sandboxed Bash inside the
Linux bwrap jail cannot read:

- repo root outside the adapter work dir
- sibling run output directories
- prior Claude transcripts under `.claude/projects`
- Hugging Face-style cache data under `.cache/huggingface`
- hidden test files or serialized task data outside the adapter work dir

It also verifies sandboxed Bash can still:

- read/write the adapter work dir
- call a localhost eval-server-style HTTP endpoint

Network policy:

- the current Linux bwrap sandbox shares the host network namespace. This keeps
  local eval-server and provider API access working, while filesystem isolation,
  isolated Claude home/cache/transcripts, and WebFetch/WebSearch denial cover
  the main leakage channels. Arbitrary outbound Bash network is part of this
  access policy and should not be pooled with stricter network-isolated runs.
  The completed Gemini smoke run verifies model API reachability under this
  policy.
