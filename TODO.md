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

4. Done: add and run a reduced benchmark smoke matrix.
   - Added `scripts/smoke_benchmark_matrix.py`.
   - The matrix covers:
     - `aime_math_mini` as `generalization`
     - `arc_agi` as `generalization`
     - `cloudcast` as `multi_task`
     - one `frontier_cs_algo_smoke` problem as `multi_task`
   - Dropped Circle Packing from this smoke matrix.
   - Verified fresh run artifacts under `outputs/smoke/benchmark_matrix_current`.
   - Confirmed meaningful evaluator output:
     - AIME hidden reduced `test_score: 1.0`
     - ARC visible reduced `best_score: 1.0`
     - Cloudcast positive cost score
     - Frontier-CS actual judge success on one problem; the empty seed scores `0.0`, as expected.
   - Confirmed Frontier-CS requires Python 3.11+ because the `frontier-cs`
     package is gated to that version range.

5. Done: remove Frontier-CS import-time side effects.
   - Problem observed during the reduced smoke matrix: unrelated tasks can still
     trigger Frontier-CS registration work at import time, including Hugging
     Face metadata/network probes.
   - Target behavior: importing/registering `terrarium.tasks` should be cheap
     and should not touch remote datasets for benchmarks that are not selected.
   - Added registry support for dynamic task-name resolvers.
   - Changed Frontier-CS registration to register only the smoke factory and a
     lazy `frontier_cs_algo_<id>` resolver; the Hugging Face problem list is
     loaded only when a selected Frontier-CS task is requested.
   - Added regression tests proving task listing does not load Frontier-CS rows
     and dynamic per-problem lookup still resolves lazily.
   - Re-ran the reduced smoke matrix under
     `outputs/smoke/benchmark_matrix_lazy_frontier`.
   - Confirmed non-Frontier smoke logs no longer contain Frontier-CS startup
     noise. AIME/ARC-AGI still perform their own expected Hugging Face dataset
     access.
   - Confirmed the one-problem Frontier-CS smoke still resolves and returns
     `EvaluationStatus.SUCCESS`.

6. Done: run reduced real-adapter smokes for Claude Code and MetaHarness.
   - Current status: contract tests cover Claude Code and MetaHarness sandbox
     materialization, and `scripts/sandbox_probe.py` verifies the live Claude
     Code-style filesystem sandbox.
   - Reduced run:
     - `aime_math_mini`
     - `task.train_limit=1`, `task.val_limit=0`, `task.test_limit=1`
     - `budget.max_evals=2`
     - Claude Code with low effort
     - MetaHarness with `adapter.max_iterations=1`,
       `adapter.max_candidates_per_iter=1`, low effort, and a cheap model
       override instead of the default Opus
   - Initial blocker resolved: after sourcing `~/.zshrc`, `ANTHROPIC_API_KEY`
     was available and both adapters could reach the Claude Code CLI.
   - Hardened failure behavior:
     - Claude Code adapter now raises on nonzero Claude subprocess exit instead
       of returning the seed candidate with zero evals.
     - MetaHarness now raises on proposer subprocess failure instead of writing
       a misleading benchmark summary with hidden seed scoring.
     - Claude Code adapter now raises if the subprocess completes without
       calling the eval server.
     - MetaHarness now raises if the proposer exits successfully but emits no
       candidates.
     - Added regression tests for these failure paths.
   - Successful smoke artifacts:
     - `outputs/smoke/claude_code_aime_mini_eval_required`
       - `total_evals: 1`
       - `best_score: 0.0`
       - `test_score: 1.0`
       - `total_cost: 0.142107`
     - `outputs/smoke/meta_harness_aime_mini_eval_required`
       - `total_evals: 1`
       - `best_score: 1.0`
       - `test_score: 1.0`
       - `total_cost: 0.1195923`
   - Confirmed each adapter launches successfully through the real Claude Code
     CLI path, eval-server budget accounting moves during search, hidden test
     remains runner-owned and post-search, and summary artifacts include
     `sandbox_scope` and access policy.
   - Re-ran the reduced real-adapter smokes with `sandbox=true`:
     - `outputs/smoke/claude_code_aime_mini_sandbox_eval_required`
       - `total_evals: 2`
       - `best_score: 1.0`
       - `test_score: 1.0`
       - `total_cost: 0.18476745`
       - `sandbox_scope.optimizer_subprocess_sandbox: true`
     - `outputs/smoke/meta_harness_aime_mini_sandbox_eval_required`
       - `total_evals: 1`
       - `best_score: 0.0`
       - `test_score: 0.0`
       - `total_cost: 0.09402405`
       - `sandbox_scope.optimizer_subprocess_sandbox: true`
   - Confirmed both real adapters can authenticate, propose, and call the eval
     server under the official filesystem sandbox path. MetaHarness produced a
     real evaluated candidate in the tiny smoke, but that candidate scored
     `0.0` on the one-example reduced AIME slice.

7. Done: resolve `adapter.max_turns` for Claude Code.
   - `adapter.max_turns` exists in the config and adapter constructor, but the
     installed Claude Code CLI (`2.1.126`) does not expose a `--max-turns`
     option in `claude --help`.
   - Removed `max_turns` from the Claude Code Hydra config.
   - Added a constructor guard so programmatic callers get a clear
     `ValueError` if they pass `max_turns`.
   - Continue bounding Claude Code runs with eval and token budgets.

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
