# Terrarium Benchmark Spec

This document defines the benchmark contract that official Terrarium runs must
follow. It is intentionally small: implementation details may vary across
adapters, but the visibility, budget, reporting, and access rules must be
explicit and comparable.

## Modes

Terrarium supports three benchmark modes.

### Single

`single` mode optimizes one fixed task/objective.

- The task YAML declares `mode: single`.
- The optimizer may evaluate candidate artifacts until the run budget is
  exhausted or an early-stop threshold is reached.
- The official score is the best score found within budget.
- Dataset splits are not part of the benchmark contract for this mode.

### Generalization

`generalization` mode optimizes using visible development data and reports final
performance on hidden test data.

- The task YAML declares `mode: generalization`.
- The task must provide `train_set`.
- Official runs must provide `test_set`.
- `val_set` is optional.
- If `benchmark.use_val=true`, `val_set` may be used during search when present.
- If `benchmark.use_val=false`, `val_set` is removed before adapters see the
  task.
- `test_set` is always hidden during search.
- The official score is the hidden-test score of the final selected artifact.

### Multi-Task Search

`multi_task` mode optimizes one artifact across a visible set of task instances.

- The task YAML declares `mode: multi_task`.
- The task must provide `train_set`.
- `val_set` is optional and visible when `benchmark.use_val=true`.
- `val_set` may intentionally be the same examples as `train_set` when the
  benchmark is a visible search benchmark rather than a generalization
  benchmark.
- `test_set` is not part of the official contract for this mode.
- The official score is the best visible-set score found within budget.

## Splits

Split visibility is part of the benchmark condition.

- `train_set` is visible during search.
- `val_set` is visible only when `benchmark.use_val=true`.
- `test_set` is never visible during search.
- Hidden test examples must not be exposed through adapter prompts, workspaces,
  eval APIs, cached dataset files, prior transcripts, or filesystem access.
- Train/val/test example IDs must not overlap for official generalization runs.
- Multi-task search may reuse visible examples across `train_set` and `val_set`
  when the task explicitly declares that policy.
- Each task should record split provenance: source dataset, split method, split
  seed when applicable, and split sizes.

## Budget

Budgets are per run.

- `budget.max_evals` caps evaluation calls.
- `budget.max_token_cost` caps adapter model spend in USD when supported by the
  adapter.
- At least one budget cap must be set.
- Train and validation evaluations consume budget.
- Crashes, timeouts, invalid outputs, and retries consume budget when they reach
  the evaluator or adapter budget mechanism.
- Hidden-test evaluation happens after search and is not available to the
  optimizer.

## Reporting

Official reports should include:

- task name and mode
- adapter name and version/config
- split policy, including `benchmark.use_val`
- enforced budget caps and budget used
- wall-clock time
- optimizer-model cost
- task/student/target-model cost when applicable
- total cost
- best in-budget score
- final hidden-test score for generalization mode
- run provenance and access policy

Paper results should use three runs and report both mean and median.

## Access Policy

Every official run must record what the optimizer was allowed to access.

The access policy must cover:

- readable paths
- writable paths
- execution permissions
- network permissions
- cache and prior-artifact access
- external tool access

Different access policies are different benchmark conditions and should be
reported separately.

By default, official runs should use clean run state and should not allow access
to hidden test data, evaluator internals, prior run transcripts, sibling run
outputs, or shared dataset caches such as Hugging Face cache.

For Claude Code adapters on Linux, the current official sandbox policy is:

- filesystem access is confined with bwrap to the adapter work directory plus
  the minimal runtime paths needed by Claude and standard tools
- each Claude subprocess gets an isolated `HOME` with copied top-level auth and
  config, an empty `.claude/projects`, and no shared `.cache`
- WebFetch and WebSearch are denied at the Claude tool layer
- the network namespace is shared with the host, so sandboxed Bash can still
  reach the local eval server and provider APIs; arbitrary outbound Bash
  network is therefore part of this access policy and must not be mixed with
  stricter network-isolated results

## Current Config Surface

The minimal current Hydra surface is:

```yaml
benchmark:
  mode: null
  use_val: true

access_policy:
  readable_paths: []
  writable_paths: []
  execution: sandboxed
  network: model_api_and_eval_server
  shared_cache: false
  prior_artifacts: false
  external_tools: []
```

`benchmark.mode: null` means the runner uses `task.mode` from the selected task
YAML. Setting `benchmark.mode` explicitly overrides the task default.

Supported modes are `single`, `multi_task`, and `generalization`.

`benchmark.use_val` controls whether validation examples are available during
search. It does not affect hidden-test evaluation in `generalization` mode.
