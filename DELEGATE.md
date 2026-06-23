# Delegate: Launch AIME GEPA vs AutoResearch Ralph Experiment

## Goal

Launch the Terrarium paper experiment comparing:

- GEPA via the optimize_anything adapter (engine=gepa)
- AutoResearch via the optimize_anything adapter (engine=autoresearch), with Ralph mode explicitly enabled

Task is full AIME generalization:

- visible development data: `AI-MO/aimo-validation-aime`, shuffled with seed 0 and split 15 train / 15 val
- hidden final test: `MathArena/aime_2025`, 30 examples

Do not inspect, expose, or use hidden test data during search. Terrarium handles final hidden-test scoring after the optimizer returns.

## Workspace

Use this repo:

```bash
cd /data/lukedhlee/omni_terrarium/terrarium
```

Terrarium imports vendored GEPA from:

```bash
/data/lukedhlee/omni_terrarium/terrarium/gepa
```

The local code already includes required uncommitted changes:

- GEPA Omni Claude Code backend supports Ralph resume mode.
- AIME config explicitly sets `++adapter.engine_config.ralph=true`.
- Launcher supports `max_parallel_runs` as the preferred name for process-level parallelism.

Do not revert local changes.

## Current Experiment Config

Config path:

```bash
benchmarks/configs/aime_gepa_ar_haiku_sonnet.yaml
```

Important settings:

- `max_parallel_runs: 8`
- `max_concurrency=32`
- `budget.max_evals=500`
- `budget.max_token_cost=10.0`
- GEPA reflection LM: `anthropic/claude-sonnet-4-6`
- AIME solver/evaluator LM: `anthropic/claude-haiku-4-5`
- GEPA internal workers: `++adapter.engine_config.engine.max_workers=32`
- AutoResearch model: `sonnet`
- AutoResearch Ralph: `++adapter.engine_config.ralph=true`
- GEPA split policy: `benchmark.split_train_val=true`
- AutoResearch split policy: `benchmark.split_train_val=false`

With the current `seeds: [0]`, there are only two runs: GEPA seed 0 and AutoResearch seed 0. `max_parallel_runs=8` matters more once more seeds are added.

## Preflight

Run these checks before launching:

```bash
uv run python -m py_compile \
  benchmarks/experiment_launcher.py \
  tests/test_experiment_launcher.py \
  gepa/src/gepa/omni/backends/claude_code.py \
  gepa/tests/test_omni_claude_code_backend.py

uv run --extra test python -m pytest tests/test_omni_claude_code_backend.py -q

uv run python -m unittest tests.test_benchmark_contract tests.test_experiment_launcher -v

uv run python benchmarks/experiment_launcher.py \
  benchmarks/configs/aime_gepa_ar_haiku_sonnet.yaml \
  --dry-run
```

Confirm the dry-run output command contains:

- GEPA: `max_concurrency=32`
- GEPA: `++adapter.engine_config.engine.max_workers=32`
- AutoResearch: `max_concurrency=32`
- AutoResearch: `++adapter.engine_config.ralph=true`

Known non-launch caveat:

- `gepa/uv.lock` currently has unrelated lockfile churn from test setup. This should be excluded or reverted before committing, but it is not a launch blocker.

## Launch

Start the experiment with:

```bash
uv run python benchmarks/experiment_launcher.py \
  benchmarks/configs/aime_gepa_ar_haiku_sonnet.yaml
```

The launcher writes an output root like:

```bash
outputs/experiments/aime_gepa_ar_haiku_sonnet/YYYYMMDD_HHMMSS/
```

Inside it, expect per-run directories:

```bash
aime_math__gepa__seed0/
aime_math__autoresearch__seed0/
```

Each run should eventually have:

```bash
summary.json
stdout.log
stderr.log
command.txt
```

The aggregate files are:

```bash
results.json
results.csv
```

## Monitoring

Find the newest experiment folder:

```bash
ls -td outputs/experiments/aime_gepa_ar_haiku_sonnet/* | head -1
```

Check aggregate status:

```bash
cat outputs/experiments/aime_gepa_ar_haiku_sonnet/*/results.json
```

Tail per-run logs:

```bash
tail -f outputs/experiments/aime_gepa_ar_haiku_sonnet/*/aime_math__gepa__seed0/stdout.log
tail -f outputs/experiments/aime_gepa_ar_haiku_sonnet/*/aime_math__autoresearch__seed0/stdout.log
```

For AutoResearch, confirm Ralph actually ran by checking the final `summary.json` metadata for `ralph_iterations > 1` when budget allowed. If `ralph_iterations == 1`, inspect the Claude Code stdout/stderr and budget status before assuming failure; it may have stopped because budget, score, error, or near-zero spend ended the loop.

## Interpreting Results

Primary reported fields:

- `best_score`: best visible/search score
- `test_score`: final hidden-test score after search
- `total_evals`
- `solver_cost_search`
- `solver_cost_test`
- `optimizer_cost`
- `total_cost`
- `wall_time`

For paper reporting, do not pool results with different access policies or split policies. GEPA and AutoResearch intentionally use different visible-data policies in this condition:

- GEPA keeps train/val separate.
- AutoResearch merges visible train+val into one train channel.

That is acceptable only if reported as part of the benchmark condition.

## If Launch Fails

Most likely issue is API pressure from `32` concurrency and `32` GEPA workers.

If rate limits, 429s, or provider instability dominate, stop the run and retry with:

```yaml
defaults:
  - max_concurrency=16

...
      - ++adapter.engine_config.engine.max_workers=16
```

Keep `budget.max_evals=500`, `budget.max_token_cost=10.0`, and `++adapter.engine_config.ralph=true` unchanged unless the experiment owner explicitly changes the condition.

## Do Not Do

- Do not launch old stopped run directories.
- Do not trust hidden-test scores from interrupted runs.
- Do not change split policy mid-experiment.
- Do not remove `++adapter.engine_config.ralph=true`.
- Do not commit `gepa/uv.lock` churn unless intentionally refreshing the GEPA lockfile.
