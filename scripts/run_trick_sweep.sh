#!/usr/bin/env bash
# scripts/run_trick_sweep.sh
#
# Thin, portable wrapper around the Hydra multirun for the trick-task
# optimizer comparison (needle_in_range / slot_machines).
#
# It is intentionally machine-agnostic: it sets NO paths, NO $HOME, NO
# $PATH, and does not source any shell rc. It assumes `python -m terrarium`
# runs in your active environment (venv/conda activated, provider API keys
# already exported) — exactly like any other terrarium invocation.
#
# All configuration is version-controlled Hydra config, so runs are
# reproducible and diffable:
#   conf/experiment/trick_{needle,slots}.yaml   — task + budget
#   conf/optimizer/{mh,gepa,autoresearch,gepa_cca}.yaml — adapter bundles
#
# Each run's research sidecar is written automatically to the per-run
# Hydra output dir (multirun/<date>/<n>/research.jsonl) — no env vars.
#
# Usage:
#   scripts/run_trick_sweep.sh needle
#   scripts/run_trick_sweep.sh slots
#   scripts/run_trick_sweep.sh needle gepa
#   scripts/run_trick_sweep.sh slots gepa,mh 42,0,1
#   scripts/run_trick_sweep.sh needle gepa,mh '' budget.max_token_cost=10.0
#
# Args 2+ are optional; anything after arg 3 is forwarded verbatim to Hydra.
#
# Equivalent raw command (the wrapper just builds this):
#   python -m terrarium -m +experiment=trick_needle \
#     +optimizer=mh,gepa,autoresearch,gepa_cca
set -euo pipefail

PY="${PYTHON:-python}"
task="${1:-}"
optimizers="${2:-mh,gepa,autoresearch,gepa_cca}"
seeds="${3:-}"
[ $# -ge 1 ] && shift
[ $# -ge 1 ] && shift
[ $# -ge 1 ] && shift

case "$task" in
  needle)
    exp=trick_needle
    seed_arg=()                       # needle has no cross-adapter seed knob
    ;;
  slots)
    exp=trick_slots
    seed_arg=("task.seed=${seeds:-42,0,1}")
    ;;
  *)
    echo "usage: $0 <needle|slots> [optimizers] [seeds] [extra hydra args]" >&2
    exit 1
    ;;
esac

set -x
exec "$PY" -m terrarium -m \
  "+experiment=${exp}" \
  "+optimizer=${optimizers}" \
  "${seed_arg[@]}" \
  "$@"
