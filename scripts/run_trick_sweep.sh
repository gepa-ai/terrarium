#!/usr/bin/env bash
# scripts/run_trick_sweep.sh
#
# Reproducible launcher for the trick-task optimizer comparison on
# needle_in_range / slot_machines across adapters and seeds.
#
# Usage:
#   scripts/run_trick_sweep.sh <task> <config> <seed> [extra hydra overrides...]
#   scripts/run_trick_sweep.sh needle-all      # all configs x seeds for needle
#   scripts/run_trick_sweep.sh slots-all       # all configs x seeds for slots
#   scripts/run_trick_sweep.sh all             # both tasks, full matrix (24 runs)
#
#   <task>   : needle | slots
#   <config> : mh | gepa | autoresearch | gepa_cca
#   <seed>   : integer
#              - needle: tags the run; used as gepa engine seed
#              - slots:  used as task.seed (the hidden environment) AND gepa seed
#
# Results land in:
#   outputs/<task>_sweep/<config>_seed<seed>/        (hydra run dir)
#   logs/<task>_sweep/<config>_seed<seed>.research.jsonl   (sidecar)
#   logs/<task>_sweep/<config>_seed<seed>.log              (stdout/stderr)
#
# All experiment knobs are pinned below so runs are reproducible and the
# config is diffable in version control. Override any of them via env vars,
# e.g.  BUDGET_MAX_TOKEN_COST=10.0 scripts/run_trick_sweep.sh needle gepa 0
#
# WARNING: `all` launches 24 concurrent runs, each capped at
# BUDGET_MAX_TOKEN_COST. Know your spend before running it.

set -euo pipefail

# ---- Pinned experiment configuration ---------------------------------------
BUDGET_MAX_EVALS="${BUDGET_MAX_EVALS:-50}"
BUDGET_MAX_TOKEN_COST="${BUDGET_MAX_TOKEN_COST:-40.0}"
NEEDLE_N="${NEEDLE_N:-50}"
SLOTS_N="${SLOTS_N:-5}"
SLOTS_M="${SLOTS_M:-50}"
GEPA_CCA_MODEL="${GEPA_CCA_MODEL:-claude_code_agent/sonnet}"
NEEDLE_SEEDS=(${NEEDLE_SEEDS:-0 1 2})
SLOTS_SEEDS=(${SLOTS_SEEDS:-42 0 1})
CONFIGS=(mh gepa autoresearch gepa_cca)

# ---- Environment -----------------------------------------------------------
# The cluster box has a broken $HOME (/home/eecs/... is missing); dspy_cache
# and the `claude` CLI need these. Override TERRARIUM_* for other machines.
export HOME="${TERRARIUM_HOME:-/data/lukedhlee}"
export PATH="/data/lukedhlee/.local/bin:$PATH"
REPO="${TERRARIUM_REPO:-/data/lukedhlee/omni_terrarium/terrarium}"
PY="${TERRARIUM_PY:-$REPO/.venv/bin/python3}"
ZSHRC="${TERRARIUM_ZSHRC:-/data/lukedhlee/.zshrc}"
[ -f "$ZSHRC" ] && eval "$(grep -E '^export (ANTHROPIC|OPENAI|OPENROUTER|GEMINI)_API_KEY=' "$ZSHRC")"
cd "$REPO"

# Adapter-specific hydra overrides. $1=config, $2=seed.
adapter_args() {
  case "$1" in
    mh)           echo "adapter=meta_harness adapter.max_iterations=${BUDGET_MAX_EVALS}" ;;
    gepa)         echo "adapter=gepa adapter.engine.seed=$2 adapter.engine.cache_evaluation=false" ;;
    autoresearch) echo "adapter=claude_code" ;;
    gepa_cca)     echo "adapter=gepa adapter.reflection.reflection_lm=${GEPA_CCA_MODEL} adapter.engine.seed=$2 adapter.engine.cache_evaluation=false" ;;
    *) echo "unknown config: $1 (want: mh|gepa|autoresearch|gepa_cca)" >&2; exit 2 ;;
  esac
}

launch() {
  local task="$1" config="$2" seed="$3"; shift 3
  local tag="${config}_seed${seed}"
  local task_args sweep logvar
  if [ "$task" = "needle" ]; then
    task_args="task=needle_in_range task.n=${NEEDLE_N}"
    sweep="needle_sweep"; logvar="TERRARIUM_NEEDLE_RESEARCH_LOG"
  elif [ "$task" = "slots" ]; then
    task_args="task=slot_machines task.n=${SLOTS_N} task.m=${SLOTS_M} task.seed=${seed}"
    sweep="slots_sweep"; logvar="TERRARIUM_SLOTS_RESEARCH_LOG"
  else
    echo "unknown task: $task (want: needle|slots)" >&2; exit 2
  fi
  local logdir="$REPO/logs/${sweep}"
  mkdir -p "$logdir"
  local sidecar="$logdir/${tag}.research.jsonl"
  local runlog="$logdir/${tag}.log"
  rm -f "$sidecar" "$runlog"
  rm -rf "$REPO/outputs/${sweep}/${tag}"
  echo "[launch] $task/$tag -> outputs/${sweep}/${tag} (cap \$${BUDGET_MAX_TOKEN_COST}, ${BUDGET_MAX_EVALS} evals)"
  env "${logvar}=${sidecar}" setsid nohup "$PY" -m terrarium \
    ${task_args} $(adapter_args "$config" "$seed") \
    budget.max_evals="${BUDGET_MAX_EVALS}" budget.max_token_cost="${BUDGET_MAX_TOKEN_COST}" \
    hydra.run.dir="outputs/${sweep}/${tag}" "$@" \
    </dev/null > "$runlog" 2>&1 &
  echo "  pid=$!  log=$runlog"
}

usage() { grep '^#' "$0" | sed 's/^#\{1,\} \{0,1\}//;1d'; }

main() {
  local mode="${1:-}"
  case "$mode" in
    "" | -h | --help) usage; exit 0 ;;
    needle-all) for c in "${CONFIGS[@]}"; do for s in "${NEEDLE_SEEDS[@]}"; do launch needle "$c" "$s"; done; done ;;
    slots-all)  for c in "${CONFIGS[@]}"; do for s in "${SLOTS_SEEDS[@]}";  do launch slots  "$c" "$s"; done; done ;;
    all)
      for c in "${CONFIGS[@]}"; do for s in "${NEEDLE_SEEDS[@]}"; do launch needle "$c" "$s"; done; done
      for c in "${CONFIGS[@]}"; do for s in "${SLOTS_SEEDS[@]}";  do launch slots  "$c" "$s"; done; done
      ;;
    needle | slots)
      [ $# -ge 3 ] || { echo "need: $0 $mode <config> <seed>" >&2; exit 1; }
      launch "$@"
      ;;
    *) echo "unknown mode: $mode" >&2; usage; exit 1 ;;
  esac
  sleep 3
  echo "--- live terrarium runs ---"
  pgrep -af "python.*terrarium" | grep -cE "needle_sweep|slots_sweep" || true
}

main "$@"
