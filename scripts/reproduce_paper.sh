#!/usr/bin/env bash
# Reproduce the prompt-optimization rows of the agent ablation
# (Fig. agent_vs_gepa) at the paper's matched 4000-eval / $400 budget.
#
# Each (task, optimizer) cell is one run. Optimizers are wrapped in the
# OMNI pipeline (`adapter=omni`) so the two cells per task share an
# identical eval budget and split-visibility setup; the only thing that
# varies is the inner adapter config (gepa vs claude_code_agent).
#
# Usage:
#   bash scripts/reproduce_paper.sh                  # all 3 tasks, both optimizers
#   bash scripts/reproduce_paper.sh finer            # one task, both optimizers
#   OPTIMIZERS="gepa"        bash scripts/reproduce_paper.sh   # only plain GEPA
#   OPTIMIZERS="claude_code_agent" bash scripts/reproduce_paper.sh   # only GEPA-Agent

set -euo pipefail

OPTIMIZERS=${OPTIMIZERS:-"gepa claude_code_agent"}
TASKS=${*:-"finer formula livebench_math"}

# Paper-faithful per-task config overrides.
declare -A SUBSAMPLE_SEED=(
  [finer]=0
  [formula]=0
  [livebench_math]=""           # frozen split; no subsample
)
declare -A SUBSAMPLE_N=(
  [finer]="100 100 150"
  [formula]="100 100 150"
  [livebench_math]=""
)

run_cell () {
  local task="$1"
  local opt="$2"
  local extra=""
  if [[ -n "${SUBSAMPLE_SEED[$task]}" ]]; then
    read -r n_tr n_va n_te <<<"${SUBSAMPLE_N[$task]}"
    extra="task.subsample_seed=${SUBSAMPLE_SEED[$task]} \
           task.subsample_train=${n_tr} \
           task.subsample_val=${n_va} \
           task.subsample_test=${n_te}"
  fi
  echo "=== ${task} | adapter.configs=[${opt}] ==="
  python -m terrarium \
    task.name="${task}" \
    adapter=omni \
    adapter.configs="[${opt}]" \
    adapter.strategy=sequential \
    benchmark.max_evals=4000 \
    benchmark.max_token_cost=400 \
    ${extra}
}

for task in ${TASKS}; do
  for opt in ${OPTIMIZERS}; do
    run_cell "${task}" "${opt}"
  done
done
