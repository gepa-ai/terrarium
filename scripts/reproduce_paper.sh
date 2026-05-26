#!/usr/bin/env bash
# Reproduce the prompt-optimization rows of the agent ablation
# (Fig. agent_vs_gepa) at the paper's matched 4000-eval / $400 budget.
#
# Each (task, optimizer) cell is one seed=0 run. Optimizers are wrapped
# in the OMNI pipeline (`adapter=omni`) so the two cells per task share
# an identical eval budget and split-visibility setup; the only thing
# that varies is the inner adapter config (gepa vs claude_code_agent).
#
# Solver LM:        openai/gpt-5-mini      (matches paper)
# Reflection LM:    claude-sonnet-4-6      (matches paper)
#
# Usage:
#   bash scripts/reproduce_paper.sh                  # all 3 tasks, both optimizers
#   bash scripts/reproduce_paper.sh finer            # one task, both optimizers
#   OPTIMIZERS="gepa"        bash scripts/reproduce_paper.sh   # only plain GEPA
#   OPTIMIZERS="claude_code_agent" bash scripts/reproduce_paper.sh   # only GEPA-Agent
#
# Override the solver or reflection LM if needed:
#   SOLVER_LM=openai/gpt-4o REFLECTION_LM=anthropic/claude-3-7-sonnet ...

set -euo pipefail

OPTIMIZERS=${OPTIMIZERS:-"gepa claude_code_agent"}
TASKS=${*:-"finer formula livebench_math"}
SOLVER_LM=${SOLVER_LM:-"openai/gpt-5-mini"}
REFLECTION_LM=${REFLECTION_LM:-"claude-sonnet-4-6"}
MAX_EVALS=${MAX_EVALS:-4000}
MAX_TOKEN_COST=${MAX_TOKEN_COST:-400}

# Paper-faithful per-task subsample config.
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

# Map optimizer name -> adapter.configs entry. claude_code_agent is the
# agentic reflection backend used for GEPA-Agent.
declare -A REFLECTION_OVERRIDE=(
  [gepa]="adapter.configs.0.config.reflection.reflection_lm=${REFLECTION_LM}"
  [claude_code_agent]="adapter.configs.0.config.model=${REFLECTION_LM}"
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
  echo "=== ${task} | adapter.configs=[${opt}] | solver=${SOLVER_LM} | reflection=${REFLECTION_LM} ==="
  python -m terrarium \
    task.name="${task}" \
    task.solver_lm="${SOLVER_LM}" \
    adapter=omni \
    adapter.configs="[{backend: ${opt}}]" \
    adapter.strategy=sequential \
    ${REFLECTION_OVERRIDE[$opt]} \
    benchmark.max_evals="${MAX_EVALS}" \
    benchmark.max_token_cost="${MAX_TOKEN_COST}" \
    ${extra}
}

for task in ${TASKS}; do
  for opt in ${OPTIMIZERS}; do
    run_cell "${task}" "${opt}"
  done
done
