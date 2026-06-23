#!/usr/bin/env bash
# Reproduce the prompt-optimization rows of the agent ablation
# (Fig. agent_vs_gepa) at the paper's matched 4000-eval / $400 budget.
#
# Each (task, optimizer) cell is one seed=0 run. Both optimizers run
# under the optimize_anything pipeline with engine=gepa; what makes one of them
# "GEPA-Agent" is the presence of the claude_code_agent.model nested
# config, which swaps the LLM reflection proposer for the agentic one.
#
# Solver LM:        openai/gpt-5-mini      (matches paper)
# Reflection / Agent model: claude-sonnet-4-6 (matches paper)
#
# Usage:
#   bash scripts/reproduce_paper.sh                  # all 3 tasks, both optimizers
#   bash scripts/reproduce_paper.sh finer            # one task, both optimizers
#   OPTIMIZERS="gepa"       bash scripts/reproduce_paper.sh   # plain GEPA only
#   OPTIMIZERS="gepa_agent" bash scripts/reproduce_paper.sh   # GEPA-Agent only
#
# Override LMs:
#   SOLVER_LM=openai/gpt-4o REFLECTION_LM=anthropic/claude-3-7-sonnet ...

set -euo pipefail

OPTIMIZERS=${OPTIMIZERS:-"gepa gepa_agent"}
TASKS=${*:-"finer formula livebench_math"}
SOLVER_LM=${SOLVER_LM:-"openai/gpt-5-mini"}
REFLECTION_LM=${REFLECTION_LM:-"claude-sonnet-4-6"}
MAX_EVALS=${MAX_EVALS:-4000}
MAX_TOKEN_COST=${MAX_TOKEN_COST:-400}
SEED=${SEED:-0}

# Paper-faithful per-task subsample config (matches paper Table tab:tasks).
declare -A SUBSAMPLE_SEED=(
  [finer]=0
  [formula]=0
  [livebench_math]=""           # frozen 100/100/168 stratified split; no subsample
)
declare -A SUBSAMPLE_N=(
  [finer]="100 100 150"
  [formula]="100 100 150"
  [livebench_math]=""
)

# Map optimizer name -> the adapter.configs payload that selects it.
# Both use engine=gepa; "gepa_agent" adds the claude_code_agent block
# to swap in the agentic reflection proposer.
configs_for () {
  case "$1" in
    gepa)
      printf '[{engine: gepa, engine_config: {engine: {seed: %s}, reflection: {reflection_lm: %s}}}]' \
        "${SEED}" "${REFLECTION_LM}"
      ;;
    gepa_agent)
      printf '[{engine: gepa, engine_config: {engine: {seed: %s}, claude_code_agent: {model: %s}}}]' \
        "${SEED}" "${REFLECTION_LM}"
      ;;
    *)
      echo "unknown optimizer: $1" >&2
      exit 1
      ;;
  esac
}

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
  local configs
  configs=$(configs_for "${opt}")
  echo "=== ${task} | ${opt} | solver=${SOLVER_LM} | reflection=${REFLECTION_LM} | seed=${SEED} ==="
  python -m terrarium \
    task.name="${task}" \
    task.solver_lm="${SOLVER_LM}" \
    adapter=optimize_anything \
    adapter.strategy=sequential \
    adapter.configs="${configs}" \
    benchmark.max_evals="${MAX_EVALS}" \
    benchmark.max_token_cost="${MAX_TOKEN_COST}" \
    ${extra}
}

for task in ${TASKS}; do
  for opt in ${OPTIMIZERS}; do
    run_cell "${task}" "${opt}"
  done
done
