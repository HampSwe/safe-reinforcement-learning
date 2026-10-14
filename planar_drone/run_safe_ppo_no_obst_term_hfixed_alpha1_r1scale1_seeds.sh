#!/usr/bin/env bash
# SafePPO (experiment_15 settings, rsl_rl/algorithms/safe_ppo.py from the 5-seed hand-off) on the planar task
# WITHOUT obstacle termination and with r1 = 30 inside the unsafe set (experiment_24 = experiment_23 + h_schedule on, 2026-09-19); 2048 envs.
# Collision rate: Episode/ep_hit_obstacle.
#
# Usage:  bash run_safe_ppo_no_obst_term_hfixed_alpha1_r1scale1_seeds.sh [iterations] [seeds...]
#         e.g. bash run_safe_ppo_no_obst_term_hfixed_alpha1_r1scale1_seeds.sh 24000 1   (24000 x 100 = 6000 x 400 env steps)
# Output: logs/expN_<run_name>/ (tensorboard events, config_used.json, exported/policy.{pt,onnx})
#         + console log per seed in logs/console/<run_name>.log
# Restart-safe: a seed whose run dir already has exported/policy.onnx is skipped.

set -uo pipefail
cd "$(dirname "$0")"

ITERS="${1:-24000}"; shift || true
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1 2 3 4 5)
CONFIG="experiment_30_safe_ppo_no_obst_term_hfixed_alpha1_r1scale1.yaml"
NAME_BASE="safe_ppo_no_obst_term_hfixed_alpha1_r1scale1_${ITERS}it"
PY="$(cd .. && pwd)/.venv-linux/bin/python"
# torch.compile (inductor) needs a C compiler; the venv has none on PATH, so borrow the conda one.
export CC="${CC:-$HOME/miniconda3/envs/drone/bin/gcc}"
export CXX="${CXX:-$HOME/miniconda3/envs/drone/bin/g++}"

mkdir -p logs/console
log() { echo "[batch $(date '+%F %T')] $*"; }

for seed in "${SEEDS[@]}"; do
    run_name="${NAME_BASE}_seed_${seed}"
    if ls -d logs/exp*_"${run_name}" >/dev/null 2>&1 && ls logs/exp*_"${run_name}"/exported/policy.onnx >/dev/null 2>&1; then
        log "seed ${seed}: already complete, skipping"; continue
    fi
    log "seed ${seed}: starting (${CONFIG}, ${ITERS} iterations)"
    t0=$(date +%s)
    "$PY" train.py --config "$CONFIG" --iterations "$ITERS" --device gpu \
        --run-name "$run_name" --seed "$seed" --no-show-dashboard-after \
        > "logs/console/${run_name}.log" 2>&1
    rc=$?
    log "seed ${seed}: exit code ${rc} after $(( ($(date +%s) - t0) / 60 )) min"
done
log "all seeds done"
