#!/usr/bin/env bash
# Linux port of run_t_no_bootstrap_seeds.ps1: sequential seeds of the planar-drone RL-SGF
# "T + no bootstrap + reset after update + 3 critic epochs" case (experiment_05), 2048 envs
# (make_env in train.py), with the per-episode termination-rate logging added to environment.py
# on 2026-09-12 (Episode/ep_term_obstacle, ep_term_crashed, ep_term_oob, ep_term_any, ep_timeout).
#
# Usage:  bash run_t_no_bootstrap_seeds.sh [iterations] [seeds...]
#         e.g. bash run_t_no_bootstrap_seeds.sh 6000 1 2 3 4 5
# Output: logs/expN_<run_name>/ (tensorboard events, config_used.json, exported/policy.{pt,onnx})
#         + console log per seed in logs/console/<run_name>.log
# Restart-safe: a seed whose run dir already has exported/policy.onnx is skipped.

set -uo pipefail
cd "$(dirname "$0")"

ITERS="${1:-6000}"; shift || true
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1 2 3 4 5)
CONFIG="experiment_05_no_bootstrap_t_reset_critic_epochs.yaml"
NAME_BASE="t_no_bootstrap_reset_critic_epochs_${ITERS}it"
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
