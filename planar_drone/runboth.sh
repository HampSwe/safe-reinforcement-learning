# runboth.sh
#!/usr/bin/env bash
#set -e                     # stop if any command fails; drop this line to run all regardless
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
python train.py --iterations 20000 --no-show-dashboard-after
python train.py --config config_b.yaml --iterations 20000 --no-show-dashboard-after

#nohup ./runboth.sh > runboth.out 2>&1 &
#tail -f runboth.out