#!/usr/bin/env bash
# Run on the login node, preferably inside a named tmux session.
set -eo pipefail
source /etc/profile
module add slurm cuda12.8/12.8
set -u
cd /share/ml/yangmin/mimiclite-chip/active-adaptation
exec salloc -N 1 -t 8:00:00 --cpus-per-task 64 --account=research --qos=lv0a \
  --job-name dexhand --gres=gpu:1 -p HGX,DGX \
  srun --ntasks=1 --kill-on-bad-exit=1 bash projects/mimic-lite/scripts/profile_three_point_chip_lambda.sh
