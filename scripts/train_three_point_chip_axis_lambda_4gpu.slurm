#!/usr/bin/env bash
#SBATCH --job-name=mimiclite-3point-axis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=64
#SBATCH --account=research
#SBATCH --qos=lv0b
#SBATCH --gres=gpu:4
#SBATCH --partition=HGX,DGX
#SBATCH --output=/share/ml/yangmin/mimiclite-chip/records/three-point-axis-%j.slurm.log

set -euo pipefail
export THREE_POINT_VARIANT=three_point_chip_axis
root="${MIMICLITE_ROOT:-/share/ml/yangmin/mimiclite-chip}"
exec bash "$root/active-adaptation/projects/mimic-lite/scripts/train_three_point_chip_lambda_4gpu.slurm"
