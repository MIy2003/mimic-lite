#!/usr/bin/env bash
#SBATCH --job-name=mimiclite-five-point
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --partition=HGX,DGX
#SBATCH --account=research
#SBATCH --qos=lv0b
#SBATCH --time=2-00:00:00
#SBATCH --output=/share/ml/yangmin/mimiclite-chip/records/chip-five-point-%j.slurm.log
set -euo pipefail
root=${MIMICLITE_ROOT:-/share/ml/yangmin/mimiclite-chip}
exec bash "$root/active-adaptation/projects/mimic-lite/scripts/run_chip_five_point_lambda_4gpu.sh" "$@"
