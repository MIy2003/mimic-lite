#!/usr/bin/env bash
#SBATCH --job-name=mimiclite-chip-axis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=64
#SBATCH --account=research
#SBATCH --qos=lv0b
#SBATCH --gres=gpu:4
#SBATCH --partition=HGX,DGX
#SBATCH --output=/share/ml/yangmin/mimiclite-chip/records/train-%j.slurm.log

set -eo pipefail
source /etc/profile
module add slurm cuda12.8/12.8
set -u
: "${SLURM_JOB_ID:?Submit using sbatch}"
root=/share/ml/yangmin/mimiclite-chip
cd "$root/active-adaptation"
export PATH="$PWD/venv/mjlab/.venv/bin:$PATH"
export HF_HOME="$root/hf" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS=0 OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="$root/cache/matplotlib" TMPDIR="$root/tmp"
export WANDB_MODE=offline WANDB_ENTITY=YM42
# Fresh run: do not inherit a checkpoint/run id from the submitting shell.
unset WANDB_RUN_ID WANDB_RESUME CHIP_CHECKPOINT
export WANDB_DIR="$root/records/train-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$WANDB_DIR"
python projects/mimic-lite/scripts/prepare_chip_loco.py
srun --ntasks=1 nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
# Slurm owns CUDA_VISIBLE_DEVICES. Never hard-code physical GPU indices here.
exec srun --ntasks=1 --kill-on-bad-exit=1 \
  torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  projects/mimic-lite/scripts/train.py \
  --config-dir "$PWD/.cache/chip_loco" task=chip_loco_train backend=mjlab \
  task.command.chip.compliance_mode=wrist_axis \
  task.num_envs=16384 total_iters=30000 checkpoint_path=null \
  wandb.mode=offline wandb.id=null \
  "exp_name=chip-axis-lambda-16384x4-30000-${SLURM_JOB_ID}" \
  "hydra.run.dir=$WANDB_DIR/output" checkpoint_interval=500 upload_interval=500 \
  2>&1 | tee "$WANDB_DIR/train.log"
