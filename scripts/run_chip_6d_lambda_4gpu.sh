#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
root=${MIMICLITE_ROOT:-/share/ml/yangmin/mimiclite-chip}
cd "$root/active-adaptation"
export PATH="$PWD/venv/mjlab/.venv/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/projects/mimic-lite:$root/any4hdmi/src"
export ACTIVE_ADAPTATION_PROJECTS_FILE="$PWD/.cache/projects.json"
export HF_HOME="$root/hf" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS=0 OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="$root/cache/matplotlib" TMPDIR="$root/tmp"
export WANDB_MODE=offline WANDB_ENTITY=${WANDB_ENTITY:-YM42}
unset WANDB_RUN_ID WANDB_RESUME CHIP_CHECKPOINT
num_envs=${NUM_ENVS:-16384}
total_iters=${TOTAL_ITERS:-30000}
run_dir="$root/records/chip-6d-${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT:-0}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$run_dir"
export WANDB_DIR="$run_dir"
python projects/mimic-lite/scripts/prepare_chip_loco.py --sixd --output "$PWD/.cache/chip_6d_loco"
srun --ntasks=1 nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
# Use Slurm's allocated CUDA devices. Short full-size check precedes production.
if [[ ${RUN_SMOKE:-1} == 1 ]]; then
    srun --ntasks=1 --kill-on-bad-exit=1 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
        projects/mimic-lite/scripts/train.py --config-dir "$PWD/.cache/chip_6d_loco" \
        task=chip_6d_loco_train backend=mjlab headless=true checkpoint_path=null \
        "task.num_envs=$num_envs" total_iters=6 wandb.mode=disabled wandb.id=null \
        task.command.chip.warmup_steps=0 task.command.chip.ramp_steps=0 \
        task.command.chip.force_interval=0 "hydra.run.dir=$run_dir/smoke" \
        "exp_name=chip-6d-size-check-${SLURM_JOB_ID}" 2>&1 | tee "$run_dir/smoke.log"
    srun --ntasks=1 nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
fi
srun --ntasks=1 --kill-on-bad-exit=1 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    projects/mimic-lite/scripts/train.py --config-dir "$PWD/.cache/chip_6d_loco" \
    task=chip_6d_loco_train backend=mjlab headless=true checkpoint_path=null \
    "task.num_envs=$num_envs" "total_iters=$total_iters" wandb.mode=offline wandb.id=null \
    "exp_name=chip-6d-lambda-${num_envs}x4-${SLURM_JOB_ID}" \
    "hydra.run.dir=$run_dir/train" checkpoint_interval=500 upload_interval=500 \
    "$@" 2>&1 | tee "$run_dir/train.log"
