#!/usr/bin/env bash
#SBATCH --job-name=mimiclite-three-point-chip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=64
#SBATCH --account=research
#SBATCH --qos=lv0b
#SBATCH --gres=gpu:4
#SBATCH --partition=HGX,DGX
#SBATCH --output=/share/ml/yangmin/mimiclite-chip/records/three-point-chip-%j.slurm.log

set -eo pipefail
source /etc/profile
module add slurm cuda12.8/12.8
set -u
: "${SLURM_JOB_ID:?Submit using sbatch}"
root="${MIMICLITE_ROOT:-/share/ml/yangmin/mimiclite-chip}"
cd "$root/active-adaptation"
python_bin="$PWD/venv/mjlab/.venv/bin/python"
export PATH="$PWD/venv/mjlab/.venv/bin:$PATH"
export HF_HOME="$root/hf" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS=0 OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="$root/cache/matplotlib" TMPDIR="$root/tmp"
export WANDB_MODE=offline WANDB_ENTITY="${WANDB_ENTITY:-YM42}"
unset WANDB_RUN_ID WANDB_RESUME CHIP_CHECKPOINT THREE_POINT_CHECKPOINT
export WANDB_DIR="$root/records/three-point-chip-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$WANDB_DIR"

# Use the single-GPU measurement, not the old MLP setting or a 1024 guess.
# An explicit NUM_ENVS override remains available for deliberate new tests.
num_envs="${NUM_ENVS:-}"
if [ -z "$num_envs" ]; then
  profile="$PWD/.cache/three_point_chip_lambda_profile.json"
  if [ ! -f "$profile" ]; then
    echo "No validated NUM_ENVS yet: wait for the single-GPU profile, or explicitly set NUM_ENVS." >&2
    exit 2
  fi
  num_envs=$("$python_bin" - "$profile" <<'PY'
import json, sys, torch
result = json.load(open(sys.argv[1]))
assert result['pass_memory'], 'Profile failed memory gate'
assert torch.cuda.device_count() == 4, 'Expected four Slurm-visible GPUs'
for i in range(4):
    assert torch.cuda.get_device_name(i) == result['device'], 'GPU type differs: re-profile this hardware'
    assert torch.cuda.get_device_properties(i).total_memory / 2**20 >= result['total_mib'] * .99, 'GPU capacity differs'
print(result['num_envs'])
PY
  )
fi
total_iters="${TOTAL_ITERS:-30000}"
if ! [[ "$num_envs" =~ ^[1-9][0-9]*$ && "$total_iters" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_ENVS and TOTAL_ITERS must be positive integers" >&2
  exit 2
fi
"$python_bin" projects/mimic-lite/scripts/prepare_chip_loco.py \
  --task-template three_point_chip --root "$root/loco_manip_physical_rollout_accepted_v1" \
  2>&1 | tee "$WANDB_DIR/prepare.log"
srun --ntasks=1 nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# Check actual force application, reward logging and one small PPO update first.
# This is an integration check, not a production-batch memory/quality benchmark.
if [ "${RUN_SMOKE:-1}" = 1 ]; then
  srun --ntasks=1 --gres=gpu:1 --kill-on-bad-exit=1 \
    "$python_bin" projects/mimic-lite/scripts/smoke_three_point.py --chip --loco \
    2>&1 | tee "$WANDB_DIR/smoke.log"
fi

# Slurm owns CUDA_VISIBLE_DEVICES; one torchrun worker per allocated GPU.
srun --ntasks=1 --gres=gpu:4 --kill-on-bad-exit=1 \
  "$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  projects/mimic-lite/scripts/train.py \
  --config-name train_three_point_chip --config-dir "$PWD/.cache/three_point_chip_loco" \
  task=three_point_chip_loco_train backend=mjlab \
  "task.num_envs=$num_envs" "total_iters=$total_iters" checkpoint_path=null \
  wandb.mode=offline wandb.id=null \
  "exp_name=three-point-chip-lambda-${num_envs}x4-${total_iters}-${SLURM_JOB_ID}" \
  "hydra.run.dir=$WANDB_DIR/output" checkpoint_interval=500 upload_interval=500 \
  2>&1 | tee "$WANDB_DIR/train.log"
