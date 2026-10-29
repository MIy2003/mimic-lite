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
run_dir="$root/records/chip-five-point-${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT:-0}"
config_dir="$run_dir/config"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$run_dir"
export WANDB_DIR="$run_dir"
printf '%s\n' preparing > "$run_dir/stage.txt"
trap 'status=$?; if (( status != 0 )); then printf "failed exit=%s\n" "$status" >> "$run_dir/stage.txt"; fi' EXIT
python projects/mimic-lite/scripts/prepare_chip_loco.py --five-point --output "$config_dir"
tar -czf "$run_dir/project-source.tar.gz" -C projects/mimic-lite \
    cfg scripts mimic_lite mimic_lite_learning tests CHIP_FIVE_POINT.md
srun --ntasks=1 nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
srun --ntasks=1 bash -c 'pgrep -u "$USER" -af "torchrun|train.py|smoke_chip" || true; tmux ls || true'

# Small four-rank run validates the new goal and DDP before a full-size probe.
for stage in small capacity; do
    if [[ $stage == small ]]; then
        stage_envs=64
        stage_iters=50
        task=chip_five_point_loco_train_smoke
    else
        stage_envs=$num_envs
        stage_iters=6
        task=chip_five_point_loco_train
    fi
    printf 'smoke_%s\n' "$stage" > "$run_dir/stage.txt"
    srun --ntasks=1 nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
    srun --ntasks=1 --kill-on-bad-exit=1 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
        projects/mimic-lite/scripts/train.py --config-dir "$config_dir" \
        "task=$task" backend=mjlab headless=true checkpoint_path=null \
        "task.num_envs=$stage_envs" "total_iters=$stage_iters" wandb.mode=disabled wandb.id=null \
        task.command.chip.warmup_steps=0 task.command.chip.ramp_steps=0 \
        task.command.chip.force_interval=0 "hydra.run.dir=$run_dir/smoke_$stage" \
        "exp_name=five-point-$stage-${SLURM_JOB_ID}" 2>&1 | tee "$run_dir/smoke_$stage.log"
    python - "$run_dir/smoke_$stage/metrics.jsonl" "$stage_iters" <<'PY'
import json
import math
from pathlib import Path
import sys

path, expected = Path(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
assert len(rows) == expected and rows[-1]['iteration'] == expected - 1, 'Smoke did not finish'
for row in rows:
    for key, value in row.items():
        if isinstance(value, (int, float)):
            assert math.isfinite(value), (key, value)
    assert row.get('env/nonfinite_rows_total', 0) == 0, row
print(f'FIVE_POINT_SMOKE_GATE_PASSED {path.parent.name}: {len(rows)} finite updates', flush=True)
PY
done

srun --ntasks=1 nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
printf '%s\n' training > "$run_dir/stage.txt"
srun --ntasks=1 --kill-on-bad-exit=1 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    projects/mimic-lite/scripts/train.py --config-dir "$config_dir" \
    task=chip_five_point_loco_train backend=mjlab headless=true checkpoint_path=null \
    "task.num_envs=$num_envs" "total_iters=$total_iters" wandb.mode=offline wandb.id=null \
    "exp_name=chip-five-point-lambda-${num_envs}x4-${SLURM_JOB_ID}" \
    "hydra.run.dir=$run_dir/train" checkpoint_interval=500 upload_interval=500 \
    "$@" 2>&1 | tee "$run_dir/train.log"
printf '%s\n' completed > "$run_dir/stage.txt"
