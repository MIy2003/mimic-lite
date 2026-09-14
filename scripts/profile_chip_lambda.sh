#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside the requested Slurm allocation}"
root="${MIMICLITE_ROOT:-/share/ml/yangmin/mimiclite-chip}"
cd "$root/active-adaptation"
export HF_HOME="$root/hf" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS=0 OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
unset WANDB_RUN_ID WANDB_RESUME
export MPLCONFIGDIR="$root/cache/matplotlib" TMPDIR="$root/tmp"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
python_bin="$PWD/venv/mjlab/.venv/bin/python"
out="$root/records/profile-${SLURM_JOB_ID}"
mkdir -p "$out"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 2 > "$out/gpu.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
"$python_bin" projects/mimic-lite/scripts/prepare_chip_loco.py
for count in 64 7168 12288 16384 20480; do
  iterations=10
  if [ "$count" = 64 ]; then iterations=50; fi
  export CHIP_PROFILE_JSONL="$out/env-${count}.jsonl"
  # Refuse to overwrite an earlier probe in the same allocation.
  test ! -e "$CHIP_PROFILE_JSONL"
  "$python_bin" projects/mimic-lite/scripts/profile_chip_lambda.py \
    --config-dir "$PWD/.cache/chip_loco" task=chip_loco_train backend=mjlab \
    "task.num_envs=$count" "total_iters=$iterations" wandb.mode=disabled \
    checkpoint_interval=100000 upload_interval=100000 \
    "hydra.run.dir=$out/env-$count" 2>&1 | tee "$out/env-$count.log"
  if ! "$python_bin" -c 'import json,sys,torch; rows=[json.loads(s) for s in open(sys.argv[1])]; peak=max(r["peak_reserved_mib"] for r in rows); limit=torch.cuda.get_device_properties(0).total_memory/2**20*.8; print("MEMORY_GATE",peak,limit); sys.exit(0 if peak<limit else 1)' "$CHIP_PROFILE_JSONL"; then
    echo "Stopping scaling at 80% reserved-memory gate."
    break
  fi
done
echo PROFILE_COMPLETE
