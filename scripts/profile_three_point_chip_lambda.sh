#!/usr/bin/env bash
# Run with srun in the user's single-GPU allocation; no production job submitted.
set -euo pipefail
run_id="${PROFILE_RUN_ID:-${SLURM_JOB_ID:-}}"
: "${run_id:?Run inside Slurm or set PROFILE_RUN_ID for a dedicated local GPU}"
platform="${PROFILE_PLATFORM:-lambda}"
root="${MIMICLITE_ROOT:-/share/ml/yangmin/mimiclite-chip}"
cd "$root/active-adaptation"
python_bin="${THREE_POINT_PYTHON:-$PWD/venv/mjlab/.venv/bin/python}"
test -x "$python_bin"
export PATH="$(dirname "$python_bin"):$PATH"
export HF_HOME="${HF_HOME:-$root/hf}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS=0 OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
unset WANDB_RUN_ID WANDB_RESUME CHIP_CHECKPOINT THREE_POINT_CHECKPOINT
export MPLCONFIGDIR="$root/cache/matplotlib" TMPDIR="$root/tmp"
out="$root/records/three-point-profile-${run_id}"
mkdir -p "$out" "$MPLCONFIGDIR" "$TMPDIR"
test ! -e "$out/smoke.log"
gpu_query=()
if [ -n "${PROFILE_GPU_UUID:-}" ]; then gpu_query=(-i "$PROFILE_GPU_UUID"); fi
nvidia-smi "${gpu_query[@]}" --query-gpu=index,uuid,name,memory.total,memory.used --format=csv
pgrep -af 'train.py|train_ppo.py|profile_three_point_chip' || true
tmux ls || true
python -c 'import torch; assert torch.cuda.device_count()==1, "Requires one allocated GPU"; f,t=torch.cuda.mem_get_info(); assert t-f<2*2**30, "Allocated GPU is already in use"; print(torch.cuda.get_device_name(0))'
python projects/mimic-lite/scripts/prepare_chip_loco.py --task-template three_point_chip \
  2>&1 | tee "$out/prepare.log"
timeout 1200 python projects/mimic-lite/scripts/smoke_three_point.py --chip --loco \
  2>&1 | tee "$out/smoke.log"

count="${PROFILE_START:-16384}"
increment="${PROFILE_INCREMENT:-2048}"
upper="${PROFILE_UPPER:-20480}"
attempts="${PROFILE_ATTEMPTS:-5}"
best=0
direction=up
for ((attempt=1; attempt<=attempts; attempt++)); do
  export THREE_POINT_PROFILE_JSONL="$out/env-${count}.jsonl"
  test ! -e "$THREE_POINT_PROFILE_JSONL"
  nvidia-smi "${gpu_query[@]}" --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 1 > "$out/env-${count}-gpu.csv" &
  monitor_pid=$!
  trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
  set +e
  timeout 1800 python projects/mimic-lite/scripts/profile_three_point_chip_lambda.py \
    --config-name train_three_point_chip --config-dir "$PWD/.cache/three_point_chip_loco" \
    task=three_point_chip_loco_train backend=mjlab "task.num_envs=$count" \
    total_iters=10 checkpoint_path=null wandb.mode=disabled wandb.id=null \
    task.command.chip.warmup_steps=0 task.command.chip.ramp_steps=0 task.command.chip.force_interval=0 \
    checkpoint_interval=100000 upload_interval=100000 \
    "hydra.run.dir=$out/env-${count}" "exp_name=three-point-profile-${run_id}-${count}" \
    2>&1 | tee "$out/env-${count}.log"
  status=${PIPESTATUS[0]}
  set -e
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  trap - EXIT
  if [ "$status" -eq 0 ]; then
    set +e
    python - "$THREE_POINT_PROFILE_JSONL" "$out/env-${count}-gpu.csv" "$count" <<'PY'
import csv, json, statistics, sys
from pathlib import Path
rows = [json.loads(s) for s in open(sys.argv[1])]
assert len(rows) >= 10, 'Expected ten completed PPO updates'
observed = [float(r[2].split()[0]) for r in list(csv.reader(open(sys.argv[2])))[1:] if len(r)==5]
peak = max([r['estimated_peak_total_mib'] for r in rows] + observed)
limit = rows[0]['total_mib'] * .8
seconds = statistics.median(r['iteration_seconds'] for r in rows[3:])
summary = dict(num_envs=int(sys.argv[3]),peak_total_mib=peak,
    limit_mib=limit,median_iter_seconds=seconds,throughput=int(sys.argv[3])*32/seconds,
    pass_memory=peak<limit,device=rows[0]['device'],total_mib=rows[0]['total_mib'])
Path(sys.argv[1]).with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PROFILE_RESULT', json.dumps(summary))
sys.exit(0 if peak<limit else 3)
PY
    gate=$?
    set -e
    if [ "$gate" -ne 0 ] && [ "$gate" -ne 3 ]; then exit "$gate"; fi
  elif grep -qiE 'out of memory|cudaErrorMemoryAllocation' "$out/env-${count}.log"; then
    gate=3
    echo "PROFILE_OOM num_envs=$count"
  else
    echo "PROFILE_FAILED num_envs=$count status=$status; not a memory scaling issue"
    exit "$status"
  fi
  if [ "$gate" -eq 0 ]; then
    best=$count
    if [ "$direction" = down ] || [ "$count" -ge "$upper" ]; then break; fi
    count=$((count + increment))
  else
    if [ "$best" -gt 0 ]; then break; fi
    direction=down
    count=$((count - increment))
    if [ "$count" -le 0 ]; then break; fi
  fi
done
if [ "$best" -eq 0 ]; then echo 'PROFILE_NO_SAFE_CANDIDATE'; exit 3; fi
python - "$out/env-${best}.summary.json" "$PWD/.cache/three_point_chip_${platform}_profile.json" <<'PY'
import json, sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text())
assert result['pass_memory']
result['profile_source'] = sys.argv[1]
Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
PY
echo "PROFILE_COMPLETE recommended_num_envs=$best output=$out"
