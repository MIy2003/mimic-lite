#!/usr/bin/env bash
# Run under salloc/srun on ONE GPU before submitting the four-GPU axis job.
set -euo pipefail
: "${SLURM_JOB_ID:?Run within a Slurm allocation}"
root="${MIMICLITE_ROOT:-/share/ml/yangmin/mimiclite-chip}"
cd "$root/active-adaptation"
export PATH="$PWD/venv/mjlab/.venv/bin:$PATH"
export HF_HOME="$root/hf" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS=0 OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
unset WANDB_RUN_ID WANDB_RESUME CHIP_CHECKPOINT
export MPLCONFIGDIR="$root/cache/matplotlib" TMPDIR="$root/tmp"
out="$root/records/axis-smoke-${SLURM_JOB_ID}"
mkdir -p "$out" "$TMPDIR" "$MPLCONFIGDIR"
test ! -e "$out/PASSED"
test ! -e "$out/env-16384.jsonl"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv
python -c 'import torch; assert torch.cuda.device_count() == 1, "Smoke requires exactly one allocated GPU"; print("SMOKE_DEVICE", torch.cuda.get_device_name(0))'
python -m unittest discover -s projects/mimic-lite/tests -p 'test_chip*.py' 2>&1 | tee "$out/tests.log"
python projects/mimic-lite/scripts/prepare_chip_loco.py 2>&1 | tee "$out/prepare.log"
# Exercise physical-force projection immediately using a tiny policy and four envs.
python projects/mimic-lite/scripts/smoke_chip.py --loco --mode wrist_axis 2>&1 | tee "$out/integration.log"
# Match production's per-GPU environment count and policy. Only the force
# curriculum is accelerated for this short check, not for production training.
export CHIP_PROFILE_JSONL="$out/env-16384.jsonl"
python projects/mimic-lite/scripts/profile_chip_lambda.py \
  --config-dir "$PWD/.cache/chip_loco" task=chip_loco_train backend=mjlab \
  task.command.chip.compliance_mode=wrist_axis \
  task.command.chip.warmup_steps=0 task.command.chip.ramp_steps=0 \
  task.command.chip.force_interval=0 \
  task.num_envs=16384 total_iters=10 checkpoint_path=null \
  wandb.mode=disabled wandb.id=null checkpoint_interval=100000 upload_interval=100000 \
  "hydra.run.dir=$out/env-16384" "exp_name=chip-axis-smoke-${SLURM_JOB_ID}" \
  2>&1 | tee "$out/env-16384.log"
python - "$CHIP_PROFILE_JSONL" <<'PY'
import json
import sys
import torch
rows = [json.loads(s) for s in open(sys.argv[1])]
assert len(rows) >= 10, f"Expected 10 completed updates, got {len(rows)}"
peak = max(r["peak_reserved_mib"] for r in rows)
limit = torch.cuda.get_device_properties(0).total_memory / 2**20 * .8
print("MEMORY_GATE", peak, limit)
assert peak < limit, "Insufficient per-GPU memory headroom for production DDP"
PY
touch "$out/PASSED"
echo "AXIS_SMOKE_PASSED job=$SLURM_JOB_ID output=$out"
