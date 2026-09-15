#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
framework_root="$(cd "${script_dir}/../../.." && pwd)"
cd "$framework_root"
python_bin="${THREE_POINT_PYTHON:-$framework_root/venv/mjlab/.venv/bin/python}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_HUB_DISABLE_TELEMETRY=1
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS="${ANY4HDMI_CACHE_BUILD_NUM_WORKERS:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
variant="${THREE_POINT_VARIANT:-three_point}"
case "$variant" in three_point|three_point_chip) ;; *) echo "Invalid THREE_POINT_VARIANT: $variant" >&2; exit 2 ;; esac
"$python_bin" "$script_dir/prepare_chip_loco.py" --task-template "$variant" \
  --root "${THREE_POINT_DATA_ROOT:-$framework_root/../loco_manip_physical_rollout_accepted_v1}" \
  --output "$framework_root/.cache/${variant}_loco"
exec "$python_bin" -m torch.distributed.run --standalone \
  --nproc-per-node="${NPROC_PER_NODE:-1}" "$script_dir/train.py" \
  --config-name "train_${variant}" --config-dir "$framework_root/.cache/${variant}_loco" \
  task="${variant}_loco_train" task.num_envs="${NUM_ENVS:-1024}" \
  total_iters="${TOTAL_ITERS:-4000}" checkpoint_path=null wandb.mode=offline "$@"
