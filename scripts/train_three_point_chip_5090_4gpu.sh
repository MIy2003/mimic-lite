#!/usr/bin/env bash
set -euo pipefail
root="${MIMICLITE_ROOT:-/media/raid/workspace/yangmin/code/MimicLite}"
cd "$root/active-adaptation"
export THREE_POINT_PYTHON="$PWD/venv/mjlab/.venv-5090/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export NUM_ENVS="${NUM_ENVS:-6208}" TOTAL_ITERS="${TOTAL_ITERS:-30000}" NPROC_PER_NODE=4
export HF_HOME="$root/.hf" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 ANY4HDMI_CACHE_BUILD_NUM_WORKERS=0
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export WANDB_MODE=online WANDB_ENTITY="${WANDB_ENTITY:-YM42}"
unset WANDB_RUN_ID WANDB_RESUME CHIP_CHECKPOINT THREE_POINT_CHECKPOINT
: "${RUN_NAME:?Set a unique RUN_NAME}"
export WANDB_DIR="$root/records/$RUN_NAME"
export TMPDIR="$root/tmp" MPLCONFIGDIR="$root/cache/matplotlib"
mkdir -p "$WANDB_DIR" "$TMPDIR" "$MPLCONFIGDIR"
"$THREE_POINT_PYTHON" -c 'import torch; assert torch.cuda.device_count()==4; [(print(i,torch.cuda.get_device_name(i),torch.cuda.mem_get_info(i)), None if torch.cuda.mem_get_info(i)[1]-torch.cuda.mem_get_info(i)[0]<2*2**30 else (_ for _ in ()).throw(RuntimeError("GPU already occupied"))) for i in range(4)]'
exec bash projects/mimic-lite/scripts/train_three_point_chip.sh \
  wandb.mode=online wandb.id=null \
  "exp_name=$RUN_NAME" "hydra.run.dir=$WANDB_DIR/output" \
  checkpoint_interval=500 upload_interval=500 "$@"
