#!/usr/bin/env bash
# Use a pre-checked idle GPU UUID; never infer GPU availability from utilization.
set -euo pipefail
: "${PROFILE_GPU_UUID:?Set the UUID of one idle GPU after checking nvidia-smi}"
: "${PROFILE_RUN_ID:?Set a unique run id for logs}"
export CUDA_VISIBLE_DEVICES="$PROFILE_GPU_UUID"
export MIMICLITE_ROOT="${MIMICLITE_ROOT:-/media/raid/workspace/yangmin/code/MimicLite}"
export THREE_POINT_PYTHON="${THREE_POINT_PYTHON:-$MIMICLITE_ROOT/active-adaptation/venv/mjlab/.venv-5090/bin/python}"
export HF_HOME="${HF_HOME:-$MIMICLITE_ROOT/.hf}"
export PROFILE_PLATFORM=5090
export PROFILE_START="${PROFILE_START:-4096}" PROFILE_INCREMENT="${PROFILE_INCREMENT:-1024}"
export PROFILE_UPPER="${PROFILE_UPPER:-12288}" PROFILE_ATTEMPTS="${PROFILE_ATTEMPTS:-9}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/profile_three_point_chip_lambda.sh"
