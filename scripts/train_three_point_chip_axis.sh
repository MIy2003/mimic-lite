#!/usr/bin/env bash
set -euo pipefail
export THREE_POINT_VARIANT=three_point_chip_axis
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_three_point.sh" "$@"
