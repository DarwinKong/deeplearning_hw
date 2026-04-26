#!/usr/bin/env bash
# 仅 CNN 四变体 ×（A2C+PPO）。全架构请用: tools/run_all_arch_ablation_parallel.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export INCLUDE_FC=0
export INCLUDE_TRANSFORMER=0
export INCLUDE_CNN=1
exec bash "${ROOT}/tools/run_all_arch_ablation_parallel.sh" "$@"
