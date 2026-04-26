#!/usr/bin/env bash
# 全架构消融：PPO 与 A2C ×（FC + CNN 五变体 a～e + Transformer 1D/2D 位置编码）
# 任务顺序：先跑完 PPO 全部子任务，再跑 A2C（见下方 AGENTS）
#
# 用法（在 deeplearning_hw 目录下）:
#   bash tools/run_all_arch_ablation_parallel.sh
#
# 环境变量（可选）:
#   MAX_JOBS=1          同时运行的最大任务数（单卡建议 1）
#   GPU_IDS=0,1         多卡轮询 CUDA_VISIBLE_DEVICES
#   N_ITER=14000 N_ENVS=64 N_STEPS=32 SEED=42 DEVICE=cuda PYTHON=python
#   N_OPTIM_STEPS_A2C=1  N_OPTIM_STEPS_PPO=4   （每算法优化轮数，覆盖 trainer YAML）
#   CNN_CONFIG=conv-policy-value-ablations.yaml
#   TF_CONFIG=transformer-policy-value-ablations.yaml
#   开关（默认全为 1）:
#     INCLUDE_FC=1 INCLUDE_CNN=1 INCLUDE_TRANSFORMER=1
#   AGENTS_ORDER="ppo a2c"  算法顺序（默认）；设为 "a2c ppo" 则先 A2C
#
# 任务数: 2 算法 × (1 FC + 5 CNN + 2 Transformer) = 16 次训练

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_JOBS="${MAX_JOBS:-1}"
GPU_IDS="${GPU_IDS:-0}"
N_ITER="${N_ITER:-14000}"
N_ENVS="${N_ENVS:-64}"
N_STEPS="${N_STEPS:-32}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python}"
N_OPTIM_STEPS_A2C="${N_OPTIM_STEPS_A2C:-1}"
N_OPTIM_STEPS_PPO="${N_OPTIM_STEPS_PPO:-4}"
CNN_CONFIG="${CNN_CONFIG:-conv-policy-value-ablations.yaml}"
TF_CONFIG="${TF_CONFIG:-transformer-policy-value-ablations.yaml}"
INCLUDE_FC="${INCLUDE_FC:-1}"
INCLUDE_CNN="${INCLUDE_CNN:-1}"
INCLUDE_TRANSFORMER="${INCLUDE_TRANSFORMER:-1}"
# 算法顺序：默认先 PPO 再 A2C；可设 AGENTS_ORDER="a2c ppo" 恢复旧顺序
read -r -a AGENTS <<< "${AGENTS_ORDER:-ppo a2c}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"

# CNN：a～d 为卷积/归一化消融；e 为 PreAct+SE 残差消融（基于 D）
CNN_VARIANTS=(
  "cnn_ablation_a"
  "cnn_ablation_b"
  "cnn_ablation_c"
  "cnn_ablation_d"
  "cnn_ablation_e"
)

TRANSFORMER_VARIANTS=(
  "transformer_pe_1d"
  "transformer_pe_2d"
)

TASKS=()
for ag in "${AGENTS[@]}"; do
  if [[ "${INCLUDE_FC}" == "1" ]]; then
    TASKS+=("${ag}|fc|")
  fi
  if [[ "${INCLUDE_CNN}" == "1" ]]; then
    for v in "${CNN_VARIANTS[@]}"; do
      TASKS+=("${ag}|conv|${v}")
    done
  fi
  if [[ "${INCLUDE_TRANSFORMER}" == "1" ]]; then
    for tv in "${TRANSFORMER_VARIANTS[@]}"; do
      TASKS+=("${ag}|transformer|${tv}")
    done
  fi
done

run_one() {
  local idx="$1"
  local spec="$2"
  local agent kind variant
  IFS='|' read -r agent kind variant _ <<< "${spec}"
  variant="${variant:-}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"

  echo "========== [${idx}] ${agent} + ${kind}${variant:+_${variant}} (CUDA_VISIBLE_DEVICES=${gpu}) =========="
  local n_opt
  if [[ "${agent}" == "a2c" ]]; then
    n_opt="${N_OPTIM_STEPS_A2C}"
  else
    n_opt="${N_OPTIM_STEPS_PPO}"
  fi

  case "${kind}" in
    fc)
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" runTorch.py \
        -an "${agent}" \
        -nn fc_policy_value \
        --n-iter "${N_ITER}" \
        --n-envs "${N_ENVS}" \
        --n-steps "${N_STEPS}" \
        --n-optim-steps "${n_opt}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
      ;;
    conv)
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" runTorch.py \
        -an "${agent}" \
        -nn conv_policy_value \
        --nn-config "${CNN_CONFIG}" \
        --nn-variant "${variant}" \
        --n-iter "${N_ITER}" \
        --n-envs "${N_ENVS}" \
        --n-steps "${N_STEPS}" \
        --n-optim-steps "${n_opt}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
      ;;
    transformer)
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" runTorch.py \
        -an "${agent}" \
        -nn transformer_policy_value \
        --nn-config "${TF_CONFIG}" \
        --nn-variant "${variant}" \
        --n-iter "${N_ITER}" \
        --n-envs "${N_ENVS}" \
        --n-steps "${N_STEPS}" \
        --n-optim-steps "${n_opt}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
      ;;
    *)
      echo "未知任务类型: ${kind}" >&2
      exit 1
      ;;
  esac
}

if [[ "${MAX_JOBS}" -lt 1 ]]; then
  echo "MAX_JOBS 必须 >= 1"
  exit 1
fi

echo "将运行 ${#TASKS[@]} 个任务: 顺序=${AGENTS[*]}  FC=${INCLUDE_FC} CNN=${INCLUDE_CNN} Transformer=${INCLUDE_TRANSFORMER}"
echo "n_optim_steps: A2C=${N_OPTIM_STEPS_A2C}  PPO=${N_OPTIM_STEPS_PPO}"

job_idx=0
for spec in "${TASKS[@]}"; do
  run_one "${job_idx}" "${spec}" &
  job_idx=$((job_idx + 1))
  if (( job_idx % MAX_JOBS == 0 )); then
    wait
  fi
done
wait

echo "全部任务已结束。日志目录: ${ROOT}/checkpoints-and-logs/local/"
