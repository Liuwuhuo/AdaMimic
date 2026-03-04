#!/usr/bin/env bash

set -e

# ==== 播放配置：在此填写，运行 ./run_play.sh 即可（无需命令行传参）====
ROBOT="adam_sp"
DATASET="adam_sp/far_jump"
ALGORITHM="adamimic/stage1"
# 要播放的模型路径（stage1 用 stage1 的 model_xxxxx.pt，stage2 用 stage2 的）
RESUME_PATH="/home/liuhongji/workspace/exp/adam_sp/far_jump/adamimic_stage1/20260303_172638_adam_sp_far_jump_adamimic_stage1_test/model_15000.pt"

# ==== 环境与 GPU ====
ENV_NAME="adamimic"
GPU_ID=""
# ==========================================================

# 自动检测 CONDA_BASE（与 run_train.sh 一致，可被环境变量 CONDA_BASE 覆盖）
if [ -z "$CONDA_BASE" ]; then
  if command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null)" || true
  fi
  if [ -z "$CONDA_BASE" ] || [ ! -d "$CONDA_BASE" ]; then
    for d in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniconda" "/opt/conda"; do
      if [ -f "$d/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$d"
        break
      fi
    done
  fi
fi
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "未找到 conda。可设置: export CONDA_BASE=/path/to/anaconda3 或安装 conda 后重试" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate "$ENV_NAME"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

if [ -n "$GPU_ID" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# 使用上面配置的参数启动 play；命令行传参会追加
PLAY_ARGS=(
  "+robot=$ROBOT"
  "+dataset=$DATASET"
  "+algorithm=$ALGORITHM"
  "resume_path=$RESUME_PATH"
)
python legged_gym/legged_gym/scripts/play.py "${PLAY_ARGS[@]}" "$@"
