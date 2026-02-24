#!/usr/bin/env bash

set -e

# ==== 训练配置：在此填写，运行 ./run_train.sh 即可（无需命令行传参）====
ROBOT="adam_sp"
DATASET="adam_sp/far_jump"
ALGORITHM="adamimic/stage1"
# checkpoint_path：stage1 从零训留空；stage2 填 stage1 的 model_xxxxx.pt；断点续训填上次保存的 pt
# CHECKPOINT_PATH="/home/liuhongji/workspace/exp/g1_dof29/high_jump/adamimic_stage1/20260210_151754_g1_dof29_high_jump_adamimic_stage1_test/model_39999.pt"
# CHECKPOINT_PATH=""

# ==== 环境与 GPU ====
ENV_NAME="adamimic"
# 指定使用的 GPU（留空则使用默认/全部）。例如: GPU_ID="0" 或 GPU_ID="1,2"
# 也可在运行时指定: GPU_ID=1 ./run_train.sh ...
GPU_ID=""
# ==========================================================

# 自动检测 CONDA_BASE（可被环境变量 CONDA_BASE 覆盖）
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

# 脚本子 shell 中必须 source conda.sh，否则 conda activate 会报 Run 'conda init'
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "未找到 conda。可设置: export CONDA_BASE=/path/to/anaconda3 或安装 conda 后重试" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 激活环境
conda activate "$ENV_NAME"

# 确保能找到 libpython3.8.so.1.0
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# 指定 GPU：若已设置 GPU_ID（脚本内或环境变量），则只使用该 GPU。
# 会生效：配置里的 sim_device/rl_device 使用 cuda:0 等逻辑编号，CUDA_VISIBLE_DEVICES 会
# 在进程内重排可见 GPU，所以 cuda:0 指向你指定的那块物理 GPU。
if [ -n "$GPU_ID" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# 使用上面配置的参数启动训练；若命令行有传参则追加（可覆盖部分项）
TRAIN_ARGS=(
  "+robot=$ROBOT"
  "+dataset=$DATASET"
  "+algorithm=$ALGORITHM"
)
[ -n "$CHECKPOINT_PATH" ] && TRAIN_ARGS+=( "checkpoint_path=$CHECKPOINT_PATH" )
python legged_gym/legged_gym/scripts/train.py "${TRAIN_ARGS[@]}" "$@"

