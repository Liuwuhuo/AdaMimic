#!/usr/bin/env bash

set -e

# ==== 播放配置：通常只需填写 RESUME_PATH，其他会自动解析 ====
# 要播放的模型路径（stage1 用 stage1 的 model_xxxxx.pt，stage2 用 stage2 的）
RESUME_PATH="/home/liuhongji/workspace/exp/adam_sp/high_jump/adamimic_stage1/20260331_110804_adam_sp_high_jump_adamimic_stage1_test/model_39999.pt"
# 从 RESUME_PATH 自动解析（格式：.../exp/<robot>/<task>/<algo>/<run>/model_x.pt）
# 如需手动覆盖，可在命令行追加：+robot=... +dataset=... +algorithm=...
ROBOT=""
DATASET=""
ALGORITHM=""
# ==== 导出配置（可选）====
# 导出 TorchScript + ONNX（true/false）
EXPORT_POLICY="true"
# 导出目录；留空时自动使用仓库下 exports（与当前终端目录无关）
EXPORT_POLICY_PATH=""
# 导出文件名前缀；留空则使用 eval.yaml 默认名称
EXPORT_POLICY_NAME="20260331_110804_adam_sp_high_jump_adamimic_stage1_test"

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

# 仓库根目录（当前脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

conda activate "$ENV_NAME"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

if [ -n "$GPU_ID" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# 若开启导出：
# 1) 未指定 EXPORT_POLICY_NAME 时，用 resume 的实验目录名作为默认名
# 2) 未指定 EXPORT_POLICY_PATH 时，默认导出到 exports/$EXPORT_POLICY_NAME
# 3) 自动创建导出目录
RESUME_DIR="$(dirname "$RESUME_PATH")"
RUN_NAME_FROM_RESUME="$(basename "$RESUME_DIR")"
ALGO_NAME_FROM_RESUME="$(basename "$(dirname "$RESUME_DIR")")"
TASK_NAME_FROM_RESUME="$(basename "$(dirname "$(dirname "$RESUME_DIR")")")"
ROBOT_NAME_FROM_RESUME="$(basename "$(dirname "$(dirname "$(dirname "$RESUME_DIR")")")")"

# 自动填充 robot/dataset/algorithm（仅当留空时）
[ -z "$ROBOT" ] && ROBOT="$ROBOT_NAME_FROM_RESUME"
[ -z "$DATASET" ] && DATASET="$ROBOT_NAME_FROM_RESUME/$TASK_NAME_FROM_RESUME"
[ -z "$ALGORITHM" ] && ALGORITHM="adamimic/${ALGO_NAME_FROM_RESUME##adamimic_}"

if [ "$EXPORT_POLICY" = "true" ]; then
  if [ -z "$EXPORT_POLICY_NAME" ]; then
    EXPORT_POLICY_NAME="$RUN_NAME_FROM_RESUME"
  fi

  if [ -z "$EXPORT_POLICY_PATH" ]; then
    EXPORT_POLICY_PATH="$SCRIPT_DIR/exports/$EXPORT_POLICY_NAME"
  fi

  mkdir -p "$EXPORT_POLICY_PATH"
fi

# 使用上面配置的参数启动 play；命令行传参会追加
PLAY_ARGS=(
  "+robot=$ROBOT"
  "+dataset=$DATASET"
  "+algorithm=$ALGORITHM"
  "resume_path=$RESUME_PATH"
)
[ -n "$EXPORT_POLICY" ] && PLAY_ARGS+=( "export_policy=$EXPORT_POLICY" )
[ -n "$EXPORT_POLICY_PATH" ] && PLAY_ARGS+=( "export_policy_path=$EXPORT_POLICY_PATH" )
[ -n "$EXPORT_POLICY_NAME" ] && PLAY_ARGS+=( "export_policy_name=$EXPORT_POLICY_NAME" )
python legged_gym/legged_gym/scripts/play.py "${PLAY_ARGS[@]}" "$@"
