#!/usr/bin/env bash

set -e

# ==== 配置区域：如需修改环境名或 Conda 路径，在这里改 ====
ENV_NAME="adamimic"
CONDA_BASE="${CONDA_BASE:-/home/liuhongji/anaconda3}"
# ==========================================================

# 加载 conda
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "找不到 conda.sh，当前 CONDA_BASE = $CONDA_BASE" >&2
  exit 1
fi

# 激活环境
conda activate "$ENV_NAME"

# 确保能找到 libpython3.8.so.1.0
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# 把所有传入参数转发给 play 脚本（Hydra 配置等）
# 示例（resume_path 换成你实际的 model_xxx.pt 路径）：
#   ./run_play.sh +robot=g1_dof27 +dataset=g1_dof27/badminton_hit +algorithm=adamimic/stage1 resume_path=../exp/g1_dof27/badminton_hit/adamimic_stage1/时间戳目录/model_500.pt
python legged_gym/legged_gym/scripts/play.py "$@"
