#!/usr/bin/env bash
# 一次性执行：为 adamimic 环境设置「激活时自动添加 LD_LIBRARY_PATH」
# 执行后，每次 conda activate adamimic 都会自动设置，无需再手动 export

set -e

ENV_NAME="adamimic"
CONDA_BASE="${CONDA_BASE:-/home/liuhongji/anaconda3}"

if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "找不到 conda.sh，请设置 CONDA_BASE" >&2
  exit 1
fi

conda activate "$ENV_NAME"

ACTIVATE_D="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_D"
echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"' > "$ACTIVATE_D/libpython.sh"

echo "已设置：此后在任意终端执行 conda activate $ENV_NAME 时会自动设置 LD_LIBRARY_PATH。"
