#!/usr/bin/env bash

# 使用 uv 创建虚拟环境并安装包，所有输出重定向到 install.log，并在后台执行
#
# 准备工作：
# 1. 安装 uv（选择以下任一方式）:
#    - 官方安装脚本: curl -LsSf https://astral.sh/uv/install.sh | sh
#    - 使用 pip: pip install uv
#    - 使用 cargo: cargo install uv
# 2. 确保 uv 在 PATH 中，或重启终端使 uv 生效
# 3. 验证安装: uv --version
#
# 注意: gptqmodel 需要 torch 和 setuptools 作为构建依赖，脚本会先安装它们然后使用 --no-build-isolation

# 检查 uv 是否已安装
if ! command -v uv &> /dev/null; then
    echo "错误: 未找到 uv 命令，请先安装 uv"
    echo "安装方法: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 虚拟环境目录
VENV_DIR=".venv"

# 检测操作系统，确定虚拟环境的 Python 路径
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    # Windows (Git Bash, MSYS2, Cygwin)
    PYTHON_PATH="$VENV_DIR/Scripts/python.exe"
    ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
else
    # Linux/Mac
    PYTHON_PATH="$VENV_DIR/bin/python"
    ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
fi

echo "开始使用 uv 创建虚拟环境并安装包，日志将保存到 install.log"
echo "进程 ID: $$"

# 在后台执行安装
{
  # 检查虚拟环境是否存在，如果不存在则创建
  if [ ! -d "$VENV_DIR" ]; then
    echo "虚拟环境不存在，正在创建..."
    uv venv "$VENV_DIR"
    echo "虚拟环境创建完成: $VENV_DIR"
  else
    echo "虚拟环境已存在: $VENV_DIR"
  fi
  
  # 使用虚拟环境安装包（uv 会自动检测当前目录下的 .venv）
  echo "开始安装 transformers==4.52.3..."
  uv pip install transformers==4.52.3

  echo "开始安装 accelerate..."
  uv pip install accelerate

  # gptqmodel 需要 torch 和 setuptools 作为构建依赖，先安装它们然后用 --no-build-isolation
  echo "开始安装 torch（为 gptqmodel 构建做准备）..."
  uv pip install torch

  echo "开始安装 setuptools（为 gptqmodel 构建做准备）..."
  uv pip install setuptools

  echo "开始安装 gptqmodel==2.0.0（使用 --no-build-isolation）..."
  uv pip install --no-build-isolation gptqmodel==2.0.0

  echo "开始安装 numpy==2.0.0..."
  uv pip install numpy==2.0.0
  
  echo "所有包安装完成！"
  echo ""
  echo "要激活虚拟环境，请运行:"
  echo "  source $ACTIVATE_SCRIPT"
} > install.log 2>&1 &

echo "安装任务已在后台启动，PID: $!"
echo "查看安装进度: tail -f install.log"
echo "虚拟环境将创建在: $VENV_DIR"