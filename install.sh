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
  echo "开始安装 PyTorch (支持 CUDA 12.1，与 CUDA 13.0 兼容)..."
  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

  echo "开始安装 transformers==4.52.3..."
  uv pip install transformers==4.52.3

  echo "开始安装 accelerate..."
  uv pip install accelerate

  # 检查是否有 CUDA 环境，如果没有则跳过 gptqmodel 安装

  echo "开始安装 autoawq==0.2.9"
  uv pip install autoawq==0.2.9

  echo "检查并安装 CUDA 开发工具包..."
  # 首先尝试安装系统级的 CUDA 工具包
  if command -v apt-get &> /dev/null; then
    echo "检测到 Ubuntu/Debian 系统，尝试安装 CUDA 工具包..."
    sudo apt update
    sudo apt install -y nvidia-cuda-toolkit || echo "系统 CUDA 工具包安装失败，可能需要手动安装"
  elif command -v yum &> /dev/null; then
    echo "检测到 CentOS/RHEL 系统，尝试安装 CUDA 工具包..."
    sudo yum install -y cuda-toolkit || echo "系统 CUDA 工具包安装失败，可能需要手动安装"
  else
    echo "未检测到支持的包管理器，请手动安装 CUDA Toolkit 13.0"
  fi

  echo "开始安装 flash_attn (支持 CUDA 13.0)..."

  # flash_attn 不在 PyTorch wheel 索引中，尝试其他安装方式
  echo "flash_attn 不在 PyTorch wheel 索引中，尝试从 PyPI 安装..."
  FLASH_ATTENTION_INSTALLED=false

  # 方法 1: 下载并安装预编译 wheel (推荐)
  echo "下载 flash_attn 预编译 wheel..."
  WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.2/flash_attn-2.6.2+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
  WHEEL_FILE="flash_attn-2.6.2+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

  if curl -L -o "$WHEEL_FILE" "$WHEEL_URL" 2>/dev/null; then
    echo "下载成功，开始安装 wheel..."
    if "$VENV_DIR/bin/pip3" install "$WHEEL_FILE"; then
      echo "成功安装 flash_attn 预编译 wheel"
      FLASH_ATTENTION_INSTALLED=true
      rm -f "$WHEEL_FILE"
    else
      echo "wheel 安装失败，清理文件"
      rm -f "$WHEEL_FILE"
    fi
  else
    echo "下载失败，尝试从 PyPI 安装"
  fi

  # 方法 2: 如果预编译 wheel 失败，从 PyPI 安装
  if [ "$FLASH_ATTENTION_INSTALLED" = false ]; then
    if "$VENV_DIR/bin/pip3" install flash_attn; then
      echo "成功从 PyPI 安装 flash_attn"
      FLASH_ATTENTION_INSTALLED=true
    fi
  fi

  # 方法 2: 如果失败，尝试特定的稳定版本
  if [ "$FLASH_ATTENTION_INSTALLED" = false ]; then
    echo "尝试安装特定的稳定版本..."
    for version in "2.5.9" "2.5.8" "2.5.6" "2.4.2"; do
      echo "尝试安装 flash_attn==$version..."
      if "$VENV_DIR/bin/pip3" install "flash_attn==$version"; then
        echo "成功安装 flash_attn $version"
        FLASH_ATTENTION_INSTALLED=true
        break
      fi
    done
  fi

  # 方法 2: 如果指定版本失败，尝试不指定版本的安装
  if [ "$FLASH_ATTENTION_INSTALLED" = false ]; then
    echo "尝试安装最新可用版本..."
    if "$VENV_DIR/bin/pip3" install flash_attn --index-url https://download.pytorch.org/whl/cu121; then
      echo "成功安装 flash_attn (最新版本)"
      FLASH_ATTENTION_INSTALLED=true
    fi
  fi

  # 方法 3: 如果仍然失败，尝试源码编译
  if [ "$FLASH_ATTENTION_INSTALLED" = false ]; then
    echo "预编译 wheel 不可用，尝试源码编译..."
    echo "设置 CUDA 环境变量..."
    # 尝试多种可能的 CUDA 路径
    if [ -d "/usr/local/cuda" ]; then
      export CUDA_HOME=/usr/local/cuda
    elif [ -d "/usr/local/cuda-13.0" ]; then
      export CUDA_HOME=/usr/local/cuda-13.0
    elif [ -d "/usr/lib/cuda" ]; then
      export CUDA_HOME=/usr/lib/cuda
    else
      echo "警告: 未找到标准 CUDA 安装路径，尝试使用系统默认路径"
      export CUDA_HOME=/usr/local/cuda
    fi
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo "CUDA_HOME 设置为: $CUDA_HOME"

    if "$VENV_DIR/bin/pip3" install flash_attn --no-build-isolation; then
      echo "源码编译安装成功"
      FLASH_ATTENTION_INSTALLED=true
    fi
  fi

  if [ "$FLASH_ATTENTION_INSTALLED" = false ]; then
    echo "警告: flash_attn 安装失败，将使用标准注意力机制"
    echo "你可以稍后手动安装: pip install flash_attn --index-url https://download.pytorch.org/whl/cu121"
  fi

  echo "开始安装 numpy==2.0.0..."
  uv pip install numpy==2.0.0

  echo "开始安装 torchvision >= 0.19.0..."
  uv pip install torchvision==0.19.0

  echo "开始安装 本项目插件qwen-omni-utils"
uv pip install -e ./qwen-omni-utils
  
  echo "所有包安装完成！"
  echo ""
  echo "要激活虚拟环境，请运行:"
  echo "  source $ACTIVATE_SCRIPT"
} > install.log 2>&1 &

echo "安装任务已在后台启动，PID: $!"
echo "查看安装进度: tail -f install.log"
echo "虚拟环境将创建在: $VENV_DIR"