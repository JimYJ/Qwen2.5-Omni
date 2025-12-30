#!/usr/bin/env bash

# 使用 uv 安装包，所有输出重定向到 install.log，并在后台执行
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

echo "开始使用 uv 安装包，日志将保存到 install.log"
echo "进程 ID: $$"

# 在后台执行安装
{
  echo "开始安装 transformers==4.52.3..."
  uv pip install transformers==4.52.3
  
  echo "开始安装 accelerate..."
  uv pip install accelerate
  
  echo "开始安装 gptqmodel==2.0.0..."
  uv pip install gptqmodel==2.0.0
  
  echo "开始安装 numpy==2.0.0..."
  uv pip install numpy==2.0.0
  
  echo "所有包安装完成！"
} > install.log 2>&1 &

echo "安装任务已在后台启动，PID: $!"
echo "查看安装进度: tail -f install.log"