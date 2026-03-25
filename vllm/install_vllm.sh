#!/bin/bash
# vLLM 安装脚本
# 务必先 proxy_on
#
# 用法: proxy_on && bash install_vllm.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== vLLM 安装 ==="
echo "工作目录: $SCRIPT_DIR"
echo "提示: 务必先 proxy_on"
echo ""

# 环境检测：glibc 2.17 的集群无法使用 vLLM 0.18 + torch 2.10（需 manylinux_2_28）
GLIBC_VER=$(python3 -c "import platform; print(platform.libc_ver()[1])" 2>/dev/null || echo "unknown")
echo "检测到 glibc: $GLIBC_VER"

# 方案 1: uv + 新 venv（适用于 glibc >= 2.28 的系统）
if command -v uv &>/dev/null; then
    echo ""
    echo "--- 方案 1: uv 新建 venv（官方推荐，需 glibc 2.28+）---"
    rm -rf .venv
    uv venv --python 3.12 --seed --managed-python 2>/dev/null || uv venv --python 3.12
    source .venv/bin/activate
    if uv pip install vllm --torch-backend=auto 2>/dev/null; then
        echo ""
        echo "=== 安装成功 (uv .venv) ==="
        python -c "import vllm; print('vLLM', vllm.__version__)"
        exit 0
    fi
    deactivate 2>/dev/null || true
fi

# 方案 2: 新建 conda 环境 vllm（与 qwen3 分离，兼容 glibc 2.17）
echo ""
echo "--- 方案 2: 新建 conda 环境 vllm（兼容本集群）---"
source ~/.bashrc 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q '^vllm '; then
    echo "创建 conda 环境 vllm (Python 3.10)..."
    conda create -n vllm python=3.10 -y
fi
conda activate vllm

echo "安装 vllm 0.4.2（适配 torch 2.x + glibc 2.17）..."
if pip install "vllm==0.4.2" 2>/dev/null; then
    echo ""
    echo "=== 安装成功 (conda vllm) ==="
    python -c "import vllm; print('vLLM', vllm.__version__)"
    echo "sbatch 将使用: conda activate vllm"
    exit 0
fi

echo ""
echo "=== 安装失败 ==="
echo "请手动："
echo "  proxy_on"
echo "  conda create -n vllm python=3.10 -y"
echo "  conda activate vllm"
echo "  pip install vllm==0.4.2"
exit 1
