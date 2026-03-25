#!/bin/bash
# 通过 sbatch 提交 vLLM 服务端，部署 Qwen3.5-27B，提供 OpenAI 兼容 API
# 模型从 S3 挂载加载
#
# 用法:
#   sbatch [--gres=gpu:N] vllm_serve_sbatch.sh [S3_URI] [模型名] [N_GPU]
# N_GPU 为张量并行卡数，须与申请的 GPU 数一致；也可用环境变量 VLLM_NUM_GPUS。
# 例（27B 建议 ≥2 卡）:
#   sbatch --gres=gpu:2 vllm_serve_sbatch.sh s3://datafrontier/leihaodong/Qwen/Qwen3.5-27B/ Qwen3.5-27B 2
# 小模型单卡:
#   sbatch --gres=gpu:1 vllm_serve_sbatch.sh s3://datafrontier/leihaodong/Qwen/Qwen2-0.5B/ Qwen2-0.5B 1
#
# 可选环境变量（进一步控显存/并发）:
#   VLLM_MAX_MODEL_LEN       如 8192、32768（否则用模型 config 里极大上下文，易 OOM）
#   VLLM_MAX_NUM_SEQS        默认 64，仍 OOM 可改为 32
#   VLLM_GPU_MEMORY_UTIL     如 0.88（默认 0.92）
#   VLLM_PORT                默认 8100

#SBATCH -J vllm_serve
#SBATCH -p DataFrontier_Knowledge
#SBATCH -N 1
# 默认 2 卡；务必与第 3 参数一致，或在命令行覆盖: sbatch --gres=gpu:4 ... ... 4
#SBATCH --gres=gpu:2
#SBATCH -o /mnt/petrelfs/leihaodong/local_model/vllm/vllm_serve_%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/local_model/vllm/vllm_serve_%j.err

set -e

VLLM_DIR="/mnt/petrelfs/leihaodong/local_model/vllm"
# 默认 Qwen2-0.5B（vllm 0.4 原生支持）；Qwen3.5 需 vllm 0.8+ 及 transformers 5.2+
# 若未下载 Qwen2，请先运行: proxy_on && bash download_qwen2_to_s3.sh
S3_URI="${1:-s3://datafrontier/leihaodong/Qwen/Qwen2-0.5B/}"
MODEL_NAME="${2:-Qwen2-0.5B}"
NGPUS="${3:-${VLLM_NUM_GPUS:-2}}"
PORT="${VLLM_PORT:-8100}"

if ! [[ "${NGPUS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: N_GPU 须为正整数，当前: ${NGPUS}"
  exit 1
fi
if [[ "${NGPUS}" -lt 1 ]] || [[ "${NGPUS}" -gt 32 ]]; then
  echo "ERROR: N_GPU 不合理: ${NGPUS}（期望 1–32）"
  exit 1
fi

# 解析 S3 路径
if [[ "$S3_URI" =~ s3://([^/]+)/(.+)$ ]]; then
    BUCKET="${BASH_REMATCH[1]}"
    BUCKET_PATH="${BASH_REMATCH[2]%/}"
else
    echo "ERROR: 无效的 S3 地址: $S3_URI"
    exit 1
fi

echo "=== vLLM 服务端启动 ==="
echo "模型: $MODEL_NAME"
echo "S3: $S3_URI"
echo "端口: $PORT"
echo "张量并行 GPU 数: ${NGPUS}（请确认 sbatch 已申请 --gres=gpu:${NGPUS}）"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# 1. 环境（不 source .bashrc，避免 set_cuda/LD_LIBRARY_PATH 与 torch 冲突）
# sbatch 在计算节点用全新 shell，不会继承你登录后 conda activate qwen35_env，须在此显式激活。
export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY=
CONDA_BASE="/mnt/petrelfs/leihaodong/anaconda3"
# 优先 P 集群公共打包环境（解压+conda-unpack 后的路径，可用环境变量覆盖）
QWEN35_ENV="${QWEN35_ENV:-${HOME}/miniconda3/envs/qwen35_env}"
USE_GCC_TRITON_HACK=1

if [[ -f "${QWEN35_ENV}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${QWEN35_ENV}/bin/activate"
  echo "使用 qwen35_env: ${QWEN35_ENV}"
  USE_GCC_TRITON_HACK=0
elif [[ -f "${VLLM_DIR}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VLLM_DIR}/.venv/bin/activate"
  echo "使用 .venv"
elif [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate vllm 2>/dev/null && echo "使用 conda vllm" || \
  { conda activate qwen3 2>/dev/null && echo "使用 conda qwen3"; } || {
    echo "ERROR: 未找到 vllm/qwen3 conda 环境，且不存在 ${QWEN35_ENV}，请解压 qwen35_env 或运行 install_vllm.sh"
    exit 1
  }
else
  echo "ERROR: 未找到 conda，请先配置 ${QWEN35_ENV} 或运行 install_vllm.sh"
  exit 1
fi

# P 集群 Qwen3.5_Env 文档：屏蔽 PyTorch Dynamo，避免与 vLLM 图编译冲突
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_SUPPRESS_ERRORS=1
export VLLM_TORCH_COMPILE_LEVEL=0

# qwen35_env 已自带编译好的 CUDA/算子，勿再注入集群 GCC，以免与 PyTorch 冲突
if [[ "${USE_GCC_TRITON_HACK}" -eq 1 ]]; then
  export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/gcc-11.2.0/lib64:/mnt/petrelfs/share/gcc/mpfr-4.1.0/lib
  export PATH=/mnt/petrelfs/share/gcc/gcc-11.2.0/bin:$PATH
fi

python -c "import vllm; print('vllm OK:', vllm.__version__)" || { echo "ERROR: vllm 未安装"; exit 1; }

# 2. 挂载 S3
MOUNT_POINT="/nvme/${USER}/s3_mount_vllm_${SLURM_JOB_ID}"
CACHE_DIR="/nvme/${USER}/s3_cache_vllm_${SLURM_JOB_ID}"
mkdir -p "$MOUNT_POINT" "$CACHE_DIR"
rm -rf "$CACHE_DIR" && mkdir -p "$CACHE_DIR"
sleep 3

echo "挂载 S3..."
/mnt/petrelfs/leihaodong/app/s3mount "$BUCKET" "$MOUNT_POINT" \
  --cache "$CACHE_DIR" --allow-delete --allow-overwrite \
  --endpoint-url http://d-ceph-ssd-inside.pjlab.org.cn \
  --force-path-style \
  --log-directory /mnt/petrelfs/leihaodong/s3mount_logs &

S3MOUNT_PID=$!
sleep 15
if ! mount | grep -q "$MOUNT_POINT"; then
  echo "ERROR: s3mount 挂载失败"
  kill $S3MOUNT_PID 2>/dev/null || true
  exit 1
fi
echo "挂载成功: $MOUNT_POINT"

MODEL_PATH="${MOUNT_POINT}/${BUCKET_PATH}"
echo "模型路径: $MODEL_PATH"
ls -la "$MODEL_PATH/" 2>/dev/null || true

# 3. 获取节点地址（供客户端连接）
NODE_HOST=$(hostname)
CONNECTION_INFO="${VLLM_DIR}/vllm_connection_${SLURM_JOB_ID}.txt"
echo "http://${NODE_HOST}:${PORT}/v1" > "$CONNECTION_INFO"
echo "Job: $SLURM_JOB_ID | Node: $NODE_HOST | Port: $PORT" >> "$CONNECTION_INFO"
echo "" >> "$CONNECTION_INFO"
echo "客户端连接地址 (base_url): http://${NODE_HOST}:${PORT}/v1" >> "$CONNECTION_INFO"
echo "连接信息已写入: $CONNECTION_INFO"

# 4. 启动 vLLM 服务（前台运行，--host 0.0.0.0 允许外部访问）
echo ""
echo "=== 启动 vLLM 服务 (Ctrl+C 或 scancel 结束) ==="
cleanup() {
  echo "卸载 S3..."
  umount "$MOUNT_POINT" 2>/dev/null || fusermount -u "$MOUNT_POINT" 2>/dev/null || true
  kill $S3MOUNT_PID 2>/dev/null || true
  rm -f "$CONNECTION_INFO"
}
trap cleanup EXIT

MAX_LEN_ARGS=()
if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
  MAX_LEN_ARGS+=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
fi
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.92}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size "${NGPUS}" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype auto \
  --trust-remote-code \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  "${MAX_LEN_ARGS[@]}"
