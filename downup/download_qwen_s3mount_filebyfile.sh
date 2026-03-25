#!/bin/bash
# 从 ModelScope 魔搭社区逐文件下载到 S3（每个文件：下载 -> 上传到挂载点 -> 删除本地）
# 支持断点续传：检测 s3://datafrontier/{S3_PREFIX}/ 中已存在且大小一致的文件会跳过
# 适用：单文件 < 本地剩余空间（如 150G）
# 注意：若单个文件超过本地空间（如 200G 的 model.safetensors），此方案无效
#
# 用法：sbatch download_qwen_s3mount_filebyfile.sh [REPO_ID] [S3_PREFIX] [PROXY_MODE]
# 示例：sbatch ... Qwen/Qwen3.5-27B leihaodong/Qwen/Qwen3.5-27B
# 需要走代理时：sbatch ... Qwen/Qwen3.5-27B leihaodong/Qwen/Qwen3.5-27B on

#SBATCH -J qwen_s3mount_fbf
#SBATCH -p DataFrontier_Knowledge
#SBATCH -c 4
#SBATCH --mem=16G

set -e

REPO_ID="${1:-Qwen/Qwen3.5-27B}"
S3_PREFIX="${2:-leihaodong/Qwen/Qwen3.5-27B}"
PROXY_MODE="${3:-off}"   # off=默认不走代理, on=走代理

BUCKET="datafrontier"
ENDPOINT="http://d-ceph-ssd-inside.pjlab.org.cn"
S3MOUNT_BIN="/mnt/petrelfs/leihaodong/app/s3mount"
NVME_BASE="/nvme/${USER}"
MOUNT_POINT="${NVME_BASE}/s3_mount_qwen"
CACHE_POINT="${NVME_BASE}/s3_cache_qwen"
TMP_SINGLE="${NVME_BASE}/tmp_single_file"   # 每次只放一个文件，用完即删
LOG_DIR="/mnt/petrelfs/leihaodong/s3mount_logs"

# sbatch 会把脚本拷贝到 /var/spool/slurmd/ 运行，$0/BASH_SOURCE 均指向 spool，必须硬编码原始路径
SCRIPT_DIR="/mnt/petrelfs/leihaodong/local_model/Qwen3.5"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
source ~/.bashrc 2>/dev/null || true

# 激活 conda 环境（需已安装 modelscope: pip install modelscope）
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate qwen3 2>/dev/null || true
fi

echo "=== 1. 创建目录 ==="
cd "${NVME_BASE}"
mkdir -p "${MOUNT_POINT}" "${TMP_SINGLE}"
rm -rf "${CACHE_POINT}" && mkdir -p "${CACHE_POINT}" "${LOG_DIR}"
sleep 3

echo "=== 2. 挂载 S3 桶 ==="
"${S3MOUNT_BIN}" "${BUCKET}" "${MOUNT_POINT}" \
  --cache "${CACHE_POINT}" --allow-delete --allow-overwrite \
  --endpoint-url "${ENDPOINT}" --force-path-style \
  --log-directory "${LOG_DIR}" &

S3MOUNT_PID=$!
for i in $(seq 1 30); do
  mount | grep -q "${MOUNT_POINT}" && break
  sleep 1
  kill -0 ${S3MOUNT_PID} 2>/dev/null || { echo "s3mount 异常退出"; exit 1; }
done
mount | grep -q "${MOUNT_POINT}" || { echo "挂载超时"; kill ${S3MOUNT_PID} 2>/dev/null; exit 1; }

cleanup() {
  umount "${MOUNT_POINT}" 2>/dev/null || fusermount -u "${MOUNT_POINT}" 2>/dev/null || true
  kill ${S3MOUNT_PID} 2>/dev/null || true
  rm -rf "${TMP_SINGLE}"
}
trap cleanup EXIT

S3_DEST="${MOUNT_POINT}/${S3_PREFIX}"
mkdir -p "${S3_DEST}"

if [ "${PROXY_MODE}" = "off" ]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  proxy_off 2>/dev/null || true
  echo "=== Proxy: 已关闭 (proxy_off) ==="
else
  proxy_on 2>/dev/null || {
    export http_proxy="http://leihaodong:Z7LqOKjqxyERzbgEBSqNYzziH3UBPt95MqaZ7mMcJBEH23Pf2RTzSyxS8K6H@10.1.20.50:23128/"
    export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
  }
  echo "=== Proxy: 已开启 ==="
fi

echo "=== 3. 逐文件下载并上传（ModelScope 魔搭，断点续传）==="
python3 "${SCRIPT_DIR}/download_qwen_to_s3_modelscope.py" \
  --repo-id "${REPO_ID}" \
  --s3-dest "${S3_DEST}" \
  --tmp-dir "${TMP_SINGLE}"

echo "模型已保存到 s3://${BUCKET}/${S3_PREFIX}/"
