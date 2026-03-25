#!/bin/bash
#SBATCH -J qwen_s3_test
#SBATCH -p DataFrontier_Knowledge
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/petrelfs/leihaodong/local_model/qwen_s3_test_%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/local_model/qwen_s3_test_%j.err

# 用法: sbatch test_qwen_s3_sbatch.sh <模型S3地址> <模型名>
# 例:   sbatch test_qwen_s3_sbatch.sh s3://datafrontier/leihaodong/Qwen/Qwen3.5-0.8B/ Qwen3.5-0.8B

set -e

# 参数
S3_URI="${1:-s3://datafrontier/leihaodong/Qwen/Qwen3.5-0.8B/}"
MODEL_NAME="${2:-Qwen3.5-0.8B}"

# 解析 S3 路径: s3://bucket/path/ -> bucket, path
if [[ "$S3_URI" =~ s3://([^/]+)/(.+)$ ]]; then
    BUCKET="${BASH_REMATCH[1]}"
    BUCKET_PATH="${BASH_REMATCH[2]%/}"  # 去掉末尾 slash
else
    echo "ERROR: 无效的 S3 地址: $S3_URI"
    exit 1
fi

echo "模型地址: $S3_URI"
echo "模型名: $MODEL_NAME"
echo "Bucket: $BUCKET, Path: $BUCKET_PATH"

# 1. 环境
source ~/.bashrc 2>/dev/null || true
export http_proxy=; export https_proxy=; export HTTP_PROXY=; export HTTPS_PROXY=
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3

# 2. 在计算节点 NVMe 上创建挂载点和缓存
MOUNT_POINT="/nvme/${USER}/s3_mount_${SLURM_JOB_ID}"
CACHE_DIR="/nvme/${USER}/s3_cache_${SLURM_JOB_ID}"

if [ ! -d "/nvme/${USER}" ]; then
    mkdir -p "/nvme/${USER}"
fi
mkdir -p "$MOUNT_POINT" "$CACHE_DIR"
sleep 5

# 3. 挂载 S3 到 NVMe
echo "挂载 S3 到 NVMe..."
/mnt/petrelfs/leihaodong/app/s3mount "$BUCKET" "$MOUNT_POINT" \
  --cache "$CACHE_DIR" \
  --allow-delete --allow-overwrite \
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

# 4. 模型路径（挂载点 + bucket 内路径）
MODEL_PATH="${MOUNT_POINT}/${BUCKET_PATH}"
export MODEL_PATH
echo "模型路径: $MODEL_PATH"
ls -la "$MODEL_PATH/" 2>/dev/null || true

# 5. 推理
echo "加载模型 [$MODEL_NAME] 并推理..."
python3 << PYEOF
import os
import sys
import glob

model_path = os.environ.get("MODEL_PATH", "")
if not model_path or not os.path.exists(model_path):
    print(f"ERROR: 模型路径不存在: {model_path}")
    sys.exit(1)

# 查找权重文件（支持多种命名）
safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
if not safetensors:
    print(f"ERROR: 未找到 .safetensors 权重文件: {model_path}")
    sys.exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")

messages = [{"role": "user", "content": "hi"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

print("\n=== 输入: hi ===")
print("=== 输出:", response, "===")
PYEOF

# 6. 卸载
echo "卸载 S3..."
fusermount -u "$MOUNT_POINT" 2>/dev/null || umount "$MOUNT_POINT" 2>/dev/null || true
kill $S3MOUNT_PID 2>/dev/null || true

echo "完成"
