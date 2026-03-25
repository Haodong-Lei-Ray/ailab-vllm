#!/bin/bash
# 通过 srun 在计算节点上运行 query_vllm.py（集群内网络，无需代理）
#
# 用法:
#   bash run_query_srun.sh              # 循环提问模式
#   bash run_query_srun.sh "你的问题"    # 单次提问
#   bash run_query_srun.sh --job 8255300 "问题"

set -e

VLLM_DIR="/mnt/petrelfs/leihaodong/local_model/vllm"
cd "$VLLM_DIR"

# 环境
export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY=
source /mnt/petrelfs/leihaodong/anaconda3/etc/profile.d/conda.sh
conda activate vllm 2>/dev/null || { echo "ERROR: conda activate vllm 失败"; exit 1; }

# srun: 同分区、无 GPU
exec srun -N 1 -p DataFrontier_Knowledge --ntasks-per-node 1 --gres=gpu:0 \
  python3 "$VLLM_DIR/query_vllm.py" "$@"
