#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/mnt/petrelfs/leihaodong/local_model/vllm/log"
mkdir -p "${LOG_DIR}"

sbatch --gres=gpu:2 \
  --output "${LOG_DIR}/vllm_serve_%j.out" \
  --error "${LOG_DIR}/vllm_serve_%j.err" \
  vllm_serve_sbatch.sh \
  s3://datafrontier/leihaodong/Qwen/Qwen3.5-27B/ \
  Qwen3.5-27B \
  2