#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/mnt/petrelfs/leihaodong/local_model/vllm/log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SERVE_SCRIPT="${SCRIPT_DIR}/../vllm_serve_sbatch.sh"
mkdir -p "${LOG_DIR}"
[[ -f "${VLLM_SERVE_SCRIPT}" ]] || { echo "ERROR: 找不到 ${VLLM_SERVE_SCRIPT}"; exit 1; }

sbatch --gres=gpu:2 \
  --export "ALL,VLLM_PORT=8108" \
  --output "${LOG_DIR}/vllm_serve_%j.out" \
  --error "${LOG_DIR}/vllm_serve_%j.err" \
  "${VLLM_SERVE_SCRIPT}" \
  s3://datafrontier/leihaodong/Qwen/Qwen3-30B-A3B-Instruct-2507/ \
  Qwen3-30B-A3B-Instruct-2507 \
  2 \
