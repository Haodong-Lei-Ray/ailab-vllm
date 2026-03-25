#!/bin/bash
# 将 Qwen/Qwen2-0.5B 下载并上传到 S3，供 vLLM 0.4.2 使用
# 用法: proxy_on && bash download_qwen2_to_s3.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ID="Qwen/Qwen2-0.5B"
S3_PREFIX="s3://datafrontier/leihaodong/Qwen/Qwen2-0.5B"
TMP_DIR="/mnt/petrelfs/leihaodong/local_model/tmp"
ENDPOINT="http://d-ceph-ssd-inside.pjlab.org.cn"

[[ -f "$SCRIPT_DIR/../Qwen3.5/token.sh" ]] && source "$SCRIPT_DIR/../Qwen3.5/token.sh" 2>/dev/null || true
[[ -z "$HF_TOKEN" ]] && unset HF_TOKEN || export HF_TOKEN

source ~/.bashrc 2>/dev/null || true
shopt -s expand_aliases 2>/dev/null || true
proxy_on 2>/dev/null || true

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "$TMP_DIR"

echo "=== 下载 Qwen2-0.5B 到 S3 ==="
python3 << PY
import os, subprocess, sys
REPO_ID, S3_PREFIX = "Qwen/Qwen2-0.5B", "s3://datafrontier/leihaodong/Qwen/Qwen2-0.5B"
TMP_DIR, ENDPOINT = "$TMP_DIR", "$ENDPOINT"
PROXY_VARS = {'http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY'}
def no_proxy(): return {k:v for k,v in os.environ.items() if k not in PROXY_VARS}

from huggingface_hub import hf_hub_download, list_repo_files
token = os.environ.get("HF_TOKEN") or None
files = list_repo_files(REPO_ID, token=token)
for i, fn in enumerate(files, 1):
    print(f"[{i}/{len(files)}] {fn}")
    local = hf_hub_download(REPO_ID, fn, local_dir=TMP_DIR, token=token or None)
    cmd = f'aws s3 cp "{local}" "{S3_PREFIX}/{fn}" --endpoint-url {ENDPOINT} --checksum-algorithm SHA256'
    subprocess.run(cmd, shell=True, env=no_proxy(), check=True)
    os.remove(local)
print("Done.")
PY
echo "完成: $S3_PREFIX"
