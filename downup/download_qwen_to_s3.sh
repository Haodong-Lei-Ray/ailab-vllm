#!/bin/bash
# 将 Qwen/Qwen3.5-0.8B 权重下载并流式上传到 s3://datafrontier/leihaodong/qwen
# 策略：下载一个文件到 tmp -> 上传到 S3 -> 删除本地 -> 下载下一个
# 代理：下载（HuggingFace）时 proxy_on，上传（S3）时 proxy_off
# 镜像：HF_ENDPOINT 使用国内镜像加速，若镜像足够快可尝试关闭代理

set -e

# 配置
REPO_ID="Qwen/Qwen3.5-0.8B"
S3_PREFIX="s3://datafrontier/leihaodong/qwen"
TMP_DIR="/mnt/petrelfs/leihaodong/local_model/tmp"
ENDPOINT="http://d-ceph-ssd-inside.pjlab.org.cn"

# HF token（从 token.sh 读取，或使用环境变量 HF_TOKEN）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "$HF_TOKEN" ] && [ -f "$SCRIPT_DIR/token.sh" ]; then
    HF_TOKEN=$(grep 'token' "$SCRIPT_DIR/token.sh" 2>/dev/null | sed 's/.*"\([^"]*\)".*/\1/' | head -1)
fi
export HF_TOKEN

# 使用 HF 国内镜像加速下载
export HF_ENDPOINT="https://hf-mirror.com"

# 下载用 proxy_on，上传用 proxy_off
# 开启代理（Python 下载 HuggingFace 时需要）
source ~/.bashrc 2>/dev/null || true
shopt -s expand_aliases 2>/dev/null || true
proxy_on 2>/dev/null || {
    export http_proxy="http://leihaodong:Z7LqOKjqxyERzbgEBSqNYzziH3UBPt95MqaZ7mMcJBEH23Pf2RTzSyxS8K6H@10.1.20.50:23128/"
    export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
}
# proxy_off
# 创建临时目录
mkdir -p "$TMP_DIR"

# Python 脚本内联执行（继承代理用于下载；上传时在子进程中关闭代理）
python3 << 'PYTHON_SCRIPT'
import os
import subprocess
import sys

REPO_ID = "Qwen/Qwen3.5-0.8B"
S3_PREFIX = "s3://datafrontier/leihaodong/qwen"
TMP_DIR = "/mnt/petrelfs/leihaodong/local_model/tmp"
ENDPOINT = "http://d-ceph-ssd-inside.pjlab.org.cn"

# 上传时需关闭代理，构造无代理的环境
PROXY_VARS = {'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY'}
def get_env_no_proxy():
    return {k: v for k, v in os.environ.items() if k not in PROXY_VARS}

def run_aws_cmd(cmd, check=True):
    """执行 aws 命令时关闭代理"""
    ret = subprocess.run(cmd, shell=True, env=get_env_no_proxy())
    if check and ret.returncode != 0:
        sys.exit(ret.returncode)
    return ret.returncode

def main():
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        print("请安装 huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("错误: 请设置 HF_TOKEN 环境变量或在 token.sh 中配置 token")
        sys.exit(1)

    print(f"正在获取 {REPO_ID} 文件列表...")
    files = list_repo_files(REPO_ID, token=token)
    total = len(files)
    print(f"共 {total} 个文件待处理\n")

    for i, filename in enumerate(files, 1):
        print(f"[{i}/{total}] 处理: {filename}")

        # 1. 下载到 tmp（返回实际下载路径）
        print(f"  -> 下载中...")
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=TMP_DIR,
                local_dir_use_symlinks=False,
                token=token,
                force_download=False,
                resume_download=True,
            )
        except Exception as e:
            print(f"  ! 下载失败: {e}")
            continue

        if not os.path.exists(local_path):
            print(f"  ! 本地文件不存在: {local_path}")
            continue

        # 2. 上传到 S3（上传时关闭代理）
        print(f"  -> 上传到 S3...")
        s3_dest = f"{S3_PREFIX}/{filename}"
        if run_aws_cmd(f'aws s3 cp "{local_path}" "{s3_dest}" --endpoint-url {ENDPOINT} --checksum-algorithm SHA256', check=False) != 0:
            print(f"  ! 上传失败")
            continue

        # 3. 删除本地文件
        os.remove(local_path)
        print(f"  -> 完成，已删除本地副本")

    print("\n全部完成！")

if __name__ == "__main__":
    main()
PYTHON_SCRIPT
