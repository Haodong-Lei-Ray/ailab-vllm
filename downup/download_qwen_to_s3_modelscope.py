#!/usr/bin/env python3
"""
从 ModelScope 魔搭社区下载模型到 S3 挂载点，支持断点续传。
用法: python download_qwen_to_s3_modelscope.py --repo-id Qwen/Qwen3.5-27B --s3-dest /path/to/mount/leihaodong/Qwen/Qwen3.5-27B [--tmp-dir /path/to/tmp]
"""
import argparse
import os
import shutil
import sys
import tempfile

try:
    from modelscope.hub.api import HubApi
    from modelscope.hub.file_download import model_file_download
    from modelscope.utils.constant import DEFAULT_MODEL_REVISION
except ImportError:
    print("请安装: pip install modelscope")
    sys.exit(1)

CHUNK_SIZE = 32 * 1024 * 1024  # 32MB，s3mount 不支持 chmod/utime 等元数据操作，用纯读写分块复制


def get_repo_files(api: HubApi, repo_id: str) -> tuple:
    """获取模型仓库文件列表（仅文件，不含目录）"""
    endpoint = api.get_endpoint_for_read(repo_id=repo_id, repo_type="model")
    cookies = api.get_cookies()
    revision = api.get_valid_revision(
        repo_id, revision=DEFAULT_MODEL_REVISION, cookies=cookies, endpoint=endpoint
    )
    repo_files = api.get_model_files(
        model_id=repo_id,
        revision=revision,
        recursive=True,
        use_cookies=False if cookies is None else cookies,
        endpoint=endpoint,
    )
    return [f for f in repo_files if f.get("Type") != "tree"], revision


def should_skip(dest_path: str, expected_size: int) -> bool:
    """检查目标文件是否已存在且大小一致，可跳过"""
    if not os.path.exists(dest_path):
        return False
    try:
        return os.path.getsize(dest_path) == expected_size
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description="从 ModelScope 下载模型到 S3 挂载点（断点续传）")
    parser.add_argument("--repo-id", required=True, help="模型 ID，如 Qwen/Qwen3.5-27B")
    parser.add_argument("--s3-dest", required=True, help="S3 挂载点下的目标目录（本地路径）")
    parser.add_argument("--tmp-dir", default=None, help="临时下载目录，默认使用系统 temp")
    args = parser.parse_args()

    repo_id = args.repo_id
    s3_dest = os.path.abspath(args.s3_dest)
    tmp_base = args.tmp_dir or tempfile.gettempdir()
    os.makedirs(tmp_base, exist_ok=True)

    print(f"\n{'='*60}")
    print("SOURCE: ModelScope 魔搭社区 (https://www.modelscope.cn)")
    print(f"MODEL:  {repo_id}")
    print(f"TARGET: {s3_dest}")
    print(f"{'='*60}\n")

    api = HubApi()
    try:
        repo_files, revision = get_repo_files(api, repo_id)
    except Exception as e:
        print(f"获取文件列表失败: {e}")
        sys.exit(1)

    total = len(repo_files)
    print(f"共 {total} 个文件\n")

    for i, rf in enumerate(repo_files, 1):
        path = rf["Path"]
        size = rf.get("Size", 0)
        dest_path = os.path.join(s3_dest, path)

        if should_skip(dest_path, size):
            print(f"[{i}/{total}] {path} (已存在，跳过)")
            continue

        print(f"[{i}/{total}] {path}")
        tmp_dir = tempfile.mkdtemp(dir=tmp_base)
        try:
            local_path = model_file_download(
                model_id=repo_id,
                file_path=path,
                revision=revision,
                local_dir=tmp_dir,
            )
            if not local_path or not os.path.exists(local_path):
                print(f"  -> 下载失败\n")
                continue
            os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
            with open(local_path, "rb") as f_in, open(dest_path, "wb") as f_out:
                while True:
                    chunk = f_in.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f_out.write(chunk)
            print("  -> 已上传并删除本地\n")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print("全部完成")


if __name__ == "__main__":
    main()
