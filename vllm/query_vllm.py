#!/usr/bin/env python3
"""
通过 OpenAI 兼容 API 查询本地 vLLM 服务

用法:
  python query_vllm.py                     # 循环提问模式（无参数时默认）
  python query_vllm.py "你的问题"           # 单次提问，自动从 vllm_connection_*.txt 读取地址
  python query_vllm.py -i                  # 显式进入循环提问模式
  python query_vllm.py --base-url http://节点名:8100/v1 "你的问题"
  python query_vllm.py --job 8255300 "你的问题"

循环模式: 输入问题回车即可，输入 q/quit/exit 退出
"""
import argparse
import glob
import os
import re
import sys

VLLM_DIR = "/mnt/petrelfs/leihaodong/local_model/vllm"


def _find_base_url_from_connection():
    """从 vllm_connection_*.txt 读取最新的 base_url"""
    pattern = os.path.join(VLLM_DIR, "vllm_connection_*.txt")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for f in files:
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if line.startswith("http://") and "/v1" in line:
                    return line.strip()
                m = re.search(r"http://[^\s]+:\d+/v1", line)
                if m:
                    return m.group(0)
    return None


def main():
    parser = argparse.ArgumentParser(description="Query vLLM server via OpenAI API")
    parser.add_argument("query", nargs="*", help="输入问题（可多个词，会拼接）")
    parser.add_argument("--base-url", default=None,
                        help="vLLM 服务地址，如 http://节点名:8100/v1")
    parser.add_argument("--job", type=int, default=None,
                        help="指定 job_id，从 vllm_connection_<job_id>.txt 读取地址")
    parser.add_argument("--model", default="", help="模型名（可选，vLLM 单模型时可省略）")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成 token 数 (default: 512)")
    parser.add_argument("-i", "--interactive", action="store_true", help="循环提问模式")
    args = parser.parse_args()

    interactive = args.interactive or not args.query
    query = " ".join(args.query).strip() if args.query else ""

    # 确定 base_url
    base_url = args.base_url
    if not base_url and args.job:
        conn_file = os.path.join(VLLM_DIR, f"vllm_connection_{args.job}.txt")
        if os.path.isfile(conn_file):
            with open(conn_file) as fp:
                for line in fp:
                    m = re.search(r"http://[^\s]+:\d+/v1", line)
                    if m:
                        base_url = m.group(0)
                        break
    if not base_url:
        base_url = _find_base_url_from_connection()
    if not base_url:
        base_url = "http://localhost:8100/v1"
        print("未找到 vllm_connection_*.txt，使用默认: " + base_url, file=sys.stderr)
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    # 集群内通信需绕过代理
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)

    try:
        from openai import OpenAI
    except ImportError:
        print("请安装: pip install openai")
        sys.exit(1)

    client = OpenAI(base_url=base_url, api_key="EMPTY")

    # 获取模型名（若未指定）
    model = args.model
    if not model:
        try:
            models = client.models.list()
            model = models.data[0].id if models.data else "default"
        except Exception:
            model = "default"

    def do_query(q):
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": q}],
            max_tokens=args.max_tokens,
        )
        return completion.choices[0].message.content

    if interactive:
        print(f"Base URL: {base_url} | 模型: {model} | 输入 q/quit/exit 退出")
        print("---")
        while True:
            try:
                q = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出")
                break
            if not q:
                continue
            if q.lower() in ("q", "quit", "exit"):
                print("退出")
                break
            try:
                ans = do_query(q)
                print("AI>", ans)
            except Exception as e:
                print(f"Error: {e}")
            print()
    else:
        print(f"Query: {query}")
        print(f"Base URL: {base_url}")
        print("---")
        try:
            answer = do_query(query)
            print("Answer:", answer)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
