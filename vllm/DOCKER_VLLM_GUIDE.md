# 用 Docker 运行 vLLM（可直接照抄）

这份文档给你一套最短路径：用 Docker 起一个 OpenAI 兼容的 vLLM 服务，然后用客户端调用。

## 1) 前置条件

- 已安装 Docker
- 节点有 NVIDIA GPU，且 Docker 能识别 GPU（`nvidia-container-toolkit` 已配置）
- 模型文件在本地目录（例如 `/path/to/model`），或你已先挂载好远端存储到本地路径

快速自检：

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

看到 GPU 信息说明容器可用 GPU。

## 2) 拉镜像

```bash
docker pull vllm/vllm-openai:latest
```

## 3) 启动 vLLM（最小可跑）

```bash
MODEL_DIR=/path/to/model
PORT=8100

docker run --rm --gpus all \
  --name vllm_server \
  -p ${PORT}:8000 \
  -v ${MODEL_DIR}:/model:ro \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model /model \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --trust-remote-code
```

说明：

- 宿主机端口用 `${PORT}`，容器内固定 `8000`
- `--ipc=host` 对大模型推理更稳
- `--trust-remote-code` 对 Qwen 等模型通常需要

## 4) 验证服务

### 4.1 看模型列表

```bash
curl http://127.0.0.1:${PORT}/v1/models
```

### 4.2 发一个聊天请求

```bash
curl http://127.0.0.1:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model",
    "messages": [{"role": "user", "content": "你好，介绍一下你自己"}],
    "temperature": 0.7
  }'
```

如果你想更稳妥，先用 `/v1/models` 返回里的真实模型名替换上面的 `"model"` 字段。

## 5) 常用性能参数（按需加）

可在启动命令末尾追加：

```bash
--gpu-memory-utilization 0.9 \
--max-model-len 8192 \
--tensor-parallel-size 1 \
--max-num-seqs 32
```

- 多卡时把 `--tensor-parallel-size` 设为 GPU 数（如 2/4）
- 显存不够时先降 `--max-model-len` 和 `--max-num-seqs`

## 6) 在 Slurm 里用 srun 启容器（你当前环境常用）

如果你先拿到交互资源：

```bash
srun -p DataFrontier_Knowledge --gres=gpu:1 --cpus-per-task=8 --time=04:00:00 --pty bash
```

进入分配节点后再运行：

```bash
MODEL_DIR=/path/to/model
PORT=8100

docker run --rm --gpus all \
  --name vllm_server \
  -p ${PORT}:8000 \
  -v ${MODEL_DIR}:/model:ro \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model /model \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --trust-remote-code
```

注意：如果你要跨机器访问，客户端 `base_url` 用 `http://<节点hostname>:${PORT}/v1`。

## 7) 常见问题

- `docker: could not select device driver "" with capabilities: [[gpu]]`
  - 说明 NVIDIA 容器运行时未配置好，检查 `nvidia-container-toolkit`
- 端口冲突
  - 换宿主机端口，如 `-p 8101:8000`
- 模型加载失败（remote code）
  - 确认加了 `--trust-remote-code`
- 权限问题
  - 确认模型目录对当前用户和 Docker 进程可读

## 8) 停止服务

前台运行时 `Ctrl+C` 即可。  
后台运行时：

```bash
docker stop vllm_server
```

