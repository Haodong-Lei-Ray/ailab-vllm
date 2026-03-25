# Qwen3.5-27B 推理部署（P 集群 `qwen35_env`）

你已执行：

```bash
mkdir -p ~/miniconda3/envs/qwen35_env
tar -xzf /mnt/phwfile/datafrontier/share/qwen35_env.tar.gz -C ~/miniconda3/envs/qwen35_env
```

下面从**第三步**开始，与《Qwen3.5_Env 高性能 P集群专属环境使用指南》一致，并只针对 **Qwen/Qwen3.5-27B、仅推理**。

---

## 0. 前置说明

- **必须在有 GPU 的计算节点上**跑 vLLM（登录节点一般不要拉服务）。
- 官方环境说明：该包针对 **A100 + CUDA 12.4**（`TORCH_CUDA_ARCH_LIST=8.0`）优化；若节点 GPU 架构不一致，可能出现 `No kernel found` 等错误。
- **27B 显存**：单卡 80GB 仍可能在 **KV + CUDA Graph 预热** 阶段 OOM（日志里 `max model len 262144` 会极大占用 KV）。**建议至少 2 卡张量并行**，并视情况限制 `--max-model-len`（见 `vllm_serve_sbatch.sh` 环境变量）。
- **`vllm_serve_sbatch.sh`**：`#SBATCH --gres=gpu:2` 为默认；**第 3 个参数 `N_GPU` 须与 `sbatch --gres=gpu:N` 一致**，并传给 `--tensor-parallel-size`。

---

## 1. 首次激活环境（解压后第一次）

不要用普通 `conda activate qwen35_env`（环境尚未注册），用**绝对路径**：

```bash
source ~/miniconda3/envs/qwen35_env/bin/activate
```

若你实际安装的是 Anaconda 且路径不同，把上面换成你的 env 根目录，例如：

```bash
source ~/anaconda3/envs/qwen35_env/bin/activate
```

---

## 2. 修复硬编码路径（必须，只做一次）

打包环境内含构建机绝对路径，**解压后首次**在**已激活**的 `qwen35_env` 里执行：

```bash
conda-unpack
```

成功后再进行任何 `python` / `vLLM` 操作。以后换机器重新解压，需要再执行一次。

---

## 3. 日常激活方式（完成 `conda-unpack` 之后）

以后登录可直接：

```bash
conda activate ~/miniconda3/envs/qwen35_env
```

（路径按你本机 env 目录调整。）

---

## 4. 推理前必加环境变量（防 PyTorch Dynamo 与 vLLM 冲突）

文档要求：**跑 vLLM 推理前**在 shell 里执行（可写进脚本或 `~/.bashrc` 里该 env 专用段落）：

```bash
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_SUPPRESS_ERRORS=1
export VLLM_TORCH_COMPILE_LEVEL=0
```

---

## 5. 最小自检（可选，建议在新节点做一次）

```bash
python -c "import vllm; print('vLLM Load Success!')"
```

文档中的 **opt-125m** 快速测速脚本（验证 Flash-Attention / CUDA Graphs）仍可直接用；若你只关心 27B，可跳过，但**第一次上机建议跑一遍**以便确认 GPU 与算子正常（见原 PDF「vLLM 性能与正确性测试」一节）。

---

## 6. 准备 Qwen3.5-27B 权重

任选其一：

### 6.1 本地目录（推荐思路清晰）

将 Hugging Face 格式的 `Qwen3.5-27B` 整目录放在可读路径，例如：

```text
/path/to/Qwen3.5-27B/
```

目录内应含 `config.json`、`tokenizer` 相关文件与权重分片等。

### 6.2 与现有脚本一致：S3 + s3mount

你仓库里 `vllm_serve_sbatch.sh` 使用 `s3mount` 把 `s3://...` 挂到本地再 `--model` 指向挂载路径。若你的 27B 已在对象存储，可把 `S3_URI` 换成 27B 前缀，**注意**：需在作业内能访问 `s3mount` 与 endpoint（与现脚本相同）。

---

## 7. 申请 GPU 并进入计算节点

### 7.1 推荐：`sbatch` 一键（与本仓库 `vllm_serve_sbatch.sh`）

脚本内已优先使用 `~/miniconda3/envs/qwen35_env`，并带齐 PDF 要求的三个 `export`。

**Qwen3.5-27B + 2 卡（默认与 `#SBATCH --gres=gpu:2` 对齐）：**

```bash
cd /mnt/petrelfs/leihaodong/local_model/vllm
sbatch vllm_serve_sbatch.sh \
  s3://datafrontier/leihaodong/Qwen/Qwen3.5-27B/ \
  Qwen3.5-27B \
  2
```

**4 卡时**：提交行必须覆盖脚本里的 `#SBATCH`，且第三个参数改为 `4`：

```bash
sbatch --gres=gpu:4 vllm_serve_sbatch.sh \
  s3://datafrontier/leihaodong/Qwen/Qwen3.5-27B/ \
  Qwen3.5-27B \
  4
```

也可用环境变量代替第 3 参数：`export VLLM_NUM_GPUS=4`（仍需 `sbatch --gres=gpu:4`）。

**仍 OOM 时**（可选）在提交前设置：

```bash
export VLLM_MAX_MODEL_LEN=32768    # 或 8192，降低 KV
export VLLM_MAX_NUM_SEQS=32
export VLLM_GPU_MEMORY_UTIL=0.88
sbatch ... Qwen3.5-27B 2
```

### 7.2 交互式：手动 `salloc` + 命令行

```bash
salloc -p DataFrontier_Knowledge --gres=gpu:2 --cpus-per-task=16 --time=08:00:00
```

进入分配节点后：

```bash
source ~/miniconda3/envs/qwen35_env/bin/activate   # 或 conda activate ~/miniconda3/envs/qwen35_env
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_SUPPRESS_ERRORS=1
export VLLM_TORCH_COMPILE_LEVEL=0
```

**说明**：27B 请保持 **`--gres=gpu:N` 与 `--tensor-parallel-size N` 一致**。

---

## 8. 启动 OpenAI 兼容 API（仅推理）

将 `MODEL_PATH` 换成你的模型根目录（本地路径或 s3mount 后的路径）。

### 8.1 双卡张量并行（40GB×2 等常见配置可尝试）

```bash
MODEL_PATH=/path/to/Qwen3.5-27B
PORT=8100

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name Qwen3.5-27B \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype auto \
  --trust-remote-code \
  --tensor-parallel-size 2
```

### 8.2 单卡（仅当显存足够时）

```bash
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name Qwen3.5-27B \
  --host 0.0.0.0 \
  --port 8100 \
  --dtype auto \
  --trust-remote-code \
  --tensor-parallel-size 1
```

### 8.3 显存仍不够时的调节方向（按需）

在命令中追加例如：

```bash
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

仍 OOM 时再适当降低 `--max-model-len` 或增加 GPU 数/提高 `tensor-parallel-size`。

---

## 9. 本机验证

另开终端（或在同一节点）：

```bash
curl http://127.0.0.1:8100/v1/models
```

聊天：

```bash
curl http://127.0.0.1:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-27B",
    "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

对外访问时，把 `127.0.0.1` 换成**计算节点 hostname**，并确认防火墙/策略允许该端口。

---

## 10. `vllm_serve_sbatch.sh`（已与本指南对齐）

`local_model/vllm/vllm_serve_sbatch.sh` 当前行为概要：

1. **优先** `~/miniconda3/envs/qwen35_env`（可用 `QWEN35_ENV` 覆盖路径）；不存在时再回退 `.venv` 或 `anaconda3` 的 `vllm`/`qwen3`。
2. 已写入第 4 节三个 `export`；`qwen35_env` 下**不**注入集群 GCC `LD_LIBRARY_PATH`。
3. **多卡**：默认 `#SBATCH --gres=gpu:2`；启动参数含 `--tensor-parallel-size`，由 **[第 3 个位置参数]** 或 **`VLLM_NUM_GPUS`** 指定，且须与 `sbatch --gres=gpu:N` 一致。
4. **`--served-model-name`** 与命令行第二个参数（展示用模型名）一致，便于 `curl` 里 `"model": "Qwen3.5-27B"`。

---

## 11. 常见问题

| 现象 | 处理 |
|------|------|
| `conda-unpack` 未做 | 先激活 env 再执行一次 `conda-unpack` |
| 未 export 三个 TORCH/VLLM 变量 | 按第 4 节补上 |
| `No kernel found` | 节点 GPU 是否与 A100/8.0 构建目标一致；换官方推荐节点 |
| OOM（如 warmup sampler / KV） | 增加 GPU 与 `tensor_parallel_size`；设 `VLLM_MAX_MODEL_LEN`；降 `VLLM_MAX_NUM_SEQS` / `VLLM_GPU_MEMORY_UTIL` |
| 端口不通 | 确认 `--host 0.0.0.0`、安全组/集群网络策略 |

---

## 12. 参考

- 原始说明 PDF：`/mnt/petrelfs/leihaodong/local_model/vllm/Qwen3.5_Env 高性能 P集群专属环境使用指南.pdf`
- 环境包：`/mnt/phwfile/datafrontier/share/qwen35_env.tar.gz`
