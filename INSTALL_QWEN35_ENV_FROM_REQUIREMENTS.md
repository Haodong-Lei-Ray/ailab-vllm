# 用 `requirements.txt` 安装推理环境（qwen35_env）

## 说明
`requirements.txt` 是从你当前机器上的 `qwen35_env`（已解压的 conda 环境）导出的 `pip freeze` 列表。
由于其中包含 `torch/vllm/flash-attn` 等强依赖 CUDA/平台的包，**在不满足 CUDA 版本与硬件架构时可能无法直接复现**。

另外需要特别注意：这个 `requirements.txt` 里可能还包含少量以 `file://...` 指向本机构建产物的条目（`pip freeze` 有时会这样记录本地安装来源）。
因此在“换机器/重建环境”的情况下，直接 `pip install -r requirements.txt` **很可能失败**，而不是因为版本问题，而是因为本地路径不存在。

下面给出通用的安装步骤；如果在安装 `torch/vllm/flash-attn` 处失败，再按常见错误做对应调整（文末有建议）。

## 安装步骤

在这台机器（或兼容的 GPU 节点）上执行：

1. 创建一个干净的 conda 环境（建议 Python 3.11）

```bash
conda create -n qwen35_env_req python=3.11 -y
conda activate qwen35_env_req
```

2. 安装 requirements（可能会失败）

```bash
pip install -U pip setuptools wheel
pip install -r /mnt/petrelfs/leihaodong/local_model/ailab-vllm/requirements.txt
```

3. 快速自检（确认 vLLM 能导入）

```bash
python -c "import vllm; print('vllm ok')"
```

## 常见问题/修正建议

### 1）提示找不到 `file://...`
如果报类似 “Local file does not exist” / “No such file or directory”（通常来自 `file://` 条目），建议：

1. 先把 `file://` 相关行过滤掉，生成一个“尽量可安装”的版本：

```bash
python - <<'PY'
from pathlib import Path
src = Path('/mnt/petrelfs/leihaodong/local_model/ailab-vllm/requirements.txt')
dst = src.with_name('requirements_no_fileurl.txt')
lines = src.read_text(encoding='utf-8').splitlines()
out = [l for l in lines if 'file://' not in l]
dst.write_text('\n'.join(out) + '\n', encoding='utf-8')
print('Wrote:', dst)
PY
```

2. 再安装过滤后的 requirements：

```bash
pip install -r /mnt/petrelfs/leihaodong/local_model/ailab-vllm/requirements_no_fileurl.txt
```

说明：过滤后可能缺少你当前环境里的某些“本地构建产物来源”的包（例如 torch/vllm/flash-attn 的具体构建来源）。如果仍缺包，你需要按你的 CUDA/平台在目标机器上重装这些关键包（把报错贴出来我可以帮你补对应命令）。

- 如果 `torch` 安装失败：
  - 确保目标机器的 CUDA 版本与 `qwen35_env` 对应（你当前环境里 `vllm` 标记里有 `cu124`，通常意味着 CUDA 12.4 生态）。
  - 需要时可以为 torch/vllm 使用对应的 wheel 索引（这一步需要你给我错误日志，我再针对性改命令）。

- 如果 `flash-attn` 安装失败：
  - 优先使用与当前 torch/CUDA 匹配的预编译 wheel；必要时需要编译环境（gcc/nvcc）齐全。

- 如果遇到 “Resolver 失败/版本冲突”：
  - 通常是因为你创建的新环境缺少某些系统库/编译依赖，或 CUDA 不一致。

如果你愿意，把你在新环境执行 `pip install -r requirements.txt` 的报错贴出来，我可以把安装命令改到你当前节点可用的版本（例如加对应的 extra-index-url / 指定 torch 安装方式）。 

