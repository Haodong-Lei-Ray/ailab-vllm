# vLLM 本地部署使用说明

通过 sbatch 在计算节点启动 vLLM 服务，s3mount 挂载 S3 上的模型权重，提供 OpenAI 兼容 API。

---

## 一、前置：S3 模型准备

**默认使用 Qwen2-0.5B**（vLLM 0.4.2 原生支持）。若 S3 上尚无，请先下载：

```bash
cd /mnt/petrelfs/leihaodong/local_model/vllm
proxy_on
bash download_qwen2_to_s3.sh
```

完成后模型位于 `s3://datafrontier/leihaodong/Qwen/Qwen2-0.5B/`。

**Qwen3.5**：需 vLLM 0.8+ 及 transformers 5.2+，本集群 glibc 2.17 下暂不可用。

---

## 二、启动服务端

```bash
cd /mnt/petrelfs/leihaodong/local_model/vllm

# 使用默认 Qwen2-0.5B
sbatch vllm_serve_sbatch.sh

# 或指定模型
sbatch vllm_serve_sbatch.sh s3://datafrontier/leihaodong/Qwen/Qwen2-0.5B/ Qwen2-0.5B
```

**说明**：
- 服务通过 s3mount 挂载 S3 模型，无需本地下载
- 启动后需等待 2–5 分钟加载模型
- 连接信息写入 `vllm_connection_<job_id>.txt`

---

## 三、查看连接地址

```bash
cat /mnt/petrelfs/leihaodong/local_model/vllm/vllm_connection_*.txt
tail -f /mnt/petrelfs/leihaodong/local_model/vllm/vllm_serve_<job_id>.out
```

---

## 四、发起查询

**循环提问模式**（推荐，无需每次重启脚本）：

```bash
cd /mnt/petrelfs/leihaodong/local_model/vllm
python query_vllm.py
# 或显式: python query_vllm.py -i
# 输入问题回车即可，输入 q/quit/exit 退出
```

**srun 在计算节点上运行**（集群内网络更稳，可 bypass 登录节点代理）：

```bash
bash run_query_srun.sh              # 循环提问
bash run_query_srun.sh "你的问题"    # 单次提问
```

**单次提问**：脚本会自动从 `vllm_connection_*.txt` 读取最新连接地址：

```bash
python query_vllm.py "你的问题"
```

**指定 job**：

```bash
python query_vllm.py --job 8255300 "用一句话介绍 Qwen2"
```

**手动指定地址**（默认端口 8100）：

```bash
python query_vllm.py --base-url http://<节点名>:8100/v1 "你的问题"
```

**其他参数**：

```bash
python query_vllm.py --max-tokens 256 "你的问题"
```

或使用 curl（需先 `unset http_proxy https_proxy` 以 bypass 代理）、SSH 端口转发。

---

## 五、安装 vLLM

**务必先 `proxy_on`。**

```bash
cd /mnt/petrelfs/leihaodong/local_model/vllm
bash install_vllm.sh
```

或手动：`conda create -n vllm python=3.10 -y && conda activate vllm && pip install vllm==0.4.2`

---

## 六、已知限制

- **环境**：当前使用 torch 2.3.0+cu121、xformers 0.0.26、triton 2.3.0，需 GCC/mpfr 库路径（脚本已配置）。
- **Qwen3.5**：需 vLLM 0.8+ 及 transformers 5.2+，glibc 2.17 下暂不可用。安装 `flash-attn` 可启用 FlashAttention 后端以提升性能。

---

## 七、结束服务

```bash
scancel -u leihaodong --job-name=vllm_serve
```
