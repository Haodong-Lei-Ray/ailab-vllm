#!/usr/bin/env python3
"""
vLLM 启动包装器：在 import 前强制使用 TORCH_SDPA 后端，
避免 xformers/triton 在计算节点上的编译问题。
用法：与 api_server 相同，例如
  python run_vllm_torch_sdpa.py --model /path/to/model --host 0.0.0.0 --port 8000
"""
import os
import runpy
import sys

os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"

# Monkey-patch 1: vllm 0.4.2 在 flash_attn 未安装时直接返回 XFORMERS，不检查 env
import vllm.attention.selector as _sel
_orig_which = _sel._which_attn_to_use

def _patched_which(dtype):
    backend = os.environ.get("VLLM_ATTENTION_BACKEND")
    if backend and backend in _sel._Backend.__members__:
        return _sel._Backend[backend]
    return _orig_which(dtype)

_sel._which_attn_to_use = _patched_which
_sel.get_attn_backend.cache_clear()

# Monkey-patch 2: TorchSDPAMetadata 不接受 max_query_len 等，且 driver 侧首次调用无 slot_mapping
import torch as _torch
import vllm.attention.backends.torch_sdpa as _tsdpa
_orig_make = _tsdpa.TorchSDPABackend.make_metadata

def _patched_make_metadata(*args, **kwargs):
    for k in ('max_query_len', 'max_seq_len', 'subquery_start_loc', 'seq_start_loc',
              'context_lens_tensor', 'use_cuda_graph'):
        kwargs.pop(k, None)
    if 'slot_mapping' not in kwargs:
        kwargs['slot_mapping'] = _torch.empty(0, dtype=_torch.long)
    return _orig_make(*args, **kwargs)

_tsdpa.TorchSDPABackend.make_metadata = staticmethod(_patched_make_metadata)

# 以 __main__ 方式运行 api_server，继承当前 argv
sys.argv[0] = "vllm.entrypoints.openai.api_server"
runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
