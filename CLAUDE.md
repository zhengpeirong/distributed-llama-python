# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
# Build C extensions in-place (required before running)
python setup.py build_ext --inplace

# Editable install
pip install -e .

# With dev dependencies (pytest, fastapi, uvicorn)
pip install -e ".[dev]"
```

## Entry Points

```bash
dllama inference --model <path> --tokenizer <path> --prompt "..."
dllama chat --model <path> --tokenizer <path>
dllama perplexity --model <path> --tokenizer <path> --prompt "..."
dllama worker --host 0.0.0.0 --port 9999
dllama-api --model <path> --tokenizer <path>
```

## Run Tests

```bash
pytest distributed_llama/test/test_tokenizer.py -v
```

## Architecture

This is a Python port of a C++ distributed LLM inference engine. It uses **tensor parallelism** — the model's weight matrices are split across multiple machines (root + N workers) along columns (row-matmul slices) and rows (col-matmul slices), with each node computing its slice and syncing partial results over TCP.

The inference pipeline has four layers:

1. **`model.py`** — Reads the custom binary model format (magic `0xA00ABCD`), parses header key-value pairs, and exposes model config (`dim`, `n_layers`, `n_heads`, `n_kv_heads`, `n_experts`, etc.). Architecture types: Llama, Qwen3, Qwen3 MoE.

2. **`graph_builder.py`** — `build_llm_net()` translates a `LlmHeader` into a complete computation graph (`NnNetConfig` + per-node `NnNodeConfig`). Each node gets a chain of segments (START → per-layer attention/FFN pairs → END), each segment containing ops (EMBEDDING, RMS_NORM, MATMUL, ROPE, MULTIHEAD_ATT, SILU, MUL, etc.) and sync points. Weight slicing functions (`slice_row_matmul`, `slice_col_matmul`) determine how weights are split across nodes.

3. **`executor.py`** — `NnNetExecution` allocates pipe buffers as `uint8` arrays. `NnCpuDeviceSegment` compiles each segment: resolves pointer configs to numpy views, pre-fills RoPE caches, loads/dequantizes weights. On `forward()`, `NnExecutor` dispatches segments to `ThreadPoolExecutor` with interleaved sync points. Pure-Python/NumPy fallbacks exist for all ops; C extensions accelerate silu, gelu, add, mul, softmax, rope, multihead_att, and Q80×Q40 matmul.

4. **`network.py`** — TCP networking layer. `NnNetwork` manages multi-socket connections, `NnRootConfigWriter`/`NnWorkerConfigReader` serialize computation graph configs, `NnRootWeightLoader`/`NnWorkerWeightReader` distribute weight slices using `split_row_matmul_weight`/`split_col_matmul_weight`. `NnNetworkNodeSynchronizer` performs per-batch AllGather-style sync: root broadcasts to workers (`SYNC_WITH_ROOT`), and nodes exchange partial slices (`SYNC_NODE_SLICES`).

**Quantization types**: `F_32` (float32), `F_16` (float16), `F_Q40` (4-bit block quantization: 2-byte F16 scale + 16× uint4), `F_Q80` (8-bit block quantization: 2-byte F16 scale + 32× int8). C extensions live in `distributed_llama/quants/quants.c` and `distributed_llama/ops/ops.c`.

**Inference modes** (`inference.py`): `inference_mode` (eval+batch benchmark), `perplexity_mode`, `chat_mode` (interactive with chat templates). The root sends control packets (position + batch_size) to workers; workers receive embeddings via `SYNC_WITH_ROOT`.

**API server** (`api.py`): Zero-dependency HTTP server using Python stdlib `http.server`. OpenAI-compatible endpoints: `POST /v1/chat/completions` (streaming SSE, tool calling, naive prompt caching), `POST /v1/completions`, `GET /v1/models`, `GET /health`.

**Tokenizer** (`tokenizer.py`): BPE tokenizer compatible with the distributed-llama binary format. FNV-1a 64-bit hash lookup for vocab matching, streaming UTF-8 decode with recovery, chat template support (Llama2, Llama3, DeepSeek3, ChatML).

### Model format constraints

- `n_nodes` must not exceed `n_kv_heads`
- Q40 weights require Q80 sync type
- The `--nthreads` flag has known thread-safety issues (FIXME in code); single-thread is the safe default
