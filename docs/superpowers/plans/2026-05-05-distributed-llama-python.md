# distributed-llama-python Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** Port distributed-llama C++ codebase to a pip-installable Python package with C extensions for compute-intensive ops.

**Architecture:** Python orchestration layer (CLI, network sync, graph building, tokenizer) with C extensions for hot-path computation (quantization, matrix multiply, RoPE, attention). TCP-based distributed inference over Ethernet.

**Tech Stack:** Python 3.10+, NumPy, C extensions (Python C API), socket + selectors for networking, setuptools for build.

---

## File Responsibility Map

| File | Responsibility |
|---|---|
| `distributed_llama/__init__.py` | Package init, version |
| `distributed_llama/quants/quants.c` + `.h` | F16/F32/Q40/Q80 conversion, quantization ops |
| `distributed_llama/quants/__init__.py` | Python wrapper for quants C extension |
| `distributed_llama/ops/ops.c` + `.h` | All NN ops: embedding, rms_norm, matmul, rope, attention, silu, softmax, moe |
| `distributed_llama/ops/__init__.py` | Python wrapper for ops C extension |
| `distributed_llama/model.py` | LlmHeader loader, binary format reader, model config |
| `distributed_llama/tokenizer.py` | BPE tokenizer with FNV-1a hash, special tokens, chat templates |
| `distributed_llama/sampler.py` | Top-p, temperature, argmax sampling |
| `distributed_llama/graph_builder.py` | Build NnNetConfig/NnNodeConfig from model header |
| `distributed_llama/executor.py` | Multithreaded graph execution engine |
| `distributed_llama/network.py` | TCP server/client for root-worker sync |
| `distributed_llama/inference.py` | Root/Worker inference loops |
| `distributed_llama/chat.py` | Chat template generation, EOS detection |
| `distributed_llama/api.py` | FastAPI HTTP server |
| `distributed_llama/cli.py` | CLI entry: inference/chat/perplexity/worker/api |
| `pyproject.toml` | Build config for pip install |

---

## Phase 1: Foundation (quants + tokenizer + model)

### Task 1: pyproject.toml and package skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `distributed_llama/__init__.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "distributed-llama-python"
version = "0.1.0"
description = "Distributed LLM inference in Python with C extensions"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[project.scripts]
dllama = "distributed_llama.cli:main"
dllama-api = "distributed_llama.api:main"

[project.optional-dependencies]
dev = ["pytest", "fastapi", "uvicorn"]

[tool.setuptools]
packages = ["distributed_llama", "distributed_llama.quants", "distributed_llama.ops", "distributed_llama.converter"]
```

- [ ] **Step 2: Write `__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Commit**

### Task 2: C extension for quantization (quants)

**Files:**
- Create: `distributed_llama/quants/quants.h`
- Create: `distributed_llama/quants/quants.c`
- Create: `distributed_llama/quants/__init__.py`

Port `src/nn/nn-quants.hpp` and `src/nn/nn-quants.cpp`.

Key functions:
- `init_lookup_tables()` — build f16→f32 lookup (65536 entries)
- `dequantize_q40_to_f32(blocks, n, n_threads, thread_idx, out)`
- `dequantize_q80_to_f32(blocks, n, n_threads, thread_idx, out)`
- `quantize_f32_to_q80(arr, n, n_threads, thread_idx, out)`
- `quantize_f32_to_q40(arr, n, n_threads, thread_idx, out)`

### Task 3: Tokenizer

**Files:**
- Create: `distributed_llama/tokenizer.py`
- Create: `distributed_llama/test/test_tokenizer.py`

Port `src/tokenizer.cpp`. BPE tokenizer with:
- Binary file reader (magic 0x567123 old / 0x567124 new format)
- FNV-1a 64-bit hash for regular vocab lookup
- BPE merge loop with vocab scores
- UTF-8 streaming decode with recovery
- Special token detection
- Chat template parsing

### Task 4: Model header loader

**Files:**
- Create: `distributed_llama/model.py`
- Create: `distributed_llama/test/test_model.py`

Port `src/llm.hpp` and `src/llm.cpp` header loading.

- Binary format: magic 0xA00ABCD, header size, key-value pairs
- LlmHeader dataclass with all fields
- `load_llm_header(path, max_seq_len, sync_type)` function
- Support Llama, Qwen3, Qwen3 MoE architectures

---

## Phase 2: Compute (ops + executor)

### Task 5: C extension for NN ops

**Files:**
- Create: `distributed_llama/ops/ops.h`
- Create: `distributed_llama/ops/ops.c`
- Create: `distributed_llama/ops/__init__.py`

Port `src/nn/nn-cpu-ops.cpp`. Each op takes: n_threads, thread_idx, batch_size, context struct.

Ops to implement:
- `op_embedding` — token → embedding vector
- `op_inv_rms` — 1/sqrt(mean(x²) + ε), per-column variant
- `op_rms_norm` — x * weight * inv_rms
- `op_matmul` — with quant types: F32_F32_F32, F32_Q40_F32, F32_Q40_Q80, F32_F32_Q80, Q80_Q80_Q80, Q80_Q80_F32, Q80_Q40_F32, Q80_F32_F32
- `op_rope` — Llama/Falcon/Llama3.1 rope variants
- `op_multihead_att` — attention with KV cache
- `op_softmax` — standard softmax
- `op_silu` — SiLU activation
- `op_mul` — element-wise multiply
- `op_merge_add` — merge with residual add
- `op_merge_sum` — merge with sum across experts
- `op_scale` — scale by expert weight
- `op_cast` — float type cast
- `op_repeat_z` — repeat across z dimension (for MoE)
- `op_shift` — shift into KV cache
- `op_moe_gate` — top-k expert selection

### Task 6: Graph builder

**Files:**
- Create: `distributed_llama/graph_builder.py`
- Create: `distributed_llama/test/test_graph_builder.py`

Port `src/llm.cpp` `buildLlmNet()`.

Define Python dataclasses for all config types:
- `NnSize3D`, `NnFloatType` enum
- `NnPipeConfig`, `NnBufferConfig`, `NnPointerConfig`
- `NnOpConfig`, `NnSyncConfig`, `NnSegmentConfig`
- `NnNetConfig`, `NnNodeConfig`
- Slice types: `NnRowMatmulSlice`, `NnColMatmulSlice`, `NnKvCacheSlice`, `NnRopeSlice`, `NnMultiHeadAttSlice`

Build the full computation graph for Llama/Qwen3/Qwen3-MoE.

### Task 7: Executor

**Files:**
- Create: `distributed_llama/executor.py`
- Create: `distributed_llama/test/test_executor.py`

Port `src/nn/nn-executor.cpp` and `src/nn/nn-cpu.cpp`.

- `NnNetExecution`: manages pipes (NumPy arrays), batch size
- `NnCpuDevice`: allocates buffers, creates segments
- `NnCpuDeviceSegment`: holds op function pointers, loads weights
- `NnExecutor`: builds step list, runs forward() with ThreadPoolExecutor
- Timer for eval/pred stats

---

## Phase 3: Distributed (network + inference)

### Task 8: Network layer

**Files:**
- Create: `distributed_llama/network.py`
- Create: `distributed_llama/test/test_network.py`

Port `src/nn/nn-network.cpp`.

- `NnSocket`: thin wrapper around Python socket
- `NnNetwork.serve(host, port)` → accepts worker connections
- `NnNetwork.connect(n_sockets, hosts, ports)` → connects to workers
- `write(socket_idx, data, size)` / `read(socket_idx, data, size)`
- `write_all(data, size)` — broadcast to all sockets
- `NnNetworkNodeSynchronizer`: sync pipe data per sync type (SYNC_WITH_ROOT, SYNC_NODE_SLICES, SYNC_NODE_SLICES_EXCEPT_ROOT)
- `NnRootConfigWriter` / `NnWorkerConfigReader`: send net/node configs
- `NnRootWeightLoader` / `NnWorkerWeightReader`: distribute weights

### Task 9: Inference loops

**Files:**
- Create: `distributed_llama/inference.py`
- Create: `distributed_llama/sampler.py`

Port `src/dllama.cpp` and `src/tokenizer.cpp` sampler.

- `RootLlmInference`: set_batch_size, set_position, set_token, forward, finish
- `WorkerLlmInference`: read control packet, wait for stop
- `inference_mode()`: eval batching + token generation with benchmark
- `perplexity_mode()`: compute perplexity
- `chat_mode()`: interactive chat with template, EOS detection

---

## Phase 4: CLI & API

### Task 10: CLI and Chat

**Files:**
- Create: `distributed_llama/cli.py`
- Create: `distributed_llama/chat.py`

Port `src/app.cpp` argument parsing and `src/dllama.cpp` main.

- Argparse for all CLI arguments
- Chat template generator (Llama2, Llama3, DeepSeek3, ChatML)
- EosDetector with streaming stop detection

### Task 11: API Server

**Files:**
- Create: `distributed_llama/api.py`

Port `src/dllama-api.cpp`.

- FastAPI server with `/v1/completions` and `/v1/chat/completions` endpoints
- OpenAI-compatible API

### Task 12: Integration testing & converter

**Files:**
- Create: `distributed_llama/converter/__init__.py`

Copy existing converter Python files from reference repo.
End-to-end test with a small model.
