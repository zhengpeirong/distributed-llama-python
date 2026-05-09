# Distributed Llama (Python)

Connect home devices into a powerful cluster to accelerate LLM inference. More devices mean faster performance — leveraging tensor parallelism and high-speed synchronization over Ethernet.

A Python port of the [distributed-llama](https://github.com/b4rtaz/distributed-llama) C++ inference engine.

## Installation

```bash
git clone https://github.com/your-org/distributed-llama-python.git
cd distributed-llama-python

# Build C extensions (required)
python setup.py build_ext --inplace

# Install
pip install -e .
```

Requirements: Python 3.10+, NumPy, GCC.

## Quick Start

```bash
# Single-node inference
dllama inference --model <model.m> --tokenizer <tokenizer.t> --prompt "Hello"

# Distributed (root + 1 worker)
# On worker machine:
dllama worker --host 0.0.0.0 --port 9999

# On root machine:
dllama inference --model <model.m> --tokenizer <tokenizer.t> \
    --prompt "Hello" --workers 192.168.1.19:9999
```

## Commands

| Command | Description |
|---------|-------------|
| `dllama inference` | Run inference with benchmark output |
| `dllama chat` | Interactive chat with history |
| `dllama perplexity` | Compute perplexity of a prompt |
| `dllama worker` | Run worker node for distributed inference |
| `dllama-api` | Start OpenAI-compatible HTTP API server |

### Inference

```bash
dllama inference \
    --model <model.m> \
    --tokenizer <tokenizer.t> \
    --prompt "The meaning of life is" \
    --steps 256 \
    --temperature 0.0
```

### Interactive Chat

```bash
dllama chat --model <model.m> --tokenizer <tokenizer.t>
```

Supports chat templates: Llama2, Llama3, DeepSeek3, ChatML (auto-detected from tokenizer).

### API Server

```bash
dllama-api --model <model.m> --tokenizer <tokenizer.t> --port 8080
```

OpenAI-compatible endpoints:
- `POST /v1/chat/completions` — streaming SSE, tool calling, prompt caching
- `POST /v1/completions`
- `GET /v1/models`
- `GET /health`

### Distributed Setup

Start a worker on each additional machine:

```bash
dllama worker --host 0.0.0.0 --port 9999
```

Then run inference on the root node, listing all worker addresses:

```bash
dllama inference --model <model.m> --tokenizer <tokenizer.t> \
    --prompt "Hello" --workers 192.168.1.10:9999 192.168.1.11:9999
```

## Model Format

This engine uses the [distributed-llama](https://github.com/b4rtaz/distributed-llama) binary model format. Convert HuggingFace models using the converter in `distributed_llama/converter/`.

Supported architectures: Llama, Qwen3, Qwen3 MoE.

Quantization: F32, F16, Q40 (4-bit), Q80 (8-bit).

## Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | 256 | Number of tokens to generate |
| `--temperature` | 0.0 | Sampling temperature (0 = greedy) |
| `--topp` | 0.9 | Top-p sampling threshold |
| `--buffer-float-type` | q80 | Sync precision (f32/f16/q40/q80) |
| `--nthreads` | 1 | CPU threads (FIXME: >1 has thread-safety issues) |
| `--nbatches` | 32 | Batch size for eval (prompt processing) |
| `--seed` | 0 | Random seed |
| `--turbo` | off | Enable TCP_NODELAY for lower latency |
| `--max-seq-len` | model default | Override max sequence length |

## Constraints

- Number of nodes must not exceed `n_kv_heads`
- Q40 weights require Q80 sync type

## Known Limitations & TODOs

- **Multi-threading is unsafe** (`--nthreads > 1`). The Q80×Q40 matmul fast path has a race condition where one thread sets `ctx.weight = None` while thread 0 still reads it. Use `--nthreads 1` for correct results.
- Single-node and distributed modes may produce slightly different tokens due to floating-point accumulation order in batch vs sequential eval. Use `--temperature 0` with `--nbatches 32` for deterministic output matching C++.
- Col-matmul weight distribution (`split_col_matmul_weight`) for WO and W2 is correct but uses indirect field access via `size.x`; should use direct fields (`d0`, `n0`) for consistency with C++ reference.
- No GPU support — CPU-only inference.
- `dllama chat` does not persist conversation history across restarts.

## Credits

This is a Python port of [distributed-llama](https://github.com/b4rtaz/distributed-llama) by Bartosz Taudul. The original C++ engine pioneered the tensor-parallel distributed inference approach, model format, and network protocol that this port builds on.

## License

MIT
