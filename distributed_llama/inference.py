"""Inference engine for root and worker nodes.

Port of src/dllama.cpp and src/app.cpp — inference modes: inference
(with benchmark), perplexity, and interactive chat.
"""

import sys
import time
import struct
import random
import numpy as np
from typing import Optional, List

from .model import (
    LlmHeader, LlmArchType,
    NnOpCode, NnSyncType, NnRopeType,
)
from .graph_builder import (
    LlmNet, NnNetConfig, NnNodeConfig,
    NnPointerSource, NnPointerType,
    build_llm_net, release_llm_net,
)
from .tokenizer import (
    Tokenizer, Sampler,
    TokenizerChatStops, ChatTemplateGenerator,
    ChatTemplateType, ChatItem, EosDetector,
)
from .quants import (
    F_32, F_Q40, F_Q80,
    get_bytes, Q40_BLOCK_SIZE, Q80_BLOCK_SIZE,
)


# Re-export for convenience
__all__ = [
    "RootLlmInference", "WorkerLlmInference",
    "run_inference_app", "run_worker_app",
    "AppCliArgs", "AppInferenceContext",
]


class AppCliArgs:
    """CLI arguments for the inference app."""

    def __init__(self):
        self.mode: str = "inference"
        self.n_threads: int = 1  # FIXME: >1 causes thread safety issues
        self.n_batches: int = 32
        self.info: bool = True
        self.help: bool = False

        # inference
        self.model_path: Optional[str] = None
        self.tokenizer_path: Optional[str] = None
        self.prompt: Optional[str] = None
        self.sync_type: int = F_32
        self.n_workers: int = 0
        self.worker_hosts: List[str] = []
        self.worker_ports: List[int] = []
        self.temperature: float = 0.0
        self.topp: float = 0.9
        self.steps: int = 256
        self.benchmark: bool = False
        self.seed: int = int(time.time())
        self.chat_template_type: int = ChatTemplateType.UNKNOWN
        self.max_seq_len: int = 0
        self.net_turbo: bool = False
        self.gpu_index: int = -1
        self.gpu_segment_from: int = -1
        self.gpu_segment_to: int = -1

        # binding
        self.host: str = "0.0.0.0"
        self.port: int = 9999


class RootLlmInference:
    """Inference context on the root node."""

    def __init__(self, net: LlmNet, execution, executor, network):
        self.logits_pipe = execution.pipes[net.logits_pipe_index].view('f4')
        self._token_pipe = execution.pipes[net.token_pipe_index].view('f4')
        self._position_pipe = execution.pipes[net.position_pipe_index].view('f4')
        self._header = net.header
        self._execution = execution
        self._executor = executor
        self._network = network
        self._batch_size = 0
        self._position = 0

    def set_batch_size(self, batch_size: int):
        self._batch_size = batch_size
        self._execution.set_batch_size(batch_size)

    def set_position(self, pos: int):
        self._position = pos
        for i in range(self._execution.batch_size):
            self._position_pipe[i] = float(pos + i)

    def set_token(self, batch_index: int, token: int):
        self._token_pipe[batch_index] = float(token)

    def forward(self):
        if self._network is not None:
            bs = self._execution.batch_size
            # C++-style LlmControlPacket: position (uint32) + batchSize (uint32)
            header = struct.pack("<II", self._position, bs)
            self._network.write_all(header)
        self._executor.forward()

    def finish(self):
        """Send stop signal (batch_size=0) to all workers."""
        if self._network is not None:
            header = struct.pack("<II", 0, 0)
            self._network.write_all(header)


class WorkerLlmInference:
    """Inference context on worker nodes."""

    def __init__(self, execution, network):
        self.is_finished = False
        self._position_pipe = execution.pipes[0].view('f4')  # first pipe is POS
        self._execution = execution
        self._network = network

    def try_read_control_packet(self) -> bool:
        """Read control packet from root. Returns True if should continue.

        Matches C++ LlmControlPacket: position (u32) + batchSize (u32) = 8 bytes.
        Token data is NOT sent — the worker gets embeddings via SYNC_WITH_ROOT.
        """
        try:
            raw_data = self._network._read_raw(0, 8)
            if raw_data is None or len(raw_data) < 8:
                self.is_finished = True
                return False
            pos, batch_size = struct.unpack("<II", raw_data[:8])
            if batch_size == 0:
                self.is_finished = True
                return False
            self._execution.set_batch_size(batch_size)
            for i in range(batch_size):
                self._position_pipe[i] = float(pos + i)
            return True
        except (ConnectionError, OSError):
            self.is_finished = True
            return False


class AppInferenceContext:
    """Full inference context for root node."""

    def __init__(self):
        self.args: Optional[AppCliArgs] = None
        self.header: Optional[LlmHeader] = None
        self.inference: Optional[RootLlmInference] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.sampler: Optional[Sampler] = None
        self.network = None
        self.executor = None


def _read_stdin(guide: str, buffer_size: int = 2048) -> str:
    """Read a line from stdin with prompt."""
    sys.stdout.write(guide)
    sys.stdout.flush()
    line = sys.stdin.readline()
    if not line:
        return ""
    return line.rstrip("\n")


def run_inference_app(args: AppCliArgs, handler):
    """Set up and run inference on root node."""
    from .model import load_llm_header, print_llm_header
    from .graph_builder import build_llm_net, print_node_required_memory
    from .executor import (
        NnNetExecution, NnCpuDevice,
        NnExecutor, NnFakeNodeSynchronizer,
    )
    from .network import (
        NnNetwork, NnNetworkNodeSynchronizer,
        NnRootConfigWriter, NnRootWeightLoader,
    )

    n_nodes = args.n_workers + 1

    print("Loading model header...")
    header = load_llm_header(args.model_path, args.max_seq_len, args.sync_type)
    print_llm_header(header)

    # Validate constraints (matching C++ reference)
    if n_nodes > header.n_kv_heads:
        raise ValueError(
            "This version does not support more nodes than the number "
            "of KV heads in the model"
        )
    if header.weight_type == F_Q40 and header.sync_type != F_Q80:
        raise ValueError(
            "This version supports only Q40 weights with Q80 sync type"
        )

    print("Building computation graph...")
    net = build_llm_net(header, n_nodes, args.n_batches)

    if args.info:
        root_node_config = net.node_configs[0]
        print_node_required_memory(net.net_config, root_node_config)

    # Set up network
    network = None
    if args.n_workers > 0:
        print(f"Connecting to {args.n_workers} workers...")
        network = NnNetwork.connect(
            args.n_workers, args.worker_hosts, args.worker_ports,
        )
        # Send configs to workers
        config_writer = NnRootConfigWriter(network)
        config_writer.write_to_workers(net.net_config, net.node_configs)
    else:
        print("Single-node mode (no workers)")

    # Create execution
    n_threads = args.n_threads
    execution = NnNetExecution(n_threads, net.net_config)

    # Create root device
    root_config = net.node_configs[0]
    device = NnCpuDevice(net.net_config, root_config, execution)

    # Create synchronizer
    if network is not None:
        synchronizer = NnNetworkNodeSynchronizer(
            network, execution, net.net_config, root_config,
        )
    else:
        synchronizer = NnFakeNodeSynchronizer()

    # Create executor
    executor = NnExecutor(
        net.net_config, root_config, device,
        execution, synchronizer, args.benchmark,
    )

    # Load weights
    print("Loading weights...")
    _load_weights(args.model_path, net, executor, network)

    # Create inference context
    inference = RootLlmInference(net, execution, executor, network)

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer_path}")
    tokenizer = Tokenizer(args.tokenizer_path)
    tokenizer.print_header()

    # Create sampler
    sampler = Sampler(header.vocab_size, args.temperature, args.topp, args.seed)

    ctx = AppInferenceContext()
    ctx.args = args
    ctx.header = header
    ctx.inference = inference
    ctx.tokenizer = tokenizer
    ctx.sampler = sampler
    ctx.network = network
    ctx.executor = executor

    try:
        handler(ctx)
    finally:
        if inference is not None:
            inference.finish()
        release_llm_net(net)


def run_worker_app(args: AppCliArgs):
    """Run as a worker node."""
    from .network import (
        NnNetwork, NnWorkerConfigReader,
        NnWorkerWeightReader, NnNetworkNodeSynchronizer,
    )
    from .executor import (
        NnNetExecution, NnCpuDevice, NnExecutor,
    )

    print(f"Worker listening on {args.host}:{args.port}...")
    network = NnNetwork.serve(args.host, args.port)

    # Read configs from root
    config_reader = NnWorkerConfigReader(network)
    net_config = config_reader.read_net()
    node_config = config_reader.read_node()

    execution = NnNetExecution(args.n_threads, net_config)
    device = NnCpuDevice(net_config, node_config, execution)
    synchronizer = NnNetworkNodeSynchronizer(
        network, execution, net_config, node_config,
    )
    executor_instance = NnExecutor(
        net_config, node_config, device,
        execution, synchronizer, False,
    )

    # Read weights
    print("Receiving weights from root...")
    weight_reader = NnWorkerWeightReader(executor_instance, network)
    weight_reader.read()

    # Main loop
    inference = WorkerLlmInference(execution, network)
    print("Worker ready, waiting for inference commands...")

    while not inference.is_finished:
        if inference.try_read_control_packet():
            executor_instance.forward()

    print("Worker finished.")


def _load_weights(model_path: str, net: LlmNet, executor, network):
    """Load model weights and distribute to workers."""
    from .network import NnRootWeightLoader

    loader = NnRootWeightLoader(executor, network, net.net_config.n_nodes)
    header = net.header

    # Keep file open and seek+read each chunk to avoid loading 5.9GB into page cache.
    # For distributed: root only keeps ~3.2GB of its own weight slice.
    f = open(model_path, "rb")
    weight_offset = header.header_size
    weight_start = weight_offset

    def _read_chunk(n_bytes):
        nonlocal weight_offset
        f.seek(weight_offset)
        chunk = f.read(n_bytes)
        weight_offset += n_bytes
        return chunk

    def _read(n_bytes):
        return _read_chunk(n_bytes)

    def _load(name, op_idx, n_bytes):
        return loader.load_all(name, op_idx, n_bytes, _read_chunk(n_bytes))

    def _load_row(name, op_idx, slice_obj):
        n_bytes = slice_obj.size.n_bytes
        return loader.load_row_matmul_slices(name, op_idx, 0, slice_obj, _read_chunk(n_bytes))

    def _load_col(name, op_idx, slice_obj):
        n_bytes = slice_obj.size.n_bytes
        return loader.load_col_matmul_slices(name, op_idx, 0, slice_obj, _read_chunk(n_bytes))

    # Embedding
    loaded = _load("embedding", 0, net.token_embedding_size.n_bytes)

    for layer_idx in range(header.n_layers):
        loaded += _load_row("block_matmul_q", layer_idx, net.q_slice)
        loaded += _load_row("block_matmul_k", layer_idx, net.k_slice)
        loaded += _load_row("block_matmul_v", layer_idx, net.v_slice)
        loaded += _load_col("block_matmul_wo", layer_idx, net.wo_slice)

        if header.n_experts > 0:
            loaded += _load("block_moe_gate", layer_idx, net.moe_gate_size.n_bytes)
            for expert_idx in range(header.n_experts):
                loaded += _load_row("block_matmul_w1", layer_idx, net.w1_slice)
                loaded += _load_col("block_matmul_w2", layer_idx, net.w2_slice)
                loaded += _load_row("block_matmul_w3", layer_idx, net.w3_slice)
        else:
            loaded += _load_row("block_matmul_w1", layer_idx, net.w1_slice)
            loaded += _load_col("block_matmul_w2", layer_idx, net.w2_slice)
            loaded += _load_row("block_matmul_w3", layer_idx, net.w3_slice)

        if header.arch_type in (LlmArchType.QWEN3, LlmArchType.QWEN3_MOE):
            loaded += _load("block_norm_q", layer_idx, net.qk_rms_norm_size.n_bytes)
            loaded += _load("block_norm_k", layer_idx, net.qk_rms_norm_size.n_bytes)

        loaded += _load("block_norm_0", layer_idx, net.rms_norm_size.n_bytes)
        loaded += _load("block_norm_1", layer_idx, net.rms_norm_size.n_bytes)

        if loaded > 10800000:
            loaded = 0
            sys.stdout.write(f"\r  Loaded {layer_idx + 1}/{header.n_layers} layers")
            sys.stdout.flush()

    loaded += _load("final_norm", 0, net.rms_norm_size.n_bytes)
    loaded += _load_row("final_matmul_logits", 0, net.wcls_slice)

    # Validate total bytes read matches expected weight data size
    expected_weight_size = header.file_size - header.header_size
    missing_bytes = (weight_offset - weight_start) - expected_weight_size
    if missing_bytes != 0:
        raise ValueError(
            f"Missing bytes in weight file: -{abs(missing_bytes)}"
            if missing_bytes < 0
            else f"Extra bytes in weight file: +{missing_bytes}"
        )

    print(f"\r  Weights loaded ({expected_weight_size} bytes)")
    f.close()
    loader.finish()


# --- Inference modes ---

def inference_mode(ctx: AppInferenceContext):
    """Run inference with benchmark output."""
    args = ctx.args
    if args.prompt is None:
        raise ValueError("Prompt is required")
    if args.steps == 0:
        raise ValueError("Number of steps is required")

    encode_result = ctx.tokenizer.encode(args.prompt, is_start=True, add_special_tokens=True)
    input_tokens = encode_result
    n_input_tokens = len(input_tokens)

    if n_input_tokens > ctx.header.seq_len:
        raise ValueError(
            f"Prompt tokens ({n_input_tokens}) exceed sequence length ({ctx.header.seq_len})"
        )
    if n_input_tokens > args.steps:
        raise ValueError(
            f"Prompt tokens ({n_input_tokens}) exceed number of steps ({args.steps})"
        )

    pos = 0
    sent_bytes = 0
    recv_bytes = 0
    eval_total_time = 0
    pred_total_time = 0

    token = input_tokens[pos]
    print(args.prompt)

    # Eval phase (process prompt tokens in batches, matching C++ behavior)
    i = 0
    while True:
        remaining = n_input_tokens - 1 - pos
        if remaining <= 0:
            break
        batch_size = remaining if remaining < args.n_batches else args.n_batches

        ctx.inference.set_batch_size(batch_size)
        ctx.inference.set_position(pos)
        for j in range(batch_size):
            ctx.inference.set_token(j, input_tokens[i + j])

        ctx.inference.forward()

        pos += batch_size
        i += batch_size
        token = input_tokens[pos] if pos < n_input_tokens else token

        if ctx.network is not None:
            sent_bytes, recv_bytes = ctx.network.get_stats()

        exec_time = ctx.executor.get_total_time_op_us() if ctx.executor else 0
        sync_time = ctx.executor.get_total_time_sync_us() if ctx.executor else 0
        eval_us = exec_time + sync_time
        eval_total_time += eval_us

        print(
            f"  Eval {int(exec_time) // 1000:5d} ms Sync {int(sync_time) // 1000:5d} ms | "
            f"Sent {sent_bytes // 1024:6d} kB Recv {recv_bytes // 1024:6d} kB | "
            f"({batch_size} tokens)"
        )

    ctx.inference.set_batch_size(1)
    ctx.tokenizer.reset_decoder()

    # Prediction phase (C++ style: while-pos loop with explicit counter)
    max_pos = min(ctx.header.seq_len, args.steps)
    n_ptr = 0
    while pos < max_pos:
        ctx.inference.set_position(pos)
        ctx.inference.set_token(0, token)
        ctx.inference.forward()

        logits = ctx.inference.logits_pipe[:ctx.header.vocab_size]
        token = ctx.sampler.sample(logits.tolist())
        piece = ctx.tokenizer.decode(token) or ""

        if ctx.network is not None:
            sent_bytes, recv_bytes = ctx.network.get_stats()

        exec_time = ctx.executor.get_total_time_op_us() if ctx.executor else 0
        sync_time = ctx.executor.get_total_time_sync_us() if ctx.executor else 0
        pred_us = exec_time + sync_time
        pred_total_time += pred_us

        print(
            f"  Pred {int(exec_time) // 1000:5d} ms Sync {int(sync_time) // 1000:5d} ms | "
            f"Sent {sent_bytes // 1024:6d} kB Recv {recv_bytes // 1024:6d} kB | "
            f"{piece if piece else '~'}"
        )
        sys.stdout.flush()

        pos += 1
        n_ptr += 1

    # Print benchmark summary
    n_eval_tokens = n_input_tokens - 1
    n_pred_tokens = n_ptr
    print()
    print("Evaluation")
    print(f"    nBatches: {args.n_batches}")
    print(f"     nTokens: {n_eval_tokens}")
    if eval_total_time > 0:
        eval_ms = eval_total_time / 1000.0
        print(f"    tokens/s: {n_eval_tokens * 1000 / eval_ms:.2f} "
              f"({eval_ms / n_eval_tokens:.2f} ms/tok)")
    print("Prediction")
    print(f"     nTokens: {n_pred_tokens}")
    if pred_total_time > 0 and n_pred_tokens > 0:
        pred_ms = pred_total_time / 1000.0
        print(f"    tokens/s: {n_pred_tokens * 1000 / pred_ms:.2f} "
              f"({pred_ms / n_pred_tokens:.2f} ms/tok)")


def perplexity_mode(ctx: AppInferenceContext):
    """Compute perplexity of a prompt."""
    args = ctx.args
    if args.prompt is None:
        raise ValueError("Prompt is required")

    encode_result = ctx.tokenizer.encode(args.prompt, is_start=True, add_special_tokens=True)
    input_tokens = encode_result
    n_input_tokens = len(input_tokens)

    print(f"Evaluating {n_input_tokens} tokens...")

    total_log_prob = 0.0

    ctx.inference.set_batch_size(1)

    for pos in range(n_input_tokens - 1):
        ctx.inference.set_position(pos)
        ctx.inference.set_token(0, input_tokens[pos])
        ctx.inference.forward()

        logits = ctx.inference.logits_pipe
        # softmax
        max_val = np.max(logits)
        exps = np.exp(logits - max_val)
        probs = exps / np.sum(exps)

        target_token = input_tokens[pos + 1]
        prob = probs[target_token]

        total_log_prob += np.log(max(prob, 1e-30))
        print(f"  {pos + 1:5d} / {n_input_tokens - 1}, prob={prob:.6f}")

    avg_log_prob = total_log_prob / (n_input_tokens - 1)
    perplexity = np.exp(-avg_log_prob)

    print()
    print("Results")
    print(f"    perplexity: {perplexity:.6f} (lower = better)")
    print(f"    avgLogProb: {avg_log_prob:.6f}")
    print(f"    bitPerToken: {-avg_log_prob / np.log(2):.6f}")


def chat_mode(ctx: AppInferenceContext):
    """Interactive chat with conversation history."""
    args = ctx.args
    seq_len = ctx.header.seq_len

    stops = TokenizerChatStops(ctx.tokenizer)
    template_gen = ChatTemplateGenerator(
        args.chat_template_type,
        ctx.tokenizer.chat_template,
        stops.stops[0] if stops.stops else "",
    )

    eos_tokens = ctx.tokenizer.eos_token_ids
    eos_pieces = [ctx.tokenizer.vocab[t] for t in eos_tokens]
    eos_detector = EosDetector(
        len(eos_tokens), eos_tokens,
        eos_pieces if eos_pieces else [""],
        stops.max_stop_length, stops.max_stop_length,
    )

    sys_prompt = _read_stdin("System prompt (optional): ")
    delta_items = []
    if sys_prompt:
        delta_items.append(ChatItem("system", sys_prompt))

    pos = 0
    while pos < seq_len:
        user_prompt = ""
        while not user_prompt:
            user_prompt = _read_stdin("\nUser\n> ")

        delta_items.append(ChatItem("user", user_prompt))
        content, public_prompt = template_gen.generate(delta_items, True)

        encode_result = ctx.tokenizer.encode(content, is_start=(pos == 0), add_special_tokens=True)
        input_tokens = encode_result
        n_input_tokens = len(input_tokens)

        # eval phase
        user_prompt_end_pos = min(seq_len, pos + n_input_tokens - 1)
        i = 0
        while True:
            remaining = user_prompt_end_pos - pos
            if remaining <= 0:
                break
            batch_size = min(remaining, args.n_batches)

            ctx.inference.set_batch_size(batch_size)
            ctx.inference.set_position(pos)
            for j in range(batch_size):
                ctx.inference.set_token(j, input_tokens[i + j])

            ctx.inference.forward()

            i += batch_size
            pos += batch_size
            token = input_tokens[i + 1] if i + 1 < len(input_tokens) else 0

        ctx.inference.set_batch_size(1)
        ctx.tokenizer.reset_decoder()

        print("\nAssistant")
        if public_prompt:
            sys.stdout.write(public_prompt)

        # prediction phase
        while pos < seq_len:
            ctx.inference.set_position(pos)
            ctx.inference.set_token(0, token)
            ctx.inference.forward()

            logits = ctx.inference.logits_pipe[:ctx.header.vocab_size]
            token = ctx.sampler.sample(logits.tolist())
            piece = ctx.tokenizer.decode(token)

            eos_type = eos_detector.append(token, piece)
            if eos_type in (EosDetector.MAYBE_EOS, EosDetector.EOS):
                delta = eos_detector.get_delta()
                if delta:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                eos_detector.reset()
            elif piece:
                sys.stdout.write(piece)
                sys.stdout.flush()

            pos += 1
            if eos_type == 1:  # EOS
                break

        delta_items.clear()
        if pos >= seq_len:
            print("\n(end of context)")

    print()
