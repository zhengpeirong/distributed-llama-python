"""LLM model header loader and config.

Port of src/llm.hpp and src/llm.cpp — reads the distributed-llama binary model
format (magic 0xA00ABCD), parses header key-value pairs, and provides model
configuration for graph building.
"""

import struct
import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from .quants import F_32, F_UNK, float_type_to_string


# --- Header key enum ---
class LlmHeaderKey(IntEnum):
    VERSION = 0
    ARCH_TYPE = 1
    DIM = 2
    HIDDEN_DIM = 3
    N_LAYERS = 4
    N_HEADS = 5
    N_KV_HEADS = 6
    N_EXPERTS = 7
    N_ACTIVE_EXPERTS = 8
    VOCAB_SIZE = 9
    SEQ_LEN = 10
    HIDDEN_ACT = 11
    ROPE_THETA = 12
    WEIGHT_FLOAT_TYPE = 13
    ROPE_SCALING_FACTOR = 14
    ROPE_SCALING_LOW_FREQ_FACTOR = 15
    ROPE_SCALING_HIGH_FREQ_FACTORY = 16
    ROPE_SCALING_ORIG_MAX_SEQ_LEN = 17
    ROPE_TYPE = 18
    HEAD_DIM = 19
    NORM_EPSILON = 20
    MOE_HIDDEN_DIM = 21


# --- Architecture types ---
class LlmArchType(IntEnum):
    LLAMA = 0xABCD00
    QWEN3 = 0xABCD01
    QWEN3_MOE = 0xABCD02


# --- Hidden activation ---
class LlmHiddenAct(IntEnum):
    GELU = 0
    SILU = 1


# --- RoPE type ---
class NnRopeType(IntEnum):
    ROPE_LLAMA = 0
    ROPE_FALCON = 1
    ROPE_LLAMA3_1 = 2


# --- Op codes ---
class NnOpCode(IntEnum):
    MERGE_ADD = 0
    MERGE_SUM = 1
    EMBEDDING = 2
    INV_RMS = 3
    RMS_NORM = 4
    MATMUL = 5
    ROPE = 6
    MULTIHEAD_ATT = 7
    GELU = 8
    SILU = 9
    MUL = 10
    SCALE = 11
    CAST = 12
    REPEAT_Z = 13
    SHIFT = 14
    SOFTMAX = 15
    MOE_GATE = 16


# --- Sync types ---
class NnSyncType(IntEnum):
    SYNC_WITH_ROOT = 0
    SYNC_NODE_SLICES = 1
    SYNC_NODE_SLICES_EXCEPT_ROOT = 2


# --- Pointer types ---
class NnPointerSource(IntEnum):
    SRC_PIPE = 0
    SRC_BUFFER = 1


class NnPointerType(IntEnum):
    PNTR_RAW = 0
    PNTR_BATCH = 1
    PNTR_BATCHED_SLICE = 2


# --- Op quant type ---
class NnOpQuantType(IntEnum):
    F32_F32_F32 = 0
    F32_Q40_F32 = 1
    F32_Q40_Q80 = 2
    F32_F32_Q80 = 3
    Q80_Q80_Q80 = 4
    Q80_Q80_F32 = 5
    Q80_Q40_F32 = 6
    Q80_F32_F32 = 7


# --- Model header ---
@dataclass
class LlmHeader:
    """Complete LLM model configuration from binary header."""

    version: int = 0
    arch_type: LlmArchType = LlmArchType.LLAMA
    dim: int = 0
    n_layers: int = 0
    n_heads: int = 0
    head_dim: int = 0
    n_kv_heads: int = 0
    n_experts: int = 0
    n_active_experts: int = 0
    orig_seq_len: int = 0
    seq_len: int = 0
    hidden_dim: int = 0
    moe_hidden_dim: int = 0
    hidden_act: LlmHiddenAct = LlmHiddenAct.SILU
    q_dim: int = 0
    kv_dim: int = 0
    vocab_size: int = 0
    rope_theta: float = 10000.0
    rope_type: NnRopeType = NnRopeType.ROPE_LLAMA
    rope_scaling_factor: float = 1.0
    rope_scaling_low_freq_factor: float = 1.0
    rope_scaling_high_freq_factor: float = 1.0
    rope_scaling_orig_max_seq_len: int = 0
    norm_epsilon: float = 1e-5
    weight_type: int = F_UNK
    sync_type: int = F_32
    header_size: int = 0
    file_size: int = 0

    @property
    def ff_dim(self) -> int:
        if self.arch_type == LlmArchType.QWEN3_MOE:
            return self.moe_hidden_dim
        return self.hidden_dim


# --- Magic number ---
MODEL_MAGIC = 0xA00ABCD


def _convert_norm_epsilon(value: int) -> float:
    if value == 5:
        return 1e-5
    if value == 6:
        return 1e-6
    raise ValueError(f"Unsupported norm epsilon value: {value}")


def load_llm_header(
    path: str,
    max_seq_len: int = 0,
    sync_type: int = F_32,
) -> LlmHeader:
    """Load model header from binary file."""

    header = LlmHeader()
    header.sync_type = sync_type

    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot open model file: {path}")

    with open(path, "rb") as f:
        magic = struct.unpack("<i", f.read(4))[0]

        if magic in (0xABCD00, 0xABCD01):
            raise ValueError("Old model format is not supported")
        if magic != MODEL_MAGIC:
            raise ValueError(f"Unsupported magic number: {magic:#x}")

        header_size = struct.unpack("<i", f.read(4))[0]
        header.header_size = header_size

        n_ints = (header_size - 2 * 4) // 4
        buffer = struct.unpack(f"<{n_ints}i", f.read(n_ints * 4))

        n_kv = n_ints // 2
        for i in range(n_kv):
            key = buffer[i * 2]
            value = buffer[i * 2 + 1]

            if key == LlmHeaderKey.VERSION:
                header.version = value
            elif key == LlmHeaderKey.ARCH_TYPE:
                header.arch_type = LlmArchType(value)
            elif key == LlmHeaderKey.DIM:
                header.dim = value
            elif key == LlmHeaderKey.HIDDEN_DIM:
                header.hidden_dim = value
            elif key == LlmHeaderKey.N_LAYERS:
                header.n_layers = value
            elif key == LlmHeaderKey.N_HEADS:
                header.n_heads = value
            elif key == LlmHeaderKey.N_KV_HEADS:
                header.n_kv_heads = value
            elif key == LlmHeaderKey.N_EXPERTS:
                header.n_experts = value
            elif key == LlmHeaderKey.N_ACTIVE_EXPERTS:
                header.n_active_experts = value
            elif key == LlmHeaderKey.VOCAB_SIZE:
                header.vocab_size = value
            elif key == LlmHeaderKey.SEQ_LEN:
                header.seq_len = value
            elif key == LlmHeaderKey.HIDDEN_ACT:
                header.hidden_act = LlmHiddenAct(value)
            elif key == LlmHeaderKey.ROPE_THETA:
                header.rope_theta = float(value)
            elif key == LlmHeaderKey.WEIGHT_FLOAT_TYPE:
                header.weight_type = value
            elif key == LlmHeaderKey.ROPE_SCALING_FACTOR:
                header.rope_scaling_factor = float(value)
            elif key == LlmHeaderKey.ROPE_SCALING_LOW_FREQ_FACTOR:
                header.rope_scaling_low_freq_factor = float(value)
            elif key == LlmHeaderKey.ROPE_SCALING_HIGH_FREQ_FACTORY:
                header.rope_scaling_high_freq_factor = float(value)
            elif key == LlmHeaderKey.ROPE_SCALING_ORIG_MAX_SEQ_LEN:
                header.rope_scaling_orig_max_seq_len = value
            elif key == LlmHeaderKey.ROPE_TYPE:
                header.rope_type = NnRopeType(value)
            elif key == LlmHeaderKey.HEAD_DIM:
                header.head_dim = value
            elif key == LlmHeaderKey.NORM_EPSILON:
                header.norm_epsilon = _convert_norm_epsilon(value)
            elif key == LlmHeaderKey.MOE_HIDDEN_DIM:
                header.moe_hidden_dim = value
            else:
                raise ValueError(f"Unsupported header key: {key}")

    if header.weight_type == F_UNK:
        raise ValueError("Model does not specify weight type")

    header.orig_seq_len = header.seq_len
    if max_seq_len > 0 and header.seq_len > max_seq_len:
        header.seq_len = max_seq_len

    if header.head_dim == 0:
        header.head_dim = header.dim // header.n_heads

    header.q_dim = header.head_dim * header.n_heads
    header.kv_dim = header.head_dim * header.n_kv_heads

    header.file_size = os.path.getsize(path)

    if header.arch_type in (LlmArchType.QWEN3, LlmArchType.QWEN3_MOE):
        header.rope_type = NnRopeType.ROPE_FALCON

    return header


def _arch_type_to_str(arch: LlmArchType) -> str:
    names = {
        LlmArchType.LLAMA: "Llama",
        LlmArchType.QWEN3: "Qwen3",
        LlmArchType.QWEN3_MOE: "Qwen3 MoE",
    }
    return names.get(arch, "Unknown")


def _hidden_act_to_str(act: LlmHiddenAct) -> str:
    names = {LlmHiddenAct.GELU: "Gelu", LlmHiddenAct.SILU: "Silu"}
    return names.get(act, "Unknown")


def _rope_type_to_str(rope: NnRopeType) -> str:
    names = {
        NnRopeType.ROPE_LLAMA: "Llama",
        NnRopeType.ROPE_LLAMA3_1: "Llama3.1",
        NnRopeType.ROPE_FALCON: "Falcon",
    }
    return names.get(rope, "Unknown")


def print_llm_header(header: LlmHeader):
    """Print model header information."""
    print(f"  Arch: {_arch_type_to_str(header.arch_type)}")
    print(f"  HiddenAct: {_hidden_act_to_str(header.hidden_act)}")
    print(f"  Dim: {header.dim}")
    print(f"  HeadDim: {header.head_dim}")
    print(f"  QDim: {header.q_dim}")
    print(f"  KvDim: {header.kv_dim}")
    print(f"  HiddenDim: {header.hidden_dim}")
    print(f"  VocabSize: {header.vocab_size}")
    print(f"  nLayers: {header.n_layers}")
    print(f"  nHeads: {header.n_heads}")
    print(f"  nKvHeads: {header.n_kv_heads}")
    if header.seq_len != header.orig_seq_len:
        print(f"  OrigSeqLen: {header.orig_seq_len}")
    if header.n_experts > 0:
        print(f"  nExperts: {header.n_experts}")
        print(f"  nActiveExperts: {header.n_active_experts}")
        print(f"  MoeHiddenDim: {header.moe_hidden_dim}")
    print(f"  SeqLen: {header.seq_len}")
    print(f"  NormEpsilon: {header.norm_epsilon}")
    print(f"  RopeType: {_rope_type_to_str(header.rope_type)}")
    print(f"  RopeTheta: {header.rope_theta:.0f}")
    if header.rope_type == NnRopeType.ROPE_LLAMA3_1:
        print(
            f"  RopeScaling: f={header.rope_scaling_factor:.1f}, "
            f"l={header.rope_scaling_low_freq_factor:.1f}, "
            f"h={header.rope_scaling_high_freq_factor:.1f}, "
            f"o={header.rope_scaling_orig_max_seq_len}"
        )
    print(f"  WeightType: {float_type_to_string(header.weight_type)}")
    print(f"  SyncType: {float_type_to_string(header.sync_type)}")
    print(f"  FileSize: {header.file_size}")
