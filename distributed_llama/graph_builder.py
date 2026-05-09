"""Computation graph builder for distributed LLM inference.

Port of src/llm.cpp buildLlmNet() and src/nn/nn-core.hpp config types.

Translates a LlmHeader into NnNetConfig + NnNodeConfig[], defining the
complete computation graph: pipes, buffers, ops, and sync points.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import math

from .quants import (
    F_32, F_16, F_Q40, F_Q80,
    Q40_BLOCK_SIZE, Q80_BLOCK_SIZE,
    get_bytes, get_block_size,
)
from .model import (
    LlmHeader, LlmArchType, LlmHiddenAct,
    NnOpCode, NnSyncType, NnPointerSource, NnPointerType, NnRopeType,
    NnOpQuantType,
)


# --- 3D size descriptor ---
@dataclass
class NnSize3D:
    float_type: int = F_32
    z: int = 1
    y: int = 1
    x: int = 1

    @property
    def length(self) -> int:
        return self.z * self.y * self.x

    @property
    def n_bytes(self) -> int:
        return get_bytes(self.float_type, self.length)

    @property
    def n_bytes_xy(self) -> int:
        return get_bytes(self.float_type, self.x * self.y)

    def __repr__(self):
        return f"NnSize3D(ft={self.float_type}, z={self.z}, y={self.y}, x={self.x})"


def size0() -> NnSize3D:
    return NnSize3D()


def size1d(ft: int, x: int) -> NnSize3D:
    return NnSize3D(float_type=ft, x=x)


def size2d(ft: int, y: int, x: int) -> NnSize3D:
    return NnSize3D(float_type=ft, y=y, x=x)


def size3d(ft: int, z: int, y: int, x: int) -> NnSize3D:
    return NnSize3D(float_type=ft, z=z, y=y, x=x)


# --- Pointer config ---
@dataclass
class NnPointerConfig:
    source: int = NnPointerSource.SRC_PIPE
    pointer_index: int = 0
    pointer_type: int = NnPointerType.PNTR_BATCH


def pointer_batch_config(source: int, index: int) -> NnPointerConfig:
    return NnPointerConfig(source=source, pointer_index=index,
                          pointer_type=NnPointerType.PNTR_BATCH)


def pointer_batched_slice_config(source: int, index: int) -> NnPointerConfig:
    return NnPointerConfig(source=source, pointer_index=index,
                          pointer_type=NnPointerType.PNTR_BATCHED_SLICE)


def pointer_raw_config(source: int, index: int) -> NnPointerConfig:
    return NnPointerConfig(source=source, pointer_index=index,
                          pointer_type=NnPointerType.PNTR_RAW)


def has_pointer_continuous_memory(config: NnPointerConfig) -> bool:
    return config.pointer_type in (
        NnPointerType.PNTR_BATCH,
        NnPointerType.PNTR_RAW,
    )


# --- Pipe and buffer configs ---
@dataclass
class NnPipeConfig:
    name: str = ""
    size: NnSize3D = field(default_factory=size0)


@dataclass
class NnBufferConfig:
    name: str = ""
    size: NnSize3D = field(default_factory=size0)


# --- Op configs ---
@dataclass
class NnOpConfig:
    code: int = 0
    name: str = ""
    index: int = 0
    input: NnPointerConfig = field(default_factory=NnPointerConfig)
    output: NnPointerConfig = field(default_factory=NnPointerConfig)
    weight_size: NnSize3D = field(default_factory=size0)
    config: Optional[object] = None


@dataclass
class NnSyncConfig:
    pipe_index: int = 0
    sync_type: int = NnSyncType.SYNC_WITH_ROOT


@dataclass
class NnPreSyncConfig:
    pipe_index: int = 0


@dataclass
class NnSegmentConfig:
    ops: List[NnOpConfig] = field(default_factory=list)
    syncs: List[NnSyncConfig] = field(default_factory=list)


@dataclass
class NnNetConfig:
    n_batches: int = 0
    n_nodes: int = 0
    pipes: List[NnPipeConfig] = field(default_factory=list)
    pre_syncs: List[NnPreSyncConfig] = field(default_factory=list)


@dataclass
class NnNodeConfig:
    node_index: int = 0
    buffers: List[NnBufferConfig] = field(default_factory=list)
    segments: List[NnSegmentConfig] = field(default_factory=list)


# --- Slice types ---
@dataclass
class NnKvCacheSlice:
    kv_dim0: int = 0
    key_size: NnSize3D = field(default_factory=size0)
    value_size: NnSize3D = field(default_factory=size0)


@dataclass
class NnRowMatmulSlice:
    type: int = F_32
    n_nodes: int = 0
    d0: int = 0
    n: int = 0
    size: NnSize3D = field(default_factory=size0)
    slice_size: NnSize3D = field(default_factory=size0)


@dataclass
class NnColMatmulSlice:
    type: int = F_32
    n_nodes: int = 0
    n: int = 0
    n0: int = 0
    d: int = 0
    size: NnSize3D = field(default_factory=size0)
    slice_size: NnSize3D = field(default_factory=size0)


@dataclass
class NnRopeSlice:
    q_dim0: int = 0
    q_dim_start: int = 0
    q_dim_end: int = 0
    q_shift: int = 0
    kv_dim: int = 0
    kv_dim0: int = 0
    kv_dim_start: int = 0
    slice_dim: int = 0
    seq_len: int = 0
    head_dim: int = 0
    n_kv_heads: int = 0
    rope_theta: float = 10000.0
    cache_size: NnSize3D = field(default_factory=size0)


@dataclass
class NnMultiHeadAttSlice:
    n_heads: int = 0
    n_heads0: int = 0
    att_size: NnSize3D = field(default_factory=size0)


# --- Op-specific configs ---
@dataclass
class NnEmbeddingOpConfig:
    pass


@dataclass
class NnInvRmsOpConfig:
    epsilon: float = 1e-5
    n_columns: int = 1


@dataclass
class NnRmsNormOpConfig:
    inv_rms_buffer_index: int = 0
    n_columns: int = 1


@dataclass
class NnMatmulOpConfig:
    n_experts: int = 0
    n_active_experts: int = 0
    active_expert_indexes_buffer_index: int = 0


@dataclass
class NnRopeOpConfig:
    rope_type: int = NnRopeType.ROPE_LLAMA
    is_q: int = 1
    position_pipe_index: int = 0
    rope_cache_buffer_index: int = 0
    rope_scaling_factor: float = 1.0
    rope_scaling_low_freq_factor: float = 1.0
    rope_scaling_high_freq_factor: float = 1.0
    rope_scaling_orig_max_seq_len: int = 0
    slice: Optional[NnRopeSlice] = None


@dataclass
class NnMultiHeadAttOpConfig:
    n_heads: int = 0
    n_heads0: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    seq_len: int = 0
    q_slice_d0: int = 0
    kv_dim0: int = 0
    position_pipe_index: int = 0
    query_buffer_index: int = 0
    key_cache_buffer_index: int = 0
    value_cache_buffer_index: int = 0
    att_buffer_index: int = 0


@dataclass
class NnMergeAddOpCodeConfig:
    pass


@dataclass
class NnMergeSumOpCodeConfig:
    pass


@dataclass
class NnSiluOpCodeConfig:
    pass


@dataclass
class NnMulOpCodeConfig:
    multiplier_buffer_index: int = 0


@dataclass
class NnScaleOpCodeConfig:
    scale_buffer_index: int = 0


@dataclass
class NnCastOpCodeConfig:
    pass


@dataclass
class NnRepeatZOpCodeConfig:
    pass


@dataclass
class NnShiftOpCodeConfig:
    index_pipe_index: int = 0


@dataclass
class NnSoftmaxOpCodeConfig:
    pass


@dataclass
class NnMoeGateOpCodeConfig:
    k: int = 0
    norm_topk: int = 1
    indexes_buffer_index: int = 0


# --- LlmNet (complete graph) ---
@dataclass
class LlmNet:
    header: LlmHeader = field(default_factory=LlmHeader)
    net_config: NnNetConfig = field(default_factory=NnNetConfig)
    node_configs: List[NnNodeConfig] = field(default_factory=list)

    # Slices for weight distribution
    q_slice: NnRowMatmulSlice = field(default_factory=NnRowMatmulSlice)
    k_slice: NnRowMatmulSlice = field(default_factory=NnRowMatmulSlice)
    v_slice: NnRowMatmulSlice = field(default_factory=NnRowMatmulSlice)
    wo_slice: NnColMatmulSlice = field(default_factory=NnColMatmulSlice)
    w1_slice: NnRowMatmulSlice = field(default_factory=NnRowMatmulSlice)
    w2_slice: NnColMatmulSlice = field(default_factory=NnColMatmulSlice)
    w3_slice: NnRowMatmulSlice = field(default_factory=NnRowMatmulSlice)
    wcls_slice: NnRowMatmulSlice = field(default_factory=NnRowMatmulSlice)

    # Pipe indices
    position_pipe_index: int = 0
    token_pipe_index: int = 0
    x_pipe_index: int = 0
    logits_pipe_index: int = 0

    # Key sizes
    token_embedding_size: NnSize3D = field(default_factory=size0)
    rms_norm_size: NnSize3D = field(default_factory=size0)
    qk_rms_norm_size: NnSize3D = field(default_factory=size0)
    moe_gate_size: NnSize3D = field(default_factory=size0)


# --- Slicing functions ---
def slice_kv_cache(kv_dim: int, seq_len: int, n_nodes: int) -> NnKvCacheSlice:
    kv_dim0 = kv_dim // n_nodes
    s = NnKvCacheSlice()
    s.kv_dim0 = kv_dim0
    s.key_size = size2d(F_32, seq_len, kv_dim0)
    s.value_size = size2d(F_32, seq_len, kv_dim0)
    return s


def slice_row_matmul(ftype: int, n_nodes: int, n: int, d: int) -> NnRowMatmulSlice:
    d0 = d // n_nodes
    s = NnRowMatmulSlice()
    s.type = ftype
    s.n_nodes = n_nodes
    s.d0 = d0
    s.n = n
    s.size = size2d(ftype, n, d)
    s.slice_size = size2d(ftype, n, d0)
    return s


def slice_col_matmul(ftype: int, n_nodes: int, n: int, d: int) -> NnColMatmulSlice:
    n0 = n // n_nodes
    s = NnColMatmulSlice()
    s.type = ftype
    s.n_nodes = n_nodes
    s.n = n
    s.n0 = n0
    s.d = d
    s.size = size2d(ftype, n, d)
    s.slice_size = size2d(ftype, n0, d)
    return s


def slice_rope(
    rope_type: int, q_dim: int, kv_dim: int,
    n_kv_heads: int, n_nodes: int, seq_len: int,
    head_dim: int, rope_theta: float, node_index: int,
) -> NnRopeSlice:
    s = NnRopeSlice()
    s.head_dim = head_dim
    s.seq_len = seq_len
    s.n_kv_heads = n_kv_heads
    s.rope_theta = rope_theta

    q_dim_per_node = q_dim // n_nodes
    kv_dim_per_node = kv_dim // n_nodes

    s.q_dim0 = q_dim_per_node
    s.kv_dim0 = kv_dim_per_node

    if rope_type in (NnRopeType.ROPE_LLAMA, NnRopeType.ROPE_LLAMA3_1):
        s.slice_dim = q_dim // n_kv_heads
        s.q_dim_start = node_index * q_dim_per_node
        s.q_dim_end = s.q_dim_start + q_dim_per_node
        s.q_shift = s.q_dim_start
        s.kv_dim = kv_dim
        s.kv_dim_start = node_index * kv_dim_per_node
        s.cache_size = size2d(F_32, seq_len, s.q_dim_end)
    elif rope_type == NnRopeType.ROPE_FALCON:
        s.slice_dim = head_dim
        s.q_dim_start = node_index * q_dim_per_node
        s.q_dim_end = s.q_dim_start + q_dim_per_node
        s.q_shift = 0
        s.kv_dim = kv_dim
        s.kv_dim_start = node_index * kv_dim_per_node
        s.cache_size = size2d(F_32, seq_len, head_dim)
    return s


def slice_multi_head_att(
    n_heads: int, seq_len: int,
    n_nodes: int, n_batches: int,
) -> NnMultiHeadAttSlice:
    s = NnMultiHeadAttSlice()
    s.n_heads = n_heads
    s.n_heads0 = n_heads // n_nodes
    s.att_size = size2d(F_32, n_batches, s.n_heads0 * seq_len)
    return s


# --- Helper builders ---
class NnSegmentConfigBuilder:
    def __init__(self):
        self._ops: List[NnOpConfig] = []
        self._syncs: List[NnSyncConfig] = []

    def add_op(self, code: int, name: str, index: int,
               input_cfg: NnPointerConfig, output_cfg: NnPointerConfig,
               weight_size: NnSize3D, op_config: object):
        self._ops.append(NnOpConfig(
            code=code, name=name, index=index,
            input=input_cfg, output=output_cfg,
            weight_size=weight_size, config=op_config,
        ))

    def add_sync(self, pipe_idx: int, sync_type: int):
        self._syncs.append(NnSyncConfig(pipe_index=pipe_idx, sync_type=sync_type))

    def build(self) -> NnSegmentConfig:
        return NnSegmentConfig(ops=list(self._ops), syncs=list(self._syncs))


class NnNetConfigBuilder:
    def __init__(self, n_nodes: int, n_batches: int):
        self._n_batches = n_batches
        self._n_nodes = n_nodes
        self._pipes: List[NnPipeConfig] = []
        self._pre_syncs: List[NnPreSyncConfig] = []

    def add_pipe(self, name: str, size: NnSize3D) -> int:
        idx = len(self._pipes)
        self._pipes.append(NnPipeConfig(name=name, size=size))
        return idx

    def add_pre_sync(self, pipe_idx: int):
        self._pre_syncs.append(NnPreSyncConfig(pipe_index=pipe_idx))

    def build(self) -> NnNetConfig:
        return NnNetConfig(
            n_batches=self._n_batches,
            n_nodes=self._n_nodes,
            pipes=list(self._pipes),
            pre_syncs=list(self._pre_syncs),
        )


class NnNodeConfigBuilder:
    def __init__(self, node_index: int):
        self._node_index = node_index
        self._buffers: List[NnBufferConfig] = []
        self._segments: List[NnSegmentConfig] = []

    def add_buffer(self, name: str, size: NnSize3D) -> int:
        idx = len(self._buffers)
        self._buffers.append(NnBufferConfig(name=name, size=size))
        return idx

    def add_segment(self, seg: NnSegmentConfig):
        self._segments.append(seg)

    def build(self) -> NnNodeConfig:
        return NnNodeConfig(
            node_index=self._node_index,
            buffers=list(self._buffers),
            segments=list(self._segments),
        )


# --- Main graph builder ---
def build_llm_net(
    header: LlmHeader,
    n_nodes: int,
    n_batches: int,
) -> LlmNet:
    """Build the complete computation graph for the model."""

    n_experts_or1 = max(header.n_experts, 1)
    n_active_experts_or1 = max(header.n_active_experts, 1)
    ff_dim = header.ff_dim

    net = LlmNet()
    net.header = header

    net.token_embedding_size = size2d(F_32, header.vocab_size, header.dim)
    net.rms_norm_size = size1d(F_32, header.dim)
    net.qk_rms_norm_size = size1d(F_32, header.head_dim)
    net.moe_gate_size = size2d(F_32, header.dim, header.n_experts)

    kv_cache_slice = slice_kv_cache(header.kv_dim, header.seq_len, n_nodes)
    multi_head_att_slice = slice_multi_head_att(
        header.n_heads, header.seq_len, n_nodes, n_batches,
    )

    net.q_slice = slice_row_matmul(header.weight_type, n_nodes, header.dim, header.q_dim)
    net.k_slice = slice_row_matmul(header.weight_type, n_nodes, header.dim, header.kv_dim)
    net.v_slice = slice_row_matmul(header.weight_type, n_nodes, header.dim, header.kv_dim)
    net.wo_slice = slice_col_matmul(header.weight_type, n_nodes, header.q_dim, header.dim)
    net.w1_slice = slice_row_matmul(header.weight_type, n_nodes, header.dim, ff_dim)
    net.w2_slice = slice_col_matmul(header.weight_type, n_nodes, ff_dim, header.dim)
    net.w3_slice = slice_row_matmul(header.weight_type, n_nodes, header.dim, ff_dim)
    net.wcls_slice = slice_row_matmul(header.weight_type, n_nodes, header.dim, header.vocab_size)

    n_q_norm_cols = 1
    n_k_norm_cols = 1
    n_inv_buffer_cols = 1
    if header.arch_type in (LlmArchType.QWEN3, LlmArchType.QWEN3_MOE):
        assert net.q_slice.d0 % header.head_dim == 0
        assert net.k_slice.d0 % header.head_dim == 0
        n_q_norm_cols = net.q_slice.d0 // header.head_dim
        n_k_norm_cols = net.k_slice.d0 // header.head_dim
        n_inv_buffer_cols = max(n_q_norm_cols, n_k_norm_cols)

    net_builder = NnNetConfigBuilder(n_nodes, n_batches)

    net.position_pipe_index = net_builder.add_pipe("POS", size2d(F_32, n_batches, 1))
    net.token_pipe_index = net_builder.add_pipe("TOK", size2d(F_32, n_batches, 1))
    net.x_pipe_index = net_builder.add_pipe("X", size2d(F_32, n_batches, header.dim))
    net.logits_pipe_index = net_builder.add_pipe("LG", size2d(F_32, n_batches, header.vocab_size))
    zq_pipe_index = net_builder.add_pipe("ZQ", size2d(header.sync_type, n_batches, header.dim * n_nodes))

    net_builder.add_pre_sync(net.position_pipe_index)

    net.net_config = net_builder.build()
    net.node_configs = []

    for node_index in range(n_nodes):
        rope_slice = slice_rope(
            header.rope_type, header.q_dim, header.kv_dim,
            header.n_kv_heads, n_nodes, header.seq_len,
            header.head_dim, header.rope_theta, node_index,
        )
        node_builder = NnNodeConfigBuilder(node_index)

        x_buf_idx = node_builder.add_buffer("x", size2d(F_32, n_batches, header.dim))
        y_buf_idx = node_builder.add_buffer("y", size2d(F_32, n_batches, header.dim))
        yq_buf_idx = (
            y_buf_idx if header.sync_type == F_32
            else node_builder.add_buffer("q_y", size2d(header.sync_type, n_batches, header.dim))
        )

        z_buf_idx = node_builder.add_buffer("z", size2d(F_32, n_batches, header.q_dim))
        zq_slice_buf_idx = node_builder.add_buffer(
            "q_z_slice", size2d(header.sync_type, n_batches, header.q_dim // n_nodes),
        )

        q_buf_idx = node_builder.add_buffer("q", size2d(F_32, n_batches, net.q_slice.d0))
        k_temp_buf_idx = node_builder.add_buffer("k_temp", size2d(F_32, n_batches, net.k_slice.d0))
        v_temp_buf_idx = node_builder.add_buffer("v_temp", size2d(F_32, n_batches, net.v_slice.d0))

        inv_rms_buf_idx = node_builder.add_buffer("inv_rms", size2d(F_32, n_batches, n_inv_buffer_cols))

        rope_cache_buf_idx = node_builder.add_buffer("rope_cache", rope_slice.cache_size)
        att_buf_idx = node_builder.add_buffer("att", multi_head_att_slice.att_size)
        logits_slice_buf_idx = node_builder.add_buffer(
            "lg", size2d(F_32, n_batches, header.vocab_size // n_nodes),
        )

        d_buf_idx = node_builder.add_buffer("d", size2d(F_32, n_batches, net.w1_slice.d0))
        dq_buf_idx = (
            d_buf_idx if header.sync_type == F_32
            else node_builder.add_buffer("q_d", size2d(header.sync_type, n_batches, net.w1_slice.d0))
        )
        l_buf_idx = node_builder.add_buffer("l", size2d(F_32, n_batches, net.w3_slice.d0))

        # MoE buffers
        moe_gt_buf_idx = node_builder.add_buffer("gt", size2d(F_32, n_batches, n_experts_or1))
        moe_expert_idx_buf_idx = node_builder.add_buffer(
            "act_exp_ix", size2d(F_32, n_batches, n_active_experts_or1),
        )
        moe_y_buf_idx = node_builder.add_buffer(
            "moe_y", size3d(F_32, n_active_experts_or1, n_batches, header.dim),
        )
        moe_yq_buf_idx = (
            moe_y_buf_idx if header.sync_type == F_32
            else node_builder.add_buffer(
                "q_moe_y", size3d(header.sync_type, n_active_experts_or1, n_batches, header.dim),
            )
        )
        moe_d_buf_idx = node_builder.add_buffer(
            "moe_d", size3d(F_32, n_active_experts_or1, n_batches, net.w1_slice.d0),
        )
        moe_dq_buf_idx = (
            moe_d_buf_idx if header.sync_type == F_32
            else node_builder.add_buffer(
                "q_moe_d", size3d(header.sync_type, n_active_experts_or1, n_batches, net.w1_slice.d0),
            )
        )
        moe_l_buf_idx = node_builder.add_buffer(
            "moe_l", size3d(F_32, n_active_experts_or1, n_batches, net.w3_slice.d0),
        )
        moe_s_buf_idx = node_builder.add_buffer(
            "moe_s", size3d(F_32, n_active_experts_or1, n_batches, 1),
        )

        # START segment
        start_seg = NnSegmentConfigBuilder()
        if node_index == 0:
            start_seg.add_op(
                NnOpCode.EMBEDDING, "embedding", 0,
                pointer_batch_config(NnPointerSource.SRC_PIPE, net.token_pipe_index),
                pointer_batch_config(NnPointerSource.SRC_PIPE, net.x_pipe_index),
                net.token_embedding_size,
                NnEmbeddingOpConfig(),
            )
        start_seg.add_sync(net.x_pipe_index, NnSyncType.SYNC_WITH_ROOT)
        node_builder.add_segment(start_seg.build())

        # PER-LAYER segments
        for layer_idx in range(header.n_layers):
            k_buf_idx = node_builder.add_buffer("k", kv_cache_slice.key_size)
            v_buf_idx = node_builder.add_buffer("v", kv_cache_slice.value_size)

            att_seg = NnSegmentConfigBuilder()
            ff_seg = NnSegmentConfigBuilder()

            # -- Attention --
            if layer_idx == 0:
                att_seg.add_op(
                    NnOpCode.CAST, "block_cast_x", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_PIPE, net.x_pipe_index),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, x_buf_idx),
                    size0(), NnCastOpCodeConfig(),
                )
            else:
                att_seg.add_op(
                    NnOpCode.MERGE_ADD, "block_merge_add", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_PIPE, zq_pipe_index),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, x_buf_idx),
                    size0(), NnMergeAddOpCodeConfig(),
                )

            att_seg.add_op(
                NnOpCode.INV_RMS, "block_norm_pre_0", layer_idx,
                pointer_batch_config(NnPointerSource.SRC_BUFFER, x_buf_idx),
                pointer_batch_config(NnPointerSource.SRC_BUFFER, inv_rms_buf_idx),
                size0(),
                NnInvRmsOpConfig(header.norm_epsilon, 1),
            )
            att_seg.add_op(
                NnOpCode.RMS_NORM, "block_norm_0", layer_idx,
                pointer_batch_config(NnPointerSource.SRC_BUFFER, x_buf_idx),
                pointer_batch_config(NnPointerSource.SRC_BUFFER, y_buf_idx),
                net.rms_norm_size,
                NnRmsNormOpConfig(inv_rms_buf_idx, 1),
            )
            if y_buf_idx != yq_buf_idx:
                att_seg.add_op(
                    NnOpCode.CAST, "block_cast_y", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, y_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, yq_buf_idx),
                    size0(), NnCastOpCodeConfig(),
                )

            # Q, K, V projections
            att_seg.add_op(
                NnOpCode.MATMUL, "block_matmul_q", layer_idx,
                pointer_batch_config(NnPointerSource.SRC_BUFFER, yq_buf_idx),
                pointer_batch_config(NnPointerSource.SRC_BUFFER, q_buf_idx),
                size2d(header.weight_type, net.q_slice.n, net.q_slice.d0),
                NnMatmulOpConfig(0, 0, moe_expert_idx_buf_idx),
            )
            att_seg.add_op(
                NnOpCode.MATMUL, "block_matmul_k", layer_idx,
                pointer_batch_config(NnPointerSource.SRC_BUFFER, yq_buf_idx),
                pointer_batch_config(NnPointerSource.SRC_BUFFER, k_temp_buf_idx),
                size2d(header.weight_type, net.k_slice.n, net.k_slice.d0),
                NnMatmulOpConfig(0, 0, moe_expert_idx_buf_idx),
            )
            att_seg.add_op(
                NnOpCode.MATMUL, "block_matmul_v", layer_idx,
                pointer_batch_config(NnPointerSource.SRC_BUFFER, yq_buf_idx),
                pointer_batch_config(NnPointerSource.SRC_BUFFER, v_temp_buf_idx),
                size2d(header.weight_type, net.v_slice.n, net.v_slice.d0),
                NnMatmulOpConfig(0, 0, moe_expert_idx_buf_idx),
            )

            # Qwen3 QK norm
            if header.arch_type in (LlmArchType.QWEN3, LlmArchType.QWEN3_MOE):
                att_seg.add_op(
                    NnOpCode.INV_RMS, "block_norm_pre_q", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, q_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, inv_rms_buf_idx),
                    size0(),
                    NnInvRmsOpConfig(header.norm_epsilon, n_q_norm_cols),
                )
                att_seg.add_op(
                    NnOpCode.RMS_NORM, "block_norm_q", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, q_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, q_buf_idx),
                    size2d(F_32, 1, header.head_dim),
                    NnRmsNormOpConfig(inv_rms_buf_idx, n_q_norm_cols),
                )
                att_seg.add_op(
                    NnOpCode.INV_RMS, "block_norm_pre_k", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, k_temp_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, inv_rms_buf_idx),
                    size0(),
                    NnInvRmsOpConfig(header.norm_epsilon, n_k_norm_cols),
                )
                att_seg.add_op(
                    NnOpCode.RMS_NORM, "block_norm_k", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, k_temp_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, k_temp_buf_idx),
                    size2d(F_32, 1, header.head_dim),
                    NnRmsNormOpConfig(inv_rms_buf_idx, n_k_norm_cols),
                )

            # RoPE
            att_seg.add_op(
                NnOpCode.ROPE, "block_rope_q", layer_idx,
                pointer_batch_config(NnPointerSource.SRC_BUFFER, q_buf_idx),
                pointer_batch_config(NnPointerSource.SRC_BUFFER, q_buf_idx),
                size0(),
                NnRopeOpConfig(
                    header.rope_type, 1, net.position_pipe_index,
                    rope_cache_buf_idx,
                    header.rope_scaling_factor,
                    header.rope_scaling_low_freq_factor,
                    header.rope_scaling_high_freq_factor,
                    header.rope_scaling_orig_max_seq_len,
                    rope_slice,
                ),
            )
            att_seg.add_op(
                NnOpCode.ROPE, "block_rope_k", layer_idx,
                pointer_batch_config(NnPointerSource.SRC_BUFFER, k_temp_buf_idx),
                pointer_batch_config(NnPointerSource.SRC_BUFFER, k_temp_buf_idx),
                size0(),
                NnRopeOpConfig(
                    header.rope_type, 0, net.position_pipe_index,
                    rope_cache_buf_idx,
                    header.rope_scaling_factor,
                    header.rope_scaling_low_freq_factor,
                    header.rope_scaling_high_freq_factor,
                    header.rope_scaling_orig_max_seq_len,
                    rope_slice,
                ),
            )

            # Shift into KV cache
            att_seg.add_op(
                NnOpCode.SHIFT, "block_shift_k", layer_idx,
                pointer_batch_config(NnPointerSource.SRC_BUFFER, k_temp_buf_idx),
                pointer_raw_config(NnPointerSource.SRC_BUFFER, k_buf_idx),
                size0(), NnShiftOpCodeConfig(net.position_pipe_index),
            )
            att_seg.add_op(
                NnOpCode.SHIFT, "block_shift_v", layer_idx,
                pointer_batch_config(NnPointerSource.SRC_BUFFER, v_temp_buf_idx),
                pointer_raw_config(NnPointerSource.SRC_BUFFER, v_buf_idx),
                size0(), NnShiftOpCodeConfig(net.position_pipe_index),
            )

            # Multi-head attention
            att_seg.add_op(
                NnOpCode.MULTIHEAD_ATT, "block_multihead_att", layer_idx,
                pointer_batched_slice_config(NnPointerSource.SRC_BUFFER, z_buf_idx),
                pointer_batched_slice_config(NnPointerSource.SRC_BUFFER, z_buf_idx),
                size0(),
                NnMultiHeadAttOpConfig(
                    multi_head_att_slice.n_heads,
                    multi_head_att_slice.n_heads0,
                    header.n_kv_heads, header.head_dim, header.seq_len,
                    net.q_slice.d0, kv_cache_slice.kv_dim0,
                    net.position_pipe_index, q_buf_idx,
                    k_buf_idx, v_buf_idx, att_buf_idx,
                ),
            )

            # Output projection
            att_seg.add_op(
                NnOpCode.CAST, "block_cast_y2", layer_idx,
                pointer_batched_slice_config(NnPointerSource.SRC_BUFFER, z_buf_idx),
                pointer_batch_config(NnPointerSource.SRC_BUFFER, zq_slice_buf_idx),
                size0(), NnCastOpCodeConfig(),
            )
            att_seg.add_op(
                NnOpCode.MATMUL, "block_matmul_wo", layer_idx,
                pointer_batch_config(NnPointerSource.SRC_BUFFER, zq_slice_buf_idx),
                pointer_batch_config(NnPointerSource.SRC_BUFFER, y_buf_idx),
                size2d(header.weight_type, net.wo_slice.n0, net.wo_slice.d),
                NnMatmulOpConfig(0, 0, moe_expert_idx_buf_idx),
            )
            att_seg.add_op(
                NnOpCode.CAST, "block_cast_d", layer_idx,
                pointer_batch_config(NnPointerSource.SRC_BUFFER, y_buf_idx),
                pointer_batched_slice_config(NnPointerSource.SRC_PIPE, zq_pipe_index),
                size0(), NnCastOpCodeConfig(),
            )
            att_seg.add_sync(zq_pipe_index, NnSyncType.SYNC_NODE_SLICES)

            # -- FFN --
            ff_seg.add_op(
                NnOpCode.MERGE_ADD, "block_merge_add2", layer_idx,
                pointer_batch_config(NnPointerSource.SRC_PIPE, zq_pipe_index),
                pointer_batch_config(NnPointerSource.SRC_BUFFER, x_buf_idx),
                size0(), NnMergeAddOpCodeConfig(),
            )
            ff_seg.add_op(
                NnOpCode.INV_RMS, "block_norm_pre_1", layer_idx,
                pointer_batch_config(NnPointerSource.SRC_BUFFER, x_buf_idx),
                pointer_batch_config(NnPointerSource.SRC_BUFFER, inv_rms_buf_idx),
                size0(),
                NnInvRmsOpConfig(header.norm_epsilon, 1),
            )
            ff_seg.add_op(
                NnOpCode.RMS_NORM, "block_norm_1", layer_idx,
                pointer_batch_config(NnPointerSource.SRC_BUFFER, x_buf_idx),
                pointer_batch_config(NnPointerSource.SRC_BUFFER, y_buf_idx),
                net.rms_norm_size,
                NnRmsNormOpConfig(inv_rms_buf_idx, 1),
            )

            if header.arch_type == LlmArchType.QWEN3_MOE:
                # MoE FFN
                ff_seg.add_op(
                    NnOpCode.REPEAT_Z, "block_moe_y_repeat", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, y_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_yq_buf_idx),
                    size0(), NnRepeatZOpCodeConfig(),
                )
                ff_seg.add_op(
                    NnOpCode.MATMUL, "block_moe_gate", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, y_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_gt_buf_idx),
                    net.moe_gate_size,
                    NnMatmulOpConfig(0, 0, moe_expert_idx_buf_idx),
                )
                ff_seg.add_op(
                    NnOpCode.SOFTMAX, "block_moe_softmax", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_gt_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_gt_buf_idx),
                    size0(), NnSoftmaxOpCodeConfig(),
                )
                ff_seg.add_op(
                    NnOpCode.MOE_GATE, "block_moe_gate2", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_gt_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_s_buf_idx),
                    size0(),
                    NnMoeGateOpCodeConfig(header.n_active_experts, 1, moe_expert_idx_buf_idx),
                )
                ff_seg.add_op(
                    NnOpCode.MATMUL, "block_matmul_w1", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_yq_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_d_buf_idx),
                    size3d(header.weight_type, header.n_experts, net.w1_slice.n, net.w1_slice.d0),
                    NnMatmulOpConfig(header.n_experts, header.n_active_experts, moe_expert_idx_buf_idx),
                )
                ff_seg.add_op(
                    NnOpCode.MATMUL, "block_matmul_w3", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_yq_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_l_buf_idx),
                    size3d(header.weight_type, header.n_experts, net.w3_slice.n, net.w3_slice.d0),
                    NnMatmulOpConfig(header.n_experts, header.n_active_experts, moe_expert_idx_buf_idx),
                )
                ff_seg.add_op(
                    NnOpCode.SILU, "block_act", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_d_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_d_buf_idx),
                    size0(), NnSiluOpCodeConfig(),
                )
                ff_seg.add_op(
                    NnOpCode.MUL, "block_mul", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_d_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_d_buf_idx),
                    size0(), NnMulOpCodeConfig(moe_l_buf_idx),
                )
                if moe_d_buf_idx != moe_dq_buf_idx:
                    ff_seg.add_op(
                        NnOpCode.CAST, "block_cast_d2", layer_idx,
                        pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_d_buf_idx),
                        pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_dq_buf_idx),
                        size0(), NnCastOpCodeConfig(),
                    )
                ff_seg.add_op(
                    NnOpCode.MATMUL, "block_matmul_w2", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_dq_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_y_buf_idx),
                    size3d(header.weight_type, header.n_experts, net.w2_slice.n0, net.w2_slice.d),
                    NnMatmulOpConfig(header.n_experts, header.n_active_experts, moe_expert_idx_buf_idx),
                )
                ff_seg.add_op(
                    NnOpCode.SCALE, "block_moe_scale", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_y_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_y_buf_idx),
                    size0(), NnScaleOpCodeConfig(moe_s_buf_idx),
                )
                ff_seg.add_op(
                    NnOpCode.MERGE_SUM, "block_moe_merge_sum", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, moe_y_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, y_buf_idx),
                    size0(), NnMergeSumOpCodeConfig(),
                )
            else:
                # Standard FFN
                if y_buf_idx != yq_buf_idx:
                    ff_seg.add_op(
                        NnOpCode.CAST, "block_cast_y3", layer_idx,
                        pointer_batch_config(NnPointerSource.SRC_BUFFER, y_buf_idx),
                        pointer_batch_config(NnPointerSource.SRC_BUFFER, yq_buf_idx),
                        size0(), NnCastOpCodeConfig(),
                    )
                ff_seg.add_op(
                    NnOpCode.MATMUL, "block_matmul_w1", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, yq_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, d_buf_idx),
                    size2d(header.weight_type, net.w1_slice.n, net.w1_slice.d0),
                    NnMatmulOpConfig(0, 0, moe_expert_idx_buf_idx),
                )
                ff_seg.add_op(
                    NnOpCode.MATMUL, "block_matmul_w3", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, yq_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, l_buf_idx),
                    size2d(header.weight_type, net.w3_slice.n, net.w3_slice.d0),
                    NnMatmulOpConfig(0, 0, moe_expert_idx_buf_idx),
                )
                ff_seg.add_op(
                    NnOpCode.SILU, "block_act", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, d_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, d_buf_idx),
                    size0(), NnSiluOpCodeConfig(),
                )
                ff_seg.add_op(
                    NnOpCode.MUL, "block_mul", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, d_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, d_buf_idx),
                    size0(), NnMulOpCodeConfig(l_buf_idx),
                )
                if d_buf_idx != dq_buf_idx:
                    ff_seg.add_op(
                        NnOpCode.CAST, "block_cast_d2", layer_idx,
                        pointer_batch_config(NnPointerSource.SRC_BUFFER, d_buf_idx),
                        pointer_batch_config(NnPointerSource.SRC_BUFFER, dq_buf_idx),
                        size0(), NnCastOpCodeConfig(),
                    )
                ff_seg.add_op(
                    NnOpCode.MATMUL, "block_matmul_w2", layer_idx,
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, dq_buf_idx),
                    pointer_batch_config(NnPointerSource.SRC_BUFFER, y_buf_idx),
                    size2d(header.weight_type, net.w2_slice.n0, net.w2_slice.d),
                    NnMatmulOpConfig(0, 0, moe_expert_idx_buf_idx),
                )

            ff_seg.add_op(
                NnOpCode.CAST, "block_cast_d3", layer_idx,
                pointer_batch_config(NnPointerSource.SRC_BUFFER, y_buf_idx),
                pointer_batched_slice_config(NnPointerSource.SRC_PIPE, zq_pipe_index),
                size0(), NnCastOpCodeConfig(),
            )
            ff_seg.add_sync(zq_pipe_index, NnSyncType.SYNC_NODE_SLICES)

            node_builder.add_segment(att_seg.build())
            node_builder.add_segment(ff_seg.build())

        # END segment
        end_seg = NnSegmentConfigBuilder()
        end_seg.add_op(
            NnOpCode.MERGE_ADD, "final_merge_add", 0,
            pointer_batch_config(NnPointerSource.SRC_PIPE, zq_pipe_index),
            pointer_batch_config(NnPointerSource.SRC_BUFFER, x_buf_idx),
            size0(), NnMergeAddOpCodeConfig(),
        )
        end_seg.add_op(
            NnOpCode.INV_RMS, "final_norm_pre", 0,
            pointer_batch_config(NnPointerSource.SRC_BUFFER, x_buf_idx),
            pointer_batch_config(NnPointerSource.SRC_BUFFER, inv_rms_buf_idx),
            size0(),
            NnInvRmsOpConfig(header.norm_epsilon, 1),
        )
        end_seg.add_op(
            NnOpCode.RMS_NORM, "final_norm", 0,
            pointer_batch_config(NnPointerSource.SRC_BUFFER, x_buf_idx),
            pointer_batch_config(NnPointerSource.SRC_BUFFER, y_buf_idx),
            net.rms_norm_size,
            NnRmsNormOpConfig(inv_rms_buf_idx, 1),
        )
        if y_buf_idx != yq_buf_idx:
            end_seg.add_op(
                NnOpCode.CAST, "final_cast_y", 0,
                pointer_batch_config(NnPointerSource.SRC_BUFFER, y_buf_idx),
                pointer_batch_config(NnPointerSource.SRC_BUFFER, yq_buf_idx),
                size0(), NnCastOpCodeConfig(),
            )
        end_seg.add_op(
            NnOpCode.MATMUL, "final_matmul_logits", 0,
            pointer_batch_config(NnPointerSource.SRC_BUFFER, yq_buf_idx),
            pointer_batch_config(NnPointerSource.SRC_BUFFER, logits_slice_buf_idx),
            size2d(header.weight_type, net.wcls_slice.n, net.wcls_slice.d0),
            NnMatmulOpConfig(),
        )
        end_seg.add_op(
            NnOpCode.CAST, "final_cast_logits", 0,
            pointer_batch_config(NnPointerSource.SRC_BUFFER, logits_slice_buf_idx),
            pointer_batched_slice_config(NnPointerSource.SRC_PIPE, net.logits_pipe_index),
            size0(), NnCastOpCodeConfig(),
        )
        end_seg.add_sync(net.logits_pipe_index, NnSyncType.SYNC_NODE_SLICES_EXCEPT_ROOT)

        node_builder.add_segment(end_seg.build())
        net.node_configs.append(node_builder.build())

    return net


def print_node_required_memory(net_config: NnNetConfig, node_config: NnNodeConfig) -> int:
    """Estimate total required memory (bytes) for a node's pipes, buffers, and weights.

    Port of C++ ``printNodeRequiredMemory``.
    """
    total = 0

    # Pipes
    for pipe in net_config.pipes:
        total += pipe.size.n_bytes

    # Buffers
    for buf in node_config.buffers:
        total += buf.size.n_bytes

    # Weights
    for segment in node_config.segments:
        for op in segment.ops:
            total += op.weight_size.n_bytes

    mb = total / (1024 * 1024)
    print(f"  Node #{node_config.node_index} required memory: {total} B ({mb:.1f} MB)")
    return total


def release_llm_net(net: LlmNet):
    """Clean up LlmNet resources."""
    net.node_configs.clear()
    net.net_config = NnNetConfig()
