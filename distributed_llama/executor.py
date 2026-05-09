"""Multithreaded graph execution engine for distributed LLM inference.

Port of src/nn/nn-executor.cpp and src/nn/nn-cpu.cpp from the reference repo.

Provides the execution runtime: pipe/buffer allocation, op compilation and
forward dispatch, step scheduling with ThreadPoolExecutor, and inter-node
synchronization abstraction.

All ops have pure-Python/NumPy fallback implementations so the engine works
even without the C extension.
"""

import math
import struct
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np

from .graph_builder import (
    NnNetConfig, NnNodeConfig, NnSegmentConfig, NnOpConfig, NnSyncConfig,
    NnPipeConfig, NnBufferConfig, NnPointerConfig, NnSize3D,
    NnPointerSource, NnPointerType, NnSyncType,
    has_pointer_continuous_memory,
)
from .model import NnOpCode, NnOpQuantType, NnRopeType
from .quants import (
    F_32, F_16, F_Q40, F_Q80, get_bytes, get_block_size,
    Q40_BLOCK_SIZE, Q80_BLOCK_SIZE,
)

# ======================================================================
# C extension (optional, for accelerated kernels)
# ======================================================================

# C extension (Python C-API bindings in ops/_ops.cpython-*.so).
# Individual ops can be enabled/disabled.
# Simple element-wise ops are verified correct (diff=0 vs NumPy).
# matmul/rope/attention/embedding have precision differences and are disabled.
_ops_c = None  # sentinel: True if ANY C ops enabled

# Individual C op availability flags
_C_SILU = False
_C_GELU = False
_C_ADD = False
_C_MUL = False
_C_SOFTMAX = False
_C_MATMUL_F32 = False
_C_MATMUL_Q80Q40 = False
_C_ROPE_LLAMA = False
_C_ROPE_FALCON = False
_C_MULTIHEAD_ATT = False

try:
    from .ops._ops import (
        init_mt         as _c_init_mt,
        op_softmax_f32 as _c_softmax,
        op_silu          as _c_silu,
        op_gelu          as _c_gelu,
        op_mul_f32       as _c_mul,
        op_add_f32       as _c_add,
        op_matmul_f32_f32_f32   as _c_matmul_f32,
        op_matmul_q80_q40_f32   as _c_matmul_q80q40,
        op_rope_llama    as _c_rope_llama,
        op_rope_falcon   as _c_rope_falcon,
        op_multihead_att as _c_multihead_att,
    )
    _C_ROPE_LLAMA = True
    _C_ROPE_FALCON = True
    _C_MULTIHEAD_ATT = True
    _C_SILU = True
    _C_GELU = True
    _C_ADD = True
    _C_MUL = True
    _C_SOFTMAX = True
    _C_MATMUL_F32 = False  # C F32 matmul is 6-16x slower than NumPy/BLAS
    _C_MATMUL_Q80Q40 = True
    _ops_c = True
except ImportError:
    pass


# ======================================================================
# F16 conversion helpers (IEEE 754 binary16)
# ======================================================================

_F16_LOOKUP = None


def _build_f16_lookup():
    """Build 64K-entry float32 lookup table for float16 conversion (idempotent)."""
    global _F16_LOOKUP
    if _F16_LOOKUP is not None:
        return
    table = np.zeros(65536, dtype=np.float32)
    for i in range(65536):
        sign = (i >> 15) & 1
        exp = (i >> 10) & 0x1F
        mant = i & 0x3FF
        if exp == 0:
            val = (mant / 1024.0) * (2.0 ** -14) if mant else 0.0
        elif exp == 31:
            val = float("inf") if mant == 0 else float("nan")
        else:
            val = (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))
        if sign:
            val = -val
        table[i] = val
    _F16_LOOKUP = table


def _f32_to_f16_raw(value: float) -> int:
    """Convert float32 to float16 and return as uint16 integer."""
    if math.isnan(value):
        return 0x7E00
    if math.isinf(value):
        return 0x7C00 if value > 0 else 0xFC00

    f32_bits = struct.unpack("I", struct.pack("f", value))[0]
    sign = (f32_bits >> 31) & 1
    exp = (f32_bits >> 23) & 0xFF
    mant = f32_bits & 0x7FFFFF

    if exp == 0:
        return sign << 15
    if exp == 0xFF:
        return (sign << 15) | 0x7C00 if mant == 0 else (sign << 15) | 0x7E00

    exp16 = exp - 127 + 15
    mant16 = mant >> 13
    if exp16 <= 0:
        mant16 = (mant | 0x800000) >> (14 - exp16)
        exp16 = 0
    elif exp16 >= 31:
        exp16 = 31
        mant16 = 0
    else:
        round_bit = (mant >> 12) & 1
        sticky = (mant & 0x1FFF) != 0
        if round_bit and (sticky or (mant16 & 1)):
            mant16 += 1
            if mant16 >= 1024:
                mant16 = 0
                exp16 += 1
    return (sign << 15) | ((exp16 & 0x1F) << 10) | (mant16 & 0x3FF)


# ======================================================================
# Dequantization / quantization (pure-Python NumPy)
# ======================================================================

def _dequantize_q80(data: np.ndarray, n_elems: int, out: np.ndarray,
                    n_threads: int = 1, thread_index: int = 0):
    """Dequantize Q80 blocks to float32 using fast C extension."""
    try:
        from .quants import dequantize_q80_to_f32
        q80_dtype = np.dtype([('d', np.uint16), ('qs', np.int8, (Q80_BLOCK_SIZE,))])
        n_blocks = n_elems // Q80_BLOCK_SIZE
        blocks = np.ascontiguousarray(data[:n_blocks * (2 + Q80_BLOCK_SIZE)]).view(q80_dtype).ravel()
        result = dequantize_q80_to_f32(blocks, n_elems, n_threads, thread_index)
        out[:n_elems] = result
    except ImportError:
        _build_f16_lookup()
        n_blocks = n_elems // Q80_BLOCK_SIZE
        b_start, b_end = _split_threads(n_blocks, n_threads, thread_index)
        for bi in range(b_start, b_end):
            bo = bi * (2 + Q80_BLOCK_SIZE)
            d_val = _F16_LOOKUP[int(data[bo:bo + 2].view(np.uint16)[0])]
            qs = data[bo + 2:bo + 2 + Q80_BLOCK_SIZE].view(np.int8)
            e0 = bi * Q80_BLOCK_SIZE
            out[e0:e0 + Q80_BLOCK_SIZE] = d_val * qs.astype(np.float32)


def _quantize_f32_to_q80(f32: np.ndarray, n_elems: int, out: np.ndarray,
                         n_threads: int = 1, thread_index: int = 0):
    """Quantize float32 to Q80 blocks using fast C extension, writing raw bytes into *out* (uint8)."""
    try:
        from .quants import quantize_f32_to_q80
        q80_dtype = np.dtype([('d', np.uint16), ('qs', np.int8, (Q80_BLOCK_SIZE,))])
        result = quantize_f32_to_q80(np.ascontiguousarray(f32[:n_elems]), n_elems, n_threads, thread_index)
        n_blocks = n_elems // Q80_BLOCK_SIZE
        out[:n_blocks * (2 + Q80_BLOCK_SIZE)] = result.view(np.uint8)
    except ImportError:
        n_blocks = n_elems // Q80_BLOCK_SIZE
        b_start, b_end = _split_threads(n_blocks, n_threads, thread_index)
        for bi in range(b_start, b_end):
            vals = f32[bi * Q80_BLOCK_SIZE:(bi + 1) * Q80_BLOCK_SIZE]
            max_abs = float(np.max(np.abs(vals)))
            if max_abs < 1e-8:
                max_abs = 1.0
            scale = max_abs / 127.0
            f16_raw = _f32_to_f16_raw(scale)
            qs = np.clip(np.round(vals / scale), -128, 127).astype(np.int8)
            bo = bi * (2 + Q80_BLOCK_SIZE)
            out[bo:bo + 2] = np.array([f16_raw], dtype=np.uint16).view(np.uint8)
            out[bo + 2:bo + 2 + Q80_BLOCK_SIZE] = qs.view(np.uint8)


# ======================================================================
# Thread-split helper  (matches C SPLIT_THREADS macro)
# ======================================================================

def _split_threads(range_len: int, n_threads: int, thread_index: int) -> Tuple[int, int]:
    """Return (start, end) slice for *thread_index* across *range_len* items."""
    if n_threads <= 0:
        n_threads = 1
    if thread_index >= n_threads:
        thread_index = thread_index % n_threads
    _slice = range_len // n_threads
    _rest = range_len % n_threads
    start = thread_index * _slice + (thread_index if thread_index < _rest else _rest)
    end = start + _slice + (1 if thread_index < _rest else 0)
    return start, end


# ======================================================================
# Stable softmax (in-place)
# ======================================================================

def _softmax_inplace(x: np.ndarray):
    if len(x) == 0:
        return
    mx = np.max(x)
    x[:] = np.exp(x - mx)
    s = np.sum(x)
    if s == 0.0:
        s = 1e-6
    x[:] /= s


# ======================================================================
# RoPE caches
# ======================================================================

def _scale_frequency_llama3(freq: float, config) -> float:
    """Llama 3.1 frequency scaling.  See llama-models reference."""
    wave_len = 2.0 * math.pi / freq
    hfw = config.rope_scaling_orig_max_seq_len / config.rope_scaling_high_freq_factor
    if wave_len < hfw:
        return freq
    lfw = config.rope_scaling_orig_max_seq_len / config.rope_scaling_low_freq_factor
    if wave_len > lfw:
        return freq / config.rope_scaling_factor
    smooth = ((config.rope_scaling_orig_max_seq_len / wave_len -
              config.rope_scaling_low_freq_factor) /
              (config.rope_scaling_high_freq_factor - config.rope_scaling_low_freq_factor))
    return (1.0 - smooth) * freq / config.rope_scaling_factor + smooth * freq


def _fill_rope_llama_cache(config, cache: np.ndarray):
    """Precompute cos/sin values for Llama-style RoPE."""
    slc = config.slice
    stride = slc.q_dim_end
    for pos in range(slc.seq_len):
        # K region: indices from kv_dim_start to kv_dim_start+kv_dim0
        k_end = slc.kv_dim_start + slc.kv_dim0
        for i in range(slc.kv_dim_start, k_end, 2):
            h = i % slc.head_dim
            freq = 1.0 / (slc.rope_theta ** (h / float(slc.head_dim)))
            if config.rope_scaling_factor != 1.0:
                freq = _scale_frequency_llama3(freq, config)
            val = pos * freq
            idx = pos * stride + (i - slc.kv_dim_start)
            cache[idx] = math.cos(val)
            cache[idx + 1] = math.sin(val)
        # Q region: indices from q_dim_start to q_dim_end
        for i in range(slc.q_dim_start, slc.q_dim_end, 2):
            h = i % slc.head_dim
            freq = 1.0 / (slc.rope_theta ** (h / float(slc.head_dim)))
            if config.rope_scaling_factor != 1.0:
                freq = _scale_frequency_llama3(freq, config)
            val = pos * freq
            idx = pos * stride + i
            cache[idx] = math.cos(val)
            cache[idx + 1] = math.sin(val)


def _fill_rope_falcon_cache(config, cache: np.ndarray):
    """Precompute cos/sin values for Falcon-style RoPE."""
    slc = config.slice
    for pos in range(slc.seq_len):
        for j in range(slc.head_dim // 2):
            freq = 1.0 / (slc.rope_theta ** (2.0 * (j / float(slc.head_dim))))
            val = pos * freq
            cache[pos * slc.head_dim + j] = math.cos(val)
            cache[pos * slc.head_dim + j + slc.head_dim // 2] = math.sin(val)


def _dequant_q40_via_c(w_raw: np.ndarray, d: int, n: int, n_blocks: int) -> np.ndarray:
    """Dequantize Q40 weight to F32 using the fast C extension.

    Args:
        w_raw: uint8 array of Q40 blocks (d * n_blocks * 18 bytes)
        d: output dimension (rows)
        n: input dimension (F32 elements)
        n_blocks: number of Q40 blocks per row (= n / 32)
    Returns:
        F32 weight matrix of shape (d, n)
    """
    from .quants import dequantize_q40_to_f32
    q40_dtype = np.dtype([('d', np.uint16), ('qs', np.uint8, (16,))])
    n_blocks_total = d * n_blocks
    # Ensure contiguous copy and correct size for dtype view
    w_contig = np.ascontiguousarray(w_raw)
    total_bytes = n_blocks_total * q40_dtype.itemsize
    w_contig = w_contig[:total_bytes]
    blocks = w_contig.view(q40_dtype).ravel()
    f32_flat = dequantize_q40_to_f32(blocks, d * n, n_threads=1, thread_index=0)
    return f32_flat.reshape(d, n)


def _dequant_q40_weight_np(raw: np.ndarray, d: int, n: int, n_blocks: int) -> np.ndarray:
    """Dequantize Q40 weight to F32 using numpy vectorization.

    Args:
        raw: uint8 array of Q40 blocks (d * n_blocks * 18 bytes)
        d: output dimension (rows)
        n: input dimension (F32 elements)
        n_blocks: number of Q40 blocks per row
    Returns:
        F32 weight matrix of shape (d, n)
    """
    block_sz = 2 + Q40_BLOCK_SIZE // 2  # 18
    f32 = np.zeros(d * n, dtype=np.float32)
    for di in range(d):
        base = di * n_blocks
        for bj in range(n_blocks):
            wbo = (base + bj) * block_sz
            sc = _F16_LOOKUP[int(raw[wbo:wbo + 2].view(np.uint16)[0])]
            wqs = raw[wbo + 2:wbo + 2 + Q40_BLOCK_SIZE // 2]
            lo = (wqs.astype(np.int32) & 0x0F) - 8
            hi = (wqs.astype(np.int32) >> 4) - 8
            off = (di * n_blocks + bj) * Q40_BLOCK_SIZE
            f32[off:off + 16] = lo * sc
            f32[off + 16:off + 32] = hi * sc
    return f32.reshape(d, n)


_rope_cache_master: np.ndarray = None
_rope_cache_master_type: int = -1


def _ensure_rope_cache(config, device_buffers: List[np.ndarray], cache_buffer_index: int):
    """Lazily fill rope cache buffer (shares cache across layers).

    In distributed mode each node owns a slice of the full cache.  The
    master cache always holds the complete (non-sliced) tensor so that
    subsequent layers on the same node can copy directly.
    """
    global _rope_cache_master, _rope_cache_master_type
    cache = device_buffers[cache_buffer_index].view(np.float32)

    slc = config.slice
    key = (config.rope_type, slc.q_dim_start, slc.q_dim_end,
           slc.kv_dim_start, slc.kv_dim0, slc.seq_len,
           slc.rope_theta, config.rope_scaling_factor)
    if key != _rope_cache_master_type or _rope_cache_master is None:
        _rope_cache_master_type = key
        if config.rope_type in (NnRopeType.ROPE_LLAMA, NnRopeType.ROPE_LLAMA3_1):
            _fill_rope_llama_cache(config, cache)
        elif config.rope_type == NnRopeType.ROPE_FALCON:
            _fill_rope_falcon_cache(config, cache)
        else:
            raise ValueError(f"Unsupported rope type: {config.rope_type}")
        _rope_cache_master = cache.copy()
    else:
        cache[:] = _rope_cache_master


# ======================================================================
# NnNetExecution
# ======================================================================

class NnNetExecution:
    """Manages pipe data as NumPy uint8 arrays.

    Pipes are shared across nodes (in multi-node configs) and hold the
    intermediate representation flowing between computation segments.
    """

    def __init__(self, n_threads: int, net_config: NnNetConfig):
        self.n_threads = n_threads
        self.n_batches = net_config.n_batches
        self.batch_size = 0  # MUST be set before calling forward()

        self.pipes: List[np.ndarray] = []
        for pc in net_config.pipes:
            self.pipes.append(np.zeros(pc.size.n_bytes, dtype=np.uint8))

    def set_batch_size(self, batch_size: int):
        if batch_size > self.n_batches:
            raise ValueError(f"batch_size {batch_size} exceeds capacity {self.n_batches}")
        self.batch_size = batch_size


# ======================================================================
# NnCpuDeviceSegment  (compiled op context + forward dispatch)
# ======================================================================

class NnCpuDeviceSegment:
    """Compiled segment: holds op contexts and forward logic for one segment.

    On construction, resolves all NnPointerConfig references into typed
    numpy views and pre-builds the rope cache for any ROPE ops.
    """

    def __init__(self,
                 segment_config: NnSegmentConfig,
                 node_config: NnNodeConfig,
                 net_config: NnNetConfig,
                 net_execution: NnNetExecution,
                 device_buffers: List[np.ndarray],
                 buffer_configs: List[NnBufferConfig],
                 pipe_configs: List[NnPipeConfig],
                 node_index: int):
        self._seg_cfg = segment_config
        self._node_cfg = node_config
        self._net_cfg = net_config
        self._net_exec = net_execution
        self._bufs = device_buffers         # raw uint8 buffers
        self._buf_cfgs = buffer_configs
        self._pipe_cfgs = pipe_configs
        self._node_idx = node_index

        self.n_ops = len(segment_config.ops)
        self._ctxs: List[_OpContext] = []
        self._build()

    # ------------------------------------------------------------------
    # resolvePointer  (Python version, returns numpy view list)
    # ------------------------------------------------------------------

    def _resolve_pointer(self, out_size: NnSize3D,
                         pcfg: NnPointerConfig) -> List[np.ndarray]:
        """Resolve a NnPointerConfig to a flat list of numpy array views.

        *out_size* is populated with the resolved NnSize3D.
        Returns one view per (z, y) for PNTR_BATCH / PNTR_BATCHED_SLICE,
        or a single-element list for PNTR_RAW.
        """
        if pcfg.source == NnPointerSource.SRC_PIPE:
            raw = self._net_exec.pipes[pcfg.pointer_index]
            src_sz = self._pipe_cfgs[pcfg.pointer_index].size
        else:
            raw = self._bufs[pcfg.pointer_index]
            src_sz = self._buf_cfgs[pcfg.pointer_index].size

        if pcfg.pointer_type == NnPointerType.PNTR_RAW:
            out_size.float_type = src_sz.float_type
            out_size.z = 1
            out_size.y = 1
            out_size.x = src_sz.length
            return [raw]

        # PNTR_BATCH or PNTR_BATCHED_SLICE
        row_bytes = get_bytes(src_sz.float_type, src_sz.x)
        ptrs = []
        for z in range(src_sz.z):
            for y in range(src_sz.y):
                off = (z * src_sz.y + y) * row_bytes
                ptrs.append(raw[off:off + row_bytes])

        if pcfg.pointer_type == NnPointerType.PNTR_BATCHED_SLICE:
            x_slice = src_sz.x // self._net_cfg.n_nodes
            x_slice_bytes = get_bytes(src_sz.float_type, x_slice)
            node_off = x_slice_bytes * self._node_idx
            for i in range(len(ptrs)):
                ptrs[i] = ptrs[i][node_off:node_off + x_slice_bytes]
            out_size.float_type = src_sz.float_type
            out_size.z = src_sz.z
            out_size.y = src_sz.y
            out_size.x = x_slice
        else:
            out_size.float_type = src_sz.float_type
            out_size.z = src_sz.z
            out_size.y = src_sz.y
            out_size.x = src_sz.x

        return ptrs

    def _build(self):
        for op_cfg in self._seg_cfg.ops:
            ctx = _OpContext()
            ctx.name = op_cfg.name
            ctx.op_config = op_cfg.config
            ctx.code = op_cfg.code
            ctx.n_batches = self._net_cfg.n_batches
            ctx.n_nodes = self._net_cfg.n_nodes
            ctx.node_index = self._node_idx

            isize = NnSize3D()
            osize = NnSize3D()
            ctx.in_ptrs = self._resolve_pointer(isize, op_cfg.input)
            ctx.out_ptrs = self._resolve_pointer(osize, op_cfg.output)
            ctx.in_size = isize
            ctx.out_size = osize
            ctx.w_size = op_cfg.weight_size
            ctx.has_in_contig = has_pointer_continuous_memory(op_cfg.input)
            ctx.has_out_contig = has_pointer_continuous_memory(op_cfg.output)
            ctx.in_ftype = isize.float_type
            ctx.out_ftype = osize.float_type
            ctx.w_ftype = op_cfg.weight_size.float_type

            # Weight buffer: raw uint8, allocated if needed
            if op_cfg.weight_size.n_bytes > 0:
                ctx.weight = np.zeros(op_cfg.weight_size.n_bytes, dtype=np.uint8)
            else:
                ctx.weight = None

            self._ctxs.append(ctx)

        # Pre-fill rope caches
        for ctx in self._ctxs:
            if ctx.code == NnOpCode.ROPE:
                _ensure_rope_cache(ctx.op_config, self._bufs,
                                   ctx.op_config.rope_cache_buffer_index)

    # ------------------------------------------------------------------
    # Buffer / pipe access helpers
    # ------------------------------------------------------------------

    def _buf_f32(self, idx: int) -> np.ndarray:
        """Return float32 view of buffer *idx*."""
        return self._bufs[idx].view(np.float32)

    def _pos(self, batch_idx: int, pipe_idx: int) -> int:
        """Read position for *batch_idx* from pipe *pipe_idx*."""
        return int(self._net_exec.pipes[pipe_idx].view(np.float32)[batch_idx])

    def _expert_idx(self, batch: int, e: int, ctx) -> int:
        cfg = ctx.op_config
        buf = self._bufs[cfg.active_expert_indexes_buffer_index]
        return int(buf.view(np.float32)[batch * cfg.n_active_experts + e])

    def _inv_rms_val(self, batch: int, col: int, ctx) -> float:
        cfg = ctx.op_config
        buf_cfg = self._buf_cfgs[cfg.inv_rms_buffer_index]
        stride = buf_cfg.size.x
        return float(self._bufs[cfg.inv_rms_buffer_index].view(np.float32)[batch * stride + col])

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weight(self, op_index: int, offset: int, n_bytes: int, weight):
        """Copy *weight* bytes into the compiled weight buffer.

        *weight* may be bytes, bytearray, or uint8 numpy array.
        """
        ctx = self._ctxs[op_index]
        if isinstance(weight, (bytes, bytearray)):
            ctx.weight[offset:offset + n_bytes] = np.frombuffer(weight, dtype=np.uint8, count=n_bytes)
        else:
            ctx.weight[offset:offset + n_bytes] = weight[:n_bytes]

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward(self, op_index: int, n_threads: int, thread_index: int, batch_size: int):
        ctx = self._ctxs[op_index]
        code = ctx.code

        if code == NnOpCode.MERGE_ADD:
            self._fwd_merge_add(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.MERGE_SUM:
            self._fwd_merge_sum(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.EMBEDDING:
            self._fwd_embedding(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.INV_RMS:
            self._fwd_inv_rms(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.RMS_NORM:
            self._fwd_rms_norm(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.MATMUL:
            self._fwd_matmul(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.ROPE:
            self._fwd_rope(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.MULTIHEAD_ATT:
            self._fwd_multihead_att(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.GELU:
            self._fwd_gelu(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.SILU:
            self._fwd_silu(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.MUL:
            self._fwd_mul(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.SCALE:
            self._fwd_scale(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.CAST:
            self._fwd_cast(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.REPEAT_Z:
            self._fwd_repeat_z(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.SHIFT:
            self._fwd_shift(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.SOFTMAX:
            self._fwd_softmax(n_threads, thread_index, batch_size, ctx)
        elif code == NnOpCode.MOE_GATE:
            self._fwd_moe_gate(n_threads, thread_index, batch_size, ctx)
        else:
            raise ValueError(f"Unsupported op code {code} ({ctx.name})")

    # ==================================================================
    # Per-op forward implementations
    # ==================================================================

    # -- MERGE_ADD -------------------------------------------------------

    def _fwd_merge_add(self, nt, ti, bs, ctx):
        it, ot = ctx.in_ftype, ctx.out_ftype
        n_slices = ctx.in_size.x // ctx.out_size.x
        xsz = ctx.out_size.x

        if it == F_32 and ot == F_32:
            for y in range(bs):
                outp = ctx.out_ptrs[y].view(np.float32)
                inp = ctx.in_ptrs[y]
                for si in range(n_slices):
                    addv = inp[si * xsz * 4:(si * xsz + xsz) * 4].view(np.float32)
                    if _C_ADD and ti == 0:
                        _c_add(outp, addv, xsz, nt, ti)
                    else:
                        s, e = _split_threads(xsz, nt, ti)
                        if e > s:
                            outp[s:e] += addv[s:e]

        elif it == F_Q80 and ot == F_32:
            _build_f16_lookup()
            nb = xsz // Q80_BLOCK_SIZE
            for y in range(bs):
                outp = ctx.out_ptrs[y].view(np.float32)
                inp = ctx.in_ptrs[y]
                for si in range(n_slices):
                    bs2, be2 = _split_threads(nb, nt, ti)
                    for bi in range(bs2, be2):
                        bo = (si * nb + bi) * (2 + Q80_BLOCK_SIZE)
                        d = _F16_LOOKUP[int(inp[bo:bo + 2].view(np.uint16)[0])]
                        qs = inp[bo + 2:bo + 2 + Q80_BLOCK_SIZE].view(np.int8)
                        e0 = bi * Q80_BLOCK_SIZE
                        outp[e0:e0 + Q80_BLOCK_SIZE] += d * qs.astype(np.float32)
        else:
            raise ValueError(f"MERGE_ADD: unsupported quant i={it} o={ot}")

    # -- MERGE_SUM -------------------------------------------------------

    def _fwd_merge_sum(self, nt, ti, bs, ctx):
        xsz = ctx.out_size.x
        nz = ctx.in_size.z
        n_ib = ctx.in_size.y
        s, e = _split_threads(xsz, nt, ti)
        if e <= s:
            return
        for y in range(bs):
            acc = np.zeros(e - s, dtype=np.float32)
            for z in range(nz):
                acc += ctx.in_ptrs[y + z * n_ib].view(np.float32)[s:e]
            ctx.out_ptrs[y].view(np.float32)[s:e] = acc

    # -- EMBEDDING -------------------------------------------------------

    def _fwd_embedding(self, nt, ti, bs, ctx):
        it, ot = ctx.in_ftype, ctx.out_ftype
        dim = ctx.out_size.x
        dim_bytes = get_bytes(ot if ot == F_Q80 else F_32, dim)

        for y in range(bs):
            tok = int(ctx.in_ptrs[y].view(np.float32)[0])
            if ot == F_Q80 and it == F_32:
                # Quantize F32 embedding to Q80 output
                if ti == 0:
                    src = tok * get_bytes(F_32, dim)
                    weight_f32 = ctx.weight[src:src + get_bytes(F_32, dim)]
                    from .quants import quantize_f32_to_q80
                    quantize_f32_to_q80(
                        np.frombuffer(weight_f32, dtype=np.float32).copy(),
                        ctx.out_ptrs[y], dim, nt, ti)
            elif ot == F_32 and it == F_32:
                s, e = _split_threads(dim_bytes, nt, ti)
                if e > s:
                    src = tok * dim_bytes + s
                    ctx.out_ptrs[y][s:e] = ctx.weight[src:src + (e - s)]
            else:
                raise ValueError(
                    f"EMBEDDING unsupported types: in={it}, out={ot}"
                )

    # -- INV_RMS ---------------------------------------------------------

    def _fwd_inv_rms(self, nt, ti, bs, ctx):
        cfg = ctx.op_config  # NnInvRmsOpConfig
        col_sz = ctx.in_size.x // cfg.n_columns
        for y in range(ti, bs, nt):
            inp = ctx.in_ptrs[y].view(np.float32)
            outp = ctx.out_ptrs[y].view(np.float32)
            for c in range(cfg.n_columns):
                cv = inp[c * col_sz:(c + 1) * col_sz]
                outp[c] = 1.0 / math.sqrt(float(np.mean(cv ** 2)) + cfg.epsilon)

    # -- RMS_NORM --------------------------------------------------------

    def _fwd_rms_norm(self, nt, ti, bs, ctx):
        cfg = ctx.op_config  # NnRmsNormOpConfig
        if ctx.in_ftype == F_32:
            w = ctx.weight.view(np.float32)
            col_sz = ctx.w_size.x
            for y in range(bs):
                inp = ctx.in_ptrs[y].view(np.float32)
                outp = ctx.out_ptrs[y].view(np.float32)
                for c in range(cfg.n_columns):
                    ir = self._inv_rms_val(y, c, ctx)
                    s, e = _split_threads(col_sz, nt, ti)
                    if e > s:
                        cs = c * col_sz
                        outp[cs + s:cs + e] = inp[cs + s:cs + e] * ir * w[s:e]

        elif ctx.in_ftype == F_Q80:
            _build_f16_lookup()
            w = ctx.weight.view(np.float32)
            nb = ctx.in_size.x // Q80_BLOCK_SIZE
            for y in range(bs):
                inp = ctx.in_ptrs[y]
                outp = ctx.out_ptrs[y].view(np.float32)
                ir = self._inv_rms_val(y, 0, ctx)
                b1, b2 = _split_threads(nb, nt, ti)
                for bi in range(b1, b2):
                    bo = bi * (2 + Q80_BLOCK_SIZE)
                    d = _F16_LOOKUP[int(inp[bo:bo + 2].view(np.uint16)[0])]
                    qs = inp[bo + 2:bo + 2 + Q80_BLOCK_SIZE].view(np.int8)
                    e0 = bi * Q80_BLOCK_SIZE
                    for j in range(Q80_BLOCK_SIZE):
                        k = e0 + j
                        outp[k] = w[k] * (ir * d * float(qs[j]))
        else:
            raise ValueError(f"RMS_NORM unsupported input type: {ctx.in_ftype}")

    # -- MATMUL ----------------------------------------------------------

    def _fwd_matmul(self, nt, ti, bs, ctx):
        cfg = ctx.op_config  # NnMatmulOpConfig
        na = max(cfg.n_active_experts, 1)
        it, wt = ctx.in_ftype, ctx.w_ftype

        # --- Q80×Q40 fast path: one C call for all batch elements ---
        # Fast-path: batch all tokens into one C call (single-node only)
        if (it == F_Q80 and wt == F_Q40
                and _C_MATMUL_Q80Q40 and ti == 0 and na == 1
                and ctx.weight is not None
                and ctx.n_nodes <= 1):
            n = ctx.w_size.y
            d = ctx.w_size.x
            n_blocks = n // Q40_BLOCK_SIZE
            blk_sz = 2 + Q80_BLOCK_SIZE  # sizeof NnBlockQ80

            x_buf = np.empty(bs * n_blocks * blk_sz, dtype=np.uint8)
            for y in range(bs):
                y_off = y * n_blocks * blk_sz
                x_buf[y_off:y_off + n_blocks * blk_sz] = \
                    ctx.in_ptrs[y][:n_blocks * blk_sz]

            w_raw = ctx.weight[:ctx.w_size.n_bytes_xy]
            w_bytes = np.frombuffer(w_raw, dtype=np.uint8)

            out_all = np.empty(bs * d, dtype=np.float32)
            _c_matmul_q80q40(out_all, x_buf, w_bytes, n, d, bs, nt, ti)
            for y in range(bs):
                ctx.out_ptrs[y].view(np.float32)[:] = out_all[y*d:(y+1)*d]
            return

        for y in range(bs):
            for e in range(na):
                ei = self._expert_idx(y, e, ctx) if cfg.n_active_experts > 0 else 0
                oi = e * ctx.out_size.y + y
                ii = e * ctx.in_size.y + y

                if it == F_32 and wt == F_32:
                    n = ctx.w_size.y   # input dim
                    d = ctx.w_size.x   # output dim
                    inp = ctx.in_ptrs[ii].view(np.float32)
                    outp = ctx.out_ptrs[oi].view(np.float32)
                    if _C_MATMUL_F32 and ti == 0:
                        w_off = ei * ctx.w_size.n_bytes_xy
                        w = ctx.weight[w_off:w_off + ctx.w_size.n_bytes_xy].view(np.float32)
                        _c_matmul_f32(outp, inp, w.reshape(d, n), n, d, 1, nt, ti)
                    else:
                        w_off = ei * ctx.w_size.n_bytes_xy
                        w = ctx.weight[w_off:w_off + ctx.w_size.n_bytes_xy].view(np.float32).reshape(d, n)
                        s, e = _split_threads(d, nt, ti)
                        if e > s:
                            outp[s:e] = np.dot(w[s:e, :], inp)

                elif it == F_32 and wt == F_Q40:
                    n = ctx.w_size.y   # input dim
                    d = ctx.w_size.x   # output dim
                    n_blocks = n // Q40_BLOCK_SIZE
                    if ctx.weight_deq is not None:
                        w_use = ctx.weight_deq.reshape(d, n)
                    else:
                        w_off_bytes = ei * ctx.w_size.n_bytes_xy
                        w_raw = ctx.weight[w_off_bytes:w_off_bytes + ctx.w_size.n_bytes_xy]
                        w_use = _dequant_q40_via_c(w_raw, d, n, n_blocks)
                        if na == 1:
                            ctx.weight_deq = w_use.ravel()
                            ctx.weight = None
                    inp = ctx.in_ptrs[ii].view(np.float32)
                    outp = ctx.out_ptrs[oi].view(np.float32)
                    s, e = _split_threads(d, nt, ti)
                    if e > s:
                        outp[s:e] = np.dot(w_use[s:e, :], inp)

                elif it == F_Q80 and wt == F_Q40:
                    n = ctx.w_size.y
                    d = ctx.w_size.x
                    n_blocks = n // Q40_BLOCK_SIZE
                    if _C_MATMUL_Q80Q40 and ti == 0 and ctx.weight is not None:
                        # Use direct Q80xQ40 matmul in C (no dequantization)
                        w_off_bytes = ei * ctx.w_size.n_bytes_xy
                        w_raw = ctx.weight[w_off_bytes:w_off_bytes + ctx.w_size.n_bytes_xy]
                        x_bytes = ctx.in_ptrs[ii]
                        inp_bytes = np.frombuffer(x_bytes[:n_blocks * (2 + Q80_BLOCK_SIZE)], dtype=np.uint8)
                        w_bytes = np.frombuffer(w_raw, dtype=np.uint8)
                        outp = ctx.out_ptrs[oi].view(np.float32)
                        _c_matmul_q80q40(outp, inp_bytes, w_bytes, n, d, 1, nt, ti)
                    else:
                        if ctx.weight_deq is not None:
                            w_use = ctx.weight_deq.reshape(d, n)
                        else:
                            w_off = ei * ctx.w_size.n_bytes_xy
                            w_raw = ctx.weight[w_off:w_off + ctx.w_size.n_bytes_xy]
                            w_use = _dequant_q40_via_c(w_raw, d, n, n_blocks)
                            if na == 1:
                                ctx.weight_deq = w_use.ravel()
                                ctx.weight = None
                        from .quants import dequantize_q80_to_f32
                        q80_dtype = np.dtype([('d', np.uint16), ('qs', np.int8, (Q80_BLOCK_SIZE,))])
                        x_bytes = ctx.in_ptrs[ii]
                        x_blocks = np.ascontiguousarray(x_bytes[:n_blocks * (2 + Q80_BLOCK_SIZE)]).view(q80_dtype).ravel()
                        inp = dequantize_q80_to_f32(x_blocks, n, n_threads=1, thread_index=0)
                        outp = ctx.out_ptrs[oi].view(np.float32)
                        s, e = _split_threads(d, nt, ti)
                        if e > s:
                            outp[s:e] = np.dot(w_use[s:e, :], inp)
                else:
                    raise ValueError(f"MATMUL unsupported: i={it} w={wt}")

    # -- ROPE ------------------------------------------------------------

    def _fwd_rope(self, nt, ti, bs, ctx):
        cfg = ctx.op_config  # NnRopeOpConfig
        cache = self._bufs[cfg.rope_cache_buffer_index].view(np.float32)
        is_q = bool(cfg.is_q)

        if _C_ROPE_LLAMA or _C_ROPE_FALCON:
            for y in range(bs):
                x = ctx.in_ptrs[y].view(np.float32)
                pos = self._pos(y, cfg.position_pipe_index)
                slc = cfg.slice
                dim0 = slc.q_dim0 if is_q else slc.kv_dim0
                shift = slc.q_shift if is_q else 0
                if cfg.rope_type in (NnRopeType.ROPE_LLAMA, NnRopeType.ROPE_LLAMA3_1):
                    _c_rope_llama(x, cache, is_q, pos, dim0, slc.q_dim_end, shift, nt, ti)
                elif cfg.rope_type == NnRopeType.ROPE_FALCON:
                    _c_rope_falcon(x, cache, pos, dim0, slc.head_dim, nt, ti)
            return

        for y in range(bs):
            x = ctx.in_ptrs[y].view(np.float32)
            pos = self._pos(y, cfg.position_pipe_index)

            if cfg.rope_type in (NnRopeType.ROPE_LLAMA, NnRopeType.ROPE_LLAMA3_1):
                slc = cfg.slice
                dim0 = slc.q_dim0 if is_q else slc.kv_dim0
                shift = slc.q_shift if is_q else 0
                dh = dim0 // 2
                s, e = _split_threads(dh, nt, ti)
                stride = slc.q_dim_end
                pc = cache[pos * stride + shift:]
                for i in range(s * 2, e * 2, 2):
                    fcr, fci = pc[i], pc[i + 1]
                    v0, v1 = x[i], x[i + 1]
                    x[i] = v0 * fcr - v1 * fci
                    x[i + 1] = v0 * fci + v1 * fcr

            elif cfg.rope_type == NnRopeType.ROPE_FALCON:
                slc = cfg.slice
                dim0 = slc.q_dim0 if is_q else slc.kv_dim0
                nh0 = dim0 // slc.head_dim
                hs, he = _split_threads(nh0, nt, ti)
                pc = cache[pos * slc.head_dim:]
                for h in range(hs, he):
                    o = h * slc.head_dim
                    hh = slc.head_dim // 2
                    for j in range(hh):
                        fcr0, fci0 = pc[j], pc[j + hh]
                        q0, q1 = x[o + j], x[o + j + hh]
                        x[o + j] = q0 * fcr0 - q1 * fci0
                        x[o + j + hh] = q0 * fci0 + q1 * fcr0
            else:
                raise ValueError(f"Unsupported rope type: {cfg.rope_type}")

    # -- MULTIHEAD_ATT ---------------------------------------------------

    def _fwd_multihead_att(self, nt, ti, bs, ctx):
        cfg = ctx.op_config  # NnMultiHeadAttOpConfig
        query = self._buf_f32(cfg.query_buffer_index)
        kc = self._buf_f32(cfg.key_cache_buffer_index)
        vc = self._buf_f32(cfg.value_cache_buffer_index)
        att_buf = self._buf_f32(cfg.att_buffer_index)

        kv_mul = cfg.n_heads // cfg.n_kv_heads
        hdr = math.sqrt(cfg.head_dim)

        for y in range(bs):
            outp = ctx.out_ptrs[y].view(np.float32)
            pos = self._pos(y, cfg.position_pipe_index)

            if _C_MULTIHEAD_ATT and ti == 0:
                q = query[y * cfg.q_slice_d0:(y + 1) * cfg.q_slice_d0]
                _c_multihead_att(
                    outp, q, att_buf, kc, vc,
                    pos, cfg.n_heads0, cfg.n_heads, cfg.n_kv_heads,
                    cfg.kv_dim0, cfg.head_dim, cfg.seq_len,
                    ctx.n_nodes, ctx.node_index,
                    nt, ti)
                continue

            q = query[y * cfg.q_slice_d0:(y + 1) * cfg.q_slice_d0]
            h0s, h0e = _split_threads(cfg.n_heads0, nt, ti)
            kv_heads_per_node = cfg.n_kv_heads // ctx.n_nodes
            for h0 in range(h0s, h0e):
                hi = (ctx.node_index * cfg.n_heads0 + h0) // kv_mul
                hq = q[h0 * cfg.head_dim:(h0 + 1) * cfg.head_dim]
                local_hi = hi - ctx.node_index * kv_heads_per_node
                hkc_base = local_hi * cfg.head_dim
                hvc_base = local_hi * cfg.head_dim

                # Scores
                ab = (y * cfg.n_heads0 + h0) * cfg.seq_len
                for t in range(pos + 1):
                    pk = kc[hkc_base + t * cfg.kv_dim0:
                            hkc_base + t * cfg.kv_dim0 + cfg.head_dim]
                    att_buf[ab + t] = np.dot(hq, pk) / hdr

                _softmax_inplace(att_buf[ab:ab + pos + 1])

                # Weighted value sum
                hy = outp[h0 * cfg.head_dim:(h0 + 1) * cfg.head_dim]
                hy.fill(0.0)
                for t in range(pos + 1):
                    pv = vc[hvc_base + t * cfg.kv_dim0:
                            hvc_base + t * cfg.kv_dim0 + cfg.head_dim]
                    hy += att_buf[ab + t] * pv

    # -- GELU ------------------------------------------------------------

    def _fwd_gelu(self, nt, ti, bs, ctx):
        for z in range(ctx.in_size.z):
            for y in range(bs):
                outp = ctx.out_ptrs[z * ctx.out_size.y + y].view(np.float32)
                n = ctx.out_size.x
                if _C_GELU and ti == 0:
                    _c_gelu(outp, n, nt, ti)
                else:
                    a = 0.7978845608028654
                    b = 0.044715
                    s, e = _split_threads(n, nt, ti)
                    if e > s:
                        x = outp[s:e].copy()
                        outp[s:e] = 0.5 * x * (1.0 + np.tanh(a * x * (1.0 + b * x * x)))

    # -- SILU ------------------------------------------------------------

    def _fwd_silu(self, nt, ti, bs, ctx):
        for z in range(ctx.in_size.z):
            for y in range(bs):
                outp = ctx.out_ptrs[z * ctx.out_size.y + y].view(np.float32)
                n = ctx.out_size.x
                if _C_SILU and ti == 0:
                    _c_silu(outp, n, nt, ti)
                else:
                    s, e = _split_threads(n, nt, ti)
                    if e > s:
                        x = outp[s:e].copy()
                        outp[s:e] = x / (1.0 + np.exp(-x))

    # -- MUL -------------------------------------------------------------

    def _fwd_mul(self, nt, ti, bs, ctx):
        cfg = ctx.op_config  # NnMulOpCodeConfig
        mul = self._buf_f32(cfg.multiplier_buffer_index)
        xsz = ctx.out_size.x
        for z in range(ctx.in_size.z):
            zo = z * ctx.in_size.y
            for y in range(bs):
                inp = ctx.in_ptrs[zo + y].view(np.float32)
                outp = ctx.out_ptrs[zo + y].view(np.float32)
                mr = mul[xsz * (zo + y):xsz * (zo + y + 1)]
                if _C_MUL and ti == 0:
                    _c_mul(outp, inp, mr, xsz, nt, ti)
                else:
                    s, e = _split_threads(xsz, nt, ti)
                    if e > s:
                        outp[s:e] = inp[s:e] * mr[s:e]

    # -- SCALE -----------------------------------------------------------

    def _fwd_scale(self, nt, ti, bs, ctx):
        cfg = ctx.op_config  # NnScaleOpCodeConfig
        sbuf = self._buf_f32(cfg.scale_buffer_index)
        for z in range(ctx.in_size.z):
            for y in range(bs):
                idx = z * ctx.in_size.y + y
                sc = sbuf[idx]
                inp = ctx.in_ptrs[idx].view(np.float32)
                outp = ctx.out_ptrs[idx].view(np.float32)
                n = ctx.in_size.x
                for j in range(ti, n, nt):
                    outp[j] = inp[j] * sc

    # -- CAST ------------------------------------------------------------

    def _fwd_cast(self, nt, ti, bs, ctx):
        it, ot = ctx.in_ftype, ctx.out_ftype
        rowb = ctx.out_size.n_bytes // ctx.out_size.y
        for z in range(ctx.in_size.z):
            zo = z * ctx.in_size.y
            for y in range(bs):
                inp = ctx.in_ptrs[zo + y]
                outp = ctx.out_ptrs[zo + y]
                if it == ot:
                    s, e = _split_threads(rowb, nt, ti)
                    if e > s:
                        outp[s:e] = inp[s:e]
                elif it == F_32 and ot == F_Q80:
                    _quantize_f32_to_q80(inp.view(np.float32), ctx.out_size.x, outp, nt, ti)
                elif it == F_Q80 and ot == F_32:
                    _dequantize_q80(inp, ctx.out_size.x, outp.view(np.float32), nt, ti)
                else:
                    raise ValueError(f"CAST unsupported: i={it} o={ot}")

    # -- REPEAT_Z --------------------------------------------------------

    def _fwd_repeat_z(self, nt, ti, bs, ctx):
        it, ot = ctx.in_ftype, ctx.out_ftype
        dimb = get_bytes(ot, ctx.out_size.x)
        for z in range(ctx.out_size.z):
            for y in range(bs):
                orow = ctx.out_ptrs[z * ctx.out_size.y + y]
                if z == 0:
                    if it == F_32 and ot == F_Q80:
                        _quantize_f32_to_q80(ctx.in_ptrs[y].view(np.float32),
                                             ctx.out_size.x, orow, nt, ti)
                    else:
                        s, e = _split_threads(dimb, nt, ti)
                        if e > s:
                            orow[s:e] = ctx.in_ptrs[y][s:e]
                else:
                    ref = ctx.out_ptrs[y]
                    s, e = _split_threads(dimb, nt, ti)
                    if e > s:
                        orow[s:e] = ref[s:e]

    # -- SHIFT -----------------------------------------------------------

    def _fwd_shift(self, nt, ti, bs, ctx):
        cfg = ctx.op_config  # NnShiftOpCodeConfig
        dimb = get_bytes(F_32, ctx.in_size.x)
        for y in range(bs):
            pos = self._pos(y, cfg.index_pipe_index)
            dst = pos * dimb
            src = ctx.in_ptrs[y]
            out_row = ctx.out_ptrs[0]
            s, e = _split_threads(dimb, nt, ti)
            if e > s:
                out_row[dst + s:dst + e] = src[s:e]

    # -- SOFTMAX ---------------------------------------------------------

    def _fwd_softmax(self, nt, ti, bs, ctx):
        if _C_SOFTMAX and ti == 0:
            for y in range(bs):
                _c_softmax(
                    ctx.out_ptrs[y].view(np.float32), ctx.out_size.x)
        else:
            for y in range(ti, bs, nt):
                _softmax_inplace(ctx.out_ptrs[y].view(np.float32))

    # -- MOE_GATE --------------------------------------------------------

    def _fwd_moe_gate(self, nt, ti, bs, ctx):
        cfg = ctx.op_config  # NnMoeGateOpCodeConfig
        idx_buf = self._buf_f32(cfg.indexes_buffer_index)
        k = cfg.k

        for y in range(ti, bs, nt):
            inp = ctx.in_ptrs[y].view(np.float32)
            top = np.argpartition(-inp, k)[:k]
            top_vals = inp[top]
            top = top[np.argsort(-top_vals)]

            denom = float(np.sum(inp[top])) if cfg.norm_topk else 1.0

            for ki in range(k):
                p = top[ki]
                idx_buf[y * k + ki] = float(p)
                outp = ctx.out_ptrs[ki * ctx.out_size.y + y].view(np.float32)
                outp[0] = float(inp[p]) / denom


# ======================================================================
# _OpContext  (compiled metadata for one op)
# ======================================================================

class _OpContext:
    """Minimal container mirroring NnCpuOpContext from nn-cpu-ops.hpp."""

    __slots__ = (
        "name", "op_config", "code",
        "in_ptrs", "out_ptrs",
        "in_size", "out_size", "w_size",
        "in_ftype", "out_ftype", "w_ftype",
        "weight", "weight_deq",
        "has_in_contig", "has_out_contig",
        "n_batches", "n_nodes", "node_index",
    )

    def __init__(self):
        self.name: str = ""
        self.op_config: object = None
        self.code: int = 0
        self.in_ptrs: List[np.ndarray] = []
        self.out_ptrs: List[np.ndarray] = []
        self.in_size: NnSize3D = NnSize3D()
        self.out_size: NnSize3D = NnSize3D()
        self.w_size: NnSize3D = NnSize3D()
        self.in_ftype: int = F_32
        self.out_ftype: int = F_32
        self.w_ftype: int = F_32
        self.weight: Optional[np.ndarray] = None  # uint8 raw bytes
        self.weight_deq: Optional[np.ndarray] = None  # pre-dequantized F32
        self.has_in_contig: bool = False
        self.has_out_contig: bool = False
        self.n_batches: int = 0
        self.n_nodes: int = 0
        self.node_index: int = 0


# ======================================================================
# NnCpuDevice
# ======================================================================

class NnCpuDevice:
    """CPU device: allocates buffers and creates compiled segments."""

    def __init__(self,
                 net_config: NnNetConfig,
                 node_config: NnNodeConfig,
                 net_execution: NnNetExecution):
        self._net_cfg = net_config
        self._node_cfg = node_config
        self._net_exec = net_execution

        self.buffers: List[np.ndarray] = []
        for bc in node_config.buffers:
            self.buffers.append(np.zeros(bc.size.n_bytes, dtype=np.uint8))

    def max_n_threads(self) -> int:
        import os
        return os.cpu_count() or 1

    def create_segment(self, segment_index: int) -> NnCpuDeviceSegment:
        seg_cfg = self._node_cfg.segments[segment_index]
        return NnCpuDeviceSegment(
            segment_config=seg_cfg,
            node_config=self._node_cfg,
            net_config=self._net_cfg,
            net_execution=self._net_exec,
            device_buffers=self.buffers,
            buffer_configs=self._node_cfg.buffers,
            pipe_configs=self._net_cfg.pipes,
            node_index=self._node_cfg.node_index,
        )


# ======================================================================
# NnNodeSynchronizer  (abstract base)
# ======================================================================

class NnNodeSynchronizer(ABC):
    """Abstract interface for inter-node synchronization."""

    @abstractmethod
    def sync(self, segment_index: int, n_threads: int, thread_index: int):
        ...


class NnFakeNodeSynchronizer(NnNodeSynchronizer):
    """No-op synchronizer for single-node deployments."""

    def sync(self, segment_index: int, n_threads: int, thread_index: int):
        pass


# ======================================================================
# NnExecutor step types
# ======================================================================

STEP_EXECUTE_OP = 0
STEP_SYNC_NODES = 1


# ======================================================================
# NnExecutor
# ======================================================================

class NnExecutor:
    """Multithreaded graph execution engine.

    Builds an interleaved step list (op execution + sync points), loads
    weights into compiled segments, and dispatches ``forward()`` using a
    ``ThreadPoolExecutor``.
    """

    def __init__(self,
                 net_config: NnNetConfig,
                 node_config: NnNodeConfig,
                 device: NnCpuDevice,
                 net_execution: NnNetExecution,
                 synchronizer: NnNodeSynchronizer,
                 benchmark: bool = False):
        self._net_cfg = net_config
        self._node_cfg = node_config
        self._device = device
        self._net_exec = net_execution
        self._sync = synchronizer
        self._benchmark = benchmark

        self._segments: List[Optional[NnCpuDeviceSegment]] = []
        self._steps: List[Tuple[int, Optional[NnCpuDeviceSegment], int, Optional[NnOpConfig]]] = []

        self._time_op_us: float = 0.0
        self._time_sync_us: float = 0.0
        self._pool = None  # cached ThreadPoolExecutor (lazy init)

        self._build_steps()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_threads(self) -> int:
        return self._net_exec.n_threads

    @property
    def batch_size(self) -> int:
        return self._net_exec.batch_size

    # ------------------------------------------------------------------
    # Step construction
    # ------------------------------------------------------------------

    def _build_steps(self):
        use_sync = self._net_cfg.n_nodes > 1
        for seg_idx, seg_cfg in enumerate(self._node_cfg.segments):
            if len(seg_cfg.ops) > 0:
                seg = self._device.create_segment(seg_idx)
                self._segments.append(seg)
                for op_i, op_cfg in enumerate(seg_cfg.ops):
                    self._steps.append((STEP_EXECUTE_OP, seg, op_i, op_cfg))
            else:
                self._segments.append(None)

            if use_sync and len(seg_cfg.syncs) > 0:
                self._steps.append((STEP_SYNC_NODES, None, seg_idx, None))

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weight(self, name: str, op_index: int, offset: int, n_bytes: int,
                    weight: np.ndarray):
        """Load raw weight bytes into all matching ops across segments.

        Searches for ops whose ``name`` and ``index`` match *name* and
        *op_index*, then copies *weight* (uint8) into the compiled buffer.
        """
        for seg_idx, seg_cfg in enumerate(self._node_cfg.segments):
            seg = self._segments[seg_idx]
            if seg is None:
                continue
            for op_i, op_cfg in enumerate(seg_cfg.ops):
                if op_cfg.index == op_index and op_cfg.name == name:
                    seg.load_weight(op_i, offset, n_bytes, weight)
                    return
        # Op not on this node — silently skip
        return 0

    # ------------------------------------------------------------------
    # Forward execution
    # ------------------------------------------------------------------

    def forward(self):
        """Execute all steps using ThreadPoolExecutor.

        Raises RuntimeError if ``batch_size`` has not been set.
        """
        bs = self._net_exec.batch_size
        if bs <= 0:
            raise RuntimeError("batch_size must be set before calling forward()")

        nt = self._net_exec.n_threads
        self._time_op_us = 0.0
        self._time_sync_us = 0.0

        # For multi-thread: initialize C-level thread pool (one-time)
        if nt > 1 and _ops_c:
            _c_init_mt(nt)

        bs = self._net_exec.batch_size
        # For single-thread: call directly, skip executor overhead
        if nt <= 1:
            for stype, seg, arg, _ in self._steps:
                t0 = time.perf_counter_ns()
                if stype == STEP_EXECUTE_OP:
                    seg.forward(arg, 1, 0, bs)
                    self._time_op_us += (time.perf_counter_ns() - t0) / 1000.0
                elif stype == STEP_SYNC_NODES:
                    self._sync.sync(arg, 1, 0)
                    self._time_sync_us += (time.perf_counter_ns() - t0) / 1000.0
            return

        # Multi-thread: lazily create cached executor
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=nt)

        ex = self._pool
        for stype, seg, arg, _ in self._steps:
            t0 = time.perf_counter_ns()
            if stype == STEP_EXECUTE_OP:
                futs = [ex.submit(seg.forward, arg, nt, ti, bs) for ti in range(nt)]
                for f in futs:
                    f.result()
                self._time_op_us += (time.perf_counter_ns() - t0) / 1000.0
            elif stype == STEP_SYNC_NODES:
                self._sync.sync(arg, nt, 0)
                self._time_sync_us += (time.perf_counter_ns() - t0) / 1000.0

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def get_total_time_op_us(self) -> float:
        return self._time_op_us

    def get_total_time_sync_us(self) -> float:
        return self._time_sync_us
