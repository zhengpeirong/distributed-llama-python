"""Q40/Q80 quantization and F16 conversion.

Uses a C extension for performance. Falls back to pure-Python NumPy
implementation if the extension is not available.
"""

import ctypes
import os
from ctypes import c_float, c_uint32, c_size_t, c_uint16, c_int8, c_uint8, POINTER, Structure

import numpy as np

Q40_BLOCK_SIZE = 32
Q80_BLOCK_SIZE = 32

F_UNK = -1
F_32 = 0
F_16 = 1
F_Q40 = 2
F_Q80 = 3


class NnBlockQ40(Structure):
    _fields_ = [
        ("d", c_uint16),
        ("qs", c_uint8 * (Q40_BLOCK_SIZE // 2)),
    ]


class NnBlockQ80(Structure):
    _fields_ = [
        ("d", c_uint16),
        ("qs", c_int8 * Q80_BLOCK_SIZE),
    ]


_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    # Load via ctypes from the .so in the same directory
    import glob
    so_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = glob.glob(os.path.join(so_dir, "_quants*.so"))
    if candidates:
        _lib = ctypes.CDLL(candidates[0], ctypes.RTLD_GLOBAL)
    else:
        _lib = _load_from_system()
    _setup_lib()
    _lib.init_quants()
    return _lib


def _load_from_system():
    import sys
    lib_name = "libdistributed_llama_quants.so"
    search_paths = [
        os.path.join(os.path.dirname(__file__), "../../"),
        os.getcwd(),
    ]
    for p in search_paths:
        try:
            return ctypes.CDLL(os.path.join(p, lib_name))
        except OSError:
            continue
    raise ImportError(
        "Cannot load quants C extension. "
        "Build it with: python setup.py build_ext --inplace"
    )


def _setup_lib():
    _lib.init_quants.argtypes = []
    _lib.init_quants.restype = None

    _lib.dequantize_q40_to_f32.argtypes = [
        POINTER(NnBlockQ40), POINTER(c_float), c_uint32,
        c_uint32, c_uint32,
    ]
    _lib.dequantize_q40_to_f32.restype = None

    _lib.dequantize_q80_to_f32.argtypes = [
        POINTER(NnBlockQ80), POINTER(c_float), c_uint32,
        c_uint32, c_uint32,
    ]
    _lib.dequantize_q80_to_f32.restype = None

    _lib.quantize_f32_to_q80.argtypes = [
        POINTER(c_float), POINTER(NnBlockQ80), c_uint32,
        c_uint32, c_uint32,
    ]
    _lib.quantize_f32_to_q80.restype = None

    _lib.quantize_f32_to_q40.argtypes = [
        POINTER(c_float), POINTER(NnBlockQ40), c_uint32,
        c_uint32, c_uint32,
    ]
    _lib.quantize_f32_to_q40.restype = None


def ensure_lib():
    if _lib is None:
        _load_lib()
    return _lib


def dequantize_q40_to_f32(blocks, n, n_threads=1, thread_index=0):
    """Dequantize Q40 blocks to float32.

    Args:
        blocks: numpy array of NnBlockQ40 (dtype must match packed struct)
        n: total number of float elements
        n_threads: number of threads
        thread_index: this thread's index (0-based)

    Returns:
        numpy array of float32, shape (n,)
    """
    lib = ensure_lib()
    out = np.empty(n, dtype=np.float32)
    blocks_p = blocks.ctypes.data_as(POINTER(NnBlockQ40))
    out_p = out.ctypes.data_as(POINTER(c_float))
    lib.dequantize_q40_to_f32(blocks_p, out_p, n, n_threads, thread_index)
    return out


def dequantize_q80_to_f32(blocks, n, n_threads=1, thread_index=0):
    """Dequantize Q80 blocks to float32."""
    lib = ensure_lib()
    out = np.empty(n, dtype=np.float32)
    blocks_p = blocks.ctypes.data_as(POINTER(NnBlockQ80))
    out_p = out.ctypes.data_as(POINTER(c_float))
    lib.dequantize_q80_to_f32(blocks_p, out_p, n, n_threads, thread_index)
    return out


def quantize_f32_to_q80(arr, n, n_threads=1, thread_index=0):
    """Quantize float32 array to Q80 blocks."""
    lib = ensure_lib()
    n_blocks = n // Q80_BLOCK_SIZE
    out = np.empty(n_blocks, dtype=np.dtype([
        ('d', np.uint16),
        ('qs', np.int8, (Q80_BLOCK_SIZE,)),
    ]))
    arr_p = arr.ctypes.data_as(POINTER(c_float))
    out_p = out.ctypes.data_as(POINTER(NnBlockQ80))
    lib.quantize_f32_to_q80(arr_p, out_p, n, n_threads, thread_index)
    return out


def quantize_f32_to_q40(arr, n, n_threads=1, thread_index=0):
    """Quantize float32 array to Q40 blocks."""
    lib = ensure_lib()
    n_blocks = n // Q40_BLOCK_SIZE
    out = np.empty(n_blocks, dtype=np.dtype([
        ('d', np.uint16),
        ('qs', np.uint8, (Q40_BLOCK_SIZE // 2,)),
    ]))
    arr_p = arr.ctypes.data_as(POINTER(c_float))
    out_p = out.ctypes.data_as(POINTER(NnBlockQ40))
    lib.quantize_f32_to_q40(arr_p, out_p, n, n_threads, thread_index)
    return out


def get_bytes(float_type, n):
    """Return number of bytes for n elements of given float type."""
    if float_type == F_32:
        return n * 4
    elif float_type == F_16:
        return n * 2
    elif float_type == F_Q40:
        return (n // Q40_BLOCK_SIZE) * (2 + Q40_BLOCK_SIZE // 2)
    elif float_type == F_Q80:
        return (n // Q80_BLOCK_SIZE) * (2 + Q80_BLOCK_SIZE)
    raise ValueError(f"Unknown float type: {float_type}")


def get_block_size(float_type):
    """Return block size for given float type."""
    if float_type in (F_32, F_16):
        return 1
    elif float_type == F_Q40:
        return Q40_BLOCK_SIZE
    elif float_type == F_Q80:
        return Q80_BLOCK_SIZE
    raise ValueError(f"Unknown float type: {float_type}")


def get_op_quant_type(input_type, weight_type, output_type):
    """Return op quant type enum value. Matches C++ getOpQuantType."""
    # If weight=F_UNK, treat as matching input type
    if weight_type == F_UNK:
        weight_type = input_type

    if input_type == F_32 and output_type == F_32:
        if weight_type == F_32:
            return 0  # F32_F32_F32
        if weight_type == F_Q40:
            return 1  # F32_Q40_F32
    if input_type == F_32 and output_type == F_Q80:
        if weight_type == F_32:
            return 3  # F32_F32_Q80
        if weight_type == F_Q40:
            return 2  # F32_Q40_Q80
    if input_type == F_Q80 and output_type == F_32:
        if weight_type == F_Q80:
            return 5  # Q80_Q80_F32
        if weight_type == F_32:
            return 7  # Q80_F32_F32
        if weight_type == F_Q40:
            return 6  # Q80_Q40_F32
    if input_type == F_Q80 and output_type == F_Q80:
        if weight_type == F_Q80:
            return 4  # Q80_Q80_Q80

    raise ValueError(
        f"Unsupported op quant: {float_type_to_string(input_type)}/"
        f"{float_type_to_string(weight_type)}/"
        f"{float_type_to_string(output_type)}"
    )


def float_type_to_string(ftype):
    names = {F_UNK: "F_UNK", F_32: "F_32", F_16: "F_16",
             F_Q40: "F_Q40", F_Q80: "F_Q80"}
    return names.get(ftype, "F_UNKNOWN")
