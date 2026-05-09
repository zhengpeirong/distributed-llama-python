#ifndef OPS_H
#define OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef uint8_t NnByte;
typedef uint32_t NnUint;
typedef size_t NnSize;

enum NnFloatType {
    F_UNK = -1,
    F_32  = 0,
    F_16  = 1,
    F_Q40 = 2,
    F_Q80 = 3,
};

enum NnOpCode {
    OP_MERGE_ADD = 0,
    OP_MERGE_SUM = 1,
    OP_EMBEDDING = 2,
    OP_INV_RMS = 3,
    OP_RMS_NORM = 4,
    OP_MATMUL = 5,
    OP_ROPE = 6,
    OP_MULTIHEAD_ATT = 7,
    OP_GELU = 8,
    OP_SILU = 9,
    OP_MUL = 10,
    OP_SCALE = 11,
    OP_CAST = 12,
    OP_REPEAT_Z = 13,
    OP_SHIFT = 14,
    OP_SOFTMAX = 15,
    OP_MOE_GATE = 16,
};

enum NnOpQuantType {
    F32_F32_F32 = 0,
    F32_Q40_F32 = 1,
    F32_Q40_Q80 = 2,
    F32_F32_Q80 = 3,
    Q80_Q80_Q80 = 4,
    Q80_Q80_F32 = 5,
    Q80_Q40_F32 = 6,
    Q80_F32_F32 = 7,
};

enum NnRopeType {
    ROPE_LLAMA = 0,
    ROPE_FALCON = 1,
    ROPE_LLAMA3_1 = 2,
};

#define Q40_BLOCK_SIZE 32
#define Q80_BLOCK_SIZE 32

#define SPLIT_THREADS(var_start, var_end, range_len, n_threads, thread_index) \
    NnUint var_start, var_end; do { \
        NnUint _slice = (range_len) / (n_threads); \
        NnUint _rest = (range_len) % (n_threads); \
        var_start = (thread_index) * _slice + ((thread_index) < _rest ? (thread_index) : _rest); \
        var_end = var_start + _slice + ((thread_index) < _rest ? 1 : 0); \
    } while(0)

/* f16 conversion is provided locally in ops.c */

/* all ops: return the output data, n_threads + thread_index for parallelism */

void op_softmax(float *x, NnUint size);

void op_embedding_f32(float *output, const float *weight, const float *input,
                      NnUint dim, NnUint batch_size,
                      NnUint n_threads, NnUint thread_index);

void op_inv_rms(float *output, const float *input, NnUint col_size,
                NnUint n_columns, float epsilon, NnUint batch_size,
                NnUint n_threads, NnUint thread_index);

void op_rms_norm_f32(float *output, const float *input, const float *weight,
                     const float *inv_rms, NnUint size, NnUint n_columns,
                     NnUint batch_size, NnUint n_threads, NnUint thread_index);

void op_matmul_f32_f32_f32(float *output, const float *x, const float *w,
                           NnUint n, NnUint d, NnUint batch_size,
                           NnUint n_threads, NnUint thread_index);

void op_matmul_q80_q40_f32(float *output, const void *x, const void *w,
                           NnUint n, NnUint d, NnUint batch_size,
                           NnUint n_threads, NnUint thread_index);

void op_rope_llama(float *x, const float *cache, bool is_q, NnUint pos,
                   NnUint dim0, NnUint slice_dim, NnUint shift,
                   NnUint n_threads, NnUint thread_index);

void op_rope_falcon(float *x, const float *cache, NnUint pos,
                    NnUint dim0, NnUint head_dim,
                    NnUint n_threads, NnUint thread_index);

void op_multihead_att(float *y, const float *q, float *att,
                      float *key_cache, float *value_cache,
                      NnUint pos, NnUint n_heads0, NnUint n_heads,
                      NnUint n_kv_heads, NnUint kv_dim0, NnUint head_dim,
                      NnUint seq_len, NnUint n_nodes, NnUint node_index,
                      NnUint n_threads, NnUint thread_index);

void op_silu(float *x, NnUint size, NnUint n_threads, NnUint thread_index);

void op_gelu(float *x, NnUint size, NnUint n_threads, NnUint thread_index);

void op_mul_f32(float *y, const float *x, const float *m, NnUint n,
                NnUint n_threads, NnUint thread_index);

void op_add_f32(float *y, const float *x, NnUint n,
                NnUint n_threads, NnUint thread_index);

void op_scale_f32(float *o, const float *i, float s, NnUint n,
                  NnUint n_threads, NnUint thread_index);

void op_copy_bytes(NnByte *dst, const NnByte *src, NnSize n,
                   NnUint n_threads, NnUint thread_index);

void op_topk(const float *x, NnUint *y, NnSize size, NnUint k);

void op_softmax_f32(float *x, NnUint size);

#endif
