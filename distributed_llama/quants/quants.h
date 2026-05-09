#ifndef QUANTS_H
#define QUANTS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef uint8_t NnByte;
typedef uint32_t NnUint;
typedef size_t NnSize;
typedef uint16_t NnFp16;

#define Q40_BLOCK_SIZE 32
#define Q80_BLOCK_SIZE 32

enum NnFloatType {
    F_UNK = -1,
    F_32  = 0,
    F_16  = 1,
    F_Q40 = 2,
    F_Q80 = 3,
};

typedef struct {
    uint16_t d;
    uint8_t qs[Q40_BLOCK_SIZE / 2];
} NnBlockQ40;

typedef struct {
    uint16_t d;
    int8_t qs[Q80_BLOCK_SIZE];
} NnBlockQ80;

extern float f16_to_f32_lookup[65536];

void init_quants(void);
float convert_f16_to_f32(NnFp16 value);
NnFp16 convert_f32_to_f16(float x);

void dequantize_q40_to_f32(const NnBlockQ40 *input, float *output, NnUint n,
                           NnUint n_threads, NnUint thread_index);
void dequantize_q80_to_f32(const NnBlockQ80 *input, float *output, NnUint n,
                           NnUint n_threads, NnUint thread_index);
void quantize_f32_to_q80(const float *input, NnBlockQ80 *output, NnUint n,
                         NnUint n_threads, NnUint thread_index);
void quantize_f32_to_q40(const float *input, NnBlockQ40 *output, NnUint n,
                         NnUint n_threads, NnUint thread_index);

#define SPLIT_THREADS(var_start, var_end, range_len, n_threads, thread_index) \
    NnUint var_start, var_end; do { \
        NnUint _slice = (range_len) / (n_threads); \
        NnUint _rest = (range_len) % (n_threads); \
        var_start = (thread_index) * _slice + ((thread_index) < _rest ? (thread_index) : _rest); \
        var_end = var_start + _slice + ((thread_index) < _rest ? 1 : 0); \
    } while(0)

const char *float_type_to_string(int type);

#endif
