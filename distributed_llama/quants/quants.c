#include "quants.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

float f16_to_f32_lookup[65536];

static float convert_f16_to_f32_impl(NnFp16 value) {
    union { uint32_t u; float f; } magic = { (254U - 15U) << 23 };
    union { uint32_t u; float f; } inf_nan = { (127U + 16U) << 23 };
    union { uint32_t u; float f; } result;
    result.u = (value & 0x7FFFU) << 13;
    result.f *= magic.f;
    if (result.f >= inf_nan.f)
        result.u |= 255U << 23;
    result.u |= (value & 0x8000U) << 16;
    return result.f;
}

static NnFp16 convert_f32_to_f16_impl(float x) {
    int i = *(int *)&x;
    int s = (i >> 16) & 0x00008000;
    int e = ((i >> 23) & 0x000000ff) - (127 - 15);
    int m = i & 0x007fffff;
    if (e <= 0) {
        if (e < -10) return s;
        m = m | 0x00800000;
        int t = 14 - e;
        int a = (1 << (t - 1)) - 1;
        int b = (m >> t) & 1;
        m = (m + a + b) >> t;
        return s | m;
    }
    if (e == 0xff - (127 - 15)) {
        if (m == 0) return s | 0x7c00;
        m >>= 13;
        return s | 0x7c00 | m | (m == 0);
    }
    m = m + 0x00000fff + ((m >> 13) & 1);
    if (m & 0x00800000) {
        m = 0;
        e += 1;
    }
    return s | (e << 10) | (m >> 13);
}

void init_quants(void) {
    for (NnUint i = 0; i < 65536; i++)
        f16_to_f32_lookup[i] = convert_f16_to_f32_impl((NnFp16)i);
}

float convert_f16_to_f32(NnFp16 value) {
    return f16_to_f32_lookup[value];
}

NnFp16 convert_f32_to_f16(float x) {
    return convert_f32_to_f16_impl(x);
}

void dequantize_q40_to_f32(const NnBlockQ40 *input, float *output, NnUint n,
                           NnUint n_threads, NnUint thread_index) {
    NnUint n_blocks = n / Q40_BLOCK_SIZE;
    SPLIT_THREADS(start, end, n_blocks, n_threads, thread_index);

    for (NnUint i = start; i < end; i++) {
        const NnBlockQ40 *b = &input[i];
        const float d = convert_f16_to_f32(b->d);

        for (int j = 0; j < Q40_BLOCK_SIZE / 2; ++j) {
            const int x0 = (b->qs[j] & 0x0F) - 8;
            const int x1 = (b->qs[j] >> 4) - 8;

            output[i * Q40_BLOCK_SIZE + j] = x0 * d;
            output[i * Q40_BLOCK_SIZE + j + Q40_BLOCK_SIZE / 2] = x1 * d;
        }
    }
}

void dequantize_q80_to_f32(const NnBlockQ80 *input, float *output, NnUint n,
                           NnUint n_threads, NnUint thread_index) {
    NnUint n_blocks = n / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, n_blocks, n_threads, thread_index);

    for (NnUint i = start; i < end; i++) {
        const float d = convert_f16_to_f32(input[i].d);
        for (int j = 0; j < Q80_BLOCK_SIZE; j++) {
            output[i * Q80_BLOCK_SIZE + j] = input[i].qs[j] * d;
        }
    }
}

void quantize_f32_to_q80(const float *input, NnBlockQ80 *output, NnUint n,
                         NnUint n_threads, NnUint thread_index) {
    NnUint n_blocks = n / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, n_blocks, n_threads, thread_index);

    for (NnUint i = start; i < end; i++) {
        const float *x = &input[i * Q80_BLOCK_SIZE];
        NnBlockQ80 *y = &output[i];

        float amax = 0.0f;
        for (NnUint j = 0; j < Q80_BLOCK_SIZE; j++) {
            float v = fabsf(x[j]);
            if (v > amax) amax = v;
        }

        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y->d = convert_f32_to_f16(d);
        for (NnUint j = 0; j < Q80_BLOCK_SIZE; ++j) {
            y->qs[j] = (int8_t)roundf(x[j] * id);
        }
    }
}

void quantize_f32_to_q40(const float *input, NnBlockQ40 *output, NnUint n,
                         NnUint n_threads, NnUint thread_index) {
    NnUint n_blocks = n / Q40_BLOCK_SIZE;
    SPLIT_THREADS(start, end, n_blocks, n_threads, thread_index);

    for (NnUint i = start; i < end; i++) {
        float amax = 0.0f;
        float max = 0.0f;
        for (NnUint j = 0; j < Q40_BLOCK_SIZE; j++) {
            float v = input[i * Q40_BLOCK_SIZE + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max = v;
            }
        }

        const float d = max / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;

        NnBlockQ40 *o = &output[i];
        o->d = convert_f32_to_f16(d);
        for (NnUint j = 0; j < Q40_BLOCK_SIZE / 2; j++) {
            const float x0 = input[i * Q40_BLOCK_SIZE + j] * id;
            const float x1 = input[i * Q40_BLOCK_SIZE + Q40_BLOCK_SIZE / 2 + j] * id;

            uint8_t xi0 = (int8_t)(x0 + 8.5f);
            uint8_t xi1 = (int8_t)(x1 + 8.5f);
            if (xi0 > 15) xi0 = 15;
            if (xi1 > 15) xi1 = 15;

            o->qs[j] = xi0 | (xi1 << 4);
        }
    }
}

const char *float_type_to_string(int type) {
    switch (type) {
        case F_UNK: return "F_UNK";
        case F_32:  return "F_32";
        case F_16:  return "F_16";
        case F_Q40: return "F_Q40";
        case F_Q80: return "F_Q80";
        default:     return "F_UNKNOWN";
    }
}
