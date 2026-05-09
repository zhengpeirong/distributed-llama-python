#include <Python.h>
#include "ops.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* llamafile SGEM (C++ wrapper, linked from sgemm_wrapper.cpp) */
extern int llamafile_sgemm_c(int64_t m, int64_t n, int64_t k,
                              const void *A, int64_t lda,
                              const void *B, int64_t ldb,
                              void *C, int64_t ldc,
                              int ith, int nth, int task,
                              int Atype, int Btype, int Ctype);
#if defined(__STDC_NO_ATOMICS__) || defined(__STDC_NO_THREADS__)
  /* No atomics/threads available - skip pthread pool */
  #define _TP_SKIP
#else
  #include <pthread.h>
  #include <stdatomic.h>
#endif

/* ===================================================================
 *  C-level persistent thread pool for CPU parallelism.
 *  Only available when pthreads + atomics are supported.
 * =================================================================== */
#ifndef _TP_SKIP

static int _tp_size = 0;
static pthread_t *_tp_threads = NULL;

typedef struct {
    void (*func)(void *ctx, NnUint ti);
    void  *ctx;
    NnUint n_threads;
    atomic_uint workers_done;
    atomic_uint go;
    NnUint *range_starts;
    NnUint *range_ends;
} TpWork;

static TpWork _tp_work = {NULL, NULL, 0, 0, 0, NULL, NULL};

static void *_tp_worker(void *arg) {
    NnUint ti = (NnUint)(uintptr_t)arg;
    while (1) {
        while (atomic_load(&_tp_work.go) == 0) {}
        if (_tp_work.func && ti < _tp_work.n_threads) {
            _tp_work.func(_tp_work.ctx, ti);
        }
        atomic_fetch_add(&_tp_work.workers_done, 1);
        while (atomic_load(&_tp_work.go) == 1) {}
    }
    return NULL;
}

void init_mt(NnUint n_threads) {
    if (n_threads <= 1 || _tp_size > 0) return;
    _tp_size = n_threads;
    _tp_threads = (pthread_t *)malloc((n_threads - 1) * sizeof(pthread_t));
    _tp_work.range_starts = (NnUint *)malloc(n_threads * sizeof(NnUint));
    _tp_work.range_ends   = (NnUint *)malloc(n_threads * sizeof(NnUint));
    atomic_store(&_tp_work.go, 0);
    atomic_store(&_tp_work.workers_done, 0);
    for (NnUint i = 1; i < n_threads; i++) {
        pthread_create(&_tp_threads[i - 1], NULL, _tp_worker, (void *)(uintptr_t)i);
    }
}

static inline void _tp_run(void (*func)(void *, NnUint), void *ctx,
                           NnUint n_threads, NnUint range_len) {
    if (_tp_size <= 0 || n_threads <= 1) { func(ctx, 0); return; }
    if (n_threads > (NnUint)_tp_size) n_threads = (NnUint)_tp_size;
    NnUint slice = range_len / n_threads;
    NnUint rest  = range_len % n_threads;
    NnUint off = 0;
    for (NnUint ti = 0; ti < n_threads; ti++) {
        NnUint s = slice + (ti < rest ? 1 : 0);
        _tp_work.range_starts[ti] = off;
        _tp_work.range_ends[ti]   = off + s;
        off += s;
    }
    _tp_work.func = func; _tp_work.ctx = ctx;
    _tp_work.n_threads = n_threads;
    atomic_store(&_tp_work.workers_done, 0);
    atomic_store(&_tp_work.go, 1);
    func(ctx, 0);
    NnUint expected = n_threads - 1;
    while (atomic_load(&_tp_work.workers_done) < expected) {}
    _tp_work.func = NULL; _tp_work.ctx = NULL;
    atomic_store(&_tp_work.go, 0);
}

#else /* _TP_SKIP */
void init_mt(NnUint n_threads) { (void)n_threads; }
#endif /* _TP_SKIP */

/* Macro: call _tp_run if thread pool is active, define args locally.
 * Usage: _MT_DISPATCH(worker_func, args_init, n_threads, range_len, fallback_code)
 *   - args_init: statement initializing a local struct passed as ctx
 *   - worker_func: void (*)(void *ctx, NnUint ti)
 *   - range_len: total work items
 *   - fallback_code: code block executed when not using MT (in a do-while(0))
 */
#define _MT_DISPATCH(wfunc, args_init, nt, rlen, fallback) \
    do { \
        if (_tp_size > 0 && (nt) > 1) { \
            args_init; \
            _tp_run(wfunc, &_mt_args, (nt), (rlen)); \
        } else { \
            fallback \
        } \
    } while (0)

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* ===================================================================
 *  ARM NEON helpers (ported from nn-cpu-ops.cpp)
 * =================================================================== */
#if defined(__ARM_NEON)
static inline float32x4_t _neon_expf(float32x4_t x) {
    const float32x4_t ln2     = vdupq_n_f32(0.69314718056f);
    const float32x4_t inv_ln2 = vdupq_n_f32(1.44269504089f);
    const float32x4_t c1 = vdupq_n_f32(1.0f);
    const float32x4_t c2 = vdupq_n_f32(0.5f);
    const float32x4_t c3 = vdupq_n_f32(0.1666666667f);
    const float32x4_t c4 = vdupq_n_f32(0.04166666667f);
    const float32x4_t c5 = vdupq_n_f32(0.008333333333f);
    x = vminq_f32(x, vdupq_n_f32(88.0f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
    float32x4_t kf = vaddq_f32(vmulq_f32(x, inv_ln2), vdupq_n_f32(0.5f));
    int32x4_t k = vcvtq_s32_f32(kf);
    kf = vcvtq_f32_s32(k);
    float32x4_t f = vmlsq_f32(x, kf, ln2);
    float32x4_t f2 = vmulq_f32(f, f);
    float32x4_t f3 = vmulq_f32(f2, f);
    float32x4_t f4 = vmulq_f32(f3, f);
    float32x4_t f5 = vmulq_f32(f4, f);
    float32x4_t p = c1;
    p = vaddq_f32(p, f);
    p = vaddq_f32(p, vmulq_f32(c2, f2));
    p = vaddq_f32(p, vmulq_f32(c3, f3));
    p = vaddq_f32(p, vmulq_f32(c4, f4));
    p = vaddq_f32(p, vmulq_f32(c5, f5));
    int32x4_t pow2k = vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(pow2k));
}
#endif

/* ===================================================================
 *  AVX2 SIMD helpers (ported from nn-cpu-ops.cpp)
 * =================================================================== */
#if defined(__AVX2__)
static inline float _avx2_hsum(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

static inline float _avx2_hmax(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1,1,1,1)));
    return _mm_cvtss_f32(m);
}

static inline __m256 _avx2_expf(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 c2 = _mm256_set1_ps(0.2402265069591007f);
    const __m256 c3 = _mm256_set1_ps(0.05550410866482158f);
    const __m256 c4 = _mm256_set1_ps(0.009618129107628477f);
    __m256 y = _mm256_mul_ps(x, log2e);
    __m256i n = _mm256_cvtps_epi32(y);
    __m256 nf = _mm256_cvtepi32_ps(n);
    __m256 f = _mm256_sub_ps(y, nf);
    __m256 p = c4;
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);
    __m256i exp = _mm256_add_epi32(n, _mm256_set1_epi32(127));
    exp = _mm256_slli_epi32(exp, 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(exp));
}
#endif

/*
 *  Quantised-block types mirroring quants.h -- redeclared locally because
 *  ops.h and quants.h share the NnFloatType enum and cannot be included
 *  together without C-name collisions on the enum tag.
 */
typedef struct { uint16_t d; uint8_t qs[16]; } NnBlockQ40;
typedef struct { uint16_t d; int8_t  qs[32]; } NnBlockQ80;

/* --------------------------------------------------------------------------
 *  f16 lookup table and conversion (local copy; avoids dependency on quants)
 * -------------------------------------------------------------------------- */
static float _f16_lookup[65536];
static int _f16_lookup_ready = 0;

#ifndef likely
#define likely(x) (x)
#endif

static float _convert_f16_to_f32(uint16_t value) {
    if (likely(_f16_lookup_ready))
        return _f16_lookup[value];
    /* one-time init */
    for (int i = 0; i < 65536; i++) {
        int sign = (i >> 15) & 1;
        int exp  = (i >> 10) & 0x1F;
        int mant = i & 0x3FF;
        if (exp == 0) {
            _f16_lookup[i] = (sign ? -1.0f : 1.0f) * ldexpf((float)mant, -24);
        } else if (exp == 31) {
            _f16_lookup[i] = mant ? NAN : (sign ? -INFINITY : INFINITY);
        } else {
            _f16_lookup[i] = (sign ? -1.0f : 1.0f) * ldexpf(1.0f + (float)mant / 1024.0f, exp - 15);
        }
    }
    _f16_lookup_ready = 1;
    return _f16_lookup[value];
}

/* --------------------------------------------------------------------------
 *  Scalar softmax: x[i] = exp(x[i] - max) / sum(exp(x[j] - max))
 * -------------------------------------------------------------------------- */
void op_softmax(float *x, NnUint size) {
    if (size == 0) return;
    float max_val, sum = 0.0f;
    NnUint i = 0;

#if defined(__ARM_NEON)
    NnUint neon_end;
    if (size >= 4) {
        float32x4_t mx = vld1q_f32(&x[0]);
        neon_end = size - (size % 4);
        for (i = 4; i < neon_end; i += 4)
            mx = vmaxq_f32(mx, vld1q_f32(&x[i]));
        max_val = vmaxvq_f32(mx);
    } else { max_val = x[0]; i = 1; neon_end = 0; }
    for (; i < size; i++) if (x[i] > max_val) max_val = x[i];
    {
        float32x4_t mv = vdupq_n_f32(max_val);
        float32x4_t sv = vdupq_n_f32(0.0f);
        i = 0;
        for (; i + 4 <= size; i += 4) {
            float32x4_t v = vsubq_f32(vld1q_f32(&x[i]), mv);
            v = _neon_expf(v);
            vst1q_f32(&x[i], v);
            sv = vaddq_f32(sv, v);
        }
        float32x2_t slo = vadd_f32(vget_low_f32(sv), vget_high_f32(sv));
        sum = vget_lane_f32(slo, 0) + vget_lane_f32(slo, 1);
    }
#elif defined(__AVX2__)
    NnUint avx_end = size - (size % 8);
    if (avx_end >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        for (i = 8; i < avx_end; i += 8) {
            mx = _mm256_max_ps(mx, _mm256_loadu_ps(&x[i]));
        }
        max_val = _avx2_hmax(mx);
    } else {
        max_val = x[0]; i = 1;
    }
#else
    max_val = x[0]; i = 1;
#endif
    for (; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

#if defined(__AVX2__)
    __m256 mv = _mm256_set1_ps(max_val);
    __m256 sv = _mm256_setzero_ps();
    i = 0;
    for (; i < avx_end; i += 8) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(&x[i]), mv);
        v = _avx2_expf(v);
        _mm256_storeu_ps(&x[i], v);
        sv = _mm256_add_ps(sv, v);
    }
    sum = _avx2_hsum(sv);
#endif
    for (; i < size; i++) {
        float v = expf(x[i] - max_val);
        x[i] = v;
        sum += v;
    }

    if (sum == 0.0f) sum = 0.000001f;
    float inv = 1.0f / sum;

#if defined(__ARM_NEON)
    {
        float32x4_t iv = vdupq_n_f32(inv);
        for (i = 0; i + 4 <= size; i += 4)
            vst1q_f32(&x[i], vmulq_f32(vld1q_f32(&x[i]), iv));
    }
#elif defined(__AVX2__)
    __m256 iv = _mm256_set1_ps(inv);
    for (i = 0; i < avx_end; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        _mm256_storeu_ps(&x[i], _mm256_mul_ps(v, iv));
    }
#endif
    for (; i < size; i++) x[i] *= inv;
}

/* alias */
void op_softmax_f32(float *x, NnUint size) {
    op_softmax(x, size);
}

/* --------------------------------------------------------------------------
 *  Embedding lookup: input[b] encodes a token id as a float; copy the
 *  corresponding weight row into the output.
 *
 *  Layout:
 *    weight  = [vocab_size][dim]         (row-major)
 *    input   = [batch_size]              (one float per batch, token id)
 *    output  = [batch_size][dim]
 * -------------------------------------------------------------------------- */
void op_embedding_f32(float *output, const float *weight, const float *input,
                      NnUint dim, NnUint batch_size,
                      NnUint n_threads, NnUint thread_index) {
    NnUint total = batch_size * dim;
    SPLIT_THREADS(start, end, total, n_threads, thread_index);

    for (NnUint i = start; i < end; i++) {
        NnUint b = i / dim;                /* batch index           */
        NnUint j = i % dim;                /* position within dim   */
        NnUint token = (NnUint)input[b];   /* float -> token id     */
        output[i] = weight[token * dim + j];
    }
}

/* --------------------------------------------------------------------------
 *  Compute 1 / sqrt(mean(x^2) + epsilon) for each column.
 *
 *  Layout:
 *    input   = [batch_size][n_columns * col_size]
 *    output  = [batch_size][n_columns]  (one scalar per column per batch)
 * -------------------------------------------------------------------------- */
void op_inv_rms(float *output, const float *input, NnUint col_size,
                NnUint n_columns, float epsilon, NnUint batch_size,
                NnUint n_threads, NnUint thread_index) {
    NnUint total_cols = batch_size * n_columns;
    NnUint row_stride  = n_columns * col_size;

    SPLIT_THREADS(start, end, total_cols, n_threads, thread_index);

    for (NnUint idx = start; idx < end; idx++) {
        NnUint b = idx / n_columns;
        NnUint c = idx % n_columns;

        const float *col = &input[b * row_stride + c * col_size];

        float sum_sq = 0.0f;
        for (NnUint j = 0U; j < col_size; j++) {
            sum_sq += col[j] * col[j];
        }
        sum_sq = sum_sq / (float)col_size + epsilon;

        output[idx] = 1.0f / sqrtf(sum_sq);
    }
}

/* --------------------------------------------------------------------------
 *  RMS normalization: output[i] = weight[i % col_size] * inv_rms[col] * input[i]
 *
 *  Layout:
 *    input    = [batch_size][size]          size = n_columns * col_size
 *    weight   = [col_size]
 *    inv_rms  = [batch_size][n_columns]     (computed by op_inv_rms)
 *    output   = [batch_size][size]
 * -------------------------------------------------------------------------- */
void op_rms_norm_f32(float *output, const float *input, const float *weight,
                     const float *inv_rms, NnUint size, NnUint n_columns,
                     NnUint batch_size, NnUint n_threads, NnUint thread_index) {
    NnUint col_size = size / n_columns;
    NnUint total    = batch_size * size;

    SPLIT_THREADS(start, end, total, n_threads, thread_index);
    NnUint i = start;

#if defined(__ARM_NEON)
    NnUint count = end - start;
    NnUint neon_end = end - (count % 4);
    for (; i < neon_end; i += 4) {
        NnUint b     = i / size;
        NnUint off   = i % size;
        NnUint col   = off / col_size;
        float32x4_t ir = vdupq_n_f32(inv_rms[b * n_columns + col]);
        float32x4_t wv = vld1q_f32(&weight[off % col_size]);
        float32x4_t iv = vld1q_f32(&input[i]);
        vst1q_f32(&output[i], vmulq_f32(vmulq_f32(wv, ir), iv));
    }
#endif
    for (; i < end; i++) {
        NnUint b     = i / size;
        NnUint off   = i % size;
        NnUint col   = off / col_size;
        NnUint w_idx = off % col_size;

        output[i] = weight[w_idx] * inv_rms[b * n_columns + col] * input[i];
    }
}

/* --------------------------------------------------------------------------
 *  Dense matrix multiply: output[b,d] = sum_j w[d,j] * x[b,j]
 *
 *  Layout:
 *    x      = [batch_size][n]
 *    w      = [d][n]   (row-major, one row per output)
 *    output = [batch_size][d]
 * -------------------------------------------------------------------------- */
void op_matmul_f32_f32_f32(float *output, const float *x, const float *w,
                           NnUint n, NnUint d, NnUint batch_size,
                           NnUint n_threads, NnUint thread_index) {
    NnUint total_out = batch_size * d;
    SPLIT_THREADS(start, end, total_out, n_threads, thread_index);

#if defined(__ARM_NEON)
    for (NnUint idx = start; idx < end; idx++) {
        NnUint b  = idx / d;
        NnUint di = idx % d;
        const float *xb = &x[b * n];
        const float *wr = &w[di * n];
        float32x4_t z = vmovq_n_f32(0.0f);
        NnUint j;
        for (j = 0U; j + 4 <= n; j += 4) {
            z = vfmaq_f32(z, vld1q_f32(&xb[j]), vld1q_f32(&wr[j]));
        }
        float sum = vaddvq_f32(z);
        for (; j < n; j++) sum += wr[j] * xb[j];
        output[idx] = sum;
    }
#elif defined(__AVX2__)
    for (NnUint idx = start; idx < end; idx++) {
        NnUint b  = idx / d;
        NnUint di = idx % d;
        const float *xb = &x[b * n];
        const float *wr = &w[di * n];
        __m256 u = _mm256_set1_ps(0.0f);
        NnUint j;
        for (j = 0U; j + 8 <= n; j += 8) {
            __m256 a = _mm256_loadu_ps(&xb[j]);
            __m256 bv = _mm256_loadu_ps(&wr[j]);
            u = _mm256_fmadd_ps(a, bv, u);
        }
        float sum = _avx2_hsum(u);
        for (; j < n; j++) sum += wr[j] * xb[j];
        output[idx] = sum;
    }
#else
    for (NnUint idx = start; idx < end; idx++) {
        NnUint b  = idx / d;
        NnUint di = idx % d;

        const float *xb = &x[b * n];
        const float *wr = &w[di * n];

        float sum = 0.0f;
        for (NnUint j = 0U; j < n; j++) {
            sum += wr[j] * xb[j];
        }
        output[idx] = sum;
    }
#endif
}

/* --------------------------------------------------------------------------
 *  Quantized matrix multiply: Q80 activations x Q40 weights -> F32 output
 *
 *  x = [batch_size][n_blocks] of NnBlockQ80  (n_blocks = n / 32)
 *  w = [d][n_blocks] of NnBlockQ40
 *
 *  For each output element (b, di):
 *    sum over blocks: s = f16tof32(wb->d) * f16tof32(xb->d)
 *    inner sum: (w0*i0 + w1*i1) * s
 *    where w0 = (qs[k] & 0x0F) - 8,  w1 = (qs[k] >> 4) - 8
 *          i0 = xb->qs[k],           i1 = xb->qs[k + 16]
 * -------------------------------------------------------------------------- */
/* Worker struct for Q80×Q40 matmul thread-pool dispatch */
typedef struct {
    float *output; const NnBlockQ80 *xq; const NnBlockQ40 *wq;
    NnUint n, d, n_blocks, batch_size;
} _Mq80q40Ctx;

static void _q80q40_worker(void *v, NnUint ti) {
    _Mq80q40Ctx *c = (_Mq80q40Ctx *)v;
    NnUint start = _tp_work.range_starts[ti];
    NnUint end   = _tp_work.range_ends[ti];
#if defined(__AVX2__)
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i bias_8  = _mm_set1_epi8(8);
    const __m256i one_16  = _mm256_set1_epi16(1);
    for (NnUint idx = start; idx < end; idx++) {
        NnUint b  = idx / c->d;
        NnUint di = idx % c->d;
        float sum = 0.0f;
        NnUint j = 0U;
        for (; j + 1 < c->n_blocks; j += 2) {
            _mm_prefetch((const char*)&c->wq[di * c->n_blocks + j + 2], _MM_HINT_T0);
            _mm_prefetch((const char*)&c->xq[b  * c->n_blocks + j + 2], _MM_HINT_T0);
            const NnBlockQ40 *wb0 = &c->wq[di * c->n_blocks + j];
            const NnBlockQ80 *xb0 = &c->xq[b  * c->n_blocks + j];
            float s0 = _convert_f16_to_f32(wb0->d) * _convert_f16_to_f32(xb0->d);
            __m128i wp0 = _mm_loadu_si128((const __m128i*)wb0->qs);
            __m128i w0_0 = _mm_sub_epi8(_mm_and_si128(wp0, mask_0f), bias_8);
            __m128i w1_0 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(wp0, 4), mask_0f), bias_8);
            __m256i i1_0_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)xb0->qs));
            __m256i i2_0_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(xb0->qs + 16)));
            __m256i s32_0 = _mm256_madd_epi16(
                _mm256_add_epi16(
                    _mm256_mullo_epi16(_mm256_cvtepi8_epi16(w0_0), i1_0_16),
                    _mm256_mullo_epi16(_mm256_cvtepi8_epi16(w1_0), i2_0_16)), one_16);
            __m128i sl0 = _mm256_castsi256_si128(s32_0);
            __m128i sh0 = _mm256_extracti128_si256(s32_0, 1);
            sl0 = _mm_add_epi32(_mm_add_epi32(sl0, sh0), _mm_hadd_epi32(sl0, sl0));
            sum += (float)_mm_extract_epi32(sl0, 0) * s0;
            /* block j+1 */
            const NnBlockQ40 *wb1 = &c->wq[di * c->n_blocks + j + 1];
            const NnBlockQ80 *xb1 = &c->xq[b  * c->n_blocks + j + 1];
            float s1 = _convert_f16_to_f32(wb1->d) * _convert_f16_to_f32(xb1->d);
            __m128i wp1 = _mm_loadu_si128((const __m128i*)wb1->qs);
            __m128i w0_1 = _mm_sub_epi8(_mm_and_si128(wp1, mask_0f), bias_8);
            __m128i w1_1 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(wp1, 4), mask_0f), bias_8);
            __m256i i1_1_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)xb1->qs));
            __m256i i2_1_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(xb1->qs + 16)));
            __m256i s32_1 = _mm256_madd_epi16(
                _mm256_add_epi16(
                    _mm256_mullo_epi16(_mm256_cvtepi8_epi16(w0_1), i1_1_16),
                    _mm256_mullo_epi16(_mm256_cvtepi8_epi16(w1_1), i2_1_16)), one_16);
            __m128i sl1 = _mm256_castsi256_si128(s32_1);
            __m128i sh1 = _mm256_extracti128_si256(s32_1, 1);
            sl1 = _mm_add_epi32(_mm_add_epi32(sl1, sh1), _mm_hadd_epi32(sl1, sl1));
            sum += (float)_mm_extract_epi32(sl1, 0) * s1;
        }
        for (; j < c->n_blocks; j++) {
            const NnBlockQ40 *wb = &c->wq[di * c->n_blocks + j];
            const NnBlockQ80 *xb = &c->xq[b  * c->n_blocks + j];
            float s = _convert_f16_to_f32(wb->d) * _convert_f16_to_f32(xb->d);
            __m128i wp = _mm_loadu_si128((const __m128i*)wb->qs);
            __m128i w0 = _mm_sub_epi8(_mm_and_si128(wp, mask_0f), bias_8);
            __m128i w1 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(wp, 4), mask_0f), bias_8);
            __m256i sp = _mm256_add_epi16(
                _mm256_mullo_epi16(_mm256_cvtepi8_epi16(w0), _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)xb->qs))),
                _mm256_mullo_epi16(_mm256_cvtepi8_epi16(w1), _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(xb->qs + 16)))));
            __m256i s32 = _mm256_madd_epi16(sp, one_16);
            __m128i sl = _mm_add_epi32(_mm256_castsi256_si128(s32), _mm256_extracti128_si256(s32, 1));
            sum += (float)_mm_extract_epi32(sl, 0) * s;
        }
        c->output[idx] = sum;
    }
#else
    for (NnUint idx = start; idx < end; idx++) {
        NnUint b  = idx / c->d; NnUint di = idx % c->d;
        float sum = 0.0f;
        for (NnUint j = 0U; j < c->n_blocks; j++) {
            const NnBlockQ40 *wb = &c->wq[di * c->n_blocks + j];
            const NnBlockQ80 *xb = &c->xq[b  * c->n_blocks + j];
            float s = _convert_f16_to_f32(wb->d) * _convert_f16_to_f32(xb->d);
            for (NnUint k = 0U; k < 16U; k++) {
                int w0 = (int)(wb->qs[k] & 0x0FU) - 8, w1 = (int)(wb->qs[k] >> 4) - 8;
                sum += (float)(w0 * (int)xb->qs[k] + w1 * (int)xb->qs[k + 16]) * s;
            }
        }
        c->output[idx] = sum;
    }
#endif
}

void op_matmul_q80_q40_f32(float *output, const void *x, const void *w,
                           NnUint n, NnUint d, NnUint batch_size,
                           NnUint n_threads, NnUint thread_index) {
    const NnBlockQ80 *xq = (const NnBlockQ80 *)x;
    const NnBlockQ40 *wq = (const NnBlockQ40 *)w;
    NnUint n_blocks = n / Q40_BLOCK_SIZE;
    NnUint total_out = batch_size * d;

    /* Try llamafile SGEM first (handles entire matmul, multi-threaded) */
    if (batch_size >= 2 && thread_index == 0) {
        if (llamafile_sgemm_c(
                d, batch_size, n_blocks,
                wq, n_blocks,   /* A = Q40 weight, lda = n_blocks */
                xq, n_blocks,   /* B = Q80 input,  ldb = n_blocks */
                output, d,      /* C = F32 output, ldc = d */
                (int)thread_index, (int)n_threads, 0,
                2, 3, 0))       /* Atype=F_Q40=2, Btype=F_Q80=3, Ctype=F_32=0 */
        {
            return;
        }
    }

    _Mq80q40Ctx _mt_args;
    _mt_args = (_Mq80q40Ctx){output, xq, wq, n, d, n_blocks, batch_size};
    if (_tp_size > 0 && n_threads > 1) {
        _tp_run(_q80q40_worker, &_mt_args, n_threads, total_out);
    } else {
        SPLIT_THREADS(start, end, total_out, n_threads, thread_index);

#if defined(__ARM_NEON)
    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t s8b  = vdupq_n_s8(0x8);

    for (NnUint idx = start; idx < end; idx++) {
        NnUint b  = idx / d;
        NnUint di = idx % d;
        float32x4_t sumv0 = vmovq_n_f32(0.0f);
        float32x4_t sumv1 = vmovq_n_f32(0.0f);
        float32x4_t sumv2 = vmovq_n_f32(0.0f);
        float32x4_t sumv3 = vmovq_n_f32(0.0f);
        NnUint j = 0U;

#if defined(__ARM_FEATURE_DOTPROD)
        for (; j + 3 < n_blocks; j += 4) {
            __builtin_prefetch(&wq[di * n_blocks + j + 4]);
            __builtin_prefetch(&xq[b  * n_blocks + j + 4]);
            const NnBlockQ40 *w0 = &wq[di * n_blocks + j];
            const NnBlockQ40 *w1 = &wq[di * n_blocks + j + 1];
            const NnBlockQ40 *w2 = &wq[di * n_blocks + j + 2];
            const NnBlockQ40 *w3 = &wq[di * n_blocks + j + 3];
            const NnBlockQ80 *x0 = &xq[b  * n_blocks + j];
            const NnBlockQ80 *x1 = &xq[b  * n_blocks + j + 1];
            const NnBlockQ80 *x2 = &xq[b  * n_blocks + j + 2];
            const NnBlockQ80 *x3 = &xq[b  * n_blocks + j + 3];
            int8x16_t w0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w0->qs), m4b)), s8b);
            int8x16_t w0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w0->qs), 4)), s8b);
            int8x16_t w1l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w1->qs), m4b)), s8b);
            int8x16_t w1h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w1->qs), 4)), s8b);
            int8x16_t w2l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w2->qs), m4b)), s8b);
            int8x16_t w2h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w2->qs), 4)), s8b);
            int8x16_t w3l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w3->qs), m4b)), s8b);
            int8x16_t w3h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w3->qs), 4)), s8b);
            const int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0l, vld1q_s8(x0->qs)), w0h, vld1q_s8(x0->qs + 16));
            const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w1l, vld1q_s8(x1->qs)), w1h, vld1q_s8(x1->qs + 16));
            const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w2l, vld1q_s8(x2->qs)), w2h, vld1q_s8(x2->qs + 16));
            const int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w3l, vld1q_s8(x3->qs)), w3h, vld1q_s8(x3->qs + 16));
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), _convert_f16_to_f32(w0->d) * _convert_f16_to_f32(x0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), _convert_f16_to_f32(w1->d) * _convert_f16_to_f32(x1->d));
            sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(p2), _convert_f16_to_f32(w2->d) * _convert_f16_to_f32(x2->d));
            sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(p3), _convert_f16_to_f32(w3->d) * _convert_f16_to_f32(x3->d));
        }
#else
        for (; j + 1 < n_blocks; j += 2) {
            const NnBlockQ40 *w0 = &wq[di * n_blocks + j];
            const NnBlockQ40 *w1 = &wq[di * n_blocks + j + 1];
            const NnBlockQ80 *x0 = &xq[b  * n_blocks + j];
            const NnBlockQ80 *x1 = &xq[b  * n_blocks + j + 1];
            const uint8x16_t w0qs = vld1q_u8(w0->qs), w1qs = vld1q_u8(w1->qs);
            int8x16_t w0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w0qs, m4b)), s8b);
            int8x16_t w0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w0qs, 4)), s8b);
            int8x16_t w1l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w1qs, m4b)), s8b);
            int8x16_t w1h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w1qs, 4)), s8b);
            const int8x16_t x0l = vld1q_s8(x0->qs), x0h = vld1q_s8(x0->qs + 16);
            const int8x16_t x1l = vld1q_s8(x1->qs), x1h = vld1q_s8(x1->qs + 16);
            const int16x8_t pl0l = vmull_s8(vget_low_s8(w0l), vget_low_s8(x0l));
            const int16x8_t pl0h = vmull_s8(vget_high_s8(w0l), vget_high_s8(x0l));
            const int16x8_t ph0l = vmull_s8(vget_low_s8(w0h), vget_low_s8(x0h));
            const int16x8_t ph0h = vmull_s8(vget_high_s8(w0h), vget_high_s8(x0h));
            const int16x8_t pl1l = vmull_s8(vget_low_s8(w1l), vget_low_s8(x1l));
            const int16x8_t pl1h = vmull_s8(vget_high_s8(w1l), vget_high_s8(x1l));
            const int16x8_t ph1l = vmull_s8(vget_low_s8(w1h), vget_low_s8(x1h));
            const int16x8_t ph1h = vmull_s8(vget_high_s8(w1h), vget_high_s8(x1h));
            const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
            const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
            const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
            const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), _convert_f16_to_f32(w0->d) * _convert_f16_to_f32(x0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), _convert_f16_to_f32(w1->d) * _convert_f16_to_f32(x1->d));
        }
#endif
        /* tail */
        for (; j < n_blocks; j++) {
            const NnBlockQ40 *wb = &wq[di * n_blocks + j];
            const NnBlockQ80 *xb = &xq[b  * n_blocks + j];
            const uint8x16_t wqs = vld1q_u8(wb->qs);
            const int8x16_t wl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(wqs, m4b)), s8b);
            const int8x16_t wh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(wqs, 4)), s8b);
            const int8x16_t xl = vld1q_s8(xb->qs), xh = vld1q_s8(xb->qs + 16);
#if defined(__ARM_FEATURE_DOTPROD)
            const int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wl, xl), wh, xh);
#else
            const int16x8_t pll = vmull_s8(vget_low_s8(wl), vget_low_s8(xl));
            const int16x8_t plh = vmull_s8(vget_high_s8(wl), vget_high_s8(xl));
            const int16x8_t phl = vmull_s8(vget_low_s8(wh), vget_low_s8(xh));
            const int16x8_t phh = vmull_s8(vget_high_s8(wh), vget_high_s8(xh));
            const int32x4_t pl = vaddq_s32(vpaddlq_s16(pll), vpaddlq_s16(plh));
            const int32x4_t ph = vaddq_s32(vpaddlq_s16(phl), vpaddlq_s16(phh));
            const int32x4_t p = vaddq_s32(pl, ph);
#endif
            float s = _convert_f16_to_f32(wb->d) * _convert_f16_to_f32(xb->d);
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), s);
        }
        output[idx] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
    }
#elif defined(__AVX2__)
    if (batch_size <= 4) {
        /* --- bs=1 fast path: row-wise dot product -------------------- */
        const __m128i m0f = _mm_set1_epi8(0x0F);
        const __m128i b8  = _mm_set1_epi8(8);
        const __m256i o16 = _mm256_set1_epi16(1);

        for (NnUint idx = start; idx < end; idx++) {
            NnUint b  = idx / d;
            NnUint di = idx % d;
            float sum = 0.0f;
            NnUint j = 0U;

            for (; j + 1 < n_blocks; j += 2) {
                _mm_prefetch((const char*)&wq[di * n_blocks + j + 2], _MM_HINT_T0);
                _mm_prefetch((const char*)&xq[b  * n_blocks + j + 2], _MM_HINT_T0);

                /* block j */
                const NnBlockQ40 *wb0 = &wq[di * n_blocks + j];
                const NnBlockQ80 *xb0 = &xq[b  * n_blocks + j];
                float s0 = _convert_f16_to_f32(wb0->d) * _convert_f16_to_f32(xb0->d);
                __m128i wp0 = _mm_loadu_si128((const __m128i*)wb0->qs);
                __m128i w0l = _mm_sub_epi8(_mm_and_si128(wp0, m0f), b8);
                __m128i w0h = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(wp0, 4), m0f), b8);
                __m256i w0l16 = _mm256_cvtepi8_epi16(w0l), w0h16 = _mm256_cvtepi8_epi16(w0h);
                __m128i xl = _mm_loadu_si128((const __m128i*)xb0->qs);
                __m128i xh = _mm_loadu_si128((const __m128i*)(xb0->qs + 16));
                __m256i xl16 = _mm256_cvtepi8_epi16(xl), xh16 = _mm256_cvtepi8_epi16(xh);
                __m256i sp = _mm256_add_epi16(_mm256_mullo_epi16(w0l16, xl16), _mm256_mullo_epi16(w0h16, xh16));
                __m256i s32 = _mm256_madd_epi16(sp, o16);
                __m128i sl = _mm256_castsi256_si128(s32), sh = _mm256_extracti128_si256(s32, 1);
                sl = _mm_add_epi32(sl, sh);
                sl = _mm_hadd_epi32(sl, sl); sl = _mm_hadd_epi32(sl, sl);
                float a0 = (float)_mm_extract_epi32(sl, 0) * s0;

                /* block j+1 */
                const NnBlockQ40 *wb1 = &wq[di * n_blocks + j + 1];
                const NnBlockQ80 *xb1 = &xq[b  * n_blocks + j + 1];
                float s1 = _convert_f16_to_f32(wb1->d) * _convert_f16_to_f32(xb1->d);
                __m128i wp1 = _mm_loadu_si128((const __m128i*)wb1->qs);
                __m128i w1l = _mm_sub_epi8(_mm_and_si128(wp1, m0f), b8);
                __m128i w1h = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(wp1, 4), m0f), b8);
                __m256i w1l16 = _mm256_cvtepi8_epi16(w1l), w1h16 = _mm256_cvtepi8_epi16(w1h);
                __m128i x1l = _mm_loadu_si128((const __m128i*)xb1->qs);
                __m128i x1h = _mm_loadu_si128((const __m128i*)(xb1->qs + 16));
                __m256i x1l16 = _mm256_cvtepi8_epi16(x1l), x1h16 = _mm256_cvtepi8_epi16(x1h);
                __m256i sp1 = _mm256_add_epi16(_mm256_mullo_epi16(w1l16, x1l16), _mm256_mullo_epi16(w1h16, x1h16));
                __m256i s32_1 = _mm256_madd_epi16(sp1, o16);
                __m128i s1l = _mm256_castsi256_si128(s32_1), s1h = _mm256_extracti128_si256(s32_1, 1);
                s1l = _mm_add_epi32(s1l, s1h);
                s1l = _mm_hadd_epi32(s1l, s1l); s1l = _mm_hadd_epi32(s1l, s1l);
                sum += a0 + (float)_mm_extract_epi32(s1l, 0) * s1;
            }
            for (; j < n_blocks; j++) {
                const NnBlockQ40 *wb = &wq[di * n_blocks + j];
                const NnBlockQ80 *xb = &xq[b  * n_blocks + j];
                float s = _convert_f16_to_f32(wb->d) * _convert_f16_to_f32(xb->d);
                __m128i wp = _mm_loadu_si128((const __m128i*)wb->qs);
                __m128i wl = _mm_sub_epi8(_mm_and_si128(wp, m0f), b8);
                __m128i wh = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(wp, 4), m0f), b8);
                __m128i xl0 = _mm_loadu_si128((const __m128i*)xb->qs);
                __m128i xh0 = _mm_loadu_si128((const __m128i*)(xb->qs + 16));
                __m256i sp = _mm256_add_epi16(
                    _mm256_mullo_epi16(_mm256_cvtepi8_epi16(wl), _mm256_cvtepi8_epi16(xl0)),
                    _mm256_mullo_epi16(_mm256_cvtepi8_epi16(wh), _mm256_cvtepi8_epi16(xh0)));
                __m256i s32 = _mm256_madd_epi16(sp, o16);
                __m128i sl = _mm256_castsi256_si128(s32), sh = _mm256_extracti128_si256(s32, 1);
                sl = _mm_add_epi32(sl, sh);
                sl = _mm_hadd_epi32(sl, sl); sl = _mm_hadd_epi32(sl, sl);
                sum += (float)_mm_extract_epi32(sl, 0) * s;
            }
            output[idx] = sum;
        }
        return;
    }

    /* ================================================================
     *  TILED path (bs > 1) – outer loop over blocks keeps weight
     *  data in L1 cache for reuse across batch elements.
     *  TILE_D rows of weight (TILE_D * 18 bytes) stay in L1 while
     *  all batch elements share them.
     * ================================================================ */
    #define TILE_D 64  /* output rows per tile (64*18=1152 B fits L1) */
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i bias_8  = _mm_set1_epi8(8);
    const __m256i one_16  = _mm256_set1_epi16(1);

    /* Split work by output ROWS: each thread owns a range of [0..d) -- */
    SPLIT_THREADS(di_start, di_stop, d, n_threads, thread_index);

    /* Zero own output slice -------------------------------------------- */
    for (NnUint b = 0U; b < batch_size; b++)
        for (NnUint di = di_start; di < di_stop; di++)
            output[b * d + di] = 0.0f;

    /* Block-level outer loop: w[d][j] reused across batch elements ---- */
    for (NnUint j = 0U; j < n_blocks; j++) {
        _mm_prefetch((const char*)&wq[j + 2], _MM_HINT_T0);

        for (NnUint di_tile = di_start; di_tile < di_stop; ) {
            NnUint tile_end = di_tile + TILE_D;
            if (tile_end > di_stop) tile_end = di_stop;

            for (; di_tile + 1 < tile_end; di_tile += 2) {
                NnUint r0 = di_tile;
                NnUint r1 = di_tile + 1;

                /* Pre-decode both weight blocks ----------------------- */
                const NnBlockQ40 *w0 = &wq[r0 * n_blocks + j];
                const NnBlockQ40 *w1 = &wq[r1 * n_blocks + j];
                float ws0 = _convert_f16_to_f32(w0->d);
                float ws1 = _convert_f16_to_f32(w1->d);

                __m128i wp0 = _mm_loadu_si128((const __m128i*)w0->qs);
                __m128i wp1 = _mm_loadu_si128((const __m128i*)w1->qs);
                __m128i w0_lo = _mm_sub_epi8(_mm_and_si128(wp0, mask_0f), bias_8);
                __m128i w0_hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(wp0, 4), mask_0f), bias_8);
                __m128i w1_lo = _mm_sub_epi8(_mm_and_si128(wp1, mask_0f), bias_8);
                __m128i w1_hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(wp1, 4), mask_0f), bias_8);
                __m256i w0l = _mm256_cvtepi8_epi16(w0_lo);
                __m256i w0h = _mm256_cvtepi8_epi16(w0_hi);
                __m256i w1l = _mm256_cvtepi8_epi16(w1_lo);
                __m256i w1h = _mm256_cvtepi8_epi16(w1_hi);

                /* Compute contribution for every batch element -------- */
                for (NnUint b = 0U; b < batch_size; b++) {
                    const NnBlockQ80 *xb = &xq[b * n_blocks + j];
                    float xs = _convert_f16_to_f32(xb->d);
                    float s0 = ws0 * xs, s1 = ws1 * xs;

                    __m128i xi0 = _mm_loadu_si128((const __m128i*)xb->qs);
                    __m128i xi1 = _mm_loadu_si128((const __m128i*)(xb->qs + 16));
                    __m256i xi0_16 = _mm256_cvtepi8_epi16(xi0);
                    __m256i xi1_16 = _mm256_cvtepi8_epi16(xi1);

                    /* Row 0 */
                    __m256i sp0 = _mm256_add_epi16(
                        _mm256_mullo_epi16(w0l, xi0_16),
                        _mm256_mullo_epi16(w0h, xi1_16));
                    __m256i s32_0 = _mm256_madd_epi16(sp0, one_16);
                    __m128i sl0 = _mm256_castsi256_si128(s32_0);
                    __m128i sh0 = _mm256_extracti128_si256(s32_0, 1);
                    sl0 = _mm_add_epi32(sl0, sh0);
                    sl0 = _mm_hadd_epi32(sl0, sl0);
                    sl0 = _mm_hadd_epi32(sl0, sl0);
                    float acc0 = (float)_mm_extract_epi32(sl0, 0) * s0;

                    /* Row 1 */
                    __m256i sp1 = _mm256_add_epi16(
                        _mm256_mullo_epi16(w1l, xi0_16),
                        _mm256_mullo_epi16(w1h, xi1_16));
                    __m256i s32_1 = _mm256_madd_epi16(sp1, one_16);
                    __m128i sl1 = _mm256_castsi256_si128(s32_1);
                    __m128i sh1 = _mm256_extracti128_si256(s32_1, 1);
                    sl1 = _mm_add_epi32(sl1, sh1);
                    sl1 = _mm_hadd_epi32(sl1, sl1);
                    sl1 = _mm_hadd_epi32(sl1, sl1);
                    float acc1 = (float)_mm_extract_epi32(sl1, 0) * s1;

                    output[b * d + r0] += acc0;
                    output[b * d + r1] += acc1;
                }
            }

            /* Single-row tail for this tile --------------------------- */
            for (; di_tile < tile_end; di_tile++) {
                NnUint r0 = di_tile;
                const NnBlockQ40 *w0 = &wq[r0 * n_blocks + j];
                float ws = _convert_f16_to_f32(w0->d);
                __m128i wp = _mm_loadu_si128((const __m128i*)w0->qs);
                __m128i w_lo = _mm_sub_epi8(_mm_and_si128(wp, mask_0f), bias_8);
                __m128i w_hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(wp, 4), mask_0f), bias_8);
                __m256i wl = _mm256_cvtepi8_epi16(w_lo);
                __m256i wh = _mm256_cvtepi8_epi16(w_hi);

                for (NnUint b = 0U; b < batch_size; b++) {
                    const NnBlockQ80 *xb = &xq[b * n_blocks + j];
                    float xs = _convert_f16_to_f32(xb->d), s = ws * xs;
                    __m128i xi0 = _mm_loadu_si128((const __m128i*)xb->qs);
                    __m128i xi1 = _mm_loadu_si128((const __m128i*)(xb->qs + 16));
                    __m256i sp = _mm256_add_epi16(
                        _mm256_mullo_epi16(wl, _mm256_cvtepi8_epi16(xi0)),
                        _mm256_mullo_epi16(wh, _mm256_cvtepi8_epi16(xi1)));
                    __m256i s32 = _mm256_madd_epi16(sp, one_16);
                    __m128i sl = _mm256_castsi256_si128(s32);
                    __m128i sh = _mm256_extracti128_si256(s32, 1);
                    sl = _mm_add_epi32(sl, sh);
                    sl = _mm_hadd_epi32(sl, sl);
                    sl = _mm_hadd_epi32(sl, sl);
                    output[b * d + r0] += (float)_mm_extract_epi32(sl, 0) * s;
                }
            }
        }
    }
    #undef TILE_D
#else
    for (NnUint idx = start; idx < end; idx++) {
        NnUint b  = idx / d;
        NnUint di = idx % d;

        float sum = 0.0f;
        for (NnUint j = 0U; j < n_blocks; j++) {
            const NnBlockQ40 *wb = &wq[di * n_blocks + j];
            const NnBlockQ80 *xb = &xq[b  * n_blocks + j];

            float s = _convert_f16_to_f32(wb->d) * _convert_f16_to_f32(xb->d);

            for (NnUint k = 0U; k < 16U; k++) {
                int w0 = (int)(wb->qs[k] & 0x0FU) - 8;
                int w1 = (int)(wb->qs[k] >> 4)    - 8;
                int i0 = (int)xb->qs[k];
                int i1 = (int)xb->qs[k + 16];
                sum += (float)(w0 * i0 + w1 * i1) * s;
            }
        }
        output[idx] = sum;
    }
#endif
    }
}

/* --------------------------------------------------------------------------
 *  RoPE (LLaMA style):  pairwise complex rotation.
 *
 *  For each pair (i, i+1):
 *    fcr = cache[pos * slice_dim + shift + i]
 *    fci = cache[pos * slice_dim + shift + i + 1]
 *    x[i]   = v0 * fcr - v1 * fci
 *    x[i+1] = v0 * fci + v1 * fcr
 * -------------------------------------------------------------------------- */
void op_rope_llama(float *x, const float *cache, bool is_q, NnUint pos,
                   NnUint dim0, NnUint slice_dim, NnUint shift,
                   NnUint n_threads, NnUint thread_index) {
    (void)is_q;
    NnUint n_pairs = dim0 / 2U;
    SPLIT_THREADS(start, end, n_pairs, n_threads, thread_index);
    const float *pos_cache = &cache[pos * slice_dim + shift];

#if defined(__AVX2__)
    NnUint p = start;
    /* Process 4 pairs (8 floats) at a time */
    for (; p + 4 <= end; p += 4) {
        NnUint i = p * 2U;
        __m256 v = _mm256_loadu_ps(&x[i]);          /* [re0,im0,re1,im1,re2,im2,re3,im3] */
        __m256 c = _mm256_loadu_ps(&pos_cache[i]);  /* [fcr0,fci0,fcr1,fci1,fcr2,fci2,fcr3,fci3] */

        __m256 v_re = _mm256_moveldup_ps(v);  /* [re0,re0,re1,re1,re2,re2,re3,re3] */
        __m256 v_im = _mm256_movehdup_ps(v);  /* [im0,im0,im1,im1,im2,im2,im3,im3] */
        __m256 fcr  = _mm256_moveldup_ps(c);  /* [fcr0,fcr0,fcr1,fcr1,fcr2,fcr2,fcr3,fcr3] */
        __m256 fci  = _mm256_movehdup_ps(c);  /* [fci0,fci0,fci1,fci1,fci2,fci2,fci3,fci3] */

        __m256 new_re = _mm256_sub_ps(_mm256_mul_ps(v_re, fcr), _mm256_mul_ps(v_im, fci));
        __m256 new_im = _mm256_add_ps(_mm256_mul_ps(v_re, fci), _mm256_mul_ps(v_im, fcr));
        /* blend: even positions from new_re, odd from new_im */
        _mm256_storeu_ps(&x[i], _mm256_blend_ps(new_re, new_im, 0xAA));
    }
    /* scalar tail */
    for (; p < end; p++) {
        NnUint i = p * 2U;
        float fcr = pos_cache[i], fci = pos_cache[i + 1U];
        float v0 = x[i], v1 = x[i + 1U];
        x[i] = v0 * fcr - v1 * fci;
        x[i + 1U] = v0 * fci + v1 * fcr;
    }
#else
    for (NnUint p = start; p < end; p++) {
        NnUint i = p * 2U;
        float fcr = pos_cache[i];
        float fci = pos_cache[i + 1U];
        float v0  = x[i];
        float v1  = x[i + 1U];
        x[i]       = v0 * fcr - v1 * fci;
        x[i + 1U]  = v0 * fci + v1 * fcr;
    }
#endif
}

/* --------------------------------------------------------------------------
 *  RoPE (Falcon style):  per-head complex rotation.
 *
 *  For each head h and each j < head_dim/2:
 *    fcr = cache[pos * head_dim + j]
 *    fci = cache[pos * head_dim + j + head_dim/2]
 *    q0 = x[h*head_dim + j]
 *    q1 = x[h*head_dim + j + head_dim/2]
 *    x[h*head_dim + j]                  = q0 * fcr - q1 * fci
 *    x[h*head_dim + j + head_dim/2]     = q0 * fci + q1 * fcr
 * -------------------------------------------------------------------------- */
void op_rope_falcon(float *x, const float *cache, NnUint pos,
                    NnUint dim0, NnUint head_dim,
                    NnUint n_threads, NnUint thread_index) {
    NnUint n_heads = dim0 / head_dim;
    SPLIT_THREADS(start, end, n_heads, n_threads, thread_index);

    const float *pos_cache = &cache[pos * head_dim];

    for (NnUint h = start; h < end; h++) {
        NnUint offset = h * head_dim;

        for (NnUint j = 0U; j < head_dim / 2U; j++) {
            float fcr = pos_cache[j];
            float fci = pos_cache[j + head_dim / 2U];

            float q0 = x[offset + j];
            float q1 = x[offset + j + head_dim / 2U];

            x[offset + j]                    = q0 * fcr - q1 * fci;
            x[offset + j + head_dim / 2U]    = q0 * fci + q1 * fcr;
        }
    }
}

/* --------------------------------------------------------------------------
 *  Multi-head attention:  for each head h0:
 *    1.  Compute attention scores  q dot k / sqrt(head_dim)  over [0..pos]
 *    2.  Softmax the scores
 *    3.  Weighted sum of values
 *
 *  kv_mul = n_heads / n_kv_heads  maps query heads to kv heads.
 * -------------------------------------------------------------------------- */
void op_multihead_att(float *y, const float *q, float *att,
                      float *key_cache, float *value_cache,
                      NnUint pos, NnUint n_heads0, NnUint n_heads,
                      NnUint n_kv_heads, NnUint kv_dim0, NnUint head_dim,
                      NnUint seq_len, NnUint n_nodes, NnUint node_index,
                      NnUint n_threads, NnUint thread_index) {
    SPLIT_THREADS(h0_start, h0_end, n_heads0, n_threads, thread_index);

    NnUint  kv_mul       = n_heads / n_kv_heads;
    NnUint  kv_heads_per_node = n_kv_heads / n_nodes;
    float   head_dim_root = sqrtf((float)head_dim);

    for (NnUint h0 = h0_start; h0 < h0_end; h0++) {
        const float *hq  = &q[h0 * head_dim];
        NnUint head_idx  = (node_index * n_heads0 + h0) / kv_mul;
        NnUint local_idx = head_idx - node_index * kv_heads_per_node;
        const float *hkc = &key_cache[local_idx * head_dim];
        const float *hvc = &value_cache[local_idx * head_dim];
        float *hatt       = &att[h0 * seq_len];

        /* --- attention scores (AVX2 dot product) ------------------------ */
#if defined(__AVX2__)
        NnUint hd_avx_end = head_dim - (head_dim % 8);
        for (NnUint t = 0U; t <= pos; t++) {
            const float *posk = &hkc[t * kv_dim0];
            __m256 u = _mm256_set1_ps(0.0f);
            NnUint j;
            for (j = 0U; j < hd_avx_end; j += 8) {
                __m256 qv = _mm256_loadu_ps(&hq[j]);
                __m256 kv = _mm256_loadu_ps(&posk[j]);
                u = _mm256_fmadd_ps(qv, kv, u);
            }
            float score = _avx2_hsum(u);
            for (; j < head_dim; j++) score += hq[j] * posk[j];
            hatt[t] = score / head_dim_root;
        }
#else
        for (NnUint t = 0U; t <= pos; t++) {
            const float *posk = &hkc[t * kv_dim0];
            float score = 0.0f;
            for (NnUint j = 0U; j < head_dim; j++) {
                score += hq[j] * posk[j];
            }
            hatt[t] = score / head_dim_root;
        }
#endif

        /* --- softmax over positions [0..pos] (AVX2) --------------------- */
        NnUint n_pos = pos + 1U;
        if (n_pos > 0U) {
            float max_val;
#if defined(__AVX2__)
            NnUint np_avx_end = n_pos - (n_pos % 8);
            if (np_avx_end >= 8) {
                __m256 mx = _mm256_loadu_ps(hatt);
                NnUint t = 8;
                for (; t < np_avx_end; t += 8)
                    mx = _mm256_max_ps(mx, _mm256_loadu_ps(&hatt[t]));
                max_val = _avx2_hmax(mx);
            } else { max_val = hatt[0]; }
#else
            max_val = hatt[0];
            for (NnUint t = 1U; t < n_pos; t++)
                if (hatt[t] > max_val) max_val = hatt[t];
#endif

            float sum = 0.0f;
#if defined(__AVX2__)
            if (np_avx_end >= 8) {
                __m256 mv = _mm256_set1_ps(max_val);
                __m256 sv = _mm256_setzero_ps();
                NnUint t = 0;
                for (; t < np_avx_end; t += 8) {
                    __m256 v = _mm256_sub_ps(_mm256_loadu_ps(&hatt[t]), mv);
                    v = _avx2_expf(v);
                    _mm256_storeu_ps(&hatt[t], v);
                    sv = _mm256_add_ps(sv, v);
                }
                sum = _avx2_hsum(sv);
                for (; t < n_pos; t++) {
                    float v = expf(hatt[t] - max_val);
                    hatt[t] = v; sum += v;
                }
            } else {
                for (NnUint t = 0U; t < n_pos; t++) {
                    float v = expf(hatt[t] - max_val);
                    hatt[t] = v; sum += v;
                }
            }
#else
            for (NnUint t = 0U; t < n_pos; t++) {
                float v = expf(hatt[t] - max_val);
                hatt[t] = v; sum += v;
            }
#endif
            if (sum == 0.0f) sum = 0.000001f;
            float inv = 1.0f / sum;
#if defined(__AVX2__)
            if (np_avx_end >= 8) {
                __m256 iv = _mm256_set1_ps(inv);
                for (NnUint t = 0; t < np_avx_end; t += 8) {
                    __m256 v = _mm256_loadu_ps(&hatt[t]);
                    _mm256_storeu_ps(&hatt[t], _mm256_mul_ps(v, iv));
                }
                for (NnUint t = np_avx_end; t < n_pos; t++) hatt[t] *= inv;
            } else {
                for (NnUint t = 0U; t < n_pos; t++) hatt[t] *= inv;
            }
#else
            for (NnUint t = 0U; t < n_pos; t++) hatt[t] *= inv;
#endif
        }

        /* --- weighted sum of values (AVX2 FMA) ----------------------- */
        float *hy = &y[h0 * head_dim];
        memset(hy, 0, head_dim * sizeof(float));

#if defined(__AVX2__)
        for (NnUint t = 0U; t <= pos; t++) {
            const float *posv = &hvc[t * kv_dim0];
            __m256 psa = _mm256_set1_ps(hatt[t]);
            NnUint j = 0;
            for (; j + 8 <= head_dim; j += 8) {
                __m256 hv = _mm256_loadu_ps(&hy[j]);
                __m256 pv = _mm256_loadu_ps(&posv[j]);
                _mm256_storeu_ps(&hy[j], _mm256_fmadd_ps(psa, pv, hv));
            }
            for (; j < head_dim; j++) hy[j] += hatt[t] * posv[j];
        }
#else
        for (NnUint t = 0U; t <= pos; t++) {
            const float *posv = &hvc[t * kv_dim0];
            float posa = hatt[t];
            for (NnUint j = 0U; j < head_dim; j++) {
                hy[j] += posa * posv[j];
            }
        }
#endif
    }
}

/* --------------------------------------------------------------------------
 *  SiLU (Sigmoid Linear Unit):  x / (1 + exp(-x))
 *  Operates in-place on x.
 * -------------------------------------------------------------------------- */
void op_silu(float *x, NnUint size, NnUint n_threads, NnUint thread_index) {
    SPLIT_THREADS(start, end, size, n_threads, thread_index);
    NnUint i = start;

#if defined(__ARM_NEON)
    NnUint count = end - start;
    NnUint neon_end = end - (count % 4);
    for (; i < neon_end; i += 4) {
        float32x4_t xv = vld1q_f32(&x[i]);
        float32x4_t en = _neon_expf(vnegq_f32(xv));
        float32x4_t d  = vaddq_f32(en, vdupq_n_f32(1.0f));
        float32x4_t recip = vrecpeq_f32(d);
        recip = vmulq_f32(recip, vsubq_f32(vdupq_n_f32(2.0f), vmulq_f32(d, recip)));
        vst1q_f32(&x[i], vmulq_f32(xv, recip));
    }
#elif defined(__AVX2__)
    NnUint count = end - start;
    NnUint avx_end = end - (count % 8);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    for (; i < avx_end; i += 8) {
        __m256 xv = _mm256_loadu_ps(&x[i]);
        __m256 nx = _mm256_sub_ps(zero, xv);
        __m256 en = _avx2_expf(nx);
        __m256 denom = _mm256_add_ps(one, en);
        _mm256_storeu_ps(&x[i], _mm256_div_ps(xv, denom));
    }
#endif
    for (; i < end; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

/* --------------------------------------------------------------------------
 *  GELU (Gaussian Error Linear Unit):
 *    0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x * x)))
 *  Operates in-place on x.
 * -------------------------------------------------------------------------- */
void op_gelu(float *x, NnUint size, NnUint n_threads, NnUint thread_index) {
#define SQRT_2_OVER_PI 0.7978845608028654f
#define GELU_COEF_A    0.044715f

    SPLIT_THREADS(start, end, size, n_threads, thread_index);

    for (NnUint i = start; i < end; i++) {
        float v = x[i];
        float t = SQRT_2_OVER_PI * v * (1.0f + GELU_COEF_A * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(t));
    }

#undef GELU_COEF_A
#undef SQRT_2_OVER_PI
}

/* --------------------------------------------------------------------------
 *  Element-wise multiply:  y[i] = x[i] * m[i]
 * -------------------------------------------------------------------------- */
void op_mul_f32(float *y, const float *x, const float *m, NnUint n,
                NnUint n_threads, NnUint thread_index) {
    SPLIT_THREADS(start, end, n, n_threads, thread_index);

    for (NnUint i = start; i < end; i++) {
        y[i] = x[i] * m[i];
    }
}

/* --------------------------------------------------------------------------
 *  Element-wise add:  y[i] += x[i]
 * -------------------------------------------------------------------------- */
void op_add_f32(float *y, const float *x, NnUint n,
                NnUint n_threads, NnUint thread_index) {
    SPLIT_THREADS(start, end, n, n_threads, thread_index);
    NnUint i = start;
#if defined(__AVX2__)
    NnUint count = end - start;
    NnUint avx_end = end - (count % 8);
    for (; i < avx_end; i += 8) {
        __m256 yv = _mm256_loadu_ps(&y[i]);
        __m256 xv = _mm256_loadu_ps(&x[i]);
        _mm256_storeu_ps(&y[i], _mm256_add_ps(yv, xv));
    }
#endif
    for (; i < end; i++) {
        y[i] += x[i];
    }
}

/* --------------------------------------------------------------------------
 *  Scale:  o[i] = i[i] * s
 * -------------------------------------------------------------------------- */
void op_scale_f32(float *o, const float *i, float s, NnUint n,
                  NnUint n_threads, NnUint thread_index) {
    SPLIT_THREADS(start, end, n, n_threads, thread_index);

    for (NnUint idx = start; idx < end; idx++) {
        o[idx] = i[idx] * s;
    }
}

/* --------------------------------------------------------------------------
 *  Parallel byte copy (memcpy split across threads).
 * -------------------------------------------------------------------------- */
void op_copy_bytes(NnByte *dst, const NnByte *src, NnSize n,
                   NnUint n_threads, NnUint thread_index) {
    SPLIT_THREADS(start, end, n, n_threads, thread_index);

    NnSize s = end - start;
    if (s != 0U) {
        memcpy(&dst[start], &src[start], (size_t)s);
    }
}

/* --------------------------------------------------------------------------
 *  Top-K:  find indices of the k largest values in x.
 *
 *  x = [size]  (float values)
 *  y = [k]     (output indices, sorted by descending value)
 *
 *  Creates an array of (value, index) pairs, sorts descending by value
 *  with qsort, then copies out the top-k indices.
 * -------------------------------------------------------------------------- */
typedef struct {
    float  val;
    NnUint idx;
} TopkPair;

static int topk_cmp_desc(const void *a, const void *b) {
    float va = ((const TopkPair *)a)->val;
    float vb = ((const TopkPair *)b)->val;
    if (va > vb) return -1;
    if (va < vb) return  1;
    return 0;
}

void op_topk(const float *x, NnUint *y, NnSize size, NnUint k) {
    if (k == 0U || size == 0U) return;
    if (k > (NnUint)size) k = (NnUint)size;

    TopkPair *pairs = (TopkPair *)malloc(size * sizeof(TopkPair));
    if (pairs == NULL) return;

    for (NnSize i = 0U; i < size; i++) {
        pairs[i].val = x[i];
        pairs[i].idx = (NnUint)i;
    }

    qsort(pairs, size, sizeof(TopkPair), topk_cmp_desc);

    for (NnUint i = 0U; i < k; i++) {
        y[i] = pairs[i].idx;
    }

    free(pairs);
}

/* ===================================================================
 *  Python C-API bindings
 * =================================================================== */
#include <Python.h>

static int _f32_rw(PyObject *obj, float **ptr) {
    Py_buffer buf;
    if (PyObject_GetBuffer(obj, &buf, PyBUF_WRITABLE | PyBUF_FORMAT) < 0) return -1;
    if (strcmp(buf.format, "f") != 0) { PyBuffer_Release(&buf); PyErr_SetString(PyExc_TypeError, "need float32"); return -1; }
    *ptr = (float *)buf.buf;
    PyBuffer_Release(&buf);
    return 0;
}
static int _f32_ro(PyObject *obj, const float **ptr) {
    Py_buffer buf;
    if (PyObject_GetBuffer(obj, &buf, PyBUF_FORMAT) < 0) return -1;
    if (strcmp(buf.format, "f") != 0) { PyBuffer_Release(&buf); PyErr_SetString(PyExc_TypeError, "need float32"); return -1; }
    *ptr = (const float *)buf.buf;
    PyBuffer_Release(&buf);
    return 0;
}
static int _bytes_ro(PyObject *obj, const void **ptr) {
    Py_buffer buf;
    if (PyObject_GetBuffer(obj, &buf, PyBUF_SIMPLE) < 0) return -1;
    *ptr = buf.buf;
    PyBuffer_Release(&buf);
    return 0;
}

#define _U(v) ((NnUint)PyLong_AsUnsignedLong(v))

static PyObject *py_softmax_f32(PyObject *s, PyObject *a) {
    PyObject *x; NnUint n; if (!PyArg_ParseTuple(a, "OI", &x, &n)) return NULL;
    float *p; if (_f32_rw(x, &p) < 0) return NULL;
    Py_BEGIN_ALLOW_THREADS op_softmax_f32(p, n); Py_END_ALLOW_THREADS Py_RETURN_NONE;
}

static PyObject *py_embedding_f32(PyObject *s, PyObject *a) {
    PyObject *o, *w, *inp; NnUint d,bs,nt,ti;
    if (!PyArg_ParseTuple(a, "OOOIIII", &o, &w, &inp, &d, &bs, &nt, &ti)) return NULL;
    float *op, *ip; const float *wp;
    if (_f32_rw(o, &op) < 0 || _f32_ro(w, &wp) < 0 || _f32_ro(inp, (const float **)&ip) < 0) return NULL;
    Py_BEGIN_ALLOW_THREADS op_embedding_f32(op, wp, ip, d, bs, nt, ti); Py_END_ALLOW_THREADS Py_RETURN_NONE;
}

static PyObject *py_matmul_f32(PyObject *s, PyObject *a) {
    PyObject *o, *x, *w; NnUint n,d,bs,nt,ti;
    if (!PyArg_ParseTuple(a, "OOOIIIII", &o, &x, &w, &n, &d, &bs, &nt, &ti)) return NULL;
    float *op, *xp; const float *wp;
    if (_f32_rw(o, &op) < 0 || _f32_ro(x, (const float **)&xp) < 0 || _f32_ro(w, &wp) < 0) return NULL;
    Py_BEGIN_ALLOW_THREADS op_matmul_f32_f32_f32(op, xp, wp, n, d, bs, nt, ti); Py_END_ALLOW_THREADS Py_RETURN_NONE;
}

static PyObject *py_matmul_q80q40(PyObject *s, PyObject *a) {
    PyObject *o, *x, *w; NnUint n,d,bs,nt,ti;
    if (!PyArg_ParseTuple(a, "OOOIIIII", &o, &x, &w, &n, &d, &bs, &nt, &ti)) return NULL;
    float *op; const void *xp, *wp;
    if (_f32_rw(o, &op) < 0 || _bytes_ro(x, &xp) < 0 || _bytes_ro(w, &wp) < 0) return NULL;
    Py_BEGIN_ALLOW_THREADS op_matmul_q80_q40_f32(op, xp, wp, n, d, bs, nt, ti); Py_END_ALLOW_THREADS Py_RETURN_NONE;
}

static PyObject *py_rms_norm(PyObject *s, PyObject *a) {
    PyObject *o, *i, *w, *ir; NnUint sz,nc,bs,nt,ti;
    if (!PyArg_ParseTuple(a, "OOOOIIIII", &o, &i, &w, &ir, &sz, &nc, &bs, &nt, &ti)) return NULL;
    float *op; const float *ip, *wp, *irp;
    if (_f32_rw(o, &op) < 0 || _f32_ro(i, &ip) < 0 || _f32_ro(w, &wp) < 0 || _f32_ro(ir, &irp) < 0) return NULL;
    Py_BEGIN_ALLOW_THREADS op_rms_norm_f32(op, ip, wp, irp, sz, nc, bs, nt, ti); Py_END_ALLOW_THREADS Py_RETURN_NONE;
}

static PyObject *py_silu(PyObject *s, PyObject *a) {
    PyObject *x; NnUint n,nt,ti;
    if (!PyArg_ParseTuple(a, "OIII", &x, &n, &nt, &ti)) return NULL;
    float *p; if (_f32_rw(x, &p) < 0) return NULL;
    Py_BEGIN_ALLOW_THREADS op_silu(p, n, nt, ti); Py_END_ALLOW_THREADS Py_RETURN_NONE;
}

static PyObject *py_gelu(PyObject *s, PyObject *a) {
    PyObject *x; NnUint n,nt,ti;
    if (!PyArg_ParseTuple(a, "OIII", &x, &n, &nt, &ti)) return NULL;
    float *p; if (_f32_rw(x, &p) < 0) return NULL;
    Py_BEGIN_ALLOW_THREADS op_gelu(p, n, nt, ti); Py_END_ALLOW_THREADS Py_RETURN_NONE;
}

static PyObject *py_mul_f32(PyObject *s, PyObject *a) {
    PyObject *y, *x, *m; NnUint n,nt,ti;
    if (!PyArg_ParseTuple(a, "OOOIII", &y, &x, &m, &n, &nt, &ti)) return NULL;
    float *yp, *xp, *mp;
    if (_f32_rw(y, &yp) < 0 || _f32_ro(x, (const float **)&xp) < 0 || _f32_ro(m, (const float **)&mp) < 0) return NULL;
    Py_BEGIN_ALLOW_THREADS op_mul_f32(yp, xp, mp, n, nt, ti); Py_END_ALLOW_THREADS Py_RETURN_NONE;
}

static PyObject *py_add_f32(PyObject *s, PyObject *a) {
    PyObject *y, *x; NnUint n,nt,ti;
    if (!PyArg_ParseTuple(a, "OOIII", &y, &x, &n, &nt, &ti)) return NULL;
    float *yp; const float *xp;
    if (_f32_rw(y, &yp) < 0 || _f32_ro(x, &xp) < 0) return NULL;
    Py_BEGIN_ALLOW_THREADS op_add_f32(yp, xp, n, nt, ti); Py_END_ALLOW_THREADS Py_RETURN_NONE;
}

static PyObject *py_rope_llama(PyObject *s, PyObject *a) {
    PyObject *x, *cache; int iq; NnUint p,d0,sd,sh,nt,ti;
    if (!PyArg_ParseTuple(a, "OOpIIIIII", &x, &cache, &iq, &p, &d0, &sd, &sh, &nt, &ti)) return NULL;
    float *xp; const float *cp;
    if (_f32_rw(x, &xp) < 0 || _f32_ro(cache, &cp) < 0) return NULL;
    Py_BEGIN_ALLOW_THREADS op_rope_llama(xp, cp, (bool)iq, p, d0, sd, sh, nt, ti); Py_END_ALLOW_THREADS Py_RETURN_NONE;
}

static PyObject *py_rope_falcon(PyObject *s, PyObject *a) {
    PyObject *x, *cache; NnUint p,d0,hd,nt,ti;
    if (!PyArg_ParseTuple(a, "OOIIIII", &x, &cache, &p, &d0, &hd, &nt, &ti)) return NULL;
    float *xp; const float *cp;
    if (_f32_rw(x, &xp) < 0 || _f32_ro(cache, &cp) < 0) return NULL;
    Py_BEGIN_ALLOW_THREADS op_rope_falcon(xp, cp, p, d0, hd, nt, ti); Py_END_ALLOW_THREADS Py_RETURN_NONE;
}

static PyObject *py_multihead_att(PyObject *s, PyObject *a) {
    PyObject *y, *q, *att, *kc, *vc;
    NnUint pos,nh0,nh,nkv,kvd0,hd,seq,nn,ni,nt,ti;
    if (!PyArg_ParseTuple(a, "OOOOOIIIIIIIIIII", &y, &q, &att, &kc, &vc,
        &pos, &nh0, &nh, &nkv, &kvd0, &hd, &seq, &nn, &ni, &nt, &ti)) return NULL;
    float *yp, *ap; const float *qp, *kp, *vp;
    if (_f32_rw(y, &yp) < 0 || _f32_ro(q, &qp) < 0 || _f32_rw(att, &ap) < 0 ||
        _f32_ro(kc, &kp) < 0 || _f32_ro(vc, &vp) < 0) return NULL;
    Py_BEGIN_ALLOW_THREADS op_multihead_att(yp, qp, ap, (float *)kp, (float *)vp, pos, nh0, nh, nkv, kvd0, hd, seq, nn, ni, nt, ti); Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject *py_scale_f32(PyObject *s, PyObject *a) {
    PyObject *o, *i; float sc; NnUint n,nt,ti;
    if (!PyArg_ParseTuple(a, "OOfIII", &o, &i, &sc, &n, &nt, &ti)) return NULL;
    float *op; const float *ip;
    if (_f32_rw(o, &op) < 0 || _f32_ro(i, &ip) < 0) return NULL;
    Py_BEGIN_ALLOW_THREADS op_scale_f32(op, ip, sc, n, nt, ti); Py_END_ALLOW_THREADS Py_RETURN_NONE;
}

static PyObject *py_copy_bytes(PyObject *s, PyObject *a) {
    PyObject *d, *src; NnUint n,nt,ti;
    if (!PyArg_ParseTuple(a, "OOIII", &d, &src, &n, &nt, &ti)) return NULL;
    Py_buffer db, sb;
    if (PyObject_GetBuffer(d, &db, PyBUF_WRITABLE | PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(src, &sb, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&db); return NULL; }
    Py_BEGIN_ALLOW_THREADS op_copy_bytes((NnByte *)db.buf, (const NnByte *)sb.buf, n, nt, ti); Py_END_ALLOW_THREADS
    PyBuffer_Release(&db); PyBuffer_Release(&sb); Py_RETURN_NONE;
}

/* Python wrapper for thread-pool initialization */
static PyObject *py_init_mt(PyObject *s, PyObject *a) {
    NnUint n; if (!PyArg_ParseTuple(a, "I", &n)) return NULL;
    init_mt(n); Py_RETURN_NONE;
}

static PyMethodDef ops_methods[] = {
    {"init_mt",           py_init_mt,        METH_VARARGS, ""},
    {"op_softmax_f32",    py_softmax_f32,    METH_VARARGS, ""},
    {"op_embedding_f32",  py_embedding_f32,  METH_VARARGS, ""},
    {"op_matmul_f32_f32_f32", py_matmul_f32, METH_VARARGS, ""},
    {"op_matmul_q80_q40_f32", py_matmul_q80q40, METH_VARARGS, ""},
    {"op_rms_norm_f32",   py_rms_norm,       METH_VARARGS, ""},
    {"op_silu",           py_silu,           METH_VARARGS, ""},
    {"op_gelu",           py_gelu,           METH_VARARGS, ""},
    {"op_mul_f32",        py_mul_f32,        METH_VARARGS, ""},
    {"op_add_f32",        py_add_f32,        METH_VARARGS, ""},
    {"op_rope_llama",     py_rope_llama,     METH_VARARGS, ""},
    {"op_rope_falcon",    py_rope_falcon,    METH_VARARGS, ""},
    {"op_multihead_att",  py_multihead_att,  METH_VARARGS, ""},
    {"op_scale_f32",      py_scale_f32,      METH_VARARGS, ""},
    {"op_copy_bytes",     py_copy_bytes,     METH_VARARGS, ""},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef ops_mod = {
    PyModuleDef_HEAD_INIT, "_ops", NULL, -1, ops_methods,
};

PyMODINIT_FUNC PyInit__ops(void) { return PyModule_Create(&ops_mod); }
