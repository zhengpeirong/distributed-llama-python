/* C-compatible wrapper around llamafile_sgemm for use from ops.c */

#include <cstdint>
#include <cmath>

/* Global f16→f32 lookup table (required by sgemm/CONVERT_F16_TO_F32) */
float f16ToF32Lookup[65536];
static bool f16_lookup_init = false;

static void init_f16_lookup() {
    if (f16_lookup_init) return;
    for (int i = 0; i < 65536; i++) {
        int sign = (i >> 15) & 1;
        int exp  = (i >> 10) & 0x1F;
        int mant = i & 0x3FF;
        if (exp == 0)
            f16ToF32Lookup[i] = (sign ? -1.0f : 1.0f) * ldexpf((float)mant, -24);
        else if (exp == 31)
            f16ToF32Lookup[i] = mant ? NAN : (sign ? -INFINITY : INFINITY);
        else
            f16ToF32Lookup[i] = (sign ? -1.0f : 1.0f) * ldexpf(1.0f + (float)mant / 1024.0f, exp - 15);
    }
    f16_lookup_init = true;
}

/* Bring in the full sgemm implementation */
#include "sgemm.cpp"

extern "C" {

int llamafile_sgemm_c(int64_t m, int64_t n, int64_t k,
                       const void *A, int64_t lda,
                       const void *B, int64_t ldb,
                       void *C, int64_t ldc,
                       int ith, int nth, int task,
                       int Atype, int Btype, int Ctype)
{
    init_f16_lookup();
    return llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc, ith, nth, task, Atype, Btype, Ctype) ? 1 : 0;
}

}
