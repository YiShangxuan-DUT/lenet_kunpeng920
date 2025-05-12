#include "sgemm4x4.h"

/* ------ 4×4 kernel：一次算 4×4 输出块 ------ */
static inline void kern4x4(int K,
    const float *A,const float *B,float *C,int lda,int ldb,int ldc)
{
    float32x4_t c0 = vdupq_n_f32(0.f);           // (row0, col0–3)
    float32x4_t c1 = vdupq_n_f32(0.f);           // (row1, col0–3)
    float32x4_t c2 = vdupq_n_f32(0.f);           // (row2, col0–3)
    float32x4_t c3 = vdupq_n_f32(0.f);           // (row3, col0–3)

    for (int k = 0; k < K; ++k) {
        float32x4_t b = vld1q_f32(B + k*ldb);    // 4 列
        float32x4_t a0 = vdupq_n_f32(A[0*lda + k]);
        float32x4_t a1 = vdupq_n_f32(A[1*lda + k]);
        float32x4_t a2 = vdupq_n_f32(A[2*lda + k]);
        float32x4_t a3 = vdupq_n_f32(A[3*lda + k]);

        c0 = vfmaq_f32(c0, b, a0);
        c1 = vfmaq_f32(c1, b, a1);
        c2 = vfmaq_f32(c2, b, a2);
        c3 = vfmaq_f32(c3, b, a3);
    }
    vst1q_f32(C + 0*ldc, vaddq_f32(vld1q_f32(C + 0*ldc), c0));
    vst1q_f32(C + 1*ldc, vaddq_f32(vld1q_f32(C + 1*ldc), c1));
    vst1q_f32(C + 2*ldc, vaddq_f32(vld1q_f32(C + 2*ldc), c2));
    vst1q_f32(C + 3*ldc, vaddq_f32(vld1q_f32(C + 3*ldc), c3));
}

/* ------ 外层包装：任意 M×N ------ */
void sgemm4x4_neon(int M,int N,int K,
                   const float *A,const float *B,float *C,
                   int lda,int ldb,int ldc)
{
    for (int m = 0; m + 3 < M; m += 4) {
        const float *a_ptr = A + m*lda;
        float *c_ptr       = C + m*ldc;
        for (int n = 0; n < N; n += 4) {
            kern4x4(K, a_ptr, B + n, c_ptr + n, lda, ldb, ldc);
        }
    }
    /* M 尾巴用标量；若想更极致可写 1×4、2×4 内核 */
    for (int m = (M&~3); m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k)
                sum += A[m*lda+k] * B[k*ldb+n];
            C[m*ldc+n] += sum;
        }
}

void sgemm4x4(int M,int N,int K,
              const float *A,const float *B,float *C)
{
    sgemm4x4_neon(M, N, K, A, B, C,
                  K,      /* lda */
                  N,      /* ldb */
                  N);     /* ldc */
}