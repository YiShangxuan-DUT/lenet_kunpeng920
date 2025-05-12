#pragma once
#ifdef __cplusplus
extern "C" {            /* 若以后被 C++ 调用可防止符号重整 */
#endif
#include <stddef.h>
#include <arm_neon.h>

/* ---------- 对外统一接口（6 个参数） ---------- */
void sgemm4x4(int M,int N,int K,
              const float *A,const float *B,float *C);

/* ---------- 真实 NEON 内核（带 lda/ldb/ldc） ---------- */
void sgemm4x4_neon(int M,int N,int K,
                   const float *A,const float *B,float *C,
                   int lda,int ldb,int ldc);

#ifdef __cplusplus
}
#endif