//gemm
#pragma once
#include <stdlib.h>
#include <stdio.h>      /* for fprintf  */
#include <errno.h>      /* for ENOMEM   */
#include <string.h>

#ifndef ALIGN_BYTES
#define ALIGN_BYTES 16      /* 可按需改 32，SVE/SIMD256 时更保险 */
#endif

/* ===== 数据类型宏 =====
 * 编译时加 -DUSE_FP16 即可把 dtype 切成 _Float16，
 * 累加器 acc_t 始终用 float 保障数值稳定。
 */
#ifdef USE_FP16
  typedef _Float16 dtype;      /* or __fp16 */
  typedef float    acc_t;
  #define DTYPE_STR "FP16"
#else
  typedef float    dtype;
  typedef float    acc_t;
  #define DTYPE_STR "FP32"
#endif

typedef struct {
    int n, c, h, w;            /* NCHW */
    dtype *data;               /* size = n*c*h*w */
} Tensor;

/* 4-D 下标宏 */
#define IDX4(t, ni, ci, hi, wi) \
    (t)->data[(((ni)*(t)->c + (ci))*(t)->h + (hi))*(t)->w + (wi)]

/* 构造 / 释放 */
static inline Tensor tensor_make(int n,int c,int h,int w){
    size_t bytes = (size_t)n*c*h*w * sizeof(dtype);
    dtype *ptr = NULL;
    if (posix_memalign((void**)&ptr, ALIGN_BYTES, bytes) != 0){
        fprintf(stderr,"tensor_make: posix_memalign failed (%s)\n",
                strerror(errno));
        exit(1);
    }
    Tensor t = { n, c, h, w, ptr };
    return t;
}
static inline void tensor_free(Tensor *t){ free(t->data); }