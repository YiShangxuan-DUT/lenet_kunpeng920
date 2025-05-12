//baseline
#pragma once
#include <stdlib.h>

typedef struct {
    int n, c, h, w;      /* NCHW */
    float *data;         /* size = n*c*h*w */
} Tensor;

/* 4D 下标宏：t 是 Tensor* */
#define IDX4(t, ni, ci, hi, wi) \
    (t)->data[(((ni)*(t)->c + (ci))*(t)->h + (hi))*(t)->w + (wi)]

/* 简易构造 / 释放 */
static inline Tensor tensor_make(int n,int c,int h,int w){
    Tensor t={n,c,h,w, (float*)malloc(sizeof(float)*n*c*h*w)};
    return t;
}
static inline void tensor_free(Tensor *t){ free(t->data); }
