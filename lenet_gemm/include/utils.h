#pragma once
#include <stddef.h>
#include <stdint.h>
#include "tensor.h"
double now_ms(void);                             /* 单位 ms */
void  *load_file(const char *path, size_t *bytes);

/* ---------- 简易 profiler ---------- */
typedef struct {
    double conv1, pool1, conv2, pool2, fc1, fc2, softmax, total;
} Prof;

extern Prof g_prof;

#define T_START(name)   double __t_##name = now_ms();
#define T_END(name)     g_prof.name += now_ms() - __t_##name;

/* 总 FLOP 按 LeNet-5 (28×28) 基线：conv×2 + fc×2            */
#define FLOP_PER_IMAGE  11.62e6       /* 11.62 MFLOP */
dtype *load_weights(const char *path, size_t *bytes);


