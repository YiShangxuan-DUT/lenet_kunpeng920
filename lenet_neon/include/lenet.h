//neon
#pragma once
#include "tensor.h"

/* ---------- 运算算子 ---------- */
void conv2d (const Tensor *in, Tensor *out,
             const Tensor *weight, const dtype *bias,
             int stride, int pad);

void avgpool2d(const Tensor *in, Tensor *out, int k);
void relu_ip  (Tensor *t);

void linear   (const Tensor *in, Tensor *out,
               const Tensor *weight, const dtype *bias);

int  argmax (const dtype *x, int n);
void softmax(dtype *x, int n);

/* ---------- LeNet-5 前向 ---------- */
int lenet_forward(const Tensor *img,
                  const dtype  *params,
                  Tensor       *work);           /* work 为一次性复用缓存 */

/* ---------- 读取 MNIST ---------- */
int  load_mnist_images (const char *path, Tensor *out);     /* returns N */
int  load_mnist_labels (const char *path, unsigned char **lab_out);

/* ---------- 复用缓存 ---------- */
typedef struct {
    Tensor x1, x2, x3, x4, x5, x6, x7;   /* 7 个中间层 */
} Workspace;

Workspace workspace_make (int batch);
void      workspace_free(Workspace *ws);

/* ---------- 批量前向 ---------- */
void lenet_forward_batch(const Tensor *img_batch,
                         const dtype   *params,
                         Workspace     *ws,
                         int           *preds);

/* Specialized 5×5, stride=1, pad=0 version (no bounds check) */
static void conv2d_5x5_s1(const Tensor *in, Tensor *out,
                           const Tensor *w, const dtype *b);