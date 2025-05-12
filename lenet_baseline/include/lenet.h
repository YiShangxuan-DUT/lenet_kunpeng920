//baseline
#pragma once
#include "tensor.h"

/* ---- 运算算子 ---- */
void conv2d(const Tensor *in, Tensor *out,
            const Tensor *weight, const float *bias,
            int stride, int pad);
void avgpool2d(const Tensor *in, Tensor *out, int k);
void relu_ip(Tensor *t);
void linear(const Tensor *in, Tensor *out,
            const Tensor *weight, const float *bias);
int  argmax(const float *x, int n);
void softmax(float *x, int n);

/* ---- LeNet-5 前向，返回预测类别 ---- */
int lenet_forward(const Tensor *img, const float *p,
                  Tensor *work);      /* work 为一次性复用缓存 */

/* ---- 读取 MNIST ---- */
int load_mnist_images(const char *path, Tensor *out); /* returns N */
int load_mnist_labels(const char *path, unsigned char **lab_out);
/* ---- 复用缓存，构造一次，循环使用 ---- */
typedef struct {
    Tensor x1, x2, x3, x4, x5, x6, x7;   /* 7 个中间层 */
} Workspace;

Workspace workspace_make(int batch);
void      workspace_free(Workspace *ws);

/* ---- 批量前向：一次处理 N=img->n 张图 ----
   preds 长度至少 N，返回每张图的预测类别               */
void lenet_forward_batch(const Tensor *img_batch,
                         const float   *params,
                         Workspace *ws,
                         int *preds);




