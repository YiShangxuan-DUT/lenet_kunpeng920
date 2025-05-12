#include "lenet.h"
#include "utils.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef PERF
extern void profile_report(int n_imgs);
#endif

int main(void)
{
    /* ---------- 读取权重 ---------- */
    // size_t wbytes;
    // float *params = (float *)load_file("weight/lenet5_fp32.bin", &wbytes);
    size_t wbytes;
    dtype *params = load_weights("weight/lenet5_fp32.bin", &wbytes);
    printf("Loaded weights: %.1f KB\n", wbytes / 1024.0);

    /* ---------- 读取测试集 ---------- */
    Tensor test_img;
    int n_test = load_mnist_images("data/t10k-images-idx3-ubyte", &test_img);
    unsigned char *labels;
    load_mnist_labels("data/t10k-labels-idx1-ubyte", &labels);
    printf("Test images: %d\n", n_test);

    /* ---------- 批量推理 ---------- */
    const int BS = 64;                     /* 批大小 */
    Workspace ws = workspace_make(BS);

    int correct = 0, preds[BS];
    double t0 = now_ms();

    for (int idx = 0; idx < n_test; idx += BS) {
        int cur = (idx + BS <= n_test) ? BS : (n_test - idx);
        Tensor batch = {cur, 1, 28, 28, &test_img.data[idx * 28 * 28]};
        lenet_forward_batch(&batch, params, &ws, preds);
        for (int i = 0; i < cur; ++i)
            if (preds[i] == labels[idx + i]) ++correct;
    }
    double elapsed = now_ms() - t0;

    printf("Accuracy      : %.2f%%\n", 100.0 * correct / n_test);
    printf("Inference time: %.2f ms\n", elapsed);
    printf("== Running LeNet5 (%s) ==\n", DTYPE_STR);


#ifdef PERF
    profile_report(n_test);
#endif

    /* ---------- 释放 ---------- */
    workspace_free(&ws);
    free(params); free(test_img.data); free(labels);
    return 0;
}
