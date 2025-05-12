#include "utils.h"
#include <stdio.h>

void profile_report(int n_imgs){
    double total = g_prof.total;
    double fps   = n_imgs * 1000.0 / total;
    double ms_img= total / n_imgs;
    double gflops= (FLOP_PER_IMAGE * n_imgs) / (total * 1e6);

    printf("\n=== Performance Summary ===\n");
    printf("Images          : %d\n", n_imgs);
    printf("Total time      : %.2f ms\n", total);
    printf("Throughput      : %.1f imgs/s\n", fps);
    printf("Latency / image : %.3f ms\n", ms_img);
    printf("GFLOPS (actual) : %.2f\n", gflops);
    printf("\n--- Layer breakdown (ms / %% of total) ---\n");
#define P(field,label) \
    printf("%-8s : %7.3f ms  %5.1f%%\n", label, g_prof.field, 100.0*g_prof.field/total);
    P(conv1,"conv1");  P(pool1,"pool1");
    P(conv2,"conv2");  P(pool2,"pool2");
    P(fc1,  "fc1");    P(fc2,  "fc2");
    P(softmax,"softmax");
#undef P
    puts("==========================================");
}

