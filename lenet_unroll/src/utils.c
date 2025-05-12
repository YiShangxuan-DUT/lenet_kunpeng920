//unroll
#include "utils.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <errno.h>

double now_ms(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1e3 + ts.tv_nsec/1e6;
}

void *load_file(const char *path, size_t *bytes){
    FILE *f=fopen(path,"rb"); if(!f){perror(path); exit(1);}
    fseek(f,0,SEEK_END); long sz=ftell(f); rewind(f);
    void *buf=NULL;
    if (posix_memalign(&buf, ALIGN_BYTES, sz)!=0){
        fprintf(stderr,"load_file: posix_memalign failed\n");
        exit(1);
    }
    fread(buf,1,sz,f); fclose(f);
    if(bytes) *bytes=sz; return buf;
}

/* ------ Profiler data ------ */
Prof g_prof = {0};




/* ---- utils.c ---- */
void f32_to_f16(const float *src, _Float16 *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) dst[i] = (_Float16)src[i];
}

/* 把磁盘上的 FP32 权重 → 内存 dtype 缓冲区 */
dtype *load_weights(const char *path, size_t *bytes_out)
{
    size_t bytes_f32;
    float *buf_f32 = (float *)load_file(path, &bytes_f32);
    if (!buf_f32) return NULL;

    size_t cnt = bytes_f32 / sizeof(float);
    dtype *buf_dtype;
    /* 16 字节对齐，利于后面手写 NEON */
    if (posix_memalign((void **)&buf_dtype, 16, cnt * sizeof(dtype)) != 0) {
        fprintf(stderr, "posix_memalign: %s\n", strerror(errno));
        free(buf_f32);
        return NULL;
    }
    for (size_t i = 0; i < cnt; ++i)
        buf_dtype[i] = (dtype)buf_f32[i];

    free(buf_f32);
    if (bytes_out) *bytes_out = cnt * sizeof(dtype);
    return buf_dtype;
}




