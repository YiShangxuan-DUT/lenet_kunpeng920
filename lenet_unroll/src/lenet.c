//循环展开
/* ----------  src/lenet.c  ---------- */
#include "lenet.h"
#include "utils.h"
#include <string.h>
#include <math.h>


/* ====== 专用 5×5, stride=1, pad=0 内核 ====== */
static void conv2d_5x5_s1(const Tensor *in, Tensor *out,
                          const Tensor *w, const dtype *bias)
{
    const int N  = in->n,   IC = in->c,  IH = in->h,  IW = in->w;
    const int OH = out->h,  OW = out->w;

    for (int n = 0; n < N; ++n) {
        const dtype *in_n  = in->data  + (size_t)n * IC * IH * IW;
        dtype       *out_n = out->data + (size_t)n * out->c * OH * OW;

        for (int oc = 0; oc < out->c; ++oc) {
            const dtype *k_oc   = w->data + (size_t)oc * IC * 25;
            dtype       *outmap = out_n   + (size_t)oc * OH * OW;

            for (int oh = 0; oh < OH; ++oh) {
                dtype *out_ptr = outmap + (size_t)oh * OW;

                for (int ow = 0; ow < OW; ++ow) {
                    acc_t sum = (acc_t)bias[oc];
                    const dtype *kptr = k_oc;

                    for (int ic = 0; ic < IC; ++ic) {
                        const dtype *base = in_n + (size_t)ic * IH * IW + (size_t)oh * IW + ow;

                        const dtype *r0 = base;
                        const dtype *r1 = r0 + IW;
                        const dtype *r2 = r1 + IW;
                        const dtype *r3 = r2 + IW;
                        const dtype *r4 = r3 + IW;

                        dtype k00 = kptr[ 0], k01 = kptr[ 1], k02 = kptr[ 2], k03 = kptr[ 3], k04 = kptr[ 4];
                        dtype k10 = kptr[ 5], k11 = kptr[ 6], k12 = kptr[ 7], k13 = kptr[ 8], k14 = kptr[ 9];
                        dtype k20 = kptr[10], k21 = kptr[11], k22 = kptr[12], k23 = kptr[13], k24 = kptr[14];
                        dtype k30 = kptr[15], k31 = kptr[16], k32 = kptr[17], k33 = kptr[18], k34 = kptr[19];
                        dtype k40 = kptr[20], k41 = kptr[21], k42 = kptr[22], k43 = kptr[23], k44 = kptr[24];
                        kptr += 25;

                        sum += (acc_t)r0[0]*k00 + (acc_t)r0[1]*k01 + (acc_t)r0[2]*k02 + (acc_t)r0[3]*k03 + (acc_t)r0[4]*k04
                             + (acc_t)r1[0]*k10 + (acc_t)r1[1]*k11 + (acc_t)r1[2]*k12 + (acc_t)r1[3]*k13 + (acc_t)r1[4]*k14
                             + (acc_t)r2[0]*k20 + (acc_t)r2[1]*k21 + (acc_t)r2[2]*k22 + (acc_t)r2[3]*k23 + (acc_t)r2[4]*k24
                             + (acc_t)r3[0]*k30 + (acc_t)r3[1]*k31 + (acc_t)r3[2]*k32 + (acc_t)r3[3]*k33 + (acc_t)r3[4]*k34
                             + (acc_t)r4[0]*k40 + (acc_t)r4[1]*k41 + (acc_t)r4[2]*k42 + (acc_t)r4[3]*k43 + (acc_t)r4[4]*k44;
                    }

                    out_ptr[ow] = (dtype)sum;
                }
            }
        }
    }
}


void avgpool2d(const Tensor *in, Tensor *out, int k)
{
    acc_t k2_inv = 1.0f / (acc_t)(k * k);
    for (int n = 0; n < in->n; n++)
    for (int c = 0; c < in->c; c++)
    for (int oh = 0; oh < out->h; oh++)
    for (int ow = 0; ow < out->w; ow++) {
        acc_t sum = 0;
        for (int kh = 0; kh < k; kh++)
        for (int kw = 0; kw < k; kw++) {
            int ih = oh * k + kh, iw = ow * k + kw;
            sum += (acc_t)IDX4(in,n,c,ih,iw);
        }
        IDX4(out,n,c,oh,ow) = (dtype)(sum * k2_inv);
    }
}

void relu_ip(Tensor *t)
{
    size_t sz = (size_t)t->n * t->c * t->h * t->w;
    for (size_t i = 0; i < sz; ++i)
        if (t->data[i] < (dtype)0) t->data[i] = (dtype)0;
}

void linear(const Tensor *in, Tensor *out,
            const Tensor *w, const dtype *b)
{
    int N = in->n, in_dim = in->c * in->h * in->w, out_dim = out->c;
    for (int n = 0; n < N; ++n)
        for (int o = 0; o < out_dim; ++o) {
            const dtype *x = in->data + (size_t)n * in_dim;
            const dtype *k = w->data + (size_t)o * in_dim;
            acc_t sum = (acc_t)b[o];
            for (int i = 0; i < in_dim; ++i)
                sum += (acc_t)x[i] * (acc_t)k[i];
            out->data[(size_t)n * out_dim + o] = (dtype)sum;
        }
}

int  argmax (const dtype *x,int n){
    int id = 0;
    for(int i = 1; i < n; ++i)
        if((acc_t)x[i] > (acc_t)x[id]) id = i;
    return id;
}

void softmax(dtype *x,int n)
{
    acc_t m = (acc_t)x[0];
    for(int i = 1; i < n; ++i)
        if((acc_t)x[i] > m) m = (acc_t)x[i];

    acc_t s = 0;
    for(int i = 0; i < n; ++i){
        acc_t t = expf((acc_t)x[i] - m);
        x[i] = (dtype)t;
        s   += t;
    }
    acc_t s_inv = 1.0f / s;
    for(int i = 0; i < n; ++i)
        x[i] = (dtype)((acc_t)x[i] * s_inv);
}


/* ====== 批量前向 ====== */
void lenet_forward_batch(const Tensor *imgs,const dtype *p,
                         Workspace *ws,int *pred)
{
    /* ---- 权重切片 ---- */
    size_t off = 0;
    Tensor w1={6,1,5,5,(dtype*)(p+off)};       off += 6*1*5*5;
    const dtype *b1 = p + off;                 off += 6;
    Tensor w2={16,6,5,5,(dtype*)(p+off)};      off += 16*6*5*5;
    const dtype *b2 = p + off;                 off += 16;
    Tensor w3={120,1,1,256,(dtype*)(p+off)};   off += 120*256;
    const dtype *b3 = p + off;                 off += 120;
    Tensor w4={10,1,1,120,(dtype*)(p+off)};
    const dtype *b4 = p + off;

#ifdef PERF
    T_START(total);   T_START(conv1);
#endif
    conv2d_5x5_s1(imgs,&ws->x1,&w1,b1);
    relu_ip(&ws->x1);
#ifdef PERF
    T_END(conv1);     T_START(pool1);
#endif
    avgpool2d(&ws->x1,&ws->x2,2);
#ifdef PERF
    T_END(pool1);     T_START(conv2);
#endif
    conv2d_5x5_s1(&ws->x2,&ws->x3,&w2,b2);
    relu_ip(&ws->x3);
#ifdef PERF
    T_END(conv2);     T_START(pool2);
#endif
    avgpool2d(&ws->x3,&ws->x4,2);
#ifdef PERF
    T_END(pool2);     T_START(fc1);
#endif
    memcpy(ws->x5.data, ws->x4.data,
           sizeof(dtype) * (size_t)imgs->n * 256);
    linear(&ws->x5,&ws->x6,&w3,b3);
    relu_ip(&ws->x6);
#ifdef PERF
    T_END(fc1);       T_START(fc2);
#endif
    linear(&ws->x6,&ws->x7,&w4,b4);
#ifdef PERF
    T_END(fc2);       T_START(softmax);
#endif
    for(int n=0;n<imgs->n;n++){
        dtype *logits = ws->x7.data + (size_t)n * 10;
        softmax(logits,10);
        pred[n]=argmax(logits,10);
    }
#ifdef PERF
    T_END(softmax);   T_END(total);
#endif
}


