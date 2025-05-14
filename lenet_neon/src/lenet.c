//neon
/* ----------  src/lenet.c  ---------- */
#include "lenet.h"
#include "utils.h"

#ifdef USE_NEON
#  include <arm_neon.h>
#endif

#include <string.h>
#include <math.h>


/* ------------------------------------------------------------- *
 * 1. 通用 2-D 卷积 (慢速参考版本，仍保留方便对拍)
 * ------------------------------------------------------------- */
void conv2d(const Tensor *in, Tensor *out,
            const Tensor *w, const dtype *b,
            int stride, int pad)
{
    for (int n = 0; n < in->n; n++)
    for (int oc = 0; oc < out->c; oc++)
    for (int oh = 0; oh < out->h; oh++)
    for (int ow = 0; ow < out->w; ow++) {
        acc_t sum = (acc_t)b[oc];
        for (int ic = 0; ic < in->c; ic++)
        for (int kh = 0; kh < w->h; kh++)
        for (int kw = 0; kw < w->w; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < in->h && iw >= 0 && iw < in->w) {
                dtype a = IDX4(in,n,ic,ih,iw);
                dtype k = IDX4(w,oc,ic,kh,kw);
                sum += (acc_t)a * (acc_t)k;
            }
        }
        IDX4(out,n,oc,oh,ow) = (dtype)sum;
    }
}

/* ------------------------------------------------------------- *
 * 2. C 版专用 5×5, stride=1, pad=0
 * ------------------------------------------------------------- */
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

/* ------------------------------------------------------------- *
 * 3. 手写 NEON 版 5×5 (IC×OC 任意，NHWC = NCHW 5×5-same)
 *    —— 4-lane 向量化 + 行尾标量累加 (ADD_TAIL 宏)
 * ------------------------------------------------------------- */
#ifdef USE_NEON
/* helper：安全地把行尾单点贡献进 accq 的 lane-0 */
#define ADD_TAIL(ptr_row, kernel_val)               \
    do {                                            \
        float tmp = (ptr_row) * (kernel_val);       \
        acc_scalar += tmp;                          \
    } while(0)

static void conv2d_5x5_s1_neon(const Tensor *in, Tensor *out,
                               const Tensor *w, const dtype *bias)
{
    const int N  = in->n,   IC = in->c,  IH = in->h,  IW = in->w;
    const int OC = out->c,  OH = out->h,  OW = out->w;

    for (int n = 0; n < N; ++n) {
        const dtype *in_n  = in->data  + (size_t)n * IC * IH * IW;
        dtype       *out_n = out->data + (size_t)n * OC * OH * OW;

        for (int oc = 0; oc < OC; ++oc) {
            const dtype *k_oc = w->data + (size_t)oc * IC * 25;
            dtype       *outm = out_n  + (size_t)oc * OH * OW;

            for (int oh = 0; oh < OH; ++oh) {
                dtype *outptr = outm + (size_t)oh * OW;

                for (int ow = 0; ow < OW; ++ow) {
                    float32x4_t accq      = vdupq_n_f32(0.f);
                    acc_t       acc_scalar = (acc_t)bias[oc];

                    for (int ic = 0; ic < IC; ++ic) {
                        const dtype *base = in_n + (size_t)ic * IH * IW + (size_t)oh * IW + ow;
                        const dtype *r0   = base;
                        const dtype *r1   = r0 + IW;
                        const dtype *r2   = r1 + IW;
                        const dtype *r3   = r2 + IW;
                        const dtype *r4   = r3 + IW;

                        const dtype *kptr = k_oc + (size_t)ic * 25;

                        /* ---- 行 0 ---- */
                        accq = vfmaq_f32(accq, vld1q_f32((float const *)r0), vld1q_f32((float const *)kptr));
                        ADD_TAIL((float)r0[4], (float)kptr[4]);
                        kptr += 5;
                        /* ---- 行 1 ---- */
                        accq = vfmaq_f32(accq, vld1q_f32((float const *)r1), vld1q_f32((float const *)kptr));
                        ADD_TAIL((float)r1[4], (float)kptr[4]);
                        kptr += 5;
                        /* ---- 行 2 ---- */
                        accq = vfmaq_f32(accq, vld1q_f32((float const *)r2), vld1q_f32((float const *)kptr));
                        ADD_TAIL((float)r2[4], (float)kptr[4]);
                        kptr += 5;
                        /* ---- 行 3 ---- */
                        accq = vfmaq_f32(accq, vld1q_f32((float const *)r3), vld1q_f32((float const *)kptr));
                        ADD_TAIL((float)r3[4], (float)kptr[4]);
                        kptr += 5;
                        /* ---- 行 4 ---- */
                        accq = vfmaq_f32(accq, vld1q_f32((float const *)r4), vld1q_f32((float const *)kptr));
                        ADD_TAIL((float)r4[4], (float)kptr[4]);
                    }

                    /* 将 NEON 寄存器累加到标量，并加上尾点累加 */
                    float32x2_t vtmp    = vadd_f32(vget_low_f32(accq), vget_high_f32(accq));
                    acc_t       sum_vec = (acc_t)(vget_lane_f32(vtmp, 0) + vget_lane_f32(vtmp, 1));
                    acc_t       total   = sum_vec + acc_scalar;

                    outptr[ow] = (dtype)total;
                }
            }
        }
    }
}
#endif /* USE_NEON */

#ifdef USE_NEON_ASM
// extern void kern25_fma4(const dtype *in, const dtype *ker,
//                         dtype *out, int ic, int iw, dtype bias);
extern void kern25_fma4(const dtype *in, const dtype *ker,
                         dtype *out, int ic, int iw, int ih, dtype bias);


static void conv2d_5x5_s1_neon_asm(const Tensor *in, Tensor *out,
                                   const Tensor *w, const dtype *bias)
{
    const int N=in->n, IC=in->c, IH=in->h, IW=in->w;
    const int OC=out->c, OH=out->h, OW=out->w;

    for (int n=0; n<N; n++){
        const dtype *in_n  = in->data  + (size_t)n*IC*IH*IW;
        dtype       *out_n = out->data + (size_t)n*OC*OH*OW;

        for (int oc=0; oc<OC; oc++){
            const dtype *k_oc = w->data + (size_t)oc*IC*25;
            dtype       *outm = out_n   + (size_t)oc*OH*OW;

            for (int oh=0; oh<OH; oh++){
                dtype *outptr = outm + (size_t)oh*OW;
                for (int ow=0; ow+3<OW; ow+=4){
                    kern25_fma4(in_n + (size_t)oh*IW + ow,
                                k_oc, outptr + ow,
                                IC, IW,IH, bias[oc]);
                }
                /* 尾列不足 4 个：退回 intrinsic 版 */
                for (int ow_tail=(OW&~3); ow_tail<OW; ow_tail++){
                    float32x4_t z = vdupq_n_f32(0.f);
                    acc_t acc = (acc_t)bias[oc];
                    for (int ic=0; ic<IC; ic++){
                        const dtype *base = in_n + (size_t)ic*IH*IW
                                                  + (size_t)oh*IW + ow_tail;
                        const dtype *kptr = k_oc + (size_t)ic*25;
                        for (int kh=0; kh<5; kh++){
                            acc += (acc_t)base[kh*IW+0] * (acc_t)kptr[kh*5+0];
                            acc += (acc_t)base[kh*IW+1] * (acc_t)kptr[kh*5+1];
                            acc += (acc_t)base[kh*IW+2] * (acc_t)kptr[kh*5+2];
                            acc += (acc_t)base[kh*IW+3] * (acc_t)kptr[kh*5+3];
                            acc += (acc_t)base[kh*IW+4] * (acc_t)kptr[kh*5+4];
                        }
                    }
                    outptr[ow_tail] = (dtype)acc;
                }
            }
        }
    }
}
#endif /* USE_NEON_ASM */



/* ------------------------------------------------------------- *
 * 4. 其余算子：池化 / ReLU / FC / Softmax
 * ------------------------------------------------------------- */
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

int argmax(const dtype *x, int n)
{
    int id = 0;
    for (int i = 1; i < n; ++i)
        if ((acc_t)x[i] > (acc_t)x[id]) id = i;
    return id;
}

void softmax(dtype *x, int n)
{
    acc_t m = (acc_t)x[0];
    for (int i = 1; i < n; ++i)
        if ((acc_t)x[i] > m) m = (acc_t)x[i];

    acc_t s = 0;
    for (int i = 0; i < n; ++i) {
        acc_t t = expf((acc_t)x[i] - m);
        x[i] = (dtype)t;
        s   += t;
    }
    acc_t s_inv = 1.0f / s;
    for (int i = 0; i < n; ++i)
        x[i] = (dtype)((acc_t)x[i] * s_inv);
}

/* ------------------------------------------------------------- *
 * 5. LeNet-5 批量前向：调用上面算子
 * ------------------------------------------------------------- */
void lenet_forward_batch(const Tensor *imgs, const dtype *p,
                         Workspace *ws, int *pred)
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

#if defined(USE_NEON_ASM)
  #define CONV5x5  conv2d_5x5_s1_neon_asm
#elif defined(USE_NEON)
  #define CONV5x5  conv2d_5x5_s1_neon
#else
  #define CONV5x5  conv2d_5x5_s1
#endif


#ifdef PERF
    T_START(total);   T_START(conv1);
#endif
    CONV5x5(imgs,      &ws->x1, &w1, b1);
    relu_ip(&ws->x1);
#ifdef PERF
    T_END(conv1);     T_START(pool1);
#endif
    avgpool2d(&ws->x1, &ws->x2, 2);
#ifdef PERF
    T_END(pool1);     T_START(conv2);
#endif
    CONV5x5(&ws->x2,   &ws->x3, &w2, b2);
    relu_ip(&ws->x3);
#ifdef PERF
    T_END(conv2);     T_START(pool2);
#endif
    avgpool2d(&ws->x3, &ws->x4, 2);
#ifdef PERF
    T_END(pool2);     T_START(fc1);
#endif
    memcpy(ws->x5.data, ws->x4.data,
           sizeof(dtype) * (size_t)imgs->n * 256);
    linear(&ws->x5, &ws->x6, &w3, b3);
    relu_ip(&ws->x6);
#ifdef PERF
    T_END(fc1);       T_START(fc2);
#endif
    linear(&ws->x6, &ws->x7, &w4, b4);
#ifdef PERF
    T_END(fc2);       T_START(softmax);
#endif
    for (int n = 0; n < imgs->n; n++) {
        dtype *logits = ws->x7.data + (size_t)n * 10;
        softmax(logits, 10);
        pred[n] = argmax(logits, 10);
    }
#ifdef PERF
    T_END(softmax);   T_END(total);
#endif
}

