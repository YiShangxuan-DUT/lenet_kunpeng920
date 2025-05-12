//baseline

/* ---------------  src/lenet.c  --------------- */
#include "lenet.h"
#include "utils.h"
#include <string.h>
#include <math.h>

/* ---------- 基础算子 ---------- */
void conv2d(const Tensor *in, Tensor *out,
            const Tensor *w, const float *b,
            int stride, int pad)
{
    for (int n = 0; n < in->n; n++)
    for (int oc = 0; oc < out->c; oc++)
    for (int oh = 0; oh < out->h; oh++)
    for (int ow = 0; ow < out->w; ow++) {
        float sum = b[oc];
        for (int ic = 0; ic < in->c; ic++)
        for (int kh = 0; kh < w->h; kh++)
        for (int kw = 0; kw < w->w; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < in->h && iw >= 0 && iw < in->w)
                sum += IDX4(in,n,ic,ih,iw) * IDX4(w,oc,ic,kh,kw);
        }
        IDX4(out,n,oc,oh,ow) = sum;
    }
}

void avgpool2d(const Tensor *in, Tensor *out, int k)
{
    for (int n = 0; n < in->n; n++)
    for (int c = 0; c < in->c; c++)
    for (int oh = 0; oh < out->h; oh++)
    for (int ow = 0; ow < out->w; ow++) {
        float sum = 0.f;
        for (int kh = 0; kh < k; kh++)
        for (int kw = 0; kw < k; kw++) {
            int ih = oh * k + kh, iw = ow * k + kw;
            sum += IDX4(in,n,c,ih,iw);
        }
        IDX4(out,n,c,oh,ow) = sum / (k * k);
    }
}

void relu_ip(Tensor *t)
{
    int sz = t->n * t->c * t->h * t->w;
    for (int i = 0; i < sz; ++i)
        if (t->data[i] < 0) t->data[i] = 0;
}

void linear(const Tensor *in, Tensor *out,
            const Tensor *w, const float *b)
{
    int N = in->n, in_dim = in->c * in->h * in->w, out_dim = out->c;
    for (int n = 0; n < N; ++n)
        for (int o = 0; o < out_dim; ++o) {
            const float *x = in->data + n * in_dim;
            const float *k = w->data + o * in_dim;
            float sum = b[o];
            for (int i = 0; i < in_dim; ++i) sum += x[i] * k[i];
            out->data[n * out_dim + o] = sum;
        }
}

int  argmax (const float *x,int n){int id=0;for(int i=1;i<n;i++)if(x[i]>x[id])id=i;return id;}
void softmax(float *x,int n){
    float m=x[0]; for(int i=1;i<n;i++) if(x[i]>m) m=x[i];
    float s=0.f;  for(int i=0;i<n;i++){ x[i]=expf(x[i]-m); s+=x[i]; }
    for(int i=0;i<n;i++) x[i]/=s;
}

/* ---------- 单张前向 (若仍需要) ---------- */
int lenet_forward(const Tensor *img,const float *p,Tensor *work)
{
    /* … 与最早版本相同，可留作对照 … */
    return 0;
}

/* ---------- 批量前向 (含 PERF 计时) ---------- */
void lenet_forward_batch(const Tensor *imgs,const float *p,
                         Workspace *ws,int *pred)
{
    /* ------------ 权重切片 ------------ */
    int off = 0;
    Tensor w1={6,1,5,5,(float*)(p+off)};  off+=6*1*5*5;  const float *b1=p+off; off+=6;
    Tensor w2={16,6,5,5,(float*)(p+off)}; off+=16*6*5*5; const float *b2=p+off; off+=16;
    Tensor w3={120,1,1,256,(float*)(p+off)}; off+=120*256; const float *b3=p+off; off+=120;
    Tensor w4={10,1,1,120,(float*)(p+off)};                  const float *b4=p+off;

#ifdef PERF
    T_START(total);   T_START(conv1);
#endif
    conv2d(imgs,&ws->x1,&w1,b1,1,0);
    relu_ip(&ws->x1);
#ifdef PERF
    T_END(conv1);     T_START(pool1);
#endif
    avgpool2d(&ws->x1,&ws->x2,2);
#ifdef PERF
    T_END(pool1);     T_START(conv2);
#endif
    conv2d(&ws->x2,&ws->x3,&w2,b2,1,0);
    relu_ip(&ws->x3);
#ifdef PERF
    T_END(conv2);     T_START(pool2);
#endif
    avgpool2d(&ws->x3,&ws->x4,2);
#ifdef PERF
    T_END(pool2);     T_START(fc1);
#endif
    memcpy(ws->x5.data, ws->x4.data,
           sizeof(float)*imgs->n*256);
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
        float *logits = ws->x7.data + n*10;
        softmax(logits,10);
        pred[n]=argmax(logits,10);
    }
#ifdef PERF
    T_END(softmax);   T_END(total);
#endif
}