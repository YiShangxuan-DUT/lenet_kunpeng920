#include "lenet.h"
#include <stdio.h>
#include <stdint.h>
#include <arpa/inet.h>

static uint32_t read_u32(FILE *f){
    uint32_t x; fread(&x,4,1,f); return ntohl(x);
}

int load_mnist_images(const char *path, Tensor *out){
    FILE *f=fopen(path,"rb"); if(!f){perror(path); exit(1);}
    uint32_t magic=read_u32(f);
    if(magic!=2051){fprintf(stderr,"bad magic\n"); exit(1);}
    int n=read_u32(f), h=read_u32(f), w=read_u32(f);
    *out = tensor_make(n,1,h,w);
    for(int i=0;i<n*h*w;i++){
        unsigned char px; fread(&px,1,1,f);
        out->data[i]=px/255.0f;
    }
    fclose(f); return n;
}

int load_mnist_labels(const char *path, unsigned char **lab_out){
    FILE *f=fopen(path,"rb"); if(!f){perror(path); exit(1);}
    uint32_t magic=read_u32(f); if(magic!=2049){fprintf(stderr,"bad label\n"); exit(1);}
    int n=read_u32(f);
    *lab_out=(unsigned char*)malloc(n);
    fread(*lab_out,1,n,f); fclose(f); return n;
}

