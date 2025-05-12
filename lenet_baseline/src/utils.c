//baseline

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
    void *buf=malloc(sz); fread(buf,1,sz,f); fclose(f);
    if(bytes) *bytes=sz; return buf;
}

/* ------ Profiler data ------ */
Prof g_prof = {0};




