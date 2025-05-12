//neon | baseline | 循环展开
#include "lenet.h"

Workspace workspace_make(int N){
    Workspace ws;
    ws.x1 = tensor_make(N,  6,24,24);
    ws.x2 = tensor_make(N,  6,12,12);
    ws.x3 = tensor_make(N, 16, 8, 8);
    ws.x4 = tensor_make(N, 16, 4, 4);
    ws.x5 = tensor_make(N,256, 1, 1);
    ws.x6 = tensor_make(N,120, 1, 1);
    ws.x7 = tensor_make(N, 10, 1, 1);
    return ws;
}
void workspace_free(Workspace *ws){
    tensor_free(&ws->x1); tensor_free(&ws->x2); tensor_free(&ws->x3);
    tensor_free(&ws->x4); tensor_free(&ws->x5); tensor_free(&ws->x6);
    tensor_free(&ws->x7);
}


