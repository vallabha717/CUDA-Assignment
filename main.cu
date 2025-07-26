#include <stdio.h>
#include <cuda_runtime.h>
#define N 1024

_global_ void reduce_sum(float* input, float* output) {
    extern _shared_ float sdata[];
    unsigned int tid = threadIdx.x;
    sdata[tid] = input[tid];
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) *output = sdata[0];
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;
    size_t size = N * sizeof(float);
    h_input = (float*) malloc(size);
    h_output = (float*) malloc(sizeof(float));

    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    reduce_sum<<<1, N, N * sizeof(float)>>>(d_input, d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Sum = %f\n", h_output[0]);

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    return 0;
}