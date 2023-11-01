#include <cuda_runtime.h>

__global__ void sgemm_kernel(int m, int n, int k, float *a, float *b, float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n)
    {
        float sum = 0.0f;
        for (int l = 0; l < k; l++)
        {
            sum += a[i * k + l] * b[l * n + j];
        }
        c[i * n + j] = sum;
    }
}

void sgemm(int m, int n, int k, float *a, float *b, float *c)
{
    float *d_a, *d_b, *d_c; // Pointers for GPU memory

    // Allocate GPU memory for a, b, and c
    cudaMalloc((void**)&d_a, m * k * sizeof(float));
    cudaMalloc((void**)&d_b, k * n * sizeof(float));
    cudaMalloc((void**)&d_c, m * n * sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Define thread block size and compute grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // Launch the SGEMM CUDA kernel
    sgemm_kernel<<<gridDim, blockDim>>>(m, n, k, d_a, d_b, d_c);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back from GPU to CPU
    cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
