#include <cuda_runtime.h>

__global__ void sgemv_kernel(int m, int n, float alpha, float *a, float *x, float beta, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (i < m)
    {
        for (int j = 0; j < n; j++)
        {
            sum += a[i * n + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}

void sgemv(int m, int n, float alpha, float *a, float *x, float beta, float *y)
{
    float *d_a, *d_x, *d_y; // Pointers for GPU memory

    // Allocate GPU memory for a, x, and y
    cudaMalloc((void**)&d_a, m * n * sizeof(float));
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, m * sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(d_a, a, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, m * sizeof(float), cudaMemcpyHostToDevice);

    // Define thread block size and compute grid dimensions
    int blockSize = 256;
    int gridSize = (m + blockSize - 1) / blockSize;

    // Launch the SGEMV CUDA kernel
    sgemv_kernel<<<gridSize, blockSize>>>(m, n, alpha, d_a, d_x, beta, d_y);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back from GPU to CPU
    cudaMemcpy(y, d_y, m * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_x);
    cudaFree(d_y);
}
