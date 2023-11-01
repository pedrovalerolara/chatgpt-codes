#include <cuda_runtime.h>

__global__ void saxpy_kernel(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        y[i] = a * x[i] + y[i];
    }
}

void saxpy(int n, float a, float *x, float *y)
{
    float *d_x, *d_y; // Pointers for GPU memory

    // Allocate GPU memory for x and y
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define thread block size and compute grid dimensions
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the SAXPY CUDA kernel
    saxpy_kernel<<<gridSize, blockSize>>>(n, a, d_x, d_y);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back from GPU to CPU
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
}
