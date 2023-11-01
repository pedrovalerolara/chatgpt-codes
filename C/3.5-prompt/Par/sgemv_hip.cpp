#include <hip/hip_runtime.h>

__global__ void sgemv_kernel(int m, int n, float alpha, float *a, float *x, float beta, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        float sum = 0.0;
        for (int j = 0; j < n; j++)
        {
            sum += a[i * n + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}

void sgemv(int m, int n, float alpha, float *a, float *x, float beta, float *y)
{
    // Define thread block size and compute grid dimensions
    int blockSize = 256;
    int gridSize = (m + blockSize - 1) / blockSize;
    // Launch the sgemv HIP kernel
    hipLaunchKernelGGL(sgemv_kernel, gridSize, blockSize, 0, 0, m, n, alpha, a, x, beta, y);
    // Wait for the kernel to finish
    hipDeviceSynchronize();
}
