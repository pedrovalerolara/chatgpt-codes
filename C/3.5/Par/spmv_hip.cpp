#include <hip/hip_runtime.h>

__global__ void csr_spmv_kernel(int n, int *row_ptr, int *col_idx, double *values, double *x, double *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];
        for (int j = row_start; j < row_end; j++)
        {
            y[i] += values[j] * x[col_idx[j]];
        }
    }
}

void csr_spmv(int n, int *row_ptr, int *col_idx, double *values, double *x, double *y)
{
    // Define thread block size and compute grid dimensions
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    // Launch the csr_spmv HIP kernel
    hipLaunchKernelGGL(csr_spmv_kernel, gridSize, blockSize, 0, 0, n, row_ptr, col_idx, values, x, y);
    hipDeviceSynchronize();
}
