#include <cuda_runtime.h>

__global__ void csr_spmv_kernel(int n, int *row_ptr, int *col_idx, double *values, double *x, double *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];
        double sum = 0.0;
        for (int j = row_start; j < row_end; j++)
        {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }
}

void csr_spmv(int n, int *row_ptr, int *col_idx, double *values, double *x, double *y)
{
    int *d_row_ptr, *d_col_idx;
    double *d_values, *d_x, *d_y; // Pointers for GPU memory

    // Allocate GPU memory for row_ptr, col_idx, values, x, and y
    cudaMalloc((void**)&d_row_ptr, (n + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_idx, row_ptr[n] * sizeof(int));
    cudaMalloc((void**)&d_values, row_ptr[n] * sizeof(double));
    cudaMalloc((void**)&d_x, n * sizeof(double));
    cudaMalloc((void**)&d_y, n * sizeof(double));

    // Copy data from CPU to GPU
    cudaMemcpy(d_row_ptr, row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx, row_ptr[n] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, row_ptr[n] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);

    // Define thread block size and compute grid dimensions
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the CSR SpMV CUDA kernel
    csr_spmv_kernel<<<gridSize, blockSize>>>(n, d_row_ptr, d_col_idx, d_values, d_x, d_y);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back from GPU to CPU
    cudaMemcpy(y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}
