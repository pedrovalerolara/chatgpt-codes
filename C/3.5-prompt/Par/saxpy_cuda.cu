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
    // Define thread block size and compute grid dimensions
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the saxpy CUDA kernel
    saxpy_kernel<<<gridSize, blockSize>>>(n, a, x, y);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();
}
