nclude <hip/hip_runtime.h>

__global__ void sgemm_kernel(int m, int n, int k, float *a, float *b, float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n)
    {
        float sum = 0.0;
        for (int l = 0; l < k; l++)
        {
            sum += a[i * k + l] * b[l * n + j];
        }
        c[i * n + j] = sum;
    }
}

void sgemm(int m, int n, int k, float *a, float *b, float *c)
{
    dim3 blockSize(16, 16); // Adjust block size as needed
    dim3 gridSize((m + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    hipLaunchKernelGGL(sgemm_kernel, gridSize, blockSize, 0, 0, m, n, k, a, b, c);

    hipDeviceSynchronize();
}
