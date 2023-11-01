#include <cuda_runtime.h>

__global__ void jacobi3D_kernel(double *in, double *out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < N - 1 && j < N - 1 && k < N - 1)
    {
        int index = i * N * N + j * N + k;
        out[index] = (in[index - 1] + in[index + 1] + in[index - N] + in[index + N] + in[index - N * N] + in[index + N * N]) / 6.0;
    }
}

void jacobi3D(double *in, double *out, int N, int T)
{
    double *d_in, *d_out;
    size_t size = N * N * N * sizeof(double);

    // Allocate GPU memory for in and out
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    // Copy data from CPU to GPU
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    dim3 blockDim(8, 8, 8); // Choose an appropriate block size
    dim3 gridDim((N - 2 + blockDim.x - 1) / blockDim.x, (N - 2 + blockDim.y - 1) / blockDim.y, (N - 2 + blockDim.z - 1) / blockDim.z);

    for (int t = 0; t < T; t++)
    {
        jacobi3D_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);

        // Swap in and out pointers
        double *temp = d_in;
        d_in = d_out;
        d_out = temp;
    }

    // Copy the result back from GPU to CPU
    cudaMemcpy(out, d_in, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_in);
    cudaFree(d_out);
}
