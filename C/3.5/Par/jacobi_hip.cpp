#include <hip/hip_runtime.h>

__global__ void jacobi3D_kernel(double ***in, double ***out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < N - 1 && j < N - 1 && k < N - 1)
    {
        out[i][j][k] = (in[i - 1][j][k] + in[i + 1][j][k] + in[i][j - 1][k] + in[i][j + 1][k] + in[i][j][k - 1] + in[i][j][k + 1]) / 6.0;
    }
}

void jacobi3D(double ***in, double ***out, int N, int T)
{
    double ***temp;

    dim3 blockDim(8, 8, 8);  // Adjust block dimensions as needed
    dim3 gridDim((N - 2 + blockDim.x - 1) / blockDim.x, (N - 2 + blockDim.y - 1) / blockDim.y, (N - 2 + blockDim.z - 1) / blockDim.z);

    for (int t = 0; t < T; t++)
    {
        jacobi3D_kernel<<<gridDim, blockDim>>>(in, out, N);
        hipDeviceSynchronize();

        // Swap in and out pointers
        temp = out;
        out = in;
        in = temp;
     }
}
