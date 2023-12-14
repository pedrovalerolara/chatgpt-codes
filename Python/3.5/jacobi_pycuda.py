import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

def jacobi_parallel(A, B, num_iterations):
    m, n, p = A.shape
    a_gpu = cuda.mem_alloc(A.nbytes)
    b_gpu = cuda.mem_alloc(B.nbytes)
    cuda.memcpy_htod(a_gpu, A)
    cuda.memcpy_htod(b_gpu, B)

    mod = SourceModule("""
        __global__ void jacobi_kernel(float *A, float *B, int m, int n, int p) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            int l = blockIdx.z * blockDim.z + threadIdx.z;
            
            if (i > 0 && i < m - 1 && j > 0 && j < n - 1 && l > 0 && l < p - 1) {
                int index = i * n * p + j * p + l;
                B[index] = 0.125f * (A[index - n * p] + A[index + n * p] + A[index - p] + A[index + p] + A[index - 1] + A[index + 1]);
            }
        }
    """)

    jacobi_kernel = mod.get_function("jacobi_kernel")

    block_size = (8, 8, 8)
    grid_size = ((m - 2) // block_size[0] + 1, (n - 2) // block_size[1] + 1, (p - 2) // block_size[2] + 1)

    for k in range(num_iterations):
        jacobi_kernel(a_gpu, b_gpu, np.int32(m), np.int32(n), np.int32(p), block=block_size, grid=grid_size)
        a_gpu, b_gpu = b_gpu, a_gpu

    result = np.empty_like(A)
    cuda.memcpy_dtoh(result, a_gpu)
    return result

# Example usage
m, n, p = 10, 10, 10
num_iterations = 100

# Create input matrices A and B
A = np.random.rand(m, n, p).astype(np.float32)
B = np.zeros((m, n, p), dtype=np.float32)

result = jacobi_parallel(A, B, num_iterations)
