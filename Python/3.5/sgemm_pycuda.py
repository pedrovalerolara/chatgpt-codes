import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Kernel code for matrix multiplication
kernel_code = """
__global__ void matrix_multiply(float *A, float *B, float *C, int N, int M, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < K) {
        float sum = 0.0f;
        for (int k = 0; k < M; ++k) {
            sum += A[i * M + k] * B[k * K + j];
        }
        C[i * K + j] = sum;
    }
}
"""

N = 1000
M = 1000
K = 1000

# Generate random matrices
A = np.random.rand(N, M).astype(np.float32)
B = np.random.rand(M, K).astype(np.float32)
C = np.zeros((N, K), dtype=np.float32)

# Allocate GPU memory
A_gpu = cuda.mem_alloc(A.nbytes)
B_gpu = cuda.mem_alloc(B.nbytes)
C_gpu = cuda.mem_alloc(C.nbytes)

# Copy matrices to GPU memory
cuda.memcpy_htod(A_gpu, A)
cuda.memcpy_htod(B_gpu, B)
cuda.memcpy_htod(C_gpu, C)

# Compile the kernel code and get function reference
mod = SourceModule(kernel_code)
matrix_multiply = mod.get_function("matrix_multiply")

# Define block and grid dimensions
block_size = (16, 16)
grid_size = ((N + block_size[0] - 1) // block_size[0], (K + block_size[1] - 1) // block_size[1])

# Call the kernel
matrix_multiply(A_gpu, B_gpu, C_gpu, np.int32(N), np.int32(M), np.int32(K), block=block_size, grid=grid_size)

# Copy the result back to the host
cuda.memcpy_dtoh(C, C_gpu)

print("Resultant matrix C:")
print(C)
