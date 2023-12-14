import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Define the CUDA kernel code
cuda_kernel_code = """
__global__ void matrixVectorMul(float *A, float *x, float *y, int N, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        for (int j = 0; j < M; ++j) {
            y[i] += A[i * M + j] * x[j];
        }
    }
}
"""

# Compile the CUDA kernel code
mod = SourceModule(cuda_kernel_code)
matrixVectorMul = mod.get_function("matrixVectorMul")

N = 1000
M = 1000

# Initialize arrays on the host
A = np.random.rand(N, M).astype(np.float32)
x = np.random.rand(M).astype(np.float32)
y = np.zeros(N, dtype=np.float32)

# Allocate memory on the device and copy data to device
A_gpu = cuda.mem_alloc(A.nbytes)
x_gpu = cuda.mem_alloc(x.nbytes)
y_gpu = cuda.mem_alloc(y.nbytes)

cuda.memcpy_htod(A_gpu, A)
cuda.memcpy_htod(x_gpu, x)
cuda.memcpy_htod(y_gpu, y)

# Define block and grid dimensions for CUDA kernel
block_size = 256
grid_size = (N + block_size - 1) // block_size

# Launch the CUDA kernel
matrixVectorMul(A_gpu, x_gpu, y_gpu, np.int32(N), np.int32(M), block=(block_size, 1, 1), grid=(grid_size, 1))

# Copy the result back to the host
cuda.memcpy_dtoh(y, y_gpu)

print(y)
