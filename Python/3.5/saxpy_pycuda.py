import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

N = 10000000
x = np.random.rand(N).astype(np.float32)
y = np.random.rand(N).astype(np.float32)
a = np.float32(2.0)

# Allocate memory on the GPU
x_gpu = cuda.mem_alloc(x.nbytes)
y_gpu = cuda.mem_alloc(y.nbytes)

# Transfer data to the GPU
cuda.memcpy_htod(x_gpu, x)
cuda.memcpy_htod(y_gpu, y)

# Define the CUDA kernel
mod = SourceModule("""
    __global__ void multiply(float *x, float *y, float a, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            y[idx] = a * x[idx] + y[idx];
        }
    }
""")

# Get the function from the compiled module
multiply = mod.get_function("multiply")

# Set up grid and block sizes
block_size = 256
grid_size = (N + block_size - 1) // block_size

# Launch the kernel
multiply(x_gpu, y_gpu, a, np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1))

# Transfer the results back to the CPU
cuda.memcpy_dtoh(y, y_gpu)

# Verify the results
for i in range(N):
    assert np.isclose(y[i], a * x[i] + y[i])

print("Computation successful!")
