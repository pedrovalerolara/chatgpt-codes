import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

# CUDA kernel for SPMV
kernel_code = """
__global__ void spmv(int n, int nnz, float *val, int *row, int *col, float *x, float *y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        for (int j = row[tid] - 1; j < row[tid + 1] - 1; ++j) {
            atomicAdd(&y[tid], val[j] * x[col[j]]);
        }
    }
}
"""

def spmv(n, nnz, val, row, col, x, y):
    # Transfer data to GPU
    val_gpu = cuda.to_device(val.astype(np.float32))
    row_gpu = cuda.to_device(row.astype(np.int32))
    col_gpu = cuda.to_device(col.astype(np.int32))
    x_gpu = cuda.to_device(x.astype(np.float32))
    y_gpu = cuda.mem_alloc(y.nbytes)
    
    cuda.memcpy_htod(y_gpu, y.astype(np.float32))

    # Compile the kernel code
    mod = SourceModule(kernel_code)
    func = mod.get_function("spmv")

    # Define block and grid sizes
    block_size = 128
    grid_size = (n + block_size - 1) // block_size

    # Execute kernel
    func(np.int32(n), np.int32(nnz), val_gpu, row_gpu, col_gpu, x_gpu, y_gpu, block=(block_size, 1, 1), grid=(grid_size, 1))

    # Copy the result back to the host
    cuda.memcpy_dtoh(y, y_gpu)

# Example usage
n = 10  # Matrix size
nnz = 15  # Number of non-zero elements
val = np.random.rand(nnz)
row = np.sort(np.random.randint(1, n, nnz + 1))  # Ensure row indices are sorted
col = np.random.randint(0, n, nnz)
x = np.random.rand(n)
y = np.zeros(n)

spmv(n, nnz, val, row, col, x, y)
print("Result:", y)
