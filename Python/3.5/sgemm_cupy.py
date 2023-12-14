import cupy as cp

N = 1000
M = 1000
K = 1000

# Generate random matrices using CuPy
A = cp.random.rand(N, M)
B = cp.random.rand(M, K)
C = cp.zeros((N, K))

# Perform matrix multiplication using CuPy
C = cp.dot(A, B)
