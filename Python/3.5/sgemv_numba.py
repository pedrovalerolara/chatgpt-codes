import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def parallel_matrix_vector_mul(A, x, y):
    N, M = A.shape
    
    for i in prange(N):
        for j in range(M):
            y[i] += A[i, j] * x[j]

N = 1000
M = 1000

A = np.random.rand(N, M)
x = np.random.rand(M)
y = np.zeros(N)

parallel_matrix_vector_mul(A, x, y)

print(y)
