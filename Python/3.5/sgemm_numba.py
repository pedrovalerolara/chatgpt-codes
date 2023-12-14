import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def matrix_multiplication(A, B):
    N, M = A.shape
    M, K = B.shape
    C = np.zeros((N, K))

    for i in prange(N):
        for j in prange(K):
            for k in range(M):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

N = 1000
M = 1000
K = 1000

A = np.random.rand(N, M)
B = np.random.rand(M, K)

result = matrix_multiplication(A, B)
