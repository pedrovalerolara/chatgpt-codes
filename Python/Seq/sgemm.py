# SGEMM using numpy arrays
import numpy as np

N = 1000
M = 1000
K = 1000

A = np.random.rand(N, M)
B = np.random.rand(M, K)
C = np.random.rand(N, K)

for i in range(N):
    for j in range(K):
        for k in range(M):
            C[i][j] += A[i][k] * B[k][j]
