# Python GEMV using numpy arrays
import numpy as np

N = 1000
M = 1000

A = np.random.rand(N, M)
x = np.random.rand(M)
y = np.random.rand(N)

for i in range(N):
    for j in range(M):
        y[i] += A[i][j] * x[j]

print(y)