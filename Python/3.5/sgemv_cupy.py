import cupy as cp

N = 1000
M = 1000

A = cp.random.rand(N, M)
x = cp.random.rand(M)
y = cp.zeros(N)

for i in range(N):
    y[i] = cp.sum(A[i] * x)

print(y.get())  # Use .get() to transfer the data back to the CPU for printing
