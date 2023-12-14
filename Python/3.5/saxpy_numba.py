import numpy as np
from numba import jit, prange

@jit(parallel=True)
def parallel_code(x, y, a):
    N = len(x)
    for i in prange(N):
        y[i] = a * x[i] + y[i]

N = 10000000
x = np.random.rand(N)
y = np.random.rand(N)
a = 2.0

parallel_code(x, y, a)
