
# Saxpy using numpy arrays
import numpy as np

N = 10000000
x = np.random.rand(N)
y = np.random.rand(N)
a = 2.0

for i in range(N):
    y[i] = a*x[i] + y[i]