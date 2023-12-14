import cupy as cp

N = 10000000
x = cp.random.rand(N)
y = cp.random.rand(N)
a = 2.0

x_gpu = cp.asarray(x)
y_gpu = cp.asarray(y)

a_gpu = cp.asarray(a)

# Perform the operation using element-wise multiplication and addition
y_gpu = a_gpu * x_gpu + y_gpu

# Transfer the result back to a NumPy array if needed
y_result = cp.asnumpy(y_gpu)
