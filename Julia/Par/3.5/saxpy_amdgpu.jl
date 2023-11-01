// saxpy_rocm.cpp
#include <hc.hpp>

extern "C" void saxpy_kernel(int n, float a, float* x, float* y) {
    hc::parallel_for_each(hc::extent<1>(n), [=](hc::index<1> idx) [[hc]] {
        int i = idx[0];
        y[i] += a * x[i];
    });
}
hcc -o saxpy_rocm.cpp -o saxpy_rocm
using CUDAnative

function saxpy_rocm(n::Int32, a::Float32, x::CuArray{Float32}, y::CuArray{Float32})
    saxpy_kernel(n, a, x, y)
end
using CUDAnative

# Input data
n = 100000
a = 2.0
x_host = rand(Float32, n)
y_host = rand(Float32, n)

# Transfer data to the GPU
x_gpu = CuArray(x_host)
y_gpu = CuArray(y_host)

# Call the ROCm HCC kernel
saxpy_rocm(n, a, x_gpu, y_gpu)

# Transfer data back to the CPU
y_result = collect(y_gpu)

# Verify the result
for i in 1:n
    @assert y_result[i] ≈ (a * x_host[i] + y_host[i], atol=1e-6)
end

