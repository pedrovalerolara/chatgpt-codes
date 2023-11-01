import Pkg
Pkg.add("CuArrays")
using CuArrays

function saxpy_cuda(n::Int32, a::Float32, x::CuVector{Float32}, y::CuVector{Float32})
    # Determine the thread index within the CUDA grid
    idx = threadIdx().x
    
    # Determine the total number of threads in the CUDA grid
    stride = blockDim().x * gridDim().x
    
    for i = idx:stride:n
        y[i] += a * x[i]
    end
    
    return nothing
end
using CuArrays
using CUDAnative

# Initialize CUDA
CUDAdrv.init()

# Input data
n = 100000
a = 2.0
x_host = rand(Float32, n)
y_host = rand(Float32, n)

# Transfer data to the GPU
x_gpu = CuArray(x_host)
y_gpu = CuArray(y_host)

# Set up the CUDA kernel
threads = 256
blocks = ceil(Int, n / threads)
@cuda threads=threads blocks=blocks saxpy_cuda(n, a, x_gpu, y_gpu)

# Transfer data back to the CPU
y_result = collect(y_gpu)

# Verify the result
for i in 1:n
    @assert y_result[i] ≈ (a * x_host[i] + y_host[i], atol=1e-6)
end

