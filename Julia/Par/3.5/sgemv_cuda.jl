using CuArrays

function sgemv_cuda(m::Int32, n::Int32, A::Matrix{Float32}, x::Vector{Float32}, y::Vector{Float32})
    # Copy data to GPU
    A_gpu = cu(A)
    x_gpu = cu(x)
    y_gpu = cu(y)

    # Determine grid and block dimensions for parallel execution
    threads_per_block = 256  # Adjust this value based on your GPU's capabilities
    blocks_per_grid = ceil(Int, m * n / threads_per_block)

    # Define a CUDA kernel to perform matrix-vector multiplication
    @cuda threads=threads_per_block blocks=blocks_per_grid sgemv_kernel(m, n, A_gpu, x_gpu, y_gpu)

    # Copy the result back to the CPU
    copyto!(y, y_gpu)
end

# Define the CUDA kernel
function sgemv_kernel(m, n, A, x, y)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i <= m
        temp = 0.0f0
        for j = 1:n
            temp += A[i, j] * x[j]
        end
        y[i] += temp
    end

    return
end

