using CUDA

function sgemm_cuda(m::Int32, n::Int32, k::Int32, A::Matrix{Float32}, B::Matrix{Float32}, C::Matrix{Float32})
    # Transfer data to the GPU
    d_A = CUDA.CuArray(A)
    d_B = CUDA.CuArray(B)
    d_C = CUDA.CuArray(C)

    # Define kernel function
    function kernel_sgemm!(d_C, d_A, d_B, m, n, k)
        i, j = threadIdx().x, blockIdx().x
        if i <= m && j <= n
            temp = 0.0f0
            for l = 1:k
                temp += d_A[i, l] * d_B[l, j]
            end
            d_C[i, j] = temp
        end
        return
    end

    # Launch kernel
    threads_per_block = 256
    blocks_per_grid = (m + threads_per_block - 1) ÷ threads_per_block

    @cuda threads=threads_per_block blocks=blocks_per_grid kernel_sgemm!(d_C, d_A, d_B, m, n, k)

    # Transfer result back to CPU
    copyto!(C, d_C)
end

