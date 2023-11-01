using Pkg
Pkg.add("ROCArray")
using ROCArray

function sgemm_amdgpu(m::Int32, n::Int32, k::Int32, A::ROCMatrix{Float32}, B::ROCMatrix{Float32}, C::ROCMatrix{Float32})
    # Define kernel function to perform matrix multiplication on AMD GPU
    function kernel_sgemm!(C, A, B, m, n, k)
        i, j = threadIdx().x, blockIdx().x
        if i <= m && j <= n
            temp = 0.0f0
            for l = 1:k
                temp += A[i, l] * B[l, j]
            end
            C[i, j] = temp
        end
        return
    end

    # Launch the kernel on AMD GPU
    threads_per_block = 256
    blocks_per_grid = (m + threads_per_block - 1) ÷ threads_per_block

    @roc threads=threads_per_block blocks=blocks_per_grid kernel_sgemm!(C, A, B, m, n, k)

    # Wait for the kernel to finish
    ROC.synchronize()
end

