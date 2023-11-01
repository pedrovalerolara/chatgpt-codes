using CUDA

function jacobi_cuda(A::CuArray{Float64, 3}, B::CuArray{Float64, 3}, num_iterations::Int)
    m, n, p = size(A)

    # Define a kernel to perform the Jacobi iteration
    function jacobi_kernel!(B, A)
        i, j, l = threadIdx().x, threadIdx().y, blockIdx().x
        if 2 <= i <= m - 1 && 2 <= j <= n - 1 && 2 <= l <= p - 1
            B[i, j, l] = 0.125 * (A[i-1, j, l] + A[i+1, j, l] + A[i, j-1, l] +
                                  A[i, j+1, l] + A[i, j, l-1] + A[i, j, l+1])
        end
        return
    end

    # Launch GPU kernels for Jacobi iterations
    for k = 1:num_iterations
        @cuda threads=(m-2, n-2) blocks=(p-2) jacobi_kernel!(B, A)

        # Swap A and B for the next iteration
        A, B = B, A
    end

    return A
end

