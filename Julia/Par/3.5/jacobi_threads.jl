using Base.Threads

function jacobi_threads(A::Array{Float64, 3}, B::Array{Float64, 3}, num_iterations::Int)
    m, n, p = size(A)

    # Create an array to store temporary results
    C = similar(A)

    for k = 1:num_iterations
        @threads for i = 2:m-1
            for j = 2:n-1
                for l = 2:p-1
                    C[i, j, l] = 0.125 * (A[i-1, j, l] + A[i+1, j, l] + A[i, j-1, l] +
                                          A[i, j+1, l] + A[i, j, l-1] + A[i, j, l+1])
                end
            end
        end

        # Swap A and C for the next iteration
        A, C = C, A
    end

    return A
end

