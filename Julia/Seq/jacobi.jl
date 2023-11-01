function jacobi(A::Array{Float64,3}, B::Array{Float64,3}, num_iterations::Int32)
    m = size(A,1)
    n = size(A,2)
    p = size(A,3)
    for k = 1:num_iterations
        for i = 2:m-1
            for j = 2:n-1
                for l = 2:p-1
                    B[i,j,l] = 0.125 * (A[i-1,j,l] + A[i+1,j,l] + A[i,j-1,l] + A[i,j+1,l] + A[i,j,l-1] + A[i,j,l+1])
                end
            end
        end
        A,B = B,A
    end
    return A
end
