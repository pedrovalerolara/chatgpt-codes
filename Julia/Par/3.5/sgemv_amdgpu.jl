using AMDGPU

function sgemv_amdgpu(m::Int32, n::Int32, A::Matrix{Float32}, x::Vector{Float32}, y::Vector{Float32})
    # Define a GPU kernel to perform matrix-vector multiplication
    @roc function sgemv_kernel(A::AbstractMatrix{Float32}, x::AbstractVector{Float32}, y::AbstractVector{Float32})
        i = workitemIdx().x  # Get the thread index

        if i <= m
            temp = 0.0f0
            for j = 1:n
                temp += A[i, j] * x[j]
            end
            y[i] += temp
        end
        return
    end

    # Launch the kernel on the GPU
    @roc sgemv_kernel(A, x, y)
end

