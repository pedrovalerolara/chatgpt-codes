using Base.Threads

function sgemm_threads(m::Int32, n::Int32, k::Int32, A::Matrix{Float32}, B::Matrix{Float32}, C::Matrix{Float32})
    num_threads = nthreads()

    @threads for i = 1:m
        for j = 1:n
            temp = 0.0f0
            for l = 1:k
                temp += A[i, l] * B[l, j]
            end
            C[i, j] = temp
        end
    end
end

