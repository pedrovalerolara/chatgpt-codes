using Base.Threads

function saxpy_parallel(n::Int32, a::Float32, x::Vector{Float32}, y::Vector{Float32})
    # Number of threads to use
    num_threads = nthreads()
    
    # Calculate the chunk size for parallelization
    chunk_size = n ÷ num_threads

    # Parallel loop
    @threads for t in 1:num_threads
        start_idx = (t - 1) * chunk_size + 1
        end_idx = t == num_threads ? n : t * chunk_size

        for i in start_idx:end_idx
            y[i] += a * x[i]
        end
    end
end
