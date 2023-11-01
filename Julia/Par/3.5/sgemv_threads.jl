using Threads

function sgemv_threaded(m::Int32, n::Int32, A::Matrix{Float32}, x::Vector{Float32}, y::Vector{Float32})
    # Determine the number of threads to use
    num_threads = nthreads()

    # Calculate the chunk size for distributing work among threads
    chunk_size = m ÷ num_threads

    # Create an array to hold the thread objects
    threads = Vector{Thread}(undef, num_threads)

    # Define a function to perform the threaded work
    function thread_work(tid)
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = tid == num_threads ? m : start_idx + chunk_size - 1

        for j = 1:n
            for i = start_idx:end_idx
                y[i] += A[i, j] * x[j]
            end
        end
    end

    # Start the threads
    for i = 1:num_threads
        threads[i] = Threads.@spawn thread_work(i)
    end

    # Wait for all threads to finish
    for thread in threads
        wait(thread)
    end
end
