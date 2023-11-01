using CUDA

function spmv_gpu(n::Int32, nnz::Int32, val::Vector{Float32}, row::Vector{Int32}, col::Vector{Int32}, x::Vector{Float32}, y::Vector{Float32})
    # Transfer data to the GPU
    val_gpu = CuArray(val)
    row_gpu = CuArray(row)
    col_gpu = CuArray(col)
    x_gpu = CuArray(x)
    y_gpu = CuArray(y)

    # Determine the number of threads (one thread per row)
    nthreads = n

    # Define a CUDA kernel function
    function kernel_spmv(y, val, row, col, x)
        i = threadIdx().x  # Get the thread index
        if i <= n
            start_idx = Int(row[i])
            end_idx = Int(row[i + 1]) - 1
            y[i] = sum(val[start_idx:end_idx] .* x[col[start_idx:end_idx]])
        end
        return
    end

    # Launch the CUDA kernel with nthreads threads
    @cuda threads=nthreads kernel_spmv(y_gpu, val_gpu, row_gpu, col_gpu, x_gpu)

    # Transfer the result back to the CPU
    CUDA.copyto!(y, y_gpu)

    return y
end

