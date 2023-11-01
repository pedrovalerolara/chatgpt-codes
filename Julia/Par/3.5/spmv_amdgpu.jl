using AMDGPU

function spmv_amd_gpu(n::Int32, nnz::Int32, val::Vector{Float32}, row::Vector{Int32}, col::Vector{Int32}, x::Vector{Float32}, y::Vector{Float32})
    # Transfer data to the AMD GPU
    val_gpu = AMDGPUArray(val)
    row_gpu = AMDGPUArray(row)
    col_gpu = AMDGPUArray(col)
    x_gpu = AMDGPUArray(x)
    y_gpu = AMDGPUArray(y)

    # Determine the number of threads (one thread per row)
    nthreads = n

    # Define the AMD GPU kernel function
    @roc function kernel_spmv(y, val, row, col, x)
        i = AMDGPU.blockIdx().x  # Get the block index
        if i <= n
            start_idx = Int(row[i])
            end_idx = Int(row[i + 1]) - 1
            for j = start_idx + AMDGPU.threadIdx().x : AMDGPU.blockDim().x : end_idx
                y[i] += val[j] * x[col[j]]
            end
        end
        return
    end

    # Launch the AMD GPU kernel with nthreads blocks
    @roc threads=nthreads kernel_spmv(y_gpu, val_gpu, row_gpu, col_gpu, x_gpu)

    # Transfer the result back to the CPU
    y = AMDGPUArray(y_gpu)

    return y
end

