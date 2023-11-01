using Base.Threads

function spmv_threaded(n::Int32, nnz::Int32, val::Vector{Float32}, row::Vector{Int32}, col::Vector{Int32}, x::Vector{Float32}, y::Vector{Float32})
    Threads.@threads for i = 1:n
        for j = row[i]:row[i+1]-1
            y[i] += val[j] * x[col[j]]
        end
    end
end

