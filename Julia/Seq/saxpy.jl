function saxpy(n::Int32, a::Float32, x::Vector{Float32}, y::Vector{Float32})
    for i in 1:n
        y[i] += a * x[i]
    end
end
