function sgemv(m::Int32, n::Int32, A::Matrix{Float32}, x::Vector{Float32}, y::Vector{Float32})
  for j=1:n
    for i=1:m
      y[i] += A[i,j]*x[j]
    end
  end
end
