function sgemm(m::Int32, n::Int32, k::Int32, A::Matrix{Float32}, B::Matrix{Float32}, C::Matrix{Float32})
 for i=1:m
  for j=1:n
    for l=1:k
      C[i,j] += A[i,l]*B[l,j]
    end
  end
end
