subroutine spmv_parallel(n, nnz, val, row, col, x, y)
  integer, intent(in) :: n, nnz
  real, intent(in) :: val(nnz), x(n)
  integer, intent(in) :: row(n+1), col(nnz)
  real, intent(out) :: y(n)
  integer i, j

  !$OMP PARALLEL DO DEFAULT(NONE) SHARED(n, nnz, val, row, col, x, y) PRIVATE(i, j)
  do i = 1, n
    y(i) = 0.0
    do j = row(i), row(i+1)-1
      y(i) = y(i) + val(j) * x(col(j))
    enddo
  enddo
  !$OMP END PARALLEL DO
end subroutine spmv_parallel

