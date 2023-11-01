subroutine jacobi (u, unew, n, niter, nthreads) implicit none
integer, intent (in) :: n, niter, nthreads
real (kind=8), dimension (n, n, n), intent (inout) :: u, unew
integer :: i, j, k, iter
do iter = 1, niter
do k = 1, n
do j = 1, n
do i = 1, n
unew(i, j, k) = 0.125 * (u(i-1, j, k) + u(i+1, j, k) + u(i, j-1, k) +
u(i, j+1, k) + u(i, j, k-1) + u(i, j, k+1) + u(i, j, k))
enddo
enddo
enddo
enddo
end subroutine jacobi

