subroutine gemv_parallel(n, A, x, y)
    implicit none
    integer, intent(in) :: n
    real, intent(in) :: A(n, n)
    real, intent(in) :: x(n)
    real, intent(out) :: y(n)
    integer :: i, j
    real :: sum

    !$OMP PARALLEL DO
    do i = 1, n
        sum = 0.0
        !$OMP DO
        do j = 1, n
            sum = sum + A(i, j) * x(j)
        end do
        !$OMP END DO
        y(i) = sum
    end do
    !$OMP END PARALLEL DO

end subroutine gemv_parallel
