subroutine gemv_parallel(n, A, x, y)
    implicit none
    integer, intent(in) :: n
    real, intent(in) :: A(n, n)
    real, intent(in) :: x(n)
    real, intent(out) :: y(n)
    integer :: i, j
    real :: sum

    !$ACC PARALLEL LOOP
    do i = 1, n
        sum = 0.0
        do j = 1, n
            sum = sum + A(i, j) * x(j)
        end do
        y(i) = sum
    end do
    !$ACC END PARALLEL LOOP

end subroutine gemv_parallel
