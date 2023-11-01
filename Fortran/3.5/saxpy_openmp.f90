subroutine saxpy_parallel(n, a, x, y)
    implicit none
    integer, intent(in) :: n
    real, intent(in) :: a
    real, intent(in) :: x(n)
    real, intent(inout) :: y(n)
    integer :: i

    !$OMP PARALLEL DO
    do i = 1, n
        y(i) = a * x(i) + y(i)
    end do
    !$OMP END PARALLEL DO

end subroutine saxpy_parallel
