subroutine saxpy(n, a, x, y)
    implicit none
    integer, intent(in) :: n
    real, intent(in) :: a
    real, intent(in) :: x(n)
    real, intent(inout) :: y(n)
    integer :: i

    !$ACC PARALLEL LOOP
    do i = 1, n
        y(i) = a * x(i) + y(i)
    end do
    !$ACC END PARALLEL LOOP

end subroutine saxpy

