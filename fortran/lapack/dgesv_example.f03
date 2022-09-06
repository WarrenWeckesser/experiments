program dgesv_example

    implicit none
    integer ipiv(3), info
    double precision a(3, 3), b(3, 2)
    integer i, j

    a(1, 1) = 1
    a(1, 2) = 3
    a(1, 3) = 6
    a(2, 1) = -1
    a(2, 2) = 0
    a(2, 3) = 1
    a(3, 1) = 4
    a(3, 2) = 4
    a(3, 3) = 5

    b(1, 1) = 1
    b(1, 2) = -1
    b(2, 1) = 0
    b(2, 2) = 3
    b(3, 1) = 0
    b(3, 2) = 0

    call dgesv(3, 2, a, 3, ipiv, b, 3, info)

    if (info .eq. 0) then
        do i = 1, 3
            write(*, '(2F8.3)') (b(i, j), j=1, 2)
        enddo
    endif

end
