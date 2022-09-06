!
! This is not a complete example of sgesdd.
! It only executes a call with lwork = -1 and prints
! the value returned in work(1).
!

program sgesdd_example

    implicit none
    external sgesdd

    integer, parameter :: n = 9537
    real, dimension(n, n) :: a
    real, dimension(n) :: s
    real, dimension(n, n) :: u
    real, dimension(n, n) :: vt
    real, dimension(1) :: work
    integer lwork
    integer, dimension(8*n) :: iwork
    integer info

    lwork = -1
    call sgesdd('A', n, n, a, n, s, u, n, vt, n, work, lwork, iwork, info)

    print *, info
    print *, work(1)

end program sgesdd_example
