!
! flute.f03 -- A couple utilities
!
! Requires Fortran 2003
!

module flute

implicit none

type, public :: combiter
    integer :: total
    integer :: nbins
    integer, dimension(:), allocatable :: counts
end type combiter

contains

function combiter_new(total, nbins) result(c)
    integer, intent(in) :: total
    integer, intent(in) :: nbins
    type(combiter) :: c

    c%total = total
    c%nbins = nbins
    allocate(c%counts(nbins))
    c%counts = 0
    c%counts(1) = total

end function combiter_new


function combiter_advance(c) result(done)
    type(combiter), intent(inout) :: c
    logical :: done
    ! Local variables
    integer :: rightmost_nonzero, i, j, p
    logical :: found_nonzero

    if (c%counts(c%nbins) .eq. c%total) then
        c%counts(c%nbins) = 0
        done = .true.
        return
    end if

    ! Find the rightmost nonzero bin.
    rightmost_nonzero = c%nbins
    found_nonzero = .false.
    do i = c%nbins, 1, -1
        if (c%counts(i) .ne. 0) then
            rightmost_nonzero = i
            found_nonzero = .true.
            exit
        end if
    end do

    if (.not. found_nonzero) then
        ! All the values in c%counts are 0.
        done = .true.
        return
    end if

    if (rightmost_nonzero .ne. c%nbins) then
        c%counts(rightmost_nonzero) = c%counts(rightmost_nonzero) - 1
        c%counts(rightmost_nonzero + 1) = c%counts(rightmost_nonzero + 1) + 1
    else
        j = rightmost_nonzero - 1
        do while (c%counts(j) .eq. 0)
            j = j - 1
        end do
        p = c%counts(c%nbins)
        c%counts(c%nbins) = 0
        c%counts(j + 1) = p + 1
        c%counts(j) = c%counts(j) - 1
    end if

    done = .false.

end function combiter_advance

!
! int_count(arr, max_int) is similar to numpy.bincount(arr, minlength)
! (but note that max_int and minlength have different meanings).
!
function int_count(arr, max_int) result(count)
    ! Parameters
    integer, intent(in), dimension(:) :: arr
    integer, optional :: max_int

    ! Return value
    integer, dimension(:), allocatable :: count

    ! Local variables
    integer m, i, v

    if (present(max_int)) then
        m = max_int
    else
        m = maxval(arr)
    end if

    allocate(count(0:m))
    count = 0

    do i = 1, size(arr, 1)
        v = arr(i)
        if (v .le. m) then
            count(v) = count(v) + 1
        end if
    end do

end function int_count

end module flute
