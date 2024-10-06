

program main
    implicit none
    character(len=100) :: name_arg
    character(len=100) :: value_arg
    integer, parameter :: dp = kind(1.0d0)
    real(dp) :: value
    integer :: i
    integer :: readstatus

    real(dp) :: abstol
    real(dp) :: reltol
    real(dp) :: stoptime

    abstol = 1d-12
    reltol = 1d-8
    stoptime = 10.0

    i = 1
    do
        call get_command_argument(i, name_arg)
        if (len(trim(name_arg)) == 0) then
            exit
        end if
        i = i + 1
        call get_command_argument(i, value_arg)
        if (len(trim(value_arg)) == 0) then
            write(*, *) "ERROR: Missing comand line value for ", name_arg
            stop -1
        end if
        i = i + 1
        read(value_arg, *, iostat=readstatus) value
        if (readstatus .ne. 0) then
            write(*, *) "ERROR: Unable to read the numerical value given for parameter ", name_arg
            write(*, *) "       The value given was ", value_arg
            stop -1
        end if
        if (name_arg .eq. "abstol") then
            abstol = value
        else if (name_arg .eq. "reltol") then
            reltol = value
        else if (name_arg .eq. "stoptime") then
            stoptime = value
        else
            write(*, *) "ERROR: Unknown parameter given on the command line: ", name_arg
            stop -1
        end if
    end do
    write(*, *) "abstol =", abstol
    write(*, *) "reltol =", reltol
    write(*, *) "stoptime =", stoptime
end program main
