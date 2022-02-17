
program int_count_example

use flute, only: int_count

integer, dimension(10) :: data = [ 2, 3, 0, 0, 0, 4, 3, 2, 2, 2 ]
integer, dimension(:), allocatable :: count, count3

print *, 'data:   ', data

count = int_count(data)
print *, 'count:  ', count

count3 = int_count(data, max_int=3)
print *, 'count3: ', count3

end program int_count_example
