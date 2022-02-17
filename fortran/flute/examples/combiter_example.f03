
program combiter_example

use flute, only: combiter, combiter_new, combiter_advance

type(combiter) :: c
logical :: done

c = combiter_new(3, 4)

done = .false.
do while (.not. done)
    print *, c%counts
    done = combiter_advance(c)
end do

end program combiter_example
