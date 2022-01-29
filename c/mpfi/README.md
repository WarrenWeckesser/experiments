mpfi
====

`mpfi` (minor page-fault investigator) is a program that provides a few
commands to allocate, write to, and free blocks of memory.  After each
command, the program prints the number of new minor page-faults that have
occurred and the program break.  The program was written to demonstrate
the behavior of glibc's malloc function.

Example
-------

In the following, a 4 mb block of memory is allocated, written to and
then freed.  Then a 2 mb block of memory is allocated, written to and
then freed, twice.  `mpfi` is using `malloc` from glibc.  Items to note
are the number of minor page faults after each write operation, and the
change in the program break pointer after the first 2 mb block is
allocated.

```
$ ./mpfi
: a 4000000
allocated    4000000 bytes in slot  0   [     7 new minor page faults; brk=0x5620501d3000]
: w
wrote to slot  0 (   4000000 bytes)     [   980 new minor page faults; brk=0x5620501d3000]
: f
freed slot  0                           [     0 new minor page faults; brk=0x5620501d3000]
: a 2000000
allocated    2000000 bytes in slot  0   [     1 new minor page faults; brk=0x5620503bb000]
: w
wrote to slot  0 (   2000000 bytes)     [   487 new minor page faults; brk=0x5620503bb000]
: f
freed slot  0                           [     0 new minor page faults; brk=0x5620503bb000]
: a 2000000
allocated    2000000 bytes in slot  0   [     0 new minor page faults; brk=0x5620503bb000]
: w
wrote to slot  0 (   2000000 bytes)     [     0 new minor page faults; brk=0x5620503bb000]
: f
freed slot  0                           [     0 new minor page faults; brk=0x5620503bb000]
: q
$
```
