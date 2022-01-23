mpfi
====

`mpfi` (minor page-fault investigator) is a program that provides a few
commands to allocate, write to, and free blocks of memory.  After each
command, the program prints the number of new minor page-faults that have
occurred and the program break.  The program was written to demonstrate
the behavior of glibc's malloc function.
