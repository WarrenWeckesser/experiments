gcc -Wall -Werror -c -O2 zgeev_example.c -o zgeev_example.o
gfortran zgeev_example.o -llapacke -llapack -lblas -o zgeev_example

gcc -Wall -Werror -c -O2 dsysv_example.c -o dsysv_example.o
gfortran dsysv_example.o -llapacke -llapack -lblas -o dsysv_example
