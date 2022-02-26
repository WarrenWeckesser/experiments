gcc -Wall -Werror -c -O2 -I$LAPACKE_INCLUDE zgeev_example.c -o zgeev_example.o
gfortran zgeev_example.o -L$LAPACK_LIB_DIR -llapacke -llapack -lrefblas -o zgeev_example
