# Hardcoded for gcc on a machine with AVX.

all: demo timing

demo: demo.o min_max.o
	gcc demo.o min_max.o -o demo

demo.o: demo.c min_max.h
	gcc -march=native -O3 -c demo.c

timing: timing.o min_max.o
	gcc timing.o min_max.o -o timing

timing.o: timing.c min_max.h
	gcc -march=native -O3 -c timing.c

min_max.o: min_max.c min_max.h
	gcc -march=native -O3 -c min_max.c

clean:
	rm -rf *.o demo timing
