# Hardcoded for gcc on a machine with AVX.

CFLAGS = -march=native -O3 -pedantic -Wall

EXECUTABLES = min_max_demo min_max_timing cabs2_demo cabs2_timing

all: $(EXECUTABLES)

min_max_demo: min_max_demo.o min_max.o
	gcc min_max_demo.o min_max.o -o min_max_demo

min_max_demo.o: min_max_demo.c min_max.h
	gcc $(CFLAGS) -c min_max_demo.c

min_max_timing: min_max_timing.o min_max.o
	gcc min_max_timing.o min_max.o -o min_max_timing

min_max_timing.o: min_max_timing.c min_max.h
	gcc $(CFLAGS) -c min_max_timing.c

min_max.o: min_max.c min_max.h
	gcc $(CFLAGS) -c min_max.c


cabs2_demo: cabs2_demo.o cabs2.o
	gcc cabs2_demo.o cabs2.o -o cabs2_demo

cabs2_demo.o: cabs2_demo.c cabs2.h
	gcc $(CFLAGS) -c cabs2_demo.c

cabs2_timing: cabs2_timing.o cabs2.o
	gcc cabs2_timing.o cabs2.o -o cabs2_timing

cabs2_timing.o: cabs2_timing.c cabs2.h
	gcc $(CFLAGS) -c cabs2_timing.c

cabs2.o: cabs2.c cabs2.h
	gcc $(CFLAGS) -c cabs2.c


clean:
	rm -rf *.o $(EXECUTABLES)
