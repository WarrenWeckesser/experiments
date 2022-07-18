# Hardcoded for gcc

all: demo

demo: demo.o randtable.o
	gcc demo.o randtable.o -o demo

demo.o: demo.c randtable.h
	gcc -march=native -O3 -c demo.c

randtable.o: randtable.c randtable.h
	gcc -march=native -O3 -c randtable.c

clean:
	rm -rf *.o demo
