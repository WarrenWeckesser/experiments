
all: demo

demo: demo.o
	g++ demo.o -o demo

demo.o: demo.cpp peaktopeak.hpp minmax.hpp
	g++ -std=c++14 -pedantic -Wall -Wextra -march=native -O3 -I $(XSIMD_INCLUDE) -c demo.cpp

clean:
	rm -rf *.o demo
