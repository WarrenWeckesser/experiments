
all: demo_minmax minmax_timing

demo_minmax: demo_minmax.o
	g++ demo_minmax.o -o demo_minmax

demo_minmax.o: demo_minmax.cpp minmax.h
	g++ -std=c++17 -pedantic -Wall -Wextra -march=native -O3 -I $(XSIMD_INCLUDE) -c demo_minmax.cpp


minmax_timing: minmax_timing.cpp minmax.h
	g++ -std=c++17 -pedantic -Wall -Wextra -march=native -O3 -I $(XSIMD_INCLUDE) minmax_timing.cpp -o minmax_timing


clean:
	rm -rf *.o demo_minmax minmax_timing
