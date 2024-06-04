
all: demo_minmax demo_peaktopeak demo_fsum

demo_minmax: demo_minmax.o
	g++ demo_minmax.o -o demo_minmax

demo_minmax.o: demo_minmax.cpp minmax.hpp
	g++ -std=c++17 -pedantic -Wall -Wextra -march=native -O3 -I $(XSIMD_INCLUDE) -c demo_minmax.cpp

demo_peaktopeak: demo_peaktopeak.o
	g++ demo_peaktopeak.o -o demo_peaktopeak

demo_peaktopeak.o: demo_peaktopeak.cpp peaktopeak.hpp
	g++ -std=c++17 -pedantic -Wall -Wextra -march=native -O3 -I $(XSIMD_INCLUDE) -c demo_peaktopeak.cpp

demo_fsum: demo_fsum.cpp fsum.hpp
	g++ -std=c++17 -pedantic -Wall -Wextra -march=native -O2 -I $(XSIMD_INCLUDE) demo_fsum.cpp -o demo_fsum

clean:
	rm -rf *.o demo_peaktopeak demo_minmax demo_fsum
