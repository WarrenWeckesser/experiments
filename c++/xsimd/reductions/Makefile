
all: test_all_same demo_minmax demo_peaktopeak demo_fsum demo_argmin demo_argmin_blocks min_argmin_timing

test_all_same: test_all_same.o
	g++ test_all_same.o -o test_all_same

test_all_same.o: test_all_same.cpp all_same.hpp
	g++ -std=c++17 -pedantic -Wall -Wextra -march=native -O3 -I $(XSIMD_INCLUDE) -c test_all_same.cpp

demo_minmax: demo_minmax.o
	g++ demo_minmax.o -o demo_minmax

demo_minmax.o: demo_minmax.cpp minmax.hpp
	g++ -std=c++17 -pedantic -Wall -Wextra -march=native -O3 -I $(XSIMD_INCLUDE) -c demo_minmax.cpp

demo_argmin: demo_argmin.cpp minmax.hpp
	g++ -std=c++17 -pedantic -Wall -Wextra -march=native -O3 -I $(XSIMD_INCLUDE) demo_argmin.cpp -o demo_argmin

demo_argmin_blocks: demo_argmin_blocks.cpp minmax.hpp
	g++ -std=c++17 -pedantic -Wall -Wextra -march=native -O3 -I $(XSIMD_INCLUDE) demo_argmin_blocks.cpp -o demo_argmin_blocks

min_argmin_timing: min_argmin_timing.cpp minmax.hpp
	g++ -std=c++17 -pedantic -Wall -Wextra -march=native -O3 -I $(XSIMD_INCLUDE) min_argmin_timing.cpp -o min_argmin_timing


demo_peaktopeak: demo_peaktopeak.o
	g++ demo_peaktopeak.o -o demo_peaktopeak

demo_peaktopeak.o: demo_peaktopeak.cpp peaktopeak.hpp
	g++ -std=c++17 -pedantic -Wall -Wextra -march=native -O3 -I $(XSIMD_INCLUDE) -c demo_peaktopeak.cpp

demo_fsum: demo_fsum.cpp fsum.hpp
	g++ -std=c++17 -pedantic -Wall -Wextra -march=native -O3 -I $(XSIMD_INCLUDE) demo_fsum.cpp -o demo_fsum

clean:
	rm -rf *.o test_all_same demo_peaktopeak demo_minmax demo_fsum demo_argmin demo_argmin_blocks min_argmin_timing
