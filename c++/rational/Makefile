
all: demo_rational rational_timing

demo_rational: demo_rational.cpp rational.hpp
	g++ -std=c++17 -pedantic -Wall -Wextra -march=native -O3 demo_rational.cpp -o demo_rational

rational_timing: rational_timing.cpp rational.hpp
	g++ -std=c++17 -pedantic -Wall -Wextra -march=native -O3 rational_timing.cpp -I$(BOOSTDIR) -o rational_timing

clean:
	rm -rf *.o demo_rational rational_timing
