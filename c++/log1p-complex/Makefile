# In the docker ppc64le container, set
#   EXTRA_CXX_FLAGS=-mabi=ieeelongdouble
# to switch the implementation of long double from IBM double-double to
# IEEE float128.

main: main.cpp log1p_complex.h print_floating_point.h
	g++ -pedantic -Wall -std=c++17 $(EXTRA_CXX_FLAGS) main.cpp -o main
