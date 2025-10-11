
all: swizzle_examples

swizzle_examples: swizzle_examples.cpp
	g++ swizzle_examples.cpp -std=c++17 -march=native -O3 -I $(XSIMD_INCLUDE) -o swizzle_examples

clean:
	rm -rf *.o swizzle_examples
