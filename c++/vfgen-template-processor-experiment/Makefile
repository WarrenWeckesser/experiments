ALL = check_simple_array abtemplate_demo

all: $(ALL)

abtemplate_demo: abtemplate_demo.cpp abtemplate.cpp abtemplate.h simple_array.h
	g++ -std=c++20 abtemplate_demo.cpp abtemplate.cpp -o abtemplate_demo

check_simple_array: check_simple_array.cpp simple_array.h
	g++ -std=c++20 check_simple_array.cpp -o check_simple_array
