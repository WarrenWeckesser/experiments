g++  -c -std=c++11 ibeta_wrap.cpp
gcc ibeta_example.c ibeta_wrap.o -lstdc++ -o ibeta_example
