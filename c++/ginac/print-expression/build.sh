g++ --std=c++17 $(pkg-config --cflags ginac cln) ocaml_code_gen.cpp $(pkg-config --libs ginac cln) -o ocaml_code_gen
