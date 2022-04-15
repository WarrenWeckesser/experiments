
EXECUTABLES = powm1_demo powm1_bad_input

all: $(EXECUTABLES)

powm1_demo: powm1_demo.cpp
	$(CXX) $(CXXFLAGS) powm1_demo.cpp -o powm1_demo

powm1_bad_input: powm1_bad_input.cpp
	$(CXX) $(CXXFLAGS) powm1_bad_input.cpp -o powm1_bad_input

clean:
	rm -rf $(EXECUTABLES)
