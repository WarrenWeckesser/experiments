
EXECUTABLES = erfinv_demo erfinv_bad_input

all: $(EXECUTABLES)

erfinv_demo: erfinv_demo.cpp
	$(CXX) $(CXXFLAGS) erfinv_demo.cpp -o erfinv_demo

erfinv_bad_input: erfinv_bad_input.cpp
	$(CXX) $(CXXFLAGS) erfinv_bad_input.cpp -o erfinv_bad_input

clean:
	rm -rf $(EXECUTABLES)
