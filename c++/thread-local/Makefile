EXECUTABLES = main main2

all: $(EXECUTABLES)

$(EXECUTABLES): %: %.cpp
	$(CXX) -std=c++17 -pedantic -Wall $< -o $@

clean:
	rm -f $(EXECUTABLES)
