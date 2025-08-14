EXECUTABLES = main tl_invcdf_demo tl_invcdf_measure_errors tl_invcdf_measure_errors_fixed_lam

all: $(EXECUTABLES)

$(EXECUTABLES): %: %.cpp tukeylambda.h
	$(CXX) -std=c++17 $< -o $@ -I$(BOOSTDIR)

clean:
	rm -f $(EXECUTABLES)
