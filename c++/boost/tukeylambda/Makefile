EXECUTABLES = main tl_invcdf_demo tl_invcdf_measure_errors

all: $(EXECUTABLES)

$(EXECUTABLES): %: %.cpp tukeylambda.h
	$(CXX) $< -o $@ -I$(BOOSTDIR)

clean:
	rm -f $(EXECUTABLES)
