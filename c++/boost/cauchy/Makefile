EXECUTABLES = cauchy_demo cauchy_cdf_alternative cauchy_quantile_demo

all: $(EXECUTABLES)

cauchy_demo: cauchy_demo.cpp
	g++ -std=c++17 -Werror -Wall -I$(BOOSTDIR) cauchy_demo.cpp -o cauchy_demo

cauchy_cdf_alternative: cauchy_cdf_alternative.cpp
	g++ -std=c++17 -Werror -Wall -I$(BOOSTDIR) cauchy_cdf_alternative.cpp -o cauchy_cdf_alternative

cauchy_quantile_demo: cauchy_quantile_demo.cpp
	g++ -std=c++17 -Werror -Wall -I$(BOOSTDIR) cauchy_quantile_demo.cpp -o cauchy_quantile_demo

clean:
	rm -rf $(EXECUTABLES)
