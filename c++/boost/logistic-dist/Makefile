EXECUTABLES = logistic_cdf logistic_quantile

all: $(EXECUTABLES)

logistic_cdf: logistic_cdf.cpp
	g++ -std=c++17 -Werror -Wall -I$(BOOSTDIR) $< -o $@

logistic_quantile: logistic_quantile.cpp
	g++ -std=c++17 -Werror -Wall -I$(BOOSTDIR) $< -o $@

clean:
	rm -rf $(EXECUTABLES)
