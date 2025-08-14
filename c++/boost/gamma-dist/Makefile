EXECUTABLES = gamma_quantile_promote_check

all: $(EXECUTABLES)

gamma_quantile_promote_check: gamma_quantile_promote_check.cpp
	g++ -std=c++17 -Werror -Wall -I$(BOOSTDIR) $< -o $@

clean:
	rm -rf $(EXECUTABLES)
