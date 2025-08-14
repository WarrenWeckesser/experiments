EXECUTABLES = logistic_cdf logistic_quantile logistic_quantile_promote_check check_policy_use

all: $(EXECUTABLES)

logistic_cdf: logistic_cdf.cpp
	g++ -std=c++17 -Werror -Wall -I$(BOOSTDIR) $< -o $@

logistic_quantile: logistic_quantile.cpp
	g++ -std=c++17 -Werror -Wall -I$(BOOSTDIR) $< -o $@

logistic_quantile_promote_check: logistic_quantile_promote_check.cpp
	g++ -std=c++17 -Werror -Wall -I$(BOOSTDIR) $< -o $@

check_policy_use: check_policy_use.cpp
	g++ -std=c++17 -Werror -Wall -I$(BOOSTDIR) $< -o $@

clean:
	rm -rf $(EXECUTABLES)
