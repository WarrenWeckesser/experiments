
CFLAGS = -Wall -Werror

BUILDIT = $(CC) $^ $(CFLAGS) $(LDFLAGS) -lgsl -lgslcblas -o $@

all: gsl_sf_hyperg_1F1_demo hyp1f1eval

gsl_sf_hyperg_1F1_demo: gsl_sf_hyperg_1F1_demo.c
	$(BUILDIT)

hyp1f1eval: hyp1f1eval.c
	$(BUILDIT)

clean:
	rm -rf gsl_sf_hyperg_1F1_demo hyp1f1eval
