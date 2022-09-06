
all: dgesv_example sgesdd_example

dgesv_example: dgesv_example.f03
	gfortran dgesv_example.f03 -L$(LAPACKDIR) -L$(BLASDIR) -llapack -lblas -o dgesv_example

sgesdd_example: sgesdd_example.f03
	gfortran sgesdd_example.f03 -L$(LAPACKDIR) -L$(BLASDIR) -llapack -lblas -o sgesdd_example

clean:
	rm -rf dgesv_example sgesdd_example
