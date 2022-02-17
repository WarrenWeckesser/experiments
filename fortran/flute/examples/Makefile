FC = gfortran

all: int_count_example combiter_example flute.o

flute.o: ../flute.f03
	$(FC) -c $(FFLAGS) ../flute.f03

int_count_example: flute.o int_count_example.o 
	$(FC) int_count_example.o flute.o -o int_count_example

combiter_example: flute.o combiter_example.o
	$(FC) combiter_example.o flute.o -o combiter_example


%.o: %.f03
	$(FC) -c $(FFLAGS) $<


clean:
	rm -f flute.o flute.mod int_count_example.o int_count_example combiter_example.o combiter_example

