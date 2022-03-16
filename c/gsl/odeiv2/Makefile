
CPPFLAGS = -Wall -Werror

rossler_solve: rossler_solve.o rossler_system.o
	$(CC) $(LDFLAGS) -o rossler_solve rossler_solve.o rossler_system.o -lgsl -lgslcblas -lm

rossler_solve.o: rossler_solve.c rossler_system.h
	$(CC) $(CPPFLAGS) -c rossler_solve.c

rossler_system.o: rossler_system.c rossler_system.h
	$(CC) $(CPPFLAGS) -c rossler_system.c

clean:
	rm -f rossler_solve rossler_solve.o rossler_system.o
