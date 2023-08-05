
all: demo_clog demo_clog1p

demo_clog: demo_clog.c
	$(CC) demo_clog.c -lm -o demo_clog

demo_clog1p: demo_clog1p.o clog1p.o
	$(CC) demo_clog1p.o clog1p.o -lm -o demo_clog1p

demo_clog1p.o: demo_clog1p.c clog1p.h
	$(CC) -c demo_clog1p.c -o demo_clog1p.o

clog1p.o: clog1p.c clog1p.h
	$(CC) -c clog1p.c

clean:
	rm -f *.o demo_clog1p demo_clog
