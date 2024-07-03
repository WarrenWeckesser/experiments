
CFLAGS = -lgmp -lm

all: fisherexact fisherexact2 fisherexact3 fisherexact4_twosided

% : %.c
	$(CC) $< $(CFLAGS) -o $@
