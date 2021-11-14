CC=gcc

main: main.c dim-expr-parser.h
	$(CC) -Wall -pedantic -std=c99 main.c -lm -o main

dim-expr-parser.h: dim-expr-grammar.owl
	owl -c dim-expr-grammar.owl -o dim-expr-parser.h

clean:
	rm -f main dim-expr-parser.h
