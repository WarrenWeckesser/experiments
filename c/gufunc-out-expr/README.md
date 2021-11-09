
The file dim-expr-grammar.owl contains an Owl grammar for the proposed
enhancement to gufuncs in which the output dimensions may be given as
functions of the input core dimensions.  This would support signatures
such as

    (n, d) -> (n*(n-1)//2)
    (m), (n) -> (m + n)
    (m), (n) -> (max(m, n) - min(m, n) - 1)

etc.

The code in main.c uses Owl to parse an expression and convert it into
a stack-based mini-language.

To create dim-expr-parser.h:

    owl -c dim-expr-grammar.owl -o dim-expr-parser.h

To compile main.c:

    gcc -Wall -pedantic -std=c99 main.c -o main
