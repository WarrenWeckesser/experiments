
The file `dim-expr-grammar.owl` contains an Owl (https://github.com/ianh/owl)
grammar for the proposed enhancement to gufuncs in which the output dimensions
may be given as functions of the input core dimensions.  This would support
signatures such as

    (n, d) -> (n*(n-1)/2)
    (m), (n) -> (m + n)
    (m), (n) -> (max(m, n) - min(m, n) - 1)

etc.

The code in `main.c` uses Owl to parse an expression and convert it into
a stack-based mini-language.


For example:

```
$ ./main "n" "n*(n - 1)/2"
Instructions
opcode  arg   symbolic opcodes and args
   11     0   PUSH VARIABLE n
   11     0   PUSH VARIABLE n
    2     1   PUSH CONSTANT 1
   10         SUBTRACT
    9         MULTIPLY
    2     2   PUSH CONSTANT 2
    6         DIVIDE

Demonstrate evaluation...
n = 3
evaluate_instructions returned 3

$ ./main "m n" "max(m, n) - min(m, n) - 1"
Instructions
opcode  arg   symbolic opcodes and args
   11     0   PUSH VARIABLE m
   11     1   PUSH VARIABLE n
    3         MAXIMUM
   11     0   PUSH VARIABLE m
   11     1   PUSH VARIABLE n
    4         MINIMUM
   10         SUBTRACT
    2     1   PUSH CONSTANT 1
   10         SUBTRACT

Demonstrate evaluation...
m = 3
n = 8
evaluate_instructions returned 4

```
