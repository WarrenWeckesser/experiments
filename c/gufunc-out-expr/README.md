
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
$ ./main "n*(n - 1)/2" n=5
Instructions
opcode  arg   symbolic opcodes and args
   12     0   PUSH VARIABLE n
   12     0   PUSH VARIABLE n
    3     1   PUSH CONSTANT 1
   11         SUBTRACT
   10         MULTIPLY
    3     2   PUSH CONSTANT 2
    7         DIVIDE

Demonstrate evaluation...
n = 5
evaluate_instructions returned 10

$ ./main "max(m, n) - min(m, n) - 1" m=5 n=12
Instructions
opcode  arg   symbolic opcodes and args
   12     0   PUSH VARIABLE m
   12     1   PUSH VARIABLE n
    4         MAXIMUM
   12     0   PUSH VARIABLE m
   12     1   PUSH VARIABLE n
    5         MINIMUM
   11         SUBTRACT
    3     1   PUSH CONSTANT 1
   11         SUBTRACT

Demonstrate evaluation...
m = 5
n = 12
evaluate_instructions returned 6

$ ./main "abs(m - n) - 1" m=5 n=12
Instructions
opcode  arg   symbolic opcodes and args
   12     0   PUSH VARIABLE m
   12     1   PUSH VARIABLE n
   11         SUBTRACT
    1         ABS
    3     1   PUSH CONSTANT 1
   11         SUBTRACT

Demonstrate evaluation...
m = 5
n = 12
evaluate_instructions returned 6
```
