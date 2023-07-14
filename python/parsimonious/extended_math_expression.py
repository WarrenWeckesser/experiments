"""
Extend the example in simple_math_expression.py with the following changes:

* allow unary + and - in front of integers;
* add 'max(expr,expr)' and 'min(expr,expr)';
* change '^' to '**'.

"""

from parsimonious.grammar import Grammar


grammar_string = """
    expr     = sum
    sum      = product (("+" / "-") product)*
    product  = power (("*" / "/") power)*
    power    = value ("**" power)?
    value    = integer / ("(" expr ")") / max / min
    max      = "max(" expr "," expr ")"
    min      = "min(" expr "," expr ")"
    integer  = ~"[+-]?[0-9]+"
"""

grammar = Grammar(grammar_string)

tree1 = grammar.parse("max(123,-45)")
tree2a = grammar.parse("10+8")
tree2b = grammar.parse("10+-8")
tree2c = grammar.parse("10--8")
tree2d = grammar.parse("10-8")
tree3 = grammar.parse("max(5,3)-2**max(-3,0)")
tree4 = grammar.parse("1+max(min(2,3),4)-5")
