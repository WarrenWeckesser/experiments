"""
Implement the example from

  https://en.wikipedia.org/wiki/Parsing_expression_grammar

with parsimonious.
"""

from parsimonious.grammar import Grammar


grammar_string = """
    expr     = sum
    sum      = product (("+" / "-") product)*
    product  = power (("*" / "/") power)*
    power    = value ("^" power)?
    value    = ~"[0-9]+" / ("(" expr ")")
"""

grammar = Grammar(grammar_string)

tree1 = grammar.parse("12345")
tree2 = grammar.parse("1+2-3+10")
tree3 = grammar.parse("(100-10)*2+12*(8-7)")

bad_cases = [
    "(100-10)*2+(12-)",
    "12a",
    "-1234",
    "100+200-",
]
for s in bad_cases:
    try:
        grammar.parse(s)
        print(f"Parsed '{s}', but it should have failed!")
    except Exception:
        print(f"Failed to parse '{s}', as expected.")
