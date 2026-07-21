Also add Examples section to the eval_* function docstrings.

See eval_legendre docstring for a pretty good example.

These could
(1) Make a simple evaluation at a few points and show the result.
(2) Use quad to demonstrate orthogonality.
(3) Show that the values at the root returned by roots_* are in fact
    (approximately) 0. (It's OK that this might repeat the demonstration
    from the roots_* docstrings.)
(4) Plot over the domain for orders 0-4 (say).
    For polynomials with unbounded domains, pick an upper bound
    that makes plots that look good.
