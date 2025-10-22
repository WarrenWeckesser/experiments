"""
Use highspy to solve the nonnegative least squares problem.

The quadratic programming problem is

   argmin_{x >= 0} (x^T @ Q @ x / 2 + c^T @ x)

where Q = A^T @ A and c = -A^T @ y.
"""

import highspy
import numpy as np
try:
    from scipy.optimize import nnls
    have_scipy = True
except ImportError:
    have_scipy = False


def highs_hessian_from_symmetric_matrix(Q):
    """
    Create the HighsHessian representation of the Hessian matrix Q.
    Q is expected to be a symmetric real numpy matrix.
    """
    n = len(Q)
    i, j = np.triu_indices(n)
    data = Q[i, j]
    mask = data != 0
    ii = i[mask]
    jj = j[mask]
    values = data[mask]
    # np.cumulative_sum() was added in NumPy 2.1.0.
    # For older versions of NumPy, replace its use with
    #   start = np.zeros(n + 1, dtype=ii.dtype)
    #   start[1:] = np.bincount(ii).cumsum()
    start = np.cumulative_sum(np.bincount(ii), include_initial=True)

    hh = highspy.HighsHessian()
    hh.dim_ = n
    hh.format_ = highspy.HessianFormat.kTriangular
    hh.start_ = start
    hh.index_ = jj
    hh.value_ = values
    return hh


def nnls_with_highs_qp(A, y):
    """
    Nonnegative least squares solver.

    Compute the nonnegative least squares solution to

        A @ x = y

    The solution is computed using the quadratic problem solver provided
    by the HiGHS library (via the `highspy` Python interface).

    The quadratic programming problem is

        argmin_{x >= 0} (x^T @ Q @ x / 2 + c^T @ x)

    where Q = A^T @ A and c = -A^T @ y.

    Note: This function is generally slower than `scipy.optimize.nnls`.
    """
    A = np.asarray(A)
    y = np.asarray(y)

    # Compute the coefficient matrices Q and c of the QP.
    At = np.transpose(A)
    Q = At @ A
    c = -At @ y

    # Create the Highs() instance.
    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    # Experimentation shows that setting qp_regularization_value to 1e-12
    # provides more precise results than when using the default (but setting
    # it to 0 gives bad results).
    h.setOptionValue("qp_regularization_value", 1e-12)
    n = A.shape[-1]
    h.addVars(n, np.zeros(n), np.full(n, fill_value=highspy.kHighsInf))
    h.changeColsCost(n, np.arange(n), c)
    h.passHessian(highs_hessian_from_symmetric_matrix(Q))

    # Solve
    h.run()

    solution = h.getSolution()
    return np.array(solution.col_value)


if __name__ == "__main__":
    from numpy.linalg import norm

    # Assorted cases.
    # Inputs are A and y.  We want the NNLS solution to A @ x = y.
    A0 = np.array([[7.0, 2.0,  0.0,  4.0],
                   [3.0, 3.0, -1.0,  0.0],
                   [4.0, 3.0,  0.0,  0.0],
                   [1.0, 4.0,  0.0, -4.0]])
    y0 = np.array([3240.0, 0.0, -360.0, 720.0])

    # When the option qp_regularization_value is left at the default
    # value, the solution found by nnls_with_highs_qp(A1, y1) is different
    # from that of scipy.optiminze.nnls(A1, y1); the residual norms
    # differ by about 2e-9.
    A1 = np.array([[5.0,  0.0, 7.0, 2.0,  0.0,  4.0],
                   [2.0,  0.0, 3.0, 3.0, -1.0,  0.0],
                   [1e-8, 0.0, 4.0, 3.0,  0.0,  0.0],
                   [6.5,  3.0, 1.0, 4.0,  0.0, -4.0]])
    y1 = np.array([30.0, 0.0, -30.0, 72.0])

    A2 = np.array([[5.0,  0.0, 7.0, 2.0,  0.0,  4.0],
                   [2.0,  0.0, 3.0, 3.0, -1.0,  0.0],
                   [1e-8, 0.0, 4.0, 3.0,  0.0,  0.5],
                   [6.5,  3.0, 1.0, 4.0,  0.0, -4.0]])
    y2 = np.array([30.0, 0.0, -30.0, 72.0])

    rng = np.random.default_rng(121263137472525314065)
    A3 = rng.choice([0, 1, 3],
                    p=[0.99, 0.005, 0.005],
                    size=(500, 75)).astype(np.float64)
    y3 = rng.integers(0, 10, size=len(A3)).astype(np.float64)

    A4 = np.array([[1.0, 3.0, 0.0, 0.0],
                   [0.0, 3.0, 0.0, 2.0],
                   [1.0, 2.0, 0.0, 1.0],
                   [5.0, 1.0, 0.5, 3.0],
                   [3.0, 0.0, 0.0, 1.0],
                   [0.0, 2.5, 0.0, 2.5],
                   [0.5, 3.0, 0.5, 3.0],
                   [0.0, 2.0, 0.0, 2.0]])
    y4 = np.array([20.0, 40.0, 20.0, 0.0, 20.0, 0.0, 40.0, 10.0])

    cases = [(A0, y0), (A1, y1), (A2, y2), (A3, y3), (A4, y4)]
    for k, (A, y) in enumerate(cases):
        print(f"Case {k}")
        x_highs = nnls_with_highs_qp(A, y)
        print("From highs:")
        print(f"  x = {x_highs!r}")
        print(f"  residual norm = {norm(A @ x_highs - y)}")
        if have_scipy:
            x_scipy, resid_norm = nnls(A, y)
            print("From scipy:")
            print(f"  x = {x_scipy!r}")
            print(f"  residual norm = {norm(A @ x_scipy - y)}")
        print()
