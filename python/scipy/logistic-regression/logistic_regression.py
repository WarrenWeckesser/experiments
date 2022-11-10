# Inspired by http://stackoverflow.com/questions/39421041/
#                 use-mle-to-estimate-parameters-in-logistic-regression

# Logistic regression using
# * scipy (least squares and MLE)
# * scikit-learn
# * statsmodels


import numpy as np
from scipy.optimize import minimize
# scipy 1.8.0 is required for log_expit
from scipy.special import expit, log_expit


def sumsqerror(c, x, y):
    """
    Least squares objective function for logistic regression.
    """
    s = expit(x.dot(c))  # theta = x.dot(c)
    sqerr = ((y - s)**2).sum()
    return sqerr


def negloglikelihood(c, x, y):
    """
    Negative log likelihood objective function for logistic regression.
    """
    return -np.sum(log_expit((2*y - 1)*x.dot(c)))


def negloglikelihood_grad(c, x, y):
    """
    Gradient (with respect to c) of ``negloglikelihood(c, x, y)``.
    """
    v = (2*y - 1)*x.T
    dll_dt = expit(-c.dot(v))
    dll_dc = v.dot(dll_dt)
    return -dll_dc


mle_minimize_kwargs = {
    'bfgs':
        {'options': {'gtol': 1e-9, 'maxiter': 1000},
         'jac': negloglikelihood_grad},
    'nelder-mead':
        {'options': {'xatol': 1e-11, 'fatol': 1e-12, 'maxiter': 1000}},
    'powell':
        {'options': {'xtol': 1e-11, 'ftol': 1e-12, 'maxiter': 1000}},
    'tnc':
        {'options': {'xtol': 1e-11, 'ftol': 1e-12, 'maxfun': 1000},
         'jac': negloglikelihood_grad},
}


def logregress_mle(x, y, c0, method=None):
    method = method.lower()
    kwargs = mle_minimize_kwargs[method]
    mle = minimize(negloglikelihood, c0, method=method, args=(x, admit),
                   **kwargs)
    if mle.success:
        return mle.x
    raise RuntimeError(f"logregress_mle: minimize with method='{method}' "
                       "failed")


def logregress_ls(x, y, c0):
    """
    Logistic regression using least-squares fit.
    """
    ls = minimize(sumsqerror, c0, method='nelder-mead', args=(x, admit),
                  options=dict(maxiter=1000, disp=False,
                               xatol=1e-11, fatol=1e-12))
    if ls.success:
        return ls.x
    raise RuntimeError(f"logregress_ls: minimize with method='{method}' "
                       "failed")


def print_params(coeffs, title):
    print(title)
    print("    intercept %11.6f" % (coeffs[0],))
    print("    rank2     %11.6f" % (coeffs[1],))
    print("    rank3     %11.6f" % (coeffs[2],))
    print("    rank4     %11.6f" % (coeffs[3],))
    print("    gre       %11.6f" % (coeffs[4],))
    print("    gpa       %11.6f" % (coeffs[5],))
    print()


admit, gre, gpa, rank = np.loadtxt('binary.csv', delimiter=',', skiprows=1,
                                   unpack=True)

# rank1 = (rank == 1).astype(int)  # We don't need this.
rank2 = (rank == 2).astype(int)
rank3 = (rank == 3).astype(int)
rank4 = (rank == 4).astype(int)

# x is the design matrix.  The column of ones provides for
# an intercept in the model.
x = np.column_stack((np.ones_like(admit), rank2, rank3, rank4, gre, gpa))

c0_ls = [-1, -1, -1, -1, 0.1, 1]
c0 = logregress_ls(x, admit, c0_ls)

c_results = []
for method in ['nelder-mead', 'powell', 'tnc', 'bfgs']:
    c = logregress_mle(x, admit, c0, method=method)
    c_results.append((method, c))


print()

print("-"*64)
print("Least squares (using scipy.optimize.minimize)")
print("-"*64)
print()

print_params(c0, "method='nelder-mead'")

print("-"*64)
print("Maximum likelihood (using scipy.optimize.minimize)")
print("-"*64)
print()

for method, result in c_results:
    print_params(result, f"method='{method}'")

try:
    import statsmodels.api as sm

    print("-"*64)
    print("statsmodels")
    print("-"*64)

    smlog = sm.Logit(admit, x).fit(disp=False)

    print_params(smlog.params, "")

except ImportError:
    pass


try:
    from sklearn.linear_model import LogisticRegression

    print("-"*64)
    print("scikit-learn")
    print("-"*64)
    print()

    # solver = 'lbfgs'
    solver = 'newton-cg'
    # Because x already has a column of ones to allow for an intercept,
    # we pass `fit_intercept=False` to LogisticRegression.
    sklog = LogisticRegression(solver=solver, penalty='none',
                               fit_intercept=False,
                               max_iter=100, tol=1e-8).fit(x, admit)

    print_params(sklog.coef_[0],
                 "fit_intercept=False; x includes 1s column\n")

    sklogi = LogisticRegression(solver=solver, penalty='none',
                                max_iter=5000, tol=1e-8).fit(x[:, 1:], admit)

    print_params(np.r_[sklogi.intercept_, sklogi.coef_[0]],
                 "fit_intercept=True\n")

except ImportError:
    pass
