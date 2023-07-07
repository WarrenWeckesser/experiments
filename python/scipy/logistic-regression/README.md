Apply logistic regression to a data set, using

* scipy (least squares and maximum likelihood estimation)
* statsmodels
* scikit-learn

The script `logistic-regression.py` outputs the following:

```

----------------------------------------------------------------
Least squares (using scipy.optimize.minimize)
----------------------------------------------------------------

method='nelder-mead'
    intercept   -3.949132
    rank2       -0.692971
    rank3       -1.363548
    rank4       -1.546927
    gre          0.002113
    gpa          0.822813

----------------------------------------------------------------
Maximum likelihood (using scipy.optimize.minimize)
----------------------------------------------------------------

method='nelder-mead'
    intercept   -3.989979
    rank2       -0.675443
    rank3       -1.340204
    rank4       -1.551464
    gre          0.002264
    gpa          0.804038

method='powell'
    intercept   -3.989979
    rank2       -0.675443
    rank3       -1.340204
    rank4       -1.551464
    gre          0.002264
    gpa          0.804038

method='tnc'
    intercept   -3.989980
    rank2       -0.675443
    rank3       -1.340204
    rank4       -1.551464
    gre          0.002264
    gpa          0.804038

method='bfgs'
    intercept   -3.989979
    rank2       -0.675443
    rank3       -1.340204
    rank4       -1.551464
    gre          0.002264
    gpa          0.804038

----------------------------------------------------------------
statsmodels
----------------------------------------------------------------

    intercept   -3.989979
    rank2       -0.675443
    rank3       -1.340204
    rank4       -1.551464
    gre          0.002264
    gpa          0.804038

----------------------------------------------------------------
scikit-learn
----------------------------------------------------------------

fit_intercept=False; x includes 1s column

    intercept   -3.989979
    rank2       -0.675443
    rank3       -1.340204
    rank4       -1.551464
    gre          0.002264
    gpa          0.804038

fit_intercept=True

    intercept   -3.989979
    rank2       -0.675443
    rank3       -1.340204
    rank4       -1.551464
    gre          0.002264
    gpa          0.804038

```