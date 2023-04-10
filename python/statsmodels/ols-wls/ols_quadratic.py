"""
Demonstrate OLS regression using statsmodels.

This script is based on version 0.3.0 of statsmodels.
"""

import numpy as np
import statsmodels.api as sm
from matplotlib.pyplot import plot, scatter, legend, xlabel, ylabel, show


# Generate some noisy observations of a quadratic function.
nsample = 80
x = np.linspace(0, 10, nsample)
X = sm.add_constant(np.column_stack((x, x**2)))
beta = np.array([1, 0.1, 10])
y = np.dot(X, beta) + np.random.normal(scale=1.5, size=nsample)

# Run the regression.
results = sm.OLS(y, X).fit()

# Print the summary of the results.
print(results.summary())

# Plot the observations and the fitted curve.
scatter(x, y, label='Observed data')
yy = np.dot(X, results.params)
plot(x, yy, 'g-', linewidth=3, label='Quadratic fit')
legend(loc='best')
xlabel('x')
ylabel('y')
show()
