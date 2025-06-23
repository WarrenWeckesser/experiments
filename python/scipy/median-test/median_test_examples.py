import numpy as np
from scipy.stats import median_test


print()

# Zar, Example 8.15
# Note: There is a mistake in the labeling of the contingency table.
# It should be "Below median", not "Not above median".  The example is doing
# an "above-below" calculation--values at the median are not counted.
# A+ = 1, A = 2, A- = 3, B+ = 4, etc.
s1 = [1, 1, 1, 2, 4, 4, 6, 6, 7, 7, 8]
s2 = [1, 1, 3, 3, 4, 5, 7, 7, 8, 10, 10, 10, 10, 11]
stat, p, m, table = median_test(s1, s2, ties='ignore')
print(f"Example 8.15:  stat = {stat:.3g}  p = {p:.3g}")

print()

# Zar, Example 10.12
# The statistic computed here matches the value shown in Zar, but the
# p-value does not.  This appears to be a mistake in the book.  Apparently
# Zar computed the p-value using 1 degree of freedom, not 3.
north = [7.1, 7.2, 7.4, 7.6, 7.6, 7.7, 7.7, 7.9, 8.1, 8.4, 8.5, 8.8]
east  = [6.9, 7.0, 7.1, 7.2, 7.3, 7.3, 7.4, 7.6, 7.8, 8.1, 8.3, 8.5]
south = [7.8, 7.9, 8.1, 8.3, 8.3, 8.4, 8.4, 8.4, 8.6, 8.9, 9.2, 9.4]
west  = [6.4, 6.6, 6.7, 7.1, 7.6, 7.8, 8.2, 8.4, 8.6, 8.7, 8.8, 8.9]
stat, p, m, table = median_test(north, east, south, west, ties='ignore')
print("Example 10.12: contingency table:")
print(table)
print(f"Example 10.12:  stat = {stat:.3g}  p = {p:.3g}")
