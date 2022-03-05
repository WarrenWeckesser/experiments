The code in `gev_return_levels.py` computes the *n*-year
return levels estimated from a 30 year sample of annual
maximum temperatures.

The return levels are computed by fitting the generalized
extreme value distribution to the data, and then using the
inverse of the distribution's survival function to compute
the return levels.

Script output:

    GEV fit parameters:
    shape: -0.96087
    loc:   21.52049
    scale: 1.05332

    Return levels:

    Period    Level
    (years)   (temp)
       5      25.06
      10      29.95
      20      39.45
      50      67.00
     100     111.53


Notes:
* The SciPy implementation of the generalized extreme value
  distribution uses a shape parameter whose sign is the
  opposite of that used in some other software.
* This example is meant to show the computational procedure
  for computing the return levels.  The actual usefulness of
  extrapolating the 50 and 100 year return levels from a
  data set with 30 samples is questionable.
