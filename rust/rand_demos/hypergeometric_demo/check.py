import numpy as np
from scipy.stats import hypergeom

rng = np.random.default_rng()

with open("out") as f:
    # The first line must be a comment containing either "InverseTransform"
    # or "RejectionAcceptance". This is printed by a modified version of
    # hypergeometric.rs.  The second line must hold "N=value K=value n=value".
    # This is printed by the program that generates the samples.
    line1 = f.readline()
    print(line1.strip())
    line2 = f.readline()
    parts = line2.strip().split()
    N = int(parts[1].split('=')[1])
    K = int(parts[2].split('=')[1])
    n = int(parts[3].split('=')[1])
    x = np.loadtxt(f, dtype=np.int64)

b = np.bincount(x)

i = np.arange(len(b))
expected = np.round(len(x)*hypergeom.pmf(i, N, K, n)).astype(int)

np.set_printoptions(linewidth=132)
print(b)
print(expected)
