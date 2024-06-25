
from mpmath import mp
import numpy as np


def compute_realpart_relerror(z, dps=100):
    w_naive = np.log(1 + z)
    with mp.workdps(dps):
        w_mp = np.array([complex(mp.log1p(t)) for t in z])
    real_relerr = np.abs((w_naive.real - w_mp.real)/w_mp.real)
    return real_relerr


def sample_rectangle(n, rng, dps=100):
    low = [-0.0001, -0.005]
    high = [0.0001, 0.005]
    z = rng.uniform(low, high, size=(n, 2)).view(np.complex128)[:, 0]
    relerr = compute_realpart_relerror(z, dps)
    return np.column_stack((z.real, z.imag, relerr))


def sample_arc(n, rng, dps=100):
    r = rng.normal(loc=1.0, scale=1e-8, size=n)
    theta = np.abs(rng.normal(loc=0.0, scale=1e-6, size=n))
    z = r*np.exp(theta*1j) - 1
    relerr = compute_realpart_relerror(z, dps)
    return np.column_stack((z.real, z.imag, relerr))


def sample_splat(n, rng, dps=100):
    # z0 = -4.999958e-05 - 0.009999833j                   # bad
    # z0 = -0.01524813 - 0.173952j                        # bad
    # z0 = -0.2 + 0.6j                                    # bad
    # z0 = -0.25 + np.sqrt(1 - 0.75**2)*1j                # bad
    # z0 = -0.4 + np.sqrt(1 - 0.6**2)*1j                  # bad
    # z0 = -0.5 + np.sqrt(1 - 0.5**2)*1j                  # bad
    x = -0.500000001
    z0 = x + np.sqrt(1 - (1 + x)**2)*1j      #
    # z0 = -0.5001 + np.sqrt(1 - (1 - 0.5001)**2)*1j      # OK
    # z0 = -0.502 + np.sqrt(1 - (1 - 0.502)**2)*1j        # OK
    # z0 = -0.51 + np.sqrt(1 - (1 - 0.51)**2)*1j          # OK
    # z0 = -0.52 + np.sqrt(1 - (1 - 0.52)**2)*1j          # OK
    # z0 = -0.53 + np.sqrt(1 - (1 - 0.53)**2)*1j          # OK
    # z0 = -0.55 + np.sqrt(1 - 0.45**2)*1j                # OK
    # z0 = -0.57113 - 0.90337j                            # OK
    # z0 = -0.6 + np.sqrt(1 - 0.4**2)*1j                  # OK
    # z0 = -1.9999999995065196 - 3.141592653092555e-05j   # OK
    rr = rng.normal(loc=[z0.real, z0.imag], scale=1e-6, size=(n, 2))
    z = rr.view(np.complex128)[:, 0]
    relerr = compute_realpart_relerror(z, dps)
    return np.column_stack((z.real, z.imag, relerr))


if __name__ == "__main__":
    from multiprocessing import Pool

    nbatches = 10
    batchsize = 200000

    rng = np.random.default_rng(121263137472525314065)
    rngs = rng.spawn(nbatches)

    with Pool(processes=nbatches) as pool:
        args = zip((batchsize,)*nbatches, rngs)
        # results = pool.starmap(sample_arc, args, chunksize=1)
        # results = pool.starmap(sample_rectangle, args, chunksize=1)
        results = pool.starmap(sample_splat, args, chunksize=1)

    fname = 'relerr.txt'
    with open(fname, 'w') as f:
        for k, result in enumerate(results):
            np.savetxt(f, result)
