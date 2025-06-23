# Experiment with a couple implementations of the Kabsch algorithm.


import numpy as np
from scipy.linalg import sqrtm, solve, inv, svd, det


def kabsch1(P, Q):
    P = P - np.mean(P, axis=0, keepdims=True)
    Q = Q - np.mean(Q, axis=0, keepdims=True)
    H = P.T @ Q
    R = sqrtm(H.T @ H) @ inv(H)
    return R


def kabsch2(P, Q):
    P = P - np.mean(P, axis=0, keepdims=True)
    Q = Q - np.mean(Q, axis=0, keepdims=True)
    H = P.T @ Q
    U, svals, Vh = svd(H)
    d = np.sign(det(U@Vh))
    E = np.diag([1.0, 1.0, d])
    E[-1, -1] = d
    R = Vh.T @ E @ U.T
    return R
