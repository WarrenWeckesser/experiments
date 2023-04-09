# Use maximum likelihood estimation to fit a mixture of an exponential
# distribution and a gamma distribution to a sample.
#
# There is no implementtion of a generic mixture here.  The code
# is written specifically for the mixture of an exponential and
# a gamma distribution.

import numpy as np
from scipy.stats import gamma
from scipy.special import expit, logit
from scipy.optimize import fmin, fmin_bfgs
import matplotlib.pyplot as plt


def exp_gamma_mixture_pdf(x, tau, rho, gamma_shape, theta):
    w = expit(tau)
    exp_scale = np.exp(rho)
    gamma_scale = np.exp(theta)
    p = (w*gamma.pdf(x, 1, loc=0, scale=exp_scale) +
         (1 - w)*gamma.pdf(x, gamma_shape, loc=0, scale=gamma_scale))
    return p


def negloglik(params, x):
    prob = exp_gamma_mixture_pdf(x, *params)
    nll = -np.log(prob).sum()
    return nll


# Generate a sample to work with.
w = 0.15  # Weight of the exponential distribution
exp_scale = 0.25
gamma_shape = 8.5
gamma_scale = 0.15

tau = logit(w)
rho = np.log(exp_scale)
theta = np.log(gamma_scale)

n = 4000
n1 = int(w*n)
n2 = int((1-w)*n)
np.random.seed(777777)
x = np.concatenate((np.random.gamma(1, exp_scale, size=n1),
                    np.random.gamma(gamma_shape, gamma_scale, size=n2)))
params = [tau, exp_scale, gamma_shape, gamma_scale]

hist, bin_edges = np.histogram(x, bins=32)

# Maximum likelihood fit
bestparams = None
bestmin = np.inf
# w0s = [0.25, 0.5, 0.75]
w0s = [0.3, 0.7]
# gammascales = [0.01, 0.05, 0.25]
gammascales = [0.02, 0.2]
for w0 in w0s:
    tau0 = logit(w)
    for expscale0 in [0.01, 0.1, 1]:
        rho0 = np.log(expscale0)
        for gammashape0 in [2, 4, 8]:
            for gammascale0 in gammascales:
                theta0 = np.log(gammascale0)
                params0 = np.array([tau0, rho0, gammashape0, theta0])
                result = fmin(negloglik, params0, args=(x,),
                              maxiter=100000, full_output=True, disp=False)
                pmle, fopt, niter, ncalls, warnflag = result
                if fopt < bestmin:
                    print('\n', pmle, fopt)
                    bestmin = fopt
                    bestparams = pmle
                else:
                    print('.', end='', flush=True)
print('')
print("###", bestparams)

result = fmin_bfgs(lambda q, y: negloglik(q, y), 0.99*bestparams, args=(x,),
                   full_output=True, disp=False, norm=1, gtol=1e-11)
pmle_b, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = result

print(">>>", pmle_b, fopt)
if warnflag > 0:
    print("fmin_bfgs returned warnflag=%d" % warnflag)

pmle = bestparams
print("MLE:")
print("w =", expit(pmle[0]))
print("exp scale =", np.exp(pmle[1]))
print("gamma shape =", pmle[2])
print("gamma scale =", np.exp(pmle[3]))


vals, bedges, patches = plt.hist(x, density=True, bins=bin_edges,
                                 facecolor='tan', edgecolor='black', alpha=0.3)
xx = np.linspace(0, 1.05*x.max(), 400)
yy = exp_gamma_mixture_pdf(xx, *pmle)
plt.plot(xx, yy, 'k-', linewidth=2.5, alpha=0.75)
w = expit(pmle[0])
exp_scale = np.exp(pmle[1])
gamma_scale = np.exp(pmle[3])
plt.plot(xx, w*gamma.pdf(xx, 1, loc=0, scale=exp_scale), 'r--', linewidth=1)
plt.plot(xx, (1-w)*gamma.pdf(xx, pmle[2], loc=0, scale=gamma_scale),
         'c--', linewidth=1)

plt.show()
