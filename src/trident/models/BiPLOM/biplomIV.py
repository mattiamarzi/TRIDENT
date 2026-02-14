"""
biplomIV.py

══════════
Log-likelihood and derivatives for BiPLOM-IV.
"""

import numpy as np
import pandas as pd

def softplus(t):
    """
    Numerically stable evaluation of log(1 + exp(t)).

    Uses the identity:
        log(1 + exp(t)) = t + log(1 + exp(-t))   if t >= 0
                        =     log(1 + exp(t))   if t <  0

    so that the exponential is always evaluated on a non-positive argument.
    """
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t)
    mask = t >= 0
    out[mask] = t[mask] + np.log1p(np.exp(-t[mask]))
    out[~mask] = np.log1p(np.exp(t[~mask]))
    return out

def logistic(t):
    """
    Numerically stable logistic function:
        sigmoid(t) = 1 / (1 + exp(-t))

    Implemented via a piecewise definition to avoid overflow:

        if t >= 0: sigmoid(t) = 1 / (1 + exp(-t))
        if t <  0: sigmoid(t) = exp(t) / (1 + exp(t))

    In both branches the exponential is evaluated on a non-positive argument.
    """
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t)
    mask = t >= 0
    # t >= 0: use exp(-t)
    out[mask] = 1.0 / (1.0 + np.exp(-t[mask]))
    # t < 0: use exp(t)
    exp_t = np.exp(t[~mask])
    out[~mask] = exp_t / (1.0 + exp_t)
    return out

def loglikelihood_prime_IV(sol, args):

    k = args[0]
    h = args[1]
    V = args[2]
    molt_k = args[3]
    molt_h = args[4]
    n = len(k)
    n1 = len(h)
    x = sol[:n]
    y = sol[n:n+n1]
    z = sol[n+n1:]

    f = np.zeros(len(sol), dtype=np.float64)

    flag = True
    for i in np.arange(n):
        f[i] -= molt_k[i]*k[i]
        f[n+n1+i] -= molt_k[i]*V[i]
        for j in np.arange(n1):
            t_ij = -x[i] - y[j] - h[j]*z[i]
            p_ij = logistic(t_ij)
            e = molt_k[i] * molt_h[j] * p_ij
            f[i] += e
            f[n+j] += e
            f[n+n1+i] += h[j]*e
            if flag:
                f[n+j] -= molt_h[j]*h[j]
        flag = False

    return -f

def loglikelihood_hessian_IV(sol, args):

    k = args[0]
    h = args[1]
    molt_k = args[3]
    molt_h = args[4]
    n = len(k)
    n1 = len(h)
    x = sol[:n]
    y = sol[n:n+n1]
    z = sol[n+n1:]
    f = np.zeros(shape=(len(sol), len(sol)), dtype=np.float64)

    for i in np.arange(n):
        for j in np.arange(n1):

            t_ij = -x[i] - y[j] - h[j]*z[i]
            p_ij = logistic(t_ij)
            g2_ij = p_ij * (1.0 - p_ij)  # = exp(t_ij)/(1+exp(t_ij))^2
            e = molt_k[i] * molt_h[j] * g2_ij

            f[i,i] += -e
            f[n+j,n+j] += -e
            f[n+n1+i,n+n1+i] += -np.power(h[j],2)*e

            f[i,n+j] = -e
            f[n+j,i] = f[i,n+j]
            f[i,n+n1+i] += -h[j]*e
            f[n+n1+i,i] = f[i,n+n1+i]
            f[n+j,i+n1+n] += -h[j]*e
            f[i+n1+n,n+j] = f[n+j,i+n1+n]

    return -f

def loglikelihood_hessian_diag_IV(sol, args):

    k = args[0]
    h = args[1]
    molt_k = args[3]
    molt_h = args[4]
    n = len(k)
    n1 = len(h)

    x = sol[:n]
    y = sol[n:n + n1]
    z = sol[n + n1:]

    f = np.zeros(len(sol), dtype=np.float64)

    for i in range(n):
        xi = x[i]
        zi = z[i]
        for j in range(n1):
            hj = h[j]
            yj = y[j]

            # t_ij = -x_i - y_j - h_j * z_i  (same as in loglikelihood_IV)
            t_ij = -xi - yj - hj * zi

            # p_ij = exp(t_ij) / (1 + exp(t_ij))
            # g''(t_ij) = p_ij * (1 - p_ij)
            exp_t = np.exp(t_ij)
            denom = 1.0 + exp_t
            p_ij = exp_t / denom
            g2_ij = p_ij * (1.0 - p_ij)

            e = molt_k[i] * molt_h[j] * g2_ij

            f[i]         += -e
            f[n + j]     += -e
            f[n + n1 + i] += -(hj ** 2) * e

    return -f

def loglikelihood_IV(sol, args):

    k = args[0]
    h = args[1]
    V = args[2]
    molt_k = args[3]
    molt_h = args[4]
    n = len(k)
    n1 = len(h)
    x = sol[:n]
    y = sol[n:n+n1]
    z = sol[n+n1:]
    f = 0.0
    flag = True

    for i in np.arange(n):
        f -= molt_k[i]*((V[i]*z[i]) + (k[i]*x[i]))
        for j in np.arange(n1):
            t_ij = -x[i] - y[j] - z[i]*h[j]
            f -= molt_k[i] * molt_h[j] * softplus(t_ij)
            if flag:
                f -= molt_h[j]*h[j]*y[j]
        flag = False

    return -f