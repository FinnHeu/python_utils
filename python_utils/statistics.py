import uncertainties.unumpy as unp
import uncertainties as unc
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

def predband(x, xd, yd, p, func, conf=.95):
    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters
    # func = function name
    alpha = 1.0 - conf    # significance
    N = xd.size          # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * \
                 np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy

    return lpb, upb

def CIs_regression(x, y):

    '''
    CIs_regression.py
    '''

    n = len(y)

    def f(x, a, b):
        return a * x + b

    popt, pcov = curve_fit(f, x, y)

    # retrieve parameter values
    a = popt[0]
    b = popt[1]
    print('Optimal Values')
    print('a: ' + str(a))
    print('b: ' + str(b))

    # compute r^2
    r2 = 1.0-(sum((y-f(x,a,b))**2)/((n-1.0)*np.var(y,ddof=1)))
    print('R^2: ' + str(r2))

    # calculate parameter confidence interval
    a,b = unc.correlated_values(popt, pcov)
    print('Uncertainty')
    print('a: ' + str(a))
    print('b: ' + str(b))

    # plot data
    plt.scatter(x, y, s=3, label='Data')

    # calculate regression confidence interval
    px = np.linspace(min(x), max(x), 100)
    py = a*px+b
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)


    lpb, upb = predband(px, x, y, popt, f, conf=.95)

    ci_l = nom - 1.96 * std
    ci_u = nom + 1.96 * std

    return px, ci_l, ci_u
