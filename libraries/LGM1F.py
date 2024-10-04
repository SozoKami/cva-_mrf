import numpy as np
import pandas as pd
import seaborn as sns

sns.set(color_codes=True)

import os
import xlrd

from scipy.stats import norm
import scipy
import scipy.interpolate
from scipy import optimize as opt
from scipy.interpolate import splev, splrep


def B_interp(x, phi, t, T, zcr_t, zcr_T, lam):
    """
    Deterministic function of the state variable.
    -----------
    """
    # zero-coupon bond
    zcb_t = np.exp(-zcr_t * t)
    zcb_T = np.exp(-zcr_T * T)

    beta = (1 - np.exp(-lam * (T - t))) / lam
    #np.apply_along_axis(lambda x : x/zcr_t,0,zcr_T) * np.exp(-0.5 * beta ** 2 * phi - beta * x)

    return (zcb_T / zcb_t) * np.exp(-0.5 * beta ** 2 * phi - beta * x)

def B_deterministic(x, phi, t, T, zcb_curve, lam):
    """
    Deterministic function of the state variable.
    -----------
    """
    zcb_t, zcb_T= zcb_curve(t) , zcb_curve(T)
    beta = (1 - np.exp(-lam * (T - t))) / lam
    return (zcb_T / zcb_t) * np.exp(-0.5 * beta ** 2 * phi - beta * x)

def blackPrice(F, K, T, vol, exercice):
    """
    Returns price calculated with Black formula

    Parameters
    ----------
    F : forward rate
    K : strike of the option
    T : maturity of the option
    vol : implied vol of the underlying
    exercice : takes 'CALL' or 'PUT'
    """

    d1 = (np.log(F / K) + 0.5 * T * vol ** 2) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    exercice = (exercice == 'CALL') * 2 - 1
    price = exercice * (F * norm.cdf(exercice * d1) - K * norm.cdf(exercice * d2))

    return price


def loss_x0(x, phiTe, T0, Te, payment_dates, zcb_curve, c, lam):
    """
    Loss function to optimize to find x0.

    Parameters
    ----------
    x : float, quantity to find
    phiTe : float, values of phi at time te
    Te : float, expiry
    fixing_date : 1D-array
    c : 1D-array
    ZCR : DataFrame maturities and zero-coupon rate

    Output : float
    """
    return ((c * B_deterministic(x, phiTe, Te, payment_dates, zcb_curve, lam)).sum() / B_deterministic(x, phiTe, Te, T0, zcb_curve, lam) - 1)

######################## Not Already used #############################################
def phi_func(sigmaTe, t, lam):
    # ---------------------------------------------------------
    b = (sigmaTe ** 2) * (1 - np.exp(-2 * lam * t)) / (2 * lam)
    return b['sigma'][0]

