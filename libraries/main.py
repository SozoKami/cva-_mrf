import pickle
import warnings

import numpy as np
import pandas as pd

from GPy.models import GPRegression
from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from scipy import optimize as opt
from scipy.interpolate import splev, splrep
from scipy.stats import norm
from scipy.integrate import quad


from LGM1F import loss_x0, blackPrice, B_deterministic
from Chebychev import eval_Barycentric , Chebyshev_points

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF

kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

warnings.filterwarnings('ignore')
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure
from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.quadrature.measures import LebesgueMeasure


####################################################
class zero_coupon_bonds:
    def __init__(self, ZCR, k):
        self.spline_deg = k
        self.initial_curve_data = ZCR
        # self.initial_curve = self.interpolate_zcr_curve()[0]
        # self.initial_forward = lambda t: self.interpolate_zcr_curve()[1](t) * t + self.initial_curve(t)
        # self.initial_zcb_curve = lambda t: np.exp(-t * self.initial_curve(t))

    def interpolate_zcr_curve(self,t):  # We interpolate the curve and the derivatives
        # interp = scipy.interpolate.CubicSpline(time_zc, rate_sc)
        spl = splrep(self.initial_curve_data["Time"], self.initial_curve_data["Rate"],
                     k=self.spline_deg)  # In vba k = 4
        return splev(t, spl),  splev(t, spl, der=1)

    def initial_curve(self,t):
        return self.interpolate_zcr_curve(t)[0]

    def initial_forward(self,t):
        return self.interpolate_zcr_curve(t)[1] * t + self.initial_curve(t)

    def initial_zcb_curve(self,t):
        return np.exp(-t * self.initial_curve(t))




# payer , reciver

class Swap:
    def __init__(self, N, K, T0, mat, freq, exercice):
        self.nominal = N
        self.strike = K
        self.T0 = T0
        self.maturity = mat
        self.freq = freq
        self.tenor = np.arange(self.T0, self.T0 + self.maturity + self.freq, self.freq)
        self.exercice = exercice

    def swap_rate(self, ZC):
        zcb = ZC.initial_zcb_curve(self.tenor)
        return (zcb[0] - zcb[-1]) / ((self.tenor[1:] - self.tenor[:-1]) * zcb[1:]).sum()

    def get_atm_rate(self, ZC):
        return self.swap_rate(ZC)

    def swaptionLGMprice(self, expiries, sigma, lam, ZCR):

        if expiries <= self.T0:
            # delta times computation
            payment_dates, reset_dates = np.array(self.tenor[1:]), np.array(self.tenor[:-1])
            delta = np.array(self.tenor[1:]) - np.array(self.tenor[:-1])

            zcb_curve = lambda t : ZCR.initial_zcb_curve(t)

            c = self.strike * delta
            c[-1] += 1

            phi = (sigma ** 2) * (1 - np.exp(-2 * lam * expiries)) / (2 * lam)

            x0 = opt.newton(loss_x0, 0,
                            args=(
                                phi, self.T0, expiries, payment_dates, zcb_curve, c,
                                lam),
                            maxiter=300)
            Ki = B_deterministic(x0, phi, expiries, payment_dates, zcb_curve, lam) / B_deterministic(x0, phi, expiries,
                                                                                                     self.T0, zcb_curve,
                                                                                                     lam)
            Fi = zcb_curve(payment_dates) / zcb_curve(self.T0)

            sigma_i = (np.exp(-lam * (self.T0 - expiries)) - np.exp(-lam * (payment_dates - expiries))) * np.sqrt(
                phi / expiries) / lam
            black_type = "CALL" if self.exercice == "reciver" else "PUT"

            return self.nominal * zcb_curve(self.T0) * ((c * blackPrice(Fi, Ki, expiries, sigma_i, black_type)).sum())

        elif self.tenor[0] < expiries <= self.tenor[-2]:  # convention a voir
            new_t0 = self.T0 + ((expiries - self.T0) - (expiries - self.T0) % self.freq)
            new_t0 = new_t0 if ((expiries - self.T0) % self.freq) == 0 else new_t0 + self.freq
            new_mat = self.tenor[-1] - new_t0

            new_swap = Swap(self.nominal, self.strike, new_t0, new_mat, self.freq, self.exercice)
            return new_swap.swaptionLGMprice(expiries, sigma, lam, ZCR)

        return 0

    def swaption_bachelier(self, expiry, sigma, lam, ZCR):

        if expiry <= self.T0:
            zeroCoupon = lambda t : ZCR.initial_zcb_curve(t)

            t_start = self.tenor[0]
            t_end = self.tenor[-1]

            level = 0
            level_exp = 0

            for tenor in self.tenor[1:]:
                level += self.freq * zeroCoupon(tenor)
                level_exp += self.freq * zeroCoupon(tenor) * np.exp(-lam * tenor)

            s0 = (zeroCoupon(t_start) - zeroCoupon(t_end)) / level
            g0 = (zeroCoupon(t_start) * np.exp(-lam * t_start) - zeroCoupon(t_end) * np.exp(-lam * t_end)) / (
                    zeroCoupon(t_start) - zeroCoupon(t_end)) - level_exp / level
            g0 /= lam

            sigma_etoile = sigma ** 2 * s0**2 * g0 ** 2 * (np.exp(2 * lam * expiry) - 1) / (2 * lam)
            sigma_etoile = np.sqrt(sigma_etoile / expiry)

            if self.exercice == 'payer':
                return self.nominal * level * ((s0 * sigma_etoile * np.sqrt(expiry)) * norm.pdf(
                    (self.strike - s0) / (s0 * sigma_etoile * np.sqrt(expiry))) +
                                               (s0 - self.strike) * (1 - norm.cdf(
                            (self.strike - s0) / (s0 * sigma_etoile * np.sqrt(expiry)))))
            else:
                return self.nominal * level * ((s0 * sigma_etoile * np.sqrt(expiry)) * norm.pdf(
                    (self.strike - s0) / (s0 * sigma_etoile * np.sqrt(expiry))) +
                                               (self.strike - s0) * norm.cdf(
                            (self.strike - s0) / (s0 * sigma_etoile * np.sqrt(expiry))))

        elif self.tenor[0] < expiry <= self.tenor[-2]:  # convention a voir
            new_t0 = self.T0 + ((expiry - self.T0) - (expiry - self.T0) % self.freq)
            new_t0 = new_t0 if ((expiry - self.T0) % self.freq) == 0 else new_t0 + self.freq
            new_mat = self.tenor[-1] - new_t0

            new_swap = Swap(self.nominal, self.strike, new_t0, new_mat, self.freq, self.exercice)
            return new_swap.swaption_bachelier(expiry, sigma, lam, ZCR)
        return 0

    def show(self):
        return {"first_reset_date": self.T0,
                "swap_freq": self.freq, "swap_N": self.nominal, "swap_maturity": self.maturity,
                "swap_fixing_date": self.tenor, "swap_fix_rate": self.strike, "swap_exercice": self.exercice}


class IrsPortfolio:
    def __init__(self, NBR_SWAPS, first_date=[0], freq=[0.25, 0.5, 1], Nominal=[10000],
                 maturity=list(np.arange(1, 7 + 0.5, 0.5)),
                 fix_rate=[0.02, 0.05], exercice=['payer'], SEED=1999):

        self.swaps_number = NBR_SWAPS
        self.nominal = np.random.RandomState(SEED).choice(Nominal, self.swaps_number)
        self.strike = np.random.RandomState(SEED).uniform(fix_rate[0], fix_rate[1], self.swaps_number)
        self.T0 = np.random.RandomState(SEED).choice(first_date, self.swaps_number)
        self.maturity = np.random.RandomState(SEED).choice(maturity, self.swaps_number)
        self.freq = np.random.RandomState(SEED).choice(freq, self.swaps_number)

        self.exercice = np.random.RandomState(SEED).choice(exercice, self.swaps_number)

        self.tenor = np.array(
            [(np.arange(self.T0[i], self.T0[i] + self.maturity[i] + self.freq[i], self.freq[i])) for i in
             range(self.swaps_number)], dtype="object")

        self.last_maturity = np.max(self.maturity + self.T0)

        self.swaps = [Swap(N, K, T0, mat, freq, exercice) for (N, K, T0, mat, freq, exercice) in
                      zip(self.nominal, self.strike, self.T0, self.maturity, self.freq, self.exercice)]

    def get_atm_rate(self, ZC):
        return np.array([swap.swap_rate(ZC) for swap in self.swaps])

    def print_as_dict(self):
        return {"first_reset_date": self.T0,
                "swap_freq": self.freq, "swap_N": self.nominal, "swap_maturity": self.maturity,
                "swap_fixing_date": self.tenor,
                "LAST_MATURITY": self.last_maturity, "swap_fix_rate": self.strike, "swap_exercice": self.exercice,
                "NBR_SWAPS": self.swaps_number}

    def Mtm(self, t, diffusion, ZC):
        p = []
        for swap in self.swaps:
            if swap.tenor[-1] <= t:
                p.append(np.zeros(diffusion.nbr_scenarios))
            else:
                swap_type = (swap.exercice == "payer") * 2 - 1
                maturities = swap.tenor[(swap.tenor < t).sum():]
                zcb = diffusion.ZCB(t, maturities, ZC)
                floating_leg = swap.nominal * (zcb[:, 0] - zcb[:, -1])
                fixing_leg = (swap.nominal * swap.strike * zcb[:, 1:] * np.diff(maturities)).sum(axis=1)
                p.append(swap_type * (floating_leg - fixing_leg))

        p = np.array(p)
        return np.sum(p.T, axis=1)

    def Mtm_mono(self, t, x, ZC, lgm_params):
        lam , sig = lgm_params
        p = []
        for swap in self.swaps:
            if swap.tenor[-1] <= t:
                p.append(np.zeros(x.shape[0]))
            else :
                swap_type = (swap.exercice == "payer") * 2 - 1
                maturities = swap.tenor[(swap.tenor < t).sum():]

                if t in (list(maturities)):
                    t = t - 0.0000000000001  # To set rate 0 maybe we have some num instabilities
                zcb_curve = lambda y: ZC.initial_zcb_curve(y)
                phi = (sig ** 2) * (1 - np.exp(-2 * lam * t)) / (2 *lam)

                zcb = [list(B_deterministic(x, phi, t, m, zcb_curve, lam)) for m in maturities]
                zcb = np.array(zcb).T

                floating_leg = swap.nominal * (zcb[:,0] - zcb[:,-1])
                fixing_leg = (swap.nominal * swap.strike * zcb[:,1:] * np.diff(maturities)).sum(axis=1)
                p.append(swap_type * (floating_leg - fixing_leg))

        p = np.array(p)
        return np.sum(p, axis=0)



    def Mtm_GPR(self,t, diffusion,ZC, nodes_nbr):

        train_range = (min(diffusion.X(t)) , max(diffusion.X(t)) )

        # Pricing to train
        train_points = np.linspace(train_range[0], train_range[1], nodes_nbr)
        mtm_points = self.Mtm_mono(t,train_points,ZC,diffusion.get_LGM_params())

        # Train
        gpr_model = train_GPR(train_points.reshape(nodes_nbr,1), mtm_points.reshape(nodes_nbr,1))

        # Mtm Proxy
        Mtm = gpr_predictor(gpr_model, diffusion.X(t).reshape(diffusion.nbr_scenarios,1))

        return Mtm.reshape(diffusion.nbr_scenarios) # Get only prediction without variance error

    def Mtm_Chebychev(self,t, diffusion,ZC, nodes_nbr):

        train_range = (min(diffusion.X(t)) , max(diffusion.X(t)) )

        # Pricing to train
        train_points = Chebyshev_points(train_range[0], train_range[1], nodes_nbr)[::-1]
        mtm_points = self.Mtm_mono(t,train_points,ZC,diffusion.get_LGM_params())

        # Mtm Proxy
        Mtm = np.array([eval_Barycentric(mtm_points, train_points, x) for x in diffusion.X(t)])

        return Mtm

    def Mtm_proxy(self, t, diffusion,ZC, GPR, Chebychev,nodes_nbr):
        train_range = (min(diffusion.X(t)), max(diffusion.X(t)))

        if GPR :
            train_points = np.linspace(train_range[0], train_range[1], nodes_nbr)
        elif Chebychev :
            train_points = Chebyshev_points(train_range[0], train_range[1], nodes_nbr)[::-1]

        mtm_points = self.Mtm_mono(t, train_points, ZC, diffusion.get_LGM_params())

        if GPR :
            if mtm_points.sum() == 0 :
                return np.zeros(diffusion.nbr_scenarios)
            gpr_model = train_GPR(train_points.reshape(nodes_nbr, 1), mtm_points.reshape(nodes_nbr, 1))
            Mtm = gpr_predictor(gpr_model, diffusion.X(t).reshape(diffusion.nbr_scenarios, 1))

            return Mtm.reshape(diffusion.nbr_scenarios) # Get only prediction without variance error

        elif Chebychev :
            Mtm = np.array([eval_Barycentric(mtm_points, train_points, x) for x in diffusion.X(t)])
            return Mtm



    def reset_specs(self, irs_specs):
        self.swaps_number, self.nominal, self.strike, self.T0, self.freq, self.maturity, self.exercice = len(irs_specs), \
                                                                                                         irs_specs[
                                                                                                             'notional'], \
                                                                                                         irs_specs[
                                                                                                             'swap_rate'], \
                                                                                                         irs_specs[
                                                                                                             'first_reset'], \
                                                                                                         irs_specs[
                                                                                                             'reset_freq'], \
                                                                                                         irs_specs[
                                                                                                             'maturity'], \
                                                                                                         irs_specs[
                                                                                                             'exercice']
        self.tenor = np.array(
            [(np.arange(self.T0[i], self.T0[i] + self.maturity[i] + self.freq[i], self.freq[i])) for i in
             range(self.swaps_number)], dtype="object")

        self.last_maturity = np.max(self.maturity + self.T0)

        self.swaps = [Swap(N, K, T0, mat, freq, exercice) for (N, K, T0, mat, freq, exercice) in
                      zip(self.nominal, self.strike, self.T0, self.maturity, self.freq, self.exercice)]
        return self


    def to_table(self):
        irs_specs = np.empty(self.swaps_number,
                             dtype=[('first_reset', '<f4'), ('reset_freq', '<f4'),
                                    ('notional', '<f4'), ('swap_rate', '<f4'),
                                    ('maturity', '<f4'), ('exercice', '<U10')])

        irs_specs['first_reset'] = self.T0
        irs_specs['reset_freq'] = self.freq
        irs_specs['notional'] = self.nominal
        irs_specs['swap_rate'] = self.strike
        irs_specs['exercice'] = self.exercice
        irs_specs['maturity'] = self.maturity

        return irs_specs

    def save_as_excel(self, path, name):
        pd.DataFrame(self.to_table()).to_excel(path + name)


    def save_as_pickle(self, path, name):
        with open(path + name, 'wb') as f1:
            pickle.dump(self, f1)

class Multi_FX_irs_portfolio:
    def __init__(self, NBR_Currency, NBR_SWAPS, first_date=[0], freq=[0.25, 0.5, 1], Nominal=[10000],
                 maturity=list(np.arange(1, 7 + 0.5, 0.5)),
                 fix_rate=[0.2, 0.5], exercice=['payer'], SEED=1999):
        self.nbr_swaps = NBR_SWAPS
        self.nbr_currency = NBR_Currency
        self.irs_specs = self.irs_specs_creator(first_date, freq, Nominal, maturity, fix_rate, exercice, SEED)

        self.irs_sub_portfolios = [IrsPortfolio(1).reset_specs(self.irs_specs[self.irs_specs['undl'] == e]) for e in
                                   range(self.nbr_currency)]

        self.last_maturity = np.max([ p.last_maturity for p in self.irs_sub_portfolios])

    def irs_specs_creator(self, first_date, freq, Nominal, maturity, fix_rate, exercice,
                          SEED=1999):  # Function to bulid a portfolio composition as in Bouazza & Hoang project ( We don't account multiple ctpy and stoch intensities
        irs_specs = np.empty(self.nbr_swaps,
                             dtype=[('first_reset', '<f4'), ('reset_freq', '<f4'),
                                    ('notional', '<f4'), ('swap_rate', '<f4'),
                                    ('maturity', '<f4'), ('exercice', '<U10'),
                                    ('undl', '<i4')])  # We set maturity instead of num_resets

        irs_specs['first_reset'] = np.random.RandomState(SEED).choice(first_date,
                                                                      self.nbr_swaps)  # First reset date in the swaps
        irs_specs['reset_freq'] = np.random.RandomState(SEED).choice(freq, self.nbr_swaps)  # Reset frequency
        irs_specs['notional'] = Nominal[0] * np.random.choice(range(1, 11), self.nbr_swaps)  # Notional of the swaps
        irs_specs['swap_rate'] = np.random.RandomState(SEED).uniform(fix_rate[0], fix_rate[1],
                                                                     self.nbr_swaps)  # Swap rate, not needed, swaps are priced at par anyway

        irs_specs['undl'] = np.random.randint(0, self.nbr_currency, self.nbr_swaps, np.int32)  # Underlying currency
        irs_specs['exercice'] = np.random.RandomState(SEED).choice(exercice, self.nbr_swaps)
        irs_specs['maturity'] = np.random.RandomState(SEED).choice(maturity, self.nbr_swaps)
        return irs_specs

    def Mtm(self, t, multi_fx_diffusion, ZC_list):
        sub_mtm = np.array([irs.Mtm(t, diff, zc) for (irs, diff, zc) in
                            zip(self.irs_sub_portfolios, multi_fx_diffusion.diffusion_objects, ZC_list)])
        fx_rate = np.array([multi_fx_diffusion.FX(t, e) for e in range(self.nbr_currency)])

        return (fx_rate * sub_mtm).sum(axis=0)

    def Mtm_GPR(self, t, multi_fx_diffusion, ZC_list, train_size):
        sub_mtm = np.array([irs.Mtm_GPR(t, diff, zc, train_size) for (irs, diff, zc) in
                            zip(self.irs_sub_portfolios, multi_fx_diffusion.diffusion_objects, ZC_list)])

        fx_rate = np.array([multi_fx_diffusion.FX(t, e) for e in range(self.nbr_currency)])

        return (fx_rate * sub_mtm).sum(axis=0)

    def Mtm_proxy(self, t, multi_fx_diffusion, ZC_list,GPR, Chebychev, train_size):
        sub_mtm = np.array([irs.Mtm_proxy(t, diff, zc, GPR, Chebychev,train_size) for (irs, diff, zc) in
                            zip(self.irs_sub_portfolios, multi_fx_diffusion.diffusion_objects, ZC_list)])

        fx_rate = np.array([multi_fx_diffusion.FX(t, e) for e in range(self.nbr_currency)])

        return (fx_rate * sub_mtm).sum(axis=0)

    def save_as_pickle(self, path, name):
        with open(path + name, 'wb') as f1:
            pickle.dump(self, f1)

    def save_as_excel(self, path, name):
        pd.DataFrame(self.irs_specs).to_excel(path + name)

    def reset_portfolio_from_excel(self, path, name):
        port = pd.read_excel(path + name)

        irs_specs = np.empty(self.nbr_swaps,
                             dtype=[('first_reset', '<f4'), ('reset_freq', '<f4'),
                                    ('notional', '<f4'), ('swap_rate', '<f4'),
                                    ('maturity', '<f4'), ('exercice', '<U10'),
                                    ('undl', '<i4')])  # We set maturity instead of num_resets

        for col in port.columns[1:]:
            irs_specs[col] = port[col]

        self.irs_specs = irs_specs
        self.irs_sub_portfolios = [IrsPortfolio(1).reset_specs(self.irs_specs[self.irs_specs['undl'] == e]) for e in
                                   range(self.nbr_currency)]

        self.last_maturity = np.max([ p.last_maturity for p in self.irs_sub_portfolios])




# It's an LGM Diffusion maybe create other diff for FX
class Diffusion:
    def __init__(self, t0, T, n, sig, lam, NBR_SCENARIOS=10 * 5, pb_measure='Terminal at t'):
        self.nbr_scenarios = NBR_SCENARIOS
        self.time_grid = np.linspace(t0, T, n)
        self.sig = sig
        self.lam = lam
        self.proba_measure = pb_measure
        self.X_diff = self.diffuse_X_lgm(prob_measure=pb_measure)

    def diffuse_X_lgm(self, prob_measure='Terminal at t'):
        X_diff = np.empty((len(self.time_grid), self.nbr_scenarios))
        X_diff[0, :] = 0
        for i, t in enumerate(self.time_grid[1:]):  # here t = time_grid[i+1] l
            dt = (self.time_grid[i + 1] - self.time_grid[i])
            if prob_measure == 'Terminal at t':
                X_diff[i + 1, :] = X_diff[i, :] - self.lam * X_diff[i, :] * dt + self.sig * np.sqrt(
                    dt) * np.random.normal(0, 1, self.nbr_scenarios)
            elif prob_measure == 'Risk Neutral':
                phi = (self.sig ** 2) * (1 - np.exp(-2 * self.lam * self.time_grid[i])) / (2 * self.lam)
                X_diff[i + 1, :] = X_diff[i, :] + (phi - self.lam * X_diff[i, :] )* dt + self.sig * np.sqrt(
                    dt) * np.random.normal(0, 1, self.nbr_scenarios)
        return X_diff

    def X(self, t):
        idx_prev = np.abs(self.time_grid - t).argmin()
        X_prev = self.X_diff[idx_prev, :]

        dt = t - self.time_grid[idx_prev]
        if self.proba_measure == 'Terminal at t':
            X = X_prev - self.lam * X_prev * dt + self.sig * np.sqrt(dt) * np.random.normal(0, 1, self.nbr_scenarios)
        elif self.proba_measure == 'Risk Neutral':
            phi = (self.sig ** 2) * (1 - np.exp(-2 * self.lam * t)) / (2 * self.lam)
            X = X_prev + (phi - self.lam * X_prev) * dt + self.sig * np.sqrt(dt) * np.random.normal(0, 1, self.nbr_scenarios)

        if np.isnan(X).sum() > 0:
            return X_prev
        return X

    def short_rate(self, t, ZC):
        return self.X(t) + ZC.initial_forward(t)

    def ZCB(self, t, maturities, ZC):  # Reconstruction formula is indepandant under measure change t,risk neutral
        """
        Risk factor :
        computation of the zero coupon bond price at time t for several maturities

        Output
        ------
        2D array (n_simulation X n_maturities)
        Each line is a simulation of a ZCB curve.

        """
        if t in (list(maturities)):
            t = t - 0.0000000000001  # To set rate 0 maybe we have some num instabilities

        zcb_curve = lambda t: ZC.initial_zcb_curve(t)
        phi = (self.sig ** 2) * (1 - np.exp(-2 * self.lam * t)) / (2 * self.lam)
        zcb_t = [list(B_deterministic(self.X(t), phi, t, m, zcb_curve, self.lam)) for m in maturities]
        # modified from B to B_deterministic
        zcb_t = np.array(zcb_t).T
        return zcb_t

    def get_LGM_params(self):
        return self.lam ,self.sig

    def save_as_pickle(self, path, name):
        with open(path + name, 'wb') as f1:
            pickle.dump(self, f1)

    def discount_factor(self, t,ZC):
        idx_prev = np.abs(self.time_grid - t).argmin()

        D = np.exp(-multiplier(np.diff(self.time_grid[:idx_prev+1]), self.X_diff[:idx_prev,:]).sum(axis=0))

        return D*ZC.initial_zcb_curve(t)






class Model_params:
    def __init__(self, num_currency, from_excel= False, path=None):
        self.nbr_currency = num_currency
        self.LGM_params = self.set_lgm_params()
        self.FX_params = self.set_fx_params(from_excel, path)


    def set_lgm_params(self):  # here params are constant
        lgm_params = np.empty(self.nbr_currency, dtype=[('mean rev', '<f4'), ('sigma', '<f4')])
        lgm_params['mean rev'] = np.abs(
            np.random.uniform(0.01, 0.02, self.nbr_currency))  #  1% to 2%
        lgm_params['sigma'] = np.abs(
            np.random.uniform(0.005, 0.025, self.nbr_currency))  #  from 0.5% to 2.5%
        return lgm_params

    def set_fx_params(self, from_excel, path):
        fx_params = np.empty(self.nbr_currency, dtype=[('rate', '<f4'), ('vol', '<f4'), ('currency', '<U3')])
        fx_params['rate'] = np.ones(self.nbr_currency)
        fx_params['vol'] = np.abs(np.random.normal(0.025, 0.025, self.nbr_currency))
        fx_params['vol'][0] = 0  # The first FX rate by convention is 1

        if from_excel :
            fx_data = self.get_fx_data(path)
            fx_params['rate'][1:] = fx_data[0]
            fx_params['currency'][1:] = fx_data[1]
            fx_params['currency'][0] = fx_data[2]

        return fx_params

    def get_fx_data(self, path):
        fx_data = pd.read_excel(path)

        pair = fx_data['pair'].apply(lambda x: x[:3])
        pair.index = range(1, 10)

        domestic = fx_data['pair'][0][4:]

        return fx_data['rate'].values , pair, domestic





class Multi_FX_Diffusion:
    def __init__(self, t0, T, n, ZC_list, model_params, NBR_SCENARIOS = 10 * 5, pb_measure = 'Risk Neutral',cov_matrix = np.array([[1]])):
        self.nbr_scenarios = NBR_SCENARIOS
        self.time_grid = np.linspace(t0, T, n)
        self.proba_measure = pb_measure

        self.ZC_data = ZC_list  # list of zero-coupon-bonds objects

        self.nbr_currencies = model_params.nbr_currency
        self.LGM_params = model_params.LGM_params
        self.FX_params = model_params.FX_params
        self.covariance_matrix = cov_matrix  # size = 2*nbr_currencies - 1 : by convention we do not include the FX domestic rate

        self.brownian_motions_driver = np.array(
            [np.random.multivariate_normal(np.zeros(len(cov_matrix)), cov_matrix, size=self.nbr_scenarios) for _ in
             self.time_grid[1:]])
        self.X_diff = np.array([self.diffuse_X_lgm(e, prob_measure=pb_measure) for e in range(0, self.nbr_currencies)])
        self.FX_diff = np.array([self.diffuse_FX_rate(e, prob_measure=pb_measure) if e > 0 else np.ones(
            (len(self.time_grid), self.nbr_scenarios)) for e in range(0, self.nbr_currencies)])

        self.diffusion_objects = self.extract_mono_diffusion()


    def diffuse_X_lgm(self, e, prob_measure):
        # Constant param in time
        lam = self.LGM_params['mean rev'][e]
        sig = self.LGM_params['sigma'][e]
        fx_vol = self.FX_params['vol'][e] if e > 0 else 0
        fx_corr = self.covariance_matrix[e, e-1 + self.nbr_currencies] if e > 0 else 0

        X_diff = np.empty((len(self.time_grid), self.nbr_scenarios))
        X_diff[0, :] = 0
        for i, t in enumerate(self.time_grid[1:]):
            dt = (self.time_grid[i + 1] - self.time_grid[i])
            if prob_measure == 'Terminal at t':
                return NotImplemented
            elif prob_measure == 'Risk Neutral':
                phi = (sig ** 2) * (1 - np.exp(-2 * lam * self.time_grid[i])) / (2 * lam)
                X_diff[i + 1, :] = X_diff[i, :] + (
                        phi - lam * X_diff[i, :] - fx_corr * fx_vol * sig) * dt + sig * np.sqrt(dt) * \
                                   self.brownian_motions_driver[i][:, e]
        return X_diff


    def diffuse_FX_rate(self, e, prob_measure):
        # work only if e>0 as there's no BM for domestic FX
        # We have an explicit sheme to simulate FX rate (GBM process) : the only approximation is the one of Xt by euler sheme
        # We start with initial value 1 ( we can include a curve of FX rate )
        if prob_measure == 'Risk Neutral':
            fx_vol = self.FX_params['vol'][e] if e > 0 else 0
            Delta_time = np.diff(self.time_grid)

            FX_diff = np.empty((len(self.time_grid), self.nbr_scenarios))
            FX_diff[0, :] = self.FX_params['rate'][e] if e >0 else 1

            exp_prod = np.exp(
                multiplier(Delta_time, self.X_diff[0][:-1] - self.X_diff[e][:-1] - 0.5 * fx_vol) + fx_vol * multiplier(
                    Delta_time,self.brownian_motions_driver[:,:, e-1 + self.nbr_currencies]))

            zc_term = (self.ZC_data[0].initial_zcb_curve(self.time_grid[0]) * self.ZC_data[e].initial_zcb_curve(
                self.time_grid[1:])) / (
                              self.ZC_data[0].initial_zcb_curve(self.time_grid[1:]) * self.ZC_data[e].initial_zcb_curve(
                          self.time_grid[0]))

            FX_diff[1:, :] = FX_diff[0, :] * multiplier(zc_term , np.cumprod(exp_prod, axis=0))

            return FX_diff

        return NotImplemented


    # def X(self, t):
    #     idx_prev = np.abs(self.time_grid - t).argmin()
    #     X_prev = self.X_diff[idx_prev, :]
    #
    #     dt = t - self.time_grid[idx_prev]
    #     X = X_prev - self.lam * X_prev * dt + self.sig * np.sqrt(dt) * np.random.normal(0, 1, self.nbr_scenarios)
    #     if np.isnan(X).sum() > 0:
    #         return X_prev
    #     return X


    def FX(self, t, e):  # We take just an euler approx to simplify the script running ( I don't think there's an impact)
        idx_prev = np.abs(self.time_grid - t).argmin()
        FX_prev = self.FX_diff[e][idx_prev, :]
        return FX_prev


    def extract_mono_diffusion(self): #IMPORTANT TO MODIFY
        objects = []
        for e in range(self.nbr_currencies):
            diffusion = Diffusion(0, 1, 5, 0.1, 0.1, NBR_SCENARIOS=5, pb_measure='Terminal at t') # IT Can impact
            diffusion.nbr_scenarios, diffusion.time_grid, diffusion.proba_measure = self.nbr_scenarios, self.time_grid, self.proba_measure
            diffusion.lam, diffusion.sig = self.LGM_params['mean rev'][e], self.LGM_params['sigma'][e]
            diffusion.X_diff = self.X_diff[e]

            objects.append(diffusion)
        return objects

    def save_as_pickle(self, path, name):
        with open(path + name, 'wb') as f1:
            pickle.dump(self, f1)

#################### Independant procedures on our objects ##########################################

def Expected_exposure_MC(t, portfolio, diffusion, ZC, quasi_analytic=False,  GPR=False, Chebychev=False, train_size=5, n_std=4, integ_pts=100):
    if isinstance(portfolio, IrsPortfolio):
        if quasi_analytic :

            mu_X_t = 0
            std_X_t = diffusion.sig * np.sqrt( (1 - np.exp(-2 * diffusion.lam *t )) / (2 * diffusion.lam) )

            train_range = ( mu_X_t-n_std*std_X_t, mu_X_t+n_std*std_X_t )

            # Pricing to train
            train_points = np.linspace(train_range[0], train_range[1], train_size)
            mtm_points = portfolio.Mtm_mono(t,train_points,ZC,diffusion.get_LGM_params())


            if (mtm_points == np.zeros(train_size) ).all():
                return 0

            gpr_model = train_GPR(train_points.reshape(train_size, 1), mtm_points.reshape(train_size, 1))

            def gpr_exposure(x):
                return max(gpr_model.predict([[x]])[0, 0], 0) * norm.pdf(x, mu_X_t, std_X_t)

            X = np.linspace(train_range[0], train_range[1], integ_pts).reshape(integ_pts,1)
            Y = np.maximum(gpr_model.predict(X), 0)*norm.pdf(X, mu_X_t, std_X_t)

            return ZC.initial_zcb_curve(t) * (np.diff(X.reshape(X.shape[0]))*Y.reshape(Y.shape[0])[1:]).sum()

            #quad(gpr_exposure, -n_std*std_X_t, n_std*std_X_t)[0]
            #BQ(X,Y,-n_std*std_X_t,n_std*std_X_t).integrate()[0]
            #

        elif GPR or Chebychev :
            Vt = portfolio.Mtm_proxy(t, diffusion, ZC, GPR, Chebychev,train_size)
        else:
            Vt = portfolio.Mtm(t, diffusion, ZC)

        if diffusion.proba_measure == 'Terminal at t' and not quasi_analytic :
            return np.mean(ZC.initial_zcb_curve(t) * np.maximum(Vt, 0))

    elif isinstance(portfolio, Multi_FX_irs_portfolio):
        if GPR or Chebychev:
            Vt = portfolio.Mtm_proxy(t, diffusion, ZC, GPR, Chebychev,train_size)
        else :
            Vt = portfolio.Mtm(t, diffusion, ZC)

        if diffusion.proba_measure == 'Risk Neutral':
            zc, X, Delta_time = ZC[0], diffusion.X_diff[0], np.diff(
                diffusion.time_grid)  # In this case ZC should be a ZC list
            discount_correction = np.exp(-multiplier(Delta_time, X[:-1, :]).sum(axis=0))
            return np.mean(zc.initial_zcb_curve(t) * (discount_correction * np.maximum(Vt, 0)))
        else :
            return NotImplemented



def CVA_from_EE(R, lamda, time_grid):
    return NotImplemented


def BQ(X, Y, lb, ub):
    # Define the model : Gaussain process Regression
    model_gpy = GPRegression(X, Y)
    model_gpy.optimize()  #(messages=True, ipython_notebook=True)  # Messages indicates you wish to see the progress of the optimizer, needs ipywidgets to be installed
    # Define the lower and upper bounds of the integral.
    integral_bounds = [(lb, ub)]
    # Load core elements for Bayesian quadrature
    emukit_measure = LebesgueMeasure.from_bounds(integral_bounds)
    emukit_qrbf = QuadratureRBFLebesgueMeasure(RBFGPy(model_gpy.kern), emukit_measure)
    emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=model_gpy)
    emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model, X=X, Y=Y)

    return emukit_method

def train_GPR(X,Y):
    gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    gpr_model.fit(X, Y)

    return gpr_model

def gpr_predictor(trained_gpr, x):
    return trained_gpr.predict(x)

def multiplier(d, A):
    return (d * A.T).T  # Return a matrix d[k]*A[k,j]


def format_table(df):
    l = list(df.columns)
    l.remove('Relative Error')
    df[l] = df[l].astype(int)
    return df.round(2)


def get_synthetic_table(df, idx, mc_df, mtm, col_name):

    res = df.copy()

    res['Relative Error'] = df[col_name].apply(lambda x: 1000 * abs(x - mc_df.iloc[idx]['CVA']) / abs(mtm))
    res['Speedup Factor'] = df['Computation Time'].apply(lambda x: int(mc_df.iloc[idx]['Computation Time'] / x))

    # bq.round(2)
    res = format_table(res)
    if 'Var' in res.columns:
        res.drop(columns='Var')
    if 'Variance' in res.columns:
        res.drop(columns='Variance')

    return format_table(res)[['Training points Number', col_name, 'Relative Error', 'Computation Time', 'Speedup Factor']]


########################## NOT USED ONLY FOR VALIDATION OF OBJECT METHOD ######################
def LGM_swaption_pricer(expiries, swap, sigma, lam, ZCR):
    if swap.tenor[0] < expiries <= swap.tenor[-2]:  # convention a voir
        new_t0 = swap.T0 + ((expiries - swap.T0) - (expiries - swap.T0) % swap.freq)
        new_t0 = new_t0 if ((expiries - swap.T0) % swap.freq) == 0 else new_t0 + swap.freq
        new_mat = swap.tenor[-1] - new_t0

        new_swap = Swap(swap.nominal, swap.strike, new_t0, new_mat, swap.freq, swap.exercice)
        return new_swap.swaptionLGMprice(expiries, sigma, lam, ZCR)
    elif expiries <= swap.tenor[0]:
        return swap.swaptionLGMprice(expiries, sigma, lam, ZCR)
    return 0


def Expected_exposure_MCGPR(t, portfolio, diffusion, ZC, n_std=6, train_size=5):
    mu_X_t = 0
    std_X_t = diffusion.sig * np.sqrt((1 - np.exp(-2 * diffusion.lam * t)) / (2 * diffusion.lam))

    train_range = (mu_X_t - n_std * std_X_t, mu_X_t + n_std * std_X_t)

    nodes_nbr = train_size

    train_points = np.linspace(train_range[0], train_range[1], nodes_nbr)
    mtm_points = portfolio.Mtm_mono(t, train_points, ZC, diffusion.get_LGM_params())
    if mtm_points.sum() == 0:
        Vt = np.zeros(diffusion.nbr_scenarios)
    else:
        gpr_model = train_GPR(train_points.reshape(nodes_nbr, 1), mtm_points.reshape(nodes_nbr, 1))
        Mtm = gpr_predictor(gpr_model, diffusion.X(t).reshape(diffusion.nbr_scenarios, 1))
        Vt = Mtm.reshape(diffusion.nbr_scenarios)

    return np.mean(ZC.initial_zcb_curve(t) * np.maximum(Vt, 0))

