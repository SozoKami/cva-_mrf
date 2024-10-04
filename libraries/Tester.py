import seaborn as sns
import  numpy as np
sns.set(color_codes=True)

import warnings

warnings.filterwarnings('ignore')

from main import *
from FinancialData import *
from Chebychev import *

def Random_correlation_matrix(n):
    rho = np.random.uniform(0, 1, (n, n))
    rho = (rho + rho.T) / 2
    np.fill_diagonal(rho, 1)
    rho = np.round(rho, 3)
    return rho

ZCR = ZC_Data_extractor("R:/DR-NATIXIS/ERM/MCRM/LeaderExpert/Stagiaires/2023")
ZC = zero_coupon_bonds(ZCR, 3)

nbr_fx =  3
nbr_swap = 10
portfolio = Multi_FX_irs_portfolio( nbr_fx, nbr_swap, first_date=[0], freq=[0.25, 0.5, 1], Nominal=[10000],
                 maturity=list(np.arange(1, 7 + 0.5, 0.5)),
                 fix_rate=[0.02, 0.05], exercice=['payer'], SEED=1999)

for e in portfolio.irs_sub_portfolios:
    print(e.print_as_dict())

model_params = Model_params(nbr_fx)

cov_matrix = Random_correlation_matrix(2*nbr_fx-1)

diffusion = Multi_FX_Diffusion(0, 7, 3000, [ZC for i in range(nbr_fx)], model_params, 10**4, pb_measure='Risk Neutral',
                 cov_matrix= cov_matrix)


Expected_exposure_MC(2, portfolio, diffusion, [ZC for i in range(nbr_fx)])



# # LGM parameter's
# sig, lam = 0.005, 0.01
# # recovery rate & defult probability parameter :
# R, lamda = 0.4, 0.005
#
# T = irs.last_maturity
# diffusion = Diffusion(0, int(T * 360) + 1, 3000, sig, lam, NBR_SCENARIOS=10 ** 4, pb_measure='Terminal at t')
#
#
# def EE_MC(t, irs, diffusion, ZC):
#     swap_dist = irs.Mtm(t, diffusion, ZC)
#     Pt = np.array(swap_dist.flatten())
#
#     return np.mean(ZC.initial_zcb_curve(t) * np.maximum(Pt, 0))
#
#
# EE_MC(1.1, irs, diffusion, ZC)

# import time
# time_grid = np.linspace(0,T,50)
# tt = time.time()
# EE_MC = np.array([EE_MC(t,irs, diffusion, ZC) for t in time_grid])
# PD_full = np.array([ lamda * np.exp(-lamda *t) for t in time_grid])
# Y_MC = (1-R)*EE_MC*PD_full
# print('Time Calcul : ', time.time()-tt)
# print((np.diff(time_grid)*Y_MC[1:]).sum())


# swap = irs.swaps[0]
# time_grid = np.linspace(0,8,2010)
# price = LGM_swaption_pricer(2,swap,0.005,0.01,ZC)
#
# plt.plot(time_grid, [swap.swaption_bachelier(t,0.005,0.01,ZC) for t in time_grid])
#
