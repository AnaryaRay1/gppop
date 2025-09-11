#!/usr/bin/env python

import numpy as np
import pymc as pm
from gppop.core import Rates_spins_with_q, Utils_spins_with_q

[scale_mean_m,scale_sd_m,scale_mean_q,scale_sd_q,scale_mean_chi,scale_sd_chi] = np.loadtxt('gp_inputs_newbins3_with_q_case_9mdo.txt')

mbins = np.loadtxt('mbins_with_q_case_9mdo.txt')
qbins = np.loadtxt('qbins_with_q_case_9mdo.txt')
chi_bins = np.loadtxt('chi_bins_new3_with_q_case_9mdo.txt')
vts = np.loadtxt('vts_GWTC4_IFAR1_m1m2chi_newbins3_with_q_case_9mdo.txt')
weights = np.loadtxt('weights_GWTC4_IFAR1_m1m2chi_newbins3_with_q_case_9mdo.txt')
wt_means = np.loadtxt('wt_means_GWTC4_newbins3_with_q_case_9mdo.txt')
wt_sigmas = np.loadtxt('wt_sigmas_GWTC4_newbins3_with_q_case_9mdo.txt')

utils = Utils_spins_with_q(mbins,qbins,chi_bins,kappa=2.7)
log_bin_centers = utils.generate_log_bin_centers_with_q()
tril_deltaLogbin = utils.arraynd_to_tril(utils.deltaLogbin_with_q())

rates = Rates_spins_with_q(mbins,qbins,chi_bins,kappa=2.7)
model = rates.make_significant_model_3d_with_q_n_eff_opt(log_bin_centers,weights,vts,tril_deltaLogbin,scale_mean_m,scale_sd_m,scale_mean_q,scale_sd_q,scale_mean_chi,scale_sd_chi,sigma_sd=1.,mu_dim=1, vt_sigmas=None,vt_accuracy_check=False, wt_means = wt_means, wt_sigmas = wt_sigmas, exponent = -44)
with model:
    trace = pm.sample(draws=6000,tune=6000,chains=6,target_accept=0.95,discard_tuned_samples=True)

trace.to_netcdf('gppop_GWTC4_posterior_newbins3_kappa_corr_with_q_n_eff44_opt_case_9mdo.nc')
