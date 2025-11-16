#!/usr/bin/env python

import numpy as np
import pymc as pm
from gppop.core_m1m2z import Rates, Utils

[scale_mean_m,scale_sd_m,scale_mean_z,scale_sd_z] = np.loadtxt('data/gp_inputs_newbins3_m1m2z.txt')

mbins = np.loadtxt('data/mbins_m1m2z.txt')
zbins = np.loadtxt('data/zbins_m1m2z.txt')
vts = np.loadtxt('data/vts_GWTC4_IFAR1_newbins3_m1m2z.txt')
weights = np.loadtxt('data/weights_GWTC4_IFAR1_newbins3_m1m2z.txt')
wt_means = np.loadtxt('data/wt_means_GWTC4_newbins3_m1m2z.txt')
wt_sigmas = np.loadtxt('data/wt_sigmas_GWTC4_newbins3_m1m2z.txt')
vt_sigmas = np.loadtxt('data/vt_sigmas_GWTC4_newbins3_m1m2z.txt')

utils = Utils(mbins,zbins)
log_bin_centers = utils.generate_log_bin_centers()
tril_deltaLogbin = utils.arraynd_to_tril(utils.deltaLogbin())

rates = Rates(mbins,zbins)
model = rates.make_significant_model_3d_n_eff_opt(log_bin_centers,weights,vts,tril_deltaLogbin,scale_mean_m,scale_sd_m,scale_mean_z,scale_sd_z,sigma_sd=1.,mu_dim=1, vt_sigmas=vt_sigmas,vt_accuracy_check=False, wt_means = wt_means, wt_sigmas = wt_sigmas, exponent = -30)
with model:
    trace = pm.sample(draws=3000,tune=3000,chains=6,target_accept=0.95,discard_tuned_samples=True)

trace.to_netcdf('output/gppop_GWTC4_posterior_newbins3_kappa_corr_m1m2z.nc')
