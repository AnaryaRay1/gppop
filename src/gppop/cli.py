#!/usr/bin/env python
"""
Driver script: fit the correlated (m1, q, chi_eff, chi_p) population with Binned Gaussian Processes.
"""

__author__="Anarya Ray <anarya.ray@ligo.org>, Claude (Anthropic)"

import argparse
import os
import numpy as np
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")



import pickle


import h5py
import torch
import arviz as az

from gppop.binned_population_model import Utils
from gppop.rates import (
    make_significant_model_3d_n_eff_opt_torch,
    run_nuts,
    run_nuts_multigpu,
    recover_deterministics_from_samples,
)

DET_SITES = ("logn_tot", "n_corr", "n_corr_physical", "N_F_exp",
             "var_pe", "var_n_det", "var_log_L")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit the Gaussian Process model to the observed source "
                    "population (torch + gpytorch + Pyro NUTS, multi-GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--metadata", type=str, required=True, metavar="METADATA",
                        help="name of metadata file")
    parser.add_argument("--output", type=str, required=True, metavar="METADATA",
                        help="path to output file")
    parser.add_argument("--ntune", type=int, required=True, metavar="NTUNE",
                        help="Number of warmup (tuning) steps for the sampler.")
    parser.add_argument("--nsteps", type=int, required=True, metavar="NSTEPS",
                        help="Number of steps to sample post tuning.")
    parser.add_argument("--target_accept", type=float, required=True, metavar="TARGET_ACCEPT",
                        help="Target acceptance fraction for the sampler.")
    parser.add_argument("--njobs", type=int, required=True, metavar="NJOBS",
                        help="number of chains; chains are distributed round-robin "
                             "over --devices (one process per chain)")
    parser.add_argument("--sigma_sd", required=False, metavar="SIGMA_SD", type=float,
                        default=1.0, help="the sd of the prior on GP sigma")
    parser.add_argument("--mu_dim", required=False, metavar="MU_DIM", type=int, default=1,
                        help="the dimensionality of the mean of the GP")
    parser.add_argument("--mc_convergence_check", required=False,
                        metavar="MC_CONVERGENCE_CHECK", type=bool, default=True,
                        help="Whether to implement Monte Carlo convergence likelihood "
                             "penalty through a variance cut")
    parser.add_argument("--exponent", required=False, metavar="EXPONENT", type=int,
                        default=-30, help="exponent to use for the likelihood penalty")
    parser.add_argument("--maximum_uncertainty", required=False,
                        metavar="MAXIMUM_UNCERTAINTY", type=float, default=1.0,
                        help="Maximum allowed variance of the log-likelihood function")
    # torch-specific options
    parser.add_argument("--devices", required=False, type=str, default=None,
                        help="comma-separated device list, e.g. "
                             "'cuda:0,cuda:1,cuda:2,cuda:3'. Default: all visible "
                             "GPUs if njobs > 1, else a single device")
    parser.add_argument("--float32", action="store_true",
                        help="run in float32 instead of float64 (float64 recommended)")
    parser.add_argument("--dense_mass", action="store_true",
                        help="use a dense mass matrix in NUTS")
    parser.add_argument("--seed", required=False, type=int, default=123456,
                        help="base rng seed (chain i uses seed + i)")
    return parser.parse_args()


def resolve_devices(args):
    if args.devices is not None:
        return [d.strip() for d in args.devices.split(",") if d.strip()]
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if args.njobs > 1:
            return [f"cuda:{i}" for i in range(min(n, args.njobs))]
        return ["cuda:0"]
    return ["cpu"]


def main():
    args = parse_args()
    np.random.seed(123456)
    devices = resolve_devices(args)
    dtype = torch.float32 if args.float32 else torch.float64

    # ---- load metadata (identical to the PyMC script) ---------------------
    with h5py.File(args.metadata, "r") as hf:
        gppop_data = hf["gppop_metadata"]
        mbins = gppop_data["mbins"][()]
        qbins = gppop_data["qbins"][()]
        chi_bins = gppop_data["chi_bins"][()]
        chip_bins = gppop_data["chip_bins"][()]
        scales_m = gppop_data["scales_m"][()]
        scales_q = gppop_data["scales_q"][()]
        scales_chi = gppop_data["scales_chi"][()]
        scales_chip = gppop_data["scales_chip"][()]
        weights = gppop_data["posterior_weights"][()]
        total_vts = gppop_data["vts"][()]
        total_vt_sigmas = gppop_data["vt_sigmas"][()]
        wt_means = gppop_data["wt_means"][()]
        wt_sigmas = np.nan_to_num(gppop_data["wt_sigmas"][()], nan=0.0)

    utils = Utils(mbins, qbins, chi_bins, chip_bins, kappa=2.7)
    log_bin_centers = utils.generate_log_bin_centers()
    deltaLogbin = utils.deltaLogbin()
    tril_deltaLogbin = utils.arraynd_to_tril(deltaLogbin, utils.construct_arg_mat())

    # ---- build model (on CPU; workers move it to their own GPU) -----------
    build_device = "cpu" if args.njobs > 1 else devices[0]
    model, meta = make_significant_model_3d_n_eff_opt_torch(
        log_bin_centers=log_bin_centers,
        weights=weights,
        tril_vts=total_vts,
        tril_deltaLogbins=tril_deltaLogbin,
        x_bins=chi_bins,
        y_bins=chip_bins,
        ls_mean_m=scales_m[0], ls_sd_m=scales_m[1],
        ls_mean_q=scales_q[0], ls_sd_q=scales_q[1],
        ls_mean_x=scales_chi[0], ls_sd_x=scales_chi[1],
        ls_mean_y=scales_chip[0], ls_sd_y=scales_chip[1],
        sigma_sd=args.sigma_sd,
        mu_dim=args.mu_dim,
        vt_sigmas=total_vt_sigmas,
        variance_cut=args.mc_convergence_check,
        wt_means=wt_means,
        wt_sigmas=wt_sigmas,
        exponent=args.exponent,
        maximum_uncertainty=args.maximum_uncertainty,
        device=build_device,
        dtype=dtype,
    )

    # ---- sample ------------------------------------------------------------
    print(f"start sampling posterior: {args.njobs} chain(s) on {devices} ({dtype})")
    if args.njobs > 1:
        posterior = run_nuts_multigpu(
            model,
            devices=devices,
            num_chains=args.njobs,
            num_samples=args.nsteps,
            warmup_steps=args.ntune,
            target_accept_prob=args.target_accept,
            seed=args.seed,
            dense_mass=args.dense_mass,
        )
    else:
        mcmc = run_nuts(
            model,
            num_samples=args.nsteps,
            warmup_steps=args.ntune,
            num_chains=1,
            target_accept_prob=args.target_accept,
            seed=args.seed,
            dense_mass=args.dense_mass,
        )
        posterior = {k: v.detach().cpu().numpy()
                     for k, v in mcmc.get_samples(group_by_chain=True).items()}
    
    with open(args.output, "wb") as pf:
        pickle.dump(posterior, pf)
        
    


if __name__ == "__main__":
    main()