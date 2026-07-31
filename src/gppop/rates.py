"""
Bayesian Hierarchical Inference of a 4-dimensional piecewise constant model for the Binary Black Hole population with a Gaussian Process Prior.
"""

__author__="Anarya Ray <anarya.ray@ligo.org>, Claude (Anthropic)"

import os

import numpy as np
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from .TruncatedNormal import TruncatedNormalTransform
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.autoguide.initialization import init_to_median

from linear_operator.operators import KroneckerProductLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky


# --------------------------------------------------------------------------- #
# Kernels                                                                     #
# --------------------------------------------------------------------------- #
def expquad(x1: torch.Tensor, x2: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
    """Unit-variance squared-exponential kernel, matching pm.gp.cov.ExpQuad.

    x1 : (n, d), x2 : (m, d), lengthscale : scalar or (d,) tensor
    returns (n, m)
    """
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    sqdist = (z1.unsqueeze(-2) - z2.unsqueeze(-3)).pow(2).sum(-1)
    return torch.exp(-0.5 * sqdist)


def kron_cholesky(Ks, jitter: float = 1e-6) -> KroneckerProductLinearOperator:
    """Kronecker product of per-factor Cholesky factors (the LKGP trick).

    chol(K1 (x) K2 (x) K3) = L1 (x) L2 (x) L3, held lazily so matvecs cost
    O(N * (n1 + n2 + n3)) instead of O(N^2).
    """
    Ls = []
    for K in Ks:
        K = K + jitter * torch.eye(K.shape[-1], dtype=K.dtype, device=K.device)
        Ls.append(psd_safe_cholesky(K))
    return KroneckerProductLinearOperator(*Ls)


# --------------------------------------------------------------------------- #
# Model (top-level class => picklable for multi-chain MCMC)                   #
# --------------------------------------------------------------------------- #
class GPPopKronModel:
    """Pyro model equivalent to the PyMC `make_significant_model_3d_n_eff_opt`.

    Instances are callable Pyro models and picklable, so they work with
    Pyro's MCMC(num_chains > 1), which spawns worker processes.
    """

    def __init__(
        self,
        log_bin_centers,
        weights,
        tril_vts,
        tril_deltaLogbins,
        x_bins,
        y_bins,
        ls_mean_m, ls_sd_m,
        ls_mean_q, ls_sd_q,
        ls_mean_x, ls_sd_x,
        ls_mean_y, ls_sd_y,
        sigma_sd=1.0,
        mu_dim=None,
        vt_sigmas=None,
        variance_cut=False,
        wt_means=None,
        wt_sigmas=None,
        exponent=-30,
        maximum_uncertainty=1.0,
        device="cuda",
        dtype=torch.float64,
        jitter=1e-6,
    ):
        device = torch.device(device)

        # ---- identical numpy preprocessing to the PyMC version ------------ #
        tril_vts = np.asarray(tril_vts) * np.asarray(tril_deltaLogbins)
        arg = tril_vts > 0.0
        keep = np.where(arg)[0]
        weights = np.asarray(weights)
        if (~arg).sum() > 0:
            tril_vts = tril_vts[keep]
            weights = weights[:, keep]

        if variance_cut:
            assert vt_sigmas is not None and wt_sigmas is not None and wt_means is not None
            wt_means = np.asarray(wt_means)[:, keep]
            wt_sigmas = np.asarray(wt_sigmas)[:, keep]
            vt_sigmas = np.asarray(vt_sigmas) * np.asarray(tril_deltaLogbins)
            vt_sigmas = vt_sigmas[keep]
        else:
            wt_sigmas = np.zeros_like(weights)
            wt_means = weights
            vt_sigmas = np.zeros_like(tril_vts)

        if mu_dim is None:
            mu_dim = len(log_bin_centers)
        assert mu_dim == 1 or mu_dim == len(log_bin_centers)

        x_bins = np.asarray(x_bins)
        y_bins = np.asarray(y_bins)
        nchi = (len(x_bins) - 1) * (len(y_bins) - 1)
        bin_centers_x = 0.5 * (x_bins[1:] + x_bins[:-1])[:, None]
        bin_centers_y = 0.5 * (y_bins[1:] + y_bins[:-1])[:, None]
        log_bin_centers_m = np.asarray(log_bin_centers)[0::nchi, :2]
        n_bins = len(log_bin_centers_m) * nchi
        assert n_bins == len(log_bin_centers), \
            "bin ordering: chi_p fastest, then chi_eff, then (m1, q)"

        # ---- store everything as tensors on the target device ------------- #
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
        self.Xm = t(log_bin_centers_m)          # (n_m, 2)
        self.Xce = t(bin_centers_x)        # (n_ce, 1)
        self.Xcp = t(bin_centers_y)          # (n_cp, 1)
        self.W = t(weights)                     # (n_events, n_keep)
        self.VT = t(tril_vts)                   # (n_keep,)
        self.WT_MU = t(wt_means)                # (n_events, n_keep)
        self.WT_SD = t(wt_sigmas)               # (n_events, n_keep)
        self.VT_SD = t(vt_sigmas)               # (n_keep,)
        self.keep_idx = torch.as_tensor(keep, dtype=torch.long, device=device)

        self.ls_mu = {k: t(v) for k, v in dict(
            m=ls_mean_m, q=ls_mean_q, x=ls_mean_x, y=ls_mean_y).items()}
        self.ls_sd = {k: t(v) for k, v in dict(
            m=ls_sd_m, q=ls_sd_q, x=ls_sd_x, y=ls_sd_y).items()}
        self.sigma_sd = t(sigma_sd)
        self.zero = t(0.0)
        self.one = t(1.0)
        self.ten = t(10.0)

        self.mu_dim = int(mu_dim)
        self.n_bins = int(n_bins)
        self.jitter = float(jitter)
        self.exponent = float(exponent)
        self.log_max_unc_sq = float(2.0 * np.log(maximum_uncertainty))
        self.variance_cut_flag = float(bool(variance_cut))
        self.device = device
        self.dtype = dtype

    def to(self, device):
        """Move all tensor attributes to `device` (in place); returns self.
        Used to ship the model to a specific GPU in multi-GPU sampling."""
        device = torch.device(device)
        for name, val in self.__dict__.items():
            if torch.is_tensor(val):
                setattr(self, name, val.to(device))
            elif isinstance(val, dict):
                setattr(self, name, {k: (v.to(device) if torch.is_tensor(v) else v)
                                     for k, v in val.items()})
        self.device = device
        return self

    def __call__(self):
        
        mu = pyro.sample(
            "mu",
            dist.TransformedDistribution(dist.Uniform(self.zero,self.one), 
                                         [TruncatedNormalTransform(self.zero, self.ten, self.zero-15.0,
                                         self.zero+15.0)]).expand([self.mu_dim]).to_event(1),
        )

        sigma = pyro.sample("sigma", dist.HalfNormal(self.sigma_sd))
        ls_m = pyro.sample("length_scale_m", dist.LogNormal(self.ls_mu["m"], self.ls_sd["m"]))
        ls_q = pyro.sample("length_scale_q", dist.LogNormal(self.ls_mu["q"], self.ls_sd["q"]))
        ls_ce = pyro.sample("length_scale_x",
                            dist.LogNormal(self.ls_mu["x"], self.ls_sd["x"]))
        ls_cp = pyro.sample("length_scale_y",
                            dist.LogNormal(self.ls_mu["y"], self.ls_sd["y"]))

        # -- LatentKron GP prior (whitened, LKGP-style Kronecker Cholesky) -- #
        K_m = expquad(self.Xm, self.Xm, torch.stack([ls_m, ls_q]))
        K_ce = expquad(self.Xce, self.Xce, ls_ce)
        K_cp = expquad(self.Xcp, self.Xcp, ls_cp)
        L = kron_cholesky([K_m, K_ce, K_cp], jitter=self.jitter)

        w = pyro.sample(
            "w_white",
            dist.Normal(self.zero, self.one).expand([self.n_bins]).to_event(1),
        )
        # total covariance amplitude in the PyMC model is sigma^2 -> field scale sigma
        logn_corr = sigma * (L @ w.unsqueeze(-1)).squeeze(-1)
        logn_tot = mu + logn_corr
        n_corr = torch.exp(logn_tot)
        n_phys = n_corr[self.keep_idx]

        pyro.deterministic("logn_tot", logn_tot)
        pyro.deterministic("n_corr", n_corr)
        pyro.deterministic("n_corr_physical", n_phys)

        # -- inhomogeneous-Poisson-style likelihood ------------------------- #
        N_F_exp = (n_phys * self.VT).sum()
        pyro.deterministic("N_F_exp", N_F_exp)
        per_event = self.W @ n_phys                     # (n_events,)
        pyro.factor("log_l", torch.log(per_event).sum() - N_F_exp)

        # -- Monte-Carlo variance diagnostics + penalty ---------------------- #
        numerator = ((self.WT_SD * n_phys) ** 2).sum(dim=1)
        denominator = (self.WT_MU @ n_phys) ** 2
        var_pe = (numerator / denominator).sum()
        var_sel = ((n_phys * self.VT_SD) ** 2).sum()
        var_log_l = var_pe + var_sel + 1e-10
        pyro.deterministic("var_pe", var_pe)
        pyro.deterministic("var_n_det", var_sel)
        pyro.deterministic("var_log_L", var_log_l)

        # -log1p((max^2/var)^exponent) == -softplus(exponent*(log max^2 - log var))
        pyro.factor(
            "variance_cut",
            -self.variance_cut_flag
            * F.softplus(self.exponent * (self.log_max_unc_sq - torch.log(var_log_l))),
        )


def make_significant_model_3d_n_eff_opt_torch(*args, **kwargs):
    """Backwards-compatible builder: returns (model, meta) like before, but the
    model is now a picklable GPPopKronModel instance."""
    model = GPPopKronModel(*args, **kwargs)
    meta = dict(
        keep_idx=model.keep_idx, VT=model.VT, W=model.W,
        n_bins=model.n_bins, mu_dim=model.mu_dim,
        device=model.device, dtype=model.dtype,
    )
    return model, meta


# --------------------------------------------------------------------------- #
# Sampling with Pyro NUTS                                                     #
# --------------------------------------------------------------------------- #
def run_nuts(
    model,
    num_samples=1000,
    warmup_steps=1000,
    num_chains=1,
    target_accept_prob=0.9,
    max_tree_depth=10,
    seed=0,
    jit_compile=True,
    dense_mass=False,
):
    """Sample the model with NUTS on whatever device the data tensors live on.

    Note on num_chains > 1: Pyro runs each chain in a separate *spawned*
    process, so (a) the model must be picklable (GPPopKronModel is), and
    (b) all chains share one GPU unless you launch separate jobs per device.
    """
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        init_strategy=init_to_median(num_samples=20),
        jit_compile=jit_compile,
        ignore_jit_warnings=True,
        full_mass=dense_mass,
    )
    mcmc = MCMC(
        kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
        mp_context="spawn",      # required with CUDA; fork will crash the child
    )
    mcmc.run()
    mcmc.summary(prob=0.9)
    return mcmc


def recover_deterministics(model, mcmc, sites=("n_corr", "n_corr_physical", "N_F_exp", "var_log_L")):
    """Replay posterior samples through the model to get deterministic sites
    (n_corr, N_F_exp, ...), since MCMC only stores sample sites."""
    posterior = mcmc.get_samples()
    pred = Predictive(model, posterior_samples=posterior, return_sites=list(sites))
    return {k: v.detach().cpu().numpy() for k, v in pred().items()}


# --------------------------------------------------------------------------- #
# Multi-GPU sampling: one independent NUTS chain per device                   #
# --------------------------------------------------------------------------- #
def _mcmc_chain_worker(rank, model, device, num_samples, warmup_steps,
                       target_accept_prob, max_tree_depth, seed,
                       jit_compile, dense_mass, out_path):
    """Worker: run one NUTS chain on `device` and save samples to out_path.
    Must be a top-level function so spawn can pickle it."""
    from tqdm.auto import tqdm

    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = model.to(device)

    pyro.clear_param_store()
    pyro.set_rng_seed(seed)

    # One progress bar per chain, pinned to its own terminal line, driven by
    # MCMC's hook_fn (Pyro's own progbar is disabled: bars from multiple
    # processes would overwrite each other).
    bar = tqdm(
        total=warmup_steps + num_samples,
        position=rank,
        desc=f"chain {rank} [{device}] warmup",
        dynamic_ncols=True,
        mininterval=0.5,
        leave=True,
    )

    def _hook(kernel, params, stage, i):
        bar.set_description(f"chain {rank} [{device}] {stage.lower()}", refresh=False)
        bar.update(1)

    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        init_strategy=init_to_median(num_samples=20),
        jit_compile=jit_compile,
        ignore_jit_warnings=True,
        full_mass=dense_mass,
    )
    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps,
                num_chains=1, disable_progbar=True, hook_fn=_hook)
    mcmc.run()
    bar.close()
    samples = {k: v.detach().cpu() for k, v in mcmc.get_samples().items()}
    torch.save(samples, out_path)


def run_nuts_multigpu(
    model,
    devices,
    num_samples=1000,
    warmup_steps=1000,
    target_accept_prob=0.9,
    max_tree_depth=10,
    seed=0,
    jit_compile=True,
    dense_mass=False,
    num_chains=None,
    tmp_dir=None,
):
    """Run independent NUTS chains in parallel, one process per chain, with
    chains assigned round-robin to `devices` (e.g. ['cuda:0', ..., 'cuda:3']).

    Returns
    -------
    posterior : dict of str -> np.ndarray with shape (n_chains, n_draws, ...)
    """
    import tempfile
    import torch.multiprocessing as mp

    if num_chains is None:
        num_chains = len(devices)
    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="gppop_chains_")
    os.makedirs(tmp_dir, exist_ok=True)

    # Ship the model to workers on CPU; each worker moves it to its own GPU.
    # (CUDA tensors unpickle onto their original device, which would put every
    # chain back on the model's build device.)
    model = model.to("cpu")

    ctx = mp.get_context("spawn")
    procs, paths = [], []
    for i in range(num_chains):
        device = devices[i % len(devices)]
        path = os.path.join(tmp_dir, f"chain_{i}.pt")
        p = ctx.Process(
            target=_mcmc_chain_worker,
            args=(i, model, device, num_samples, warmup_steps,
                  target_accept_prob, max_tree_depth, seed + i,
                  jit_compile, dense_mass, path),
        )
        p.start()
        procs.append(p)
        paths.append(path)

    for p in procs:
        p.join()
    failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"chains {failed} exited with nonzero status; "
                           f"see their tracebacks above")

    chains = [torch.load(path, map_location="cpu") for path in paths]
    posterior = {
        k: np.stack([c[k].numpy() for c in chains], axis=0)   # (chain, draw, ...)
        for k in chains[0].keys()
    }
    return posterior


def recover_deterministics_from_samples(
    model, posterior,
    sites=("n_corr", "n_corr_physical", "N_F_exp", "var_log_L"),
    device="cpu",
    batch_size=200,
):
    """Like recover_deterministics, but takes a (chain, draw, ...) numpy
    posterior dict (as returned by run_nuts_multigpu) and replays it in
    chunks of `batch_size` draws, moving results to CPU numpy immediately.
 
    Runs on CPU by default: the replay is cheap (no gradients, one forward
    pass per draw) but stacking every draw of bin-sized deterministics
    (n_draws x n_bins in float64) can exceed GPU memory.
    """
    device = torch.device(device)
    model = model.to(device)
    n_chains, n_draws = next(iter(posterior.values())).shape[:2]
    total = n_chains * n_draws
 
    # flatten to (chain*draw, ...) but keep on CPU; batches move to `device`
    flat = {
        k: torch.as_tensor(v, dtype=model.dtype).reshape(
            (total,) + v.shape[2:])
        for k, v in posterior.items()
    }
 
    chunks = {k: [] for k in sites}
    with torch.no_grad():
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            batch = {k: v[start:stop].to(device) for k, v in flat.items()}
            pred = Predictive(model, posterior_samples=batch,
                              return_sites=list(sites))
            for k, v in pred().items():
                chunks[k].append(v.detach().cpu().numpy())
            del batch
 
    out = {}
    for k, parts in chunks.items():
        v = np.concatenate(parts, axis=0)
        out[k] = v.reshape((n_chains, n_draws) + v.shape[1:])
    return out