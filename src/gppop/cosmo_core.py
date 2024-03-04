__author__="Anarya Ray <anarya.ray@ligo.org>; Ignacio Maga\~na Hernandez <imhernan@andrew.cmu.edu>; Siddharth Mohite <siddharth.mohite@ligo.org>"

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import warnings
import h5py
from tqdm import tqdm

import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC
from numpyro.infer import NUTS

import pymc as pm
import pytensor.tensor as at
import pytensor as ae
from pymc import sampling_jax
from pytensor.link.jax.dispatch import jax_funcify

import jax
from jax import jit, vmap
import jax.numpy as jnp
import jax.scipy as jsp

from astropy.cosmology import Planck15, FlatLambdaCDM
from astropy import constants, units as u

from jaxinterp2d import interp2d, CartesianGrid

from logbesselk.jax import log_bessel_k as log_k


log_k_vec = jax.jit(jax.vmap(log_k,(None,0),0))

jax.config.update("jax_enable_x64", True)

warnings.warn("Warning... gppop-cosmo is an experimental module. Needs to be debugged, tested, and further optimized before it can produce correct results")

seed = np.random.randint(1000)
key = jax.random.PRNGKey(1000)

zMax = 10
H0Planck = Planck15.H0.value
Om0Planck = Planck15.Om0

cosmo = FlatLambdaCDM(H0=H0Planck,Om0=Planck15.Om0)
speed_of_light = constants.c.to('km/s').value
zgrid = np.expm1(np.linspace(np.log(1), np.log(zMax+1), 5000))

rs = []
Om0grid = jnp.linspace(Om0Planck-0.15,Om0Planck+0.15,100)

for Om0 in tqdm(Om0grid):
    cosmo = FlatLambdaCDM(H0=H0Planck,Om0=Om0)
    rs.append(cosmo.comoving_distance(zgrid).to(u.Mpc).value)

zgrid = jnp.array(zgrid)
rs = jnp.asarray(rs)
rs = rs.reshape(len(Om0grid),len(zgrid))

@jit
def E(z,Om0=Om0Planck):
    return jnp.sqrt(Om0*(1+z)**3 + (1.0-Om0))


@jit
def r_of_z(z,H0,Om0=Om0Planck):
    return interp2d(Om0,z,Om0grid,zgrid,rs)*(H0Planck/H0)


@jit
def dL_of_z(z,H0,Om0=Om0Planck):
    return (1+z)*r_of_z(z,H0,Om0)


@jit
def z_of_dL(dL,H0,Om0=Om0Planck):
    return jnp.interp(dL,dL_of_z(zgrid,H0,Om0),zgrid)


@jit
def dV_of_z(z,H0,Om0=Om0Planck):
    return speed_of_light*r_of_z(z,H0,Om0)**2/(H0*E(z,Om0))


@jit
def ddL_of_z(z,dL,H0,Om0=Om0Planck):
    return dL/(1+z) + speed_of_light*(1+z)/(H0*E(z,Om0))


def bin_edges(mbins):
    m1 = mbins
    m2 = mbins
    edge_array = []
    for i in range(len(m1)-1):
        for j in range(len(m2)-1):
            if(m2[j] > m1[i]):
                continue
            edge_array.append([[m1[i],m2[j]],[m1[i+1],m2[j+1]]])
    return jnp.array(edge_array)


def generate_logM_bin_centers(mbins):
    log_m1 = np.log(np.asarray(mbins))
    log_m2 = np.log(np.asarray(mbins))
    nbin = len(log_m1) - 1
    logm1_bin_centres = np.asarray([0.5*(log_m1[i+1]+log_m1[i]) for i in range(nbin)])
    logm2_bin_centres = np.asarray([0.5*(log_m2[i+1]+log_m2[i]) for i in range(nbin)])
    l1,l2 = jnp.meshgrid(logm1_bin_centres,logm2_bin_centres)
    logM = np.concatenate((l1.reshape([nbin*nbin,1]),l2.reshape([nbin*nbin,1])),axis=1)
    logM_lower_tri = np.asarray([a for a in logM if a[1] <= a[0]])
    logM_lower_tri_sorted = np.asarray([logM_lower_tri[i] for i in np.argsort(logM_lower_tri[:,0],kind='mergesort')])
    return jnp.asarray(logM_lower_tri_sorted)


@jit
def compute_weight_single_ev(samples, mbins, H0=H0Planck, Om0=Om0Planck, kappa=3):
    m1d_samples = samples[:,0]
    m2d_samples = samples[:,1]
    d_samples = samples[:,2]
    
    nbins_m = len(mbins)-1
    nsamples = len(samples)
    sindices = jnp.arange(nsamples)
    weights = jnp.zeros((nsamples,nbins_m,nbins_m))
    
    z_samples = z_of_dL(d_samples,H0=H0,Om0=Om0)
    m1s_samples = m1d_samples/(1+z_samples)
    m2s_samples = m2d_samples/(1+z_samples)

    pz_pop = dV_of_z(z_samples,H0=H0,Om0=Om0)*(1+z_samples)**(kappa-1) #uniform in comoving-volume
    p_pop = pz_pop/m1s_samples/m2s_samples
  
    ddL_dz = ddL_of_z(z_samples,d_samples,H0=H0,Om0=Om0)
    jac = (1+z_samples)**2 *  ddL_dz
    
    weight = p_pop/d_samples**2/jac
    
    m1_indices = jnp.clip(jnp.searchsorted(mbins, m1s_samples, side='right')- 1, a_min=0, a_max=nbins_m - 1)
    m2_indices = jnp.clip(jnp.searchsorted(mbins, m2s_samples, side='right')- 1, a_min=0, a_max=nbins_m - 1)
 
    weight_means = jnp.sum(weights.at[sindices,m1_indices,m2_indices].set(weight),axis=0)/nsamples    
    weight_vars = jnp.sum(weights.at[sindices,m1_indices,m2_indices].set(weight**2),axis=0)/nsamples**2 - weight_means**2/nsamples

    return [weight_means[jnp.tril_indices(len(weight_means))], jnp.sqrt(weight_vars[jnp.tril_indices(len(weight_means))])]


@jit
def VT_numerical(det_samples, p_draw, Ndraw, mbins, H0=H0Planck, Om0=Om0Planck, kappa=3, Tobs=1, mixture_weights=1.0):
    m1d_samples = det_samples[:,0]
    m2d_samples = det_samples[:,1]
    d_samples = det_samples[:,2]
    
    nbins_m = len(mbins)-1
    nsamples = len(m1d_samples)
    sindices = jnp.arange(nsamples)
    vts = jnp.zeros((nsamples,nbins_m,nbins_m))
    
    z_samples = z_of_dL(d_samples,H0=H0,Om0=Om0)
    m1s_samples = m1d_samples/(1+z_samples)
    m2s_samples = m2d_samples/(1+z_samples)
    
    pz_pop = Tobs*dV_of_z(z_samples,H0=H0,Om0=Om0)*(1+z_samples)**(kappa-1)
    p_pop = pz_pop/m1s_samples/m2s_samples
    
    ddL_dz = ddL_of_z(z_samples,d_samples,H0=H0,Om0=Om0)
    jac = (1+z_samples)**2 * ddL_dz

    weight = mixture_weights*p_pop/p_draw/jac
    
    m1_indices = jnp.clip(jnp.searchsorted(mbins, m1s_samples, side='right')- 1, a_min=0, a_max=nbins_m - 1)
    m2_indices = jnp.clip(jnp.searchsorted(mbins, m2s_samples, side='right')- 1, a_min=0, a_max=nbins_m - 1)
 
    vt_means = jnp.sum(vts.at[sindices,m1_indices,m2_indices].set(weight),axis=0)/(Ndraw)
    vt_vars = jnp.sum(vts.at[sindices,m1_indices,m2_indices].set(weight**2),axis=0)/(Ndraw**2) - vt_means**2/Ndraw
    
    return vt_means[jnp.tril_indices(nbins_m)], jnp.sqrt(vt_vars)[jnp.tril_indices(nbins_m)]


def compute_gp_inputs(mbins):
    logm_bin_centers = generate_logM_bin_centers(mbins)
    dist_array = jnp.zeros(int(len(logm_bin_centers)*(len(logm_bin_centers)+1)/2))

    k=0
    for i in range(len(logm_bin_centers)):
        for j in range(i+1):
            dist_array=dist_array.at[k].set(jnp.linalg.norm(logm_bin_centers[i]-logm_bin_centers[j]))
            k+=1

    scale_min = jnp.log(jnp.min(dist_array[dist_array!=0.]))
    scale_max = jnp.log(jnp.max(dist_array))

    scale_mean = 0.5*(scale_min + scale_max) # chosen to give coverage over the bin-grid
    scale_sd = (scale_max - scale_min)/4

    return scale_mean, scale_sd, logm_bin_centers


def make_gp_spectral_siren_model_pymc(samples, det_samples, p_draw, Ndraw, mbins, Tobs, mu_dim=None, H0min=20, H0max=140, kappamin=-10, kappamax=10):
    
    samples = np.asarray(samples)
    det_samples = np.asarray(det_samples)
    p_draw = np.asarray(p_draw)
    mbins = np.asarray(mbins)
    
    scale_mean, scale_sd, logm_bin_centers = compute_gp_inputs(mbins)
    scale_mean, scale_sd, logm_bin_centers = np.asarray(scale_mean), np.asarray(scale_sd), np.asarray(logm_bin_centers)
    
    mu_dim = len(logm_bin_centers) if mu_dim is None else 1.
    
    hmin = H0min/100
    hmax = H0max/100
    
    H0grid = jnp.linspace(H0min,H0max,50)
    kappagrid = jnp.linspace(kappamin,kappamax,50)
    
    nbins_m = int(len(mbins)*(len(mbins)-1)/2)
    bingrid = jnp.arange(nbins_m)
    
    event_grid = jnp.arange(len(samples))
    
    VT_means = []
    VT_sigmas = []
    for k in tqdm(range(len(H0grid))):
        for j in range(len(kappagrid)):
            VT_mean, VT_sigma = VT_numerical(det_samples, p_draw, Ndraw, mbins, H0=H0grid[k], Om0=Om0Planck, kappa=kappagrid[j], Tobs=Tobs, mixture_weights=1.0)
            VT_means.append(VT_mean)
            VT_sigmas.append(VT_sigma)

    VT_means = jnp.array(VT_means).reshape(len(H0grid),len(kappagrid),len(bingrid))
    VT_sigmas = jnp.array(VT_sigmas).reshape(len(H0grid),len(kappagrid),len(bingrid))

    limits_VT = [(H0grid[0], H0grid[-1]), (kappagrid[0], kappagrid[-1]), (bingrid[0], bingrid[-1])]

    VT_means_grid = CartesianGrid(limits_VT, VT_means, mode='nearest')
    VT_sigmas_grid = CartesianGrid(limits_VT, VT_sigmas, mode='nearest')

    def VT_means_(H0,kappa,mbin):
        return VT_means_grid(H0,kappa,mbin)

    def VT_sigmas_(H0,kappa,mbin):
        return VT_sigmas_grid(H0,kappa,mbin)

    VT_means_H0kappa = jit(vmap(VT_means_, in_axes=(None,None,0), out_axes=0))
    VT_sigmas_H0kappa = jit(vmap(VT_sigmas_, in_axes=(None,None,0), out_axes=0))

    def VT_numerical_grid(H0,kappa,bingrid):
        return VT_means_H0kappa(H0,kappa,bingrid), VT_sigmas_H0kappa(H0,kappa,bingrid)

    weights_means = []
    weights_sigmas = []

    for i in tqdm(range(len(event_grid))):
        for k in range(len(H0grid)):
            for j in range(len(kappagrid)):
                weight_mean, weight_sigma = compute_weight_single_ev(samples[i], mbins, H0=H0grid[k], Om0=Om0Planck, kappa=kappagrid[j])
                weights_means.append(weight_mean)
                weights_sigmas.append(weight_sigma)

    weights_means = jnp.array(weights_means).reshape(len(event_grid),len(H0grid),len(kappagrid),len(bingrid))
    weights_sigmas = jnp.array(weights_sigmas).reshape(len(event_grid),len(H0grid),len(kappagrid),len(bingrid))

    limits_weights = [(event_grid[0], event_grid[-1]), (H0grid[0], H0grid[-1]), (kappagrid[0], kappagrid[-1]), (bingrid[0], bingrid[-1])]

    weights_means_grid = CartesianGrid(limits_weights, weights_means, mode='nearest')
    weights_sigmas_grid = CartesianGrid(limits_weights, weights_sigmas, mode='nearest')

    def weights_means_(event,H0,kappa,mbin):
        return weights_means_grid(event,H0,kappa,mbin)

    def weights_sigmas_(event,H0,kappa,mbin):
        return weights_sigmas_grid(event,H0,kappa,mbin)

    weights_means_H0kappa = jit(vmap(weights_means_, in_axes=(None,None,None,0), out_axes=0))
    weights_means_eventH0kappa = jit(vmap(weights_means_H0kappa, in_axes=(0,None,None,None), out_axes=0))

    weights_sigmas_H0kappa = jit(vmap(weights_sigmas_, in_axes=(None,None,None,0), out_axes=0))
    weights_sigmas_eventH0kappa = jit(vmap(weights_sigmas_H0kappa, in_axes=(0,None,None,None), out_axes=0))

    def weights_numerical_grid(event_grid,H0,kappa,bingrid):
        return [weights_means_eventH0kappa(event_grid,H0,kappa,bingrid), weights_sigmas_eventH0kappa(event_grid,H0,kappa,bingrid)]
    
    class ComputeWeightsOp(at.Op):
        itypes = [at.dvector, at.dscalar, at.dscalar, at.dvector]
        otypes = [at.dmatrix, at.dmatrix]

        def make_node(self, event_grid, H0, kappa, bingrid):
            event_grid = at.as_tensor_variable(event_grid)
            H0 = at.as_tensor_variable(H0)
            kappa = at.as_tensor_variable(kappa)
            bingrid = at.as_tensor_variable(bingrid)

            return ae.graph.basic.Apply(self, [event_grid, H0, kappa, bingrid], [at.dmatrix(),at.dmatrix()])

        def perform(self, node, inputs, outputs):
            event_grid, H0, kappa, bingrid = inputs

            out = weights_numerical_grid(event_grid, H0, kappa, bingrid)

            outputs[0][0] = np.asarray(out[0])
            outputs[1][0] = np.asarray(out[1])

        def grad(self, inputs, gradients):
            event_grid, H0, kappa, bingrid = inputs
            grad_event_grid = at.zeros_like(event_grid)
            grad_H0 = at.zeros_like(H0)
            grad_kappa = at.zeros_like(kappa)
            grad_bingrid = at.zeros_like(bingrid)
            return [grad_event_grid, grad_H0, grad_kappa, grad_bingrid]

    class ComputeVTsOp(at.Op):
        itypes = [at.dscalar, at.dscalar, at.dvector]
        otypes = [at.dvector, at.dvector]

        def make_node(self, H0, kappa, bingrid):
            H0 = at.as_tensor_variable(H0)
            kappa = at.as_tensor_variable(kappa)
            bingrid = at.as_tensor_variable(bingrid)

            return ae.graph.basic.Apply(self, [H0, kappa, bingrid], [at.dvector(), at.dvector()])

        def perform(self, node, inputs, outputs):
            H0, kappa, bingrid = inputs

            out = VT_numerical_grid(H0, kappa, bingrid)

            outputs[0][0] = np.asarray(out[0])
            outputs[1][0] = np.asarray(out[1])

        def grad(self, inputs, gradients):
            H0, kappa, bingrid = inputs
            grad_H0 = at.zeros_like(H0)
            grad_kappa = at.zeros_like(kappa)
            grad_bingrid = at.zeros_like(bingrid)
            return [ grad_H0, grad_kappa, grad_bingrid]

    compute_weights_op = ComputeWeightsOp()
    compute_VTs_op = ComputeVTsOp()
    
    @jax_funcify.register(ComputeWeightsOp)
    def jax_funcify_compute_weights_op(op,**kwargs):
        def compute_weights_op(event_grid, H0, kappa, bingrid):
              return weights_numerical_grid(event_grid, H0, kappa, bingrid)
        return compute_weights_op
    
    @jax_funcify.register(ComputeVTsOp)
    def jax_funcify_compute_weights_op(op,**kwargs):
        def compute_VTs_op(H0, kappa, bingrid):
              return VT_numerical_grid(H0, kappa, bingrid)
        return compute_VTs_op

    bingrid = np.asarray(bingrid)
    event_grid = np.asarray(event_grid)

    with pm.Model() as model:
        h = pm.Uniform('h', hmin, hmax)
        H0 = 100*h
        
        kappa = pm.Uniform('kappa', kappamin, kappamax)
        Om0 = pm.Deterministic('Om0', at.as_tensor_variable(Om0Planck))
        
        mu = pm.Normal('mu', mu=0, sigma=5, shape=mu_dim)
        sigma = pm.HalfNormal('sigma', sigma=1)
        length_scale = pm.Lognormal('length_scale', mu=scale_mean, sigma=scale_sd)
        
        covariance = sigma**2*pm.gp.cov.ExpQuad(input_dim=2, ls=length_scale)
        gp = pm.gp.Latent(cov_func=covariance)
        
        logn_corr = gp.prior('logn_corr', X=logm_bin_centers)
        logn_tot = pm.Deterministic('logn_tot', mu+logn_corr)
        n_corr = pm.Deterministic('n_corr', at.exp(logn_tot))
        
        [weights, weight_sigmas] = compute_weights_op(event_grid,H0,kappa,bingrid)
        [vts, vt_sigmas] = compute_VTs_op(H0,kappa,bingrid)
        
        N_F_exp = pm.Deterministic('N_F_exp', at.sum(n_corr*vts))
        
        log_l = pm.Potential('log_l', at.sum(at.log(at.dot(weights,n_corr))) - N_F_exp)
        
        return model


def sample_pymc(model, njobs=1, ndraw=1000, ntune=1000, target_accept=0.7):
    with model:
        trace = pm.sampling_jax.sample_numpyro_nuts(draws=ndraw,tune=ntune,chains=njobs,
                                                    target_accept=target_accept)
        return trace

