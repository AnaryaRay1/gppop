__author__="Anarya Ray <anarya.ray@ligo.org>; Ignacio Maga\~na Hernandez <imhernan@andrew.cmu.edu>; Siddharth Mohite <siddharth.mohite@ligo.org>"

import numpy as np
import scipy
import scipy.stats as ss
import matplotlib.pyplot as plt
from pylab import *
from functools import partial
import warnings
import h5py

import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC
from numpyro.infer import NUTS

import jax
import jax.numpy as jnp
import jax.scipy.stats as jss

jax.config.update("jax_enable_x64", True)


import pymc as pm
import pytensor.tensor as at
import pytensor as ae
from pymc import sampling_jax
from pytensor.link.jax.dispatch import jax_funcify

import arviz as az
from astropy.cosmology import Planck15, FlatLambdaCDM
from astropy import constants, units as u

from jaxinterp2d import interp2d, CartesianGrid
import tqdm
from jax import jit, vmap

jax.config.update("jax_enable_x64", True)

warnings.warn("Warning... gppop-cosmo is an experimental module. Needs to be debugged, tested, and further optimized before it can produce correct results")

seed = np.random.randint(1000)
key = jax.random.PRNGKey(1000)


zMax = 5
H0Planck = Planck15.H0.value
Om0Planck = Planck15.Om0

cosmo = FlatLambdaCDM(H0=H0Planck,Om0=Planck15.Om0)
speed_of_light = constants.c.to('km/s').value
zgrid = np.expm1(np.linspace(np.log(1), np.log(zMax+1), 5000))

rs = []

Om0grid = jnp.linspace(Om0Planck-0.1,Om0Planck+0.1,50)

for Om0 in tqdm.tqdm(Om0grid):
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
            if(m2[j]>m1[i]):
                continue
            edge_array.append([[m1[i],m2[j]],[m1[i+1],m2[j+1]]])
    return jnp.array(edge_array)


def generate_logM_bin_centers(mbins):
        
        log_m1 = np.log(np.asarray(mbins))
        log_m2 = np.log(np.asarray(mbins))
        nbin = len(log_m1) - 1
        logm1_bin_centres = np.asarray([0.5*(log_m1[i+1]+log_m1[i])for i in range(nbin)])
        logm2_bin_centres = np.asarray([0.5*(log_m2[i+1]+log_m2[i])for i in range(nbin)])
        l1,l2 = jnp.meshgrid(logm1_bin_centres,logm2_bin_centres)
        logM = np.concatenate((l1.reshape([nbin*nbin,1]),l2.reshape([nbin*nbin,1])),axis=1)
        logM_lower_tri = np.asarray([a for a in logM if a[1]<=a[0]])
        logM_lower_tri_sorted = np.asarray([logM_lower_tri[i] for i in np.argsort(logM_lower_tri[:,0],kind='mergesort')])
        return jnp.asarray(logM_lower_tri_sorted)


@jit
def compute_weight_single_ev(samples,H=70.,Om0 = Om0Planck,mbins=10.,kappa=3):
    
    m1d_samples = samples[:,0]
    m2d_samples = samples[:,1]
    d_samples = samples[:,2]
    
    nbins_m = len(mbins)-1
    nsamples = len(samples)
    sindices = jnp.arange(nsamples)
    weight = jnp.zeros((nsamples,nbins_m,nbins_m))
    z_samples = z_of_dL(d_samples,H,Om0=Om0)
        
    m1_indices = jnp.clip(jnp.searchsorted(mbins, m1d_samples/(1+z_samples), side='right')- 1, a_min=0, a_max=nbins_m - 1)
    m2_indices = jnp.clip(jnp.searchsorted(mbins, m2d_samples/(1+z_samples), side='right')- 1, a_min=0, a_max=nbins_m - 1)
    pz_pop = dV_of_z(z_samples,H,Om0=Om0) *(1+z_samples)**(kappa-1) #uniform in comoving-volume
    ddL_dz = ddL_of_z(z_samples,d_samples,H,Om0=Om0) 
    pz_PE = (1+z_samples)**2 * d_samples**2 * ddL_dz # default PE prior - flat in det frame masses and dL**2 in distance
    pz_weight = pz_pop/pz_PE
 
    weight = jnp.sum(weight.at[sindices,m1_indices,m2_indices].set( pz_weight *(1+z_samples)**2/ (m1d_samples*m2d_samples)),axis=0)
    normalized_weight = weight/nsamples
    return normalized_weight[jnp.tril_indices(len(weight))]

def draw_thetas(N=10000):
    """Draw `N` random angular factors for the SNR.

    Theta is as defined in [Finn & Chernoff
    (1993)](https://ui.adsabs.harvard.edu/#abs/1993PhRvD..47.2198F/abstract).

    Author: Will Farr
    """

    cos_thetas = np.random.uniform(low=-1, high=1, size=N)
    cos_incs = np.random.uniform(low=-1, high=1, size=N)
    phis = np.random.uniform(low=0, high=2*np.pi, size=N)
    zetas = np.random.uniform(low=0, high=2*np.pi, size=N)

    Fps = 0.5*np.cos(2*zetas)*(1 + np.square(cos_thetas))*np.cos(2*phis) - np.sin(2*zetas)*cos_thetas*np.sin(2*phis)
    Fxs = 0.5*np.sin(2*zetas)*(1 + np.square(cos_thetas))*np.cos(2*phis) + np.cos(2*zetas)*cos_thetas*np.sin(2*phis)

    return np.sqrt(0.25*np.square(Fps)*np.square(1 + np.square(cos_incs)) + np.square(Fxs)*np.square(cos_incs))

    return jnp.sqrt(0.25*jnp.square(Fps)*jnp.square(1 + jnp.square(cos_incs)) + jnp.square(Fxs)*jnp.square(cos_incs))

thetas = jnp.array(draw_thetas(10000))
rns = jnp.array(np.random.randn(10000))

@jit
def Pdet_msdet(m1det, m2det, dL, ms, rhos, ref_dist_Gpc = 1.0, dist_unit = u.Mpc, rand_noise = False, thresh=8.0):
    """
    m1det: detector-frame mass 1
    m2det: detector-frame mass 2
    dL: luminosity distance in dist_unit
    osnr_interp: a 2-d spline object constructed by interpolate_optimal_snr_grid
    ref_dist_Gpc: the reference distance at which osnr_interp was calculated
    rand_noise: add random N(0,1)
    """
    dL_Gpc = dL*((1.*dist_unit).to(u.Gpc)).value
    
    if rand_noise:
        noise = rns
    else:
        noise = 0.0
    #Implement Eq. (i)
    return jnp.mean((interp2d(m1det,m2det,ms,ms,rhos)*ref_dist_Gpc/dL_Gpc)[:,:,:,jnp.newaxis]*(thetas[jnp.newaxis,jnp.newaxis,jnp.newaxis,:])+rns[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]>thresh,axis=-1)


@jit
def VT_bin(edges,H=70.,Om0 = Om0Planck, ms=10.0,rhos=15.0,T=1,ngrid=10,kappa=3):
    m1_low,m1_high,m2_low,m2_high,z_low,z_high = edges[0,0],edges[1,0],edges[0,1],edges[1,1],edges[0,2],edges[1,2]
    z_grid = jnp.linspace(z_low,z_high,ngrid)
    m1_grid = jnp.linspace(m1_low,m1_high,ngrid)
    m2_grid = jnp.linspace(m2_low,m2_high,ngrid)
    
    m1s,m2s,zs = jnp.meshgrid(m1_grid,m2_grid,z_grid,indexing='ij')
    v = dV_of_z(zs,H,Om0=Om0)*(1+zs)**(kappa-1)*((u.Mpc**3).to(u.Gpc**3))
    ds = dL_of_z(zs,H,Om0=Om0)
    VT = Pdet_msdet(m1s*(1+zs),m2s*(1+zs),ds,ms,rhos)*v*T/(m1s*m2s)
    
    return jnp.trapz(jnp.trapz(jnp.trapz(VT,z_grid,axis=2),m2_grid,axis=1),m1_grid)/(1.+(m1_low==m2_low))

@jit
def jax_compute_weights_vts_op(Samples, H, Om0, mbins, edges, ms, rhos, T, kappa):
    compute_weight_single_ev_partial = partial(compute_weight_single_ev, H=H, Om0=Om0, mbins=mbins, kappa=kappa)
    VT_bin_partial = partial(VT_bin, H=H, Om0=Om0, ms=ms, rhos=rhos, T=T, kappa=kappa)
    
    # Use vmap for vectorizing computations
    compute_weight_results = vmap(compute_weight_single_ev_partial)(Samples)
    VT_bin_results = vmap(VT_bin_partial)(edges)
    
    return [compute_weight_results, VT_bin_results]

class ComputeWeightsVtsOp(at.Op):
    itypes = [at.dtensor3, at.dscalar, at.dscalar, at.dvector, at.dtensor3, at.dvector, at.dmatrix, at.dscalar, at.dscalar]
    otypes = [at.dmatrix,at.dvector]

    def make_node(self, Samples,H,Om0,mbins,edges,ms,rhos,T,kappa):
        Samples = at.as_tensor_variable(Samples)
        H = at.as_tensor_variable(H)
        mbins = at.as_tensor_variable(mbins)
        edges = at.as_tensor_variable(edges)
        ms = at.as_tensor_variable(ms)
        rhos = at.as_tensor_variable(rhos)
        T = at.as_tensor_variable(T)
        kappa = at.as_tensor_variable(kappa)
        Om0 = at.as_tensor_variable(Om0)
        return ae.graph.basic.Apply(self, [Samples,H,Om0,mbins,edges,ms,rhos,T,kappa], [at.dmatrix(),at.dvector()])

    def perform(self, node, inputs, outputs):
        Samples,H,Om0,mbins,edges,ms,rhos,T,kappa = inputs
        out = jax_compute_weights_vts_op(Samples,H,Om0,mbins,edges,ms,rhos,T,kappa)
        outputs[0][0] = np.asarray(out[0])
        outputs[1][0] = np.asarray(out[1])
        
    def grad(self, inputs, gradients):
        Samples,H,Om0,mbins,edges,ms,rhos,T,kappa = inputs
        grad_Samples = at.zeros_like(Samples)
        grad_H = at.zeros_like(H)
        grad_Om0 = at.zeros_like(Om0)
        grad_mbins = at.zeros_like(mbins)
        grad_edges = at.zeros_like(edges)
        grad_ms = at.zeros_like(ms)
        grad_rhos = at.zeros_like(rhos)
        grad_T = at.zeros_like(T)
        grad_kappa = at.zeros_like(kappa)
        return [grad_Samples,grad_H,grad_Om0,grad_mbins,grad_edges,grad_ms,grad_rhos,grad_T,grad_kappa]

compute_weights_vts_op = ComputeWeightsVtsOp()

@jax_funcify.register(ComputeWeightsVtsOp)
def jax_funcify_compute_weights_vts_op(op,**kwargs):
    def compute_weights_vts_op(Samples,H,Om0,mbins,edges,ms,rhos,T,kappa):
      return jax_compute_weights_vts_op(Samples,H,Om0,mbins,edges,ms,rhos,T,kappa)
    return compute_weights_vts_op

def compute_gp_inputs(mbins):
    logm_bin_centers = generate_logM_bin_centers(mbins)
    k=0
    dist_array = jnp.zeros(int(len(logm_bin_centers)*(len(logm_bin_centers)+1)/2))
    for i in range(len(logm_bin_centers)):
        for j in range(i+1):
            dist_array=dist_array.at[k].set(jnp.linalg.norm(logm_bin_centers[i]-logm_bin_centers[j]))
            k+=1


    scale_min = jnp.log(jnp.min(dist_array[dist_array!=0.]))
    scale_max = jnp.log(jnp.max(dist_array))
    scale_mean = 0.5*(scale_min + scale_max) # chosen to give coverage over the bin-grid
    scale_sd = (scale_max - scale_min)/4
    
    return scale_mean,scale_sd, logm_bin_centers


def make_gp_spectral_siren_model_pymc(Samples, mbins, ms, rhos, T,z_low,z_high):
    
    scale_mean,scale_sd, logm_bin_centers = compute_gp_inputs(mbins)
    scale_mean,scale_sd, logm_bin_centers = jnp.asarray(scale_mean),jnp.asarray(scale_sd),jnp.asarray( logm_bin_centers)
    Samples_var = ae.shared(Samples, borrow=True)
    mbins_var = ae.shared(mbins, borrow=True)
    ms_var = ae.shared(ms, borrow=True)
    rhos_var = ae.shared(rhos, borrow=True)
    T_var = ae.shared(T, borrow=True)
    edges = bin_edges(mbins)
    edges = np.array([[[e[0,0],e[0,1],z_low],[e[1,0],e[1,1],z_high]] for e in edges]).astype('float32')
    edges_var = ae.shared(edges,borrow=True)
    
    with pm.Model() as model:
        H = pm.Uniform('H_0',60,80)
        kappa= pm.Deterministic('kappa', at.as_tensor_variable(3.0)) # pm.Uniform('kappa',0.,5.)
        Om0 = pm.Deterministic('Om0',at.as_tensor_variable(Om0Planck)) # pm.Uniform('Om0',0.,1.)
        mu = pm.Normal('mu',mu=0,sigma=5)
        sigma = pm.HalfNormal('sigma',sigma=1)
        length_scale = pm.Lognormal('length_scale',mu=scale_mean,sigma=scale_sd)
        covariance = sigma**2*pm.gp.cov.ExpQuad(input_dim=2,ls=length_scale)
        gp = pm.gp.Latent(cov_func=covariance)
        logn_corr = gp.prior('logn_corr',X=logm_bin_centers)
        logn_tot = pm.Deterministic('logn_tot', mu+logn_corr)
        n_corr = pm.Deterministic('n_corr',at.exp(logn_tot))
        
        
        [weights,vts] = compute_weights_vts_op(Samples_var,H,Om0,mbins_var,edges_var,ms_var,rhos_var,T_var,kappa)
        N_F_exp = pm.Deterministic('N_F_exp',at.sum(n_corr*vts))
        
        log_l = pm.Potential('log_l',at.sum(at.log(at.dot(weights,n_corr)))-N_F_exp)
        
        
        return model

def sample_pymc(model,njobs=1,ndraw=1000,ntune=1000,target_accept = 0.9):
    with model:
        trace = pm.sampling_jax.sample_numpyro_nuts(draws=ndraw,tune=ntune,
                      chains=njobs,
                      target_accept=target_accept)
        
        return trace

@jit
def kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    deltaXsq = jnp.sum(jnp.power((X[jnp.newaxis,:,:]-X[:,jnp.newaxis,:])/length, 2.0),axis=-1)
    k = var * jnp.exp(-0.5 * deltaXsq)
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k

        
def gp_spectral_siren_model_numpyro(Samples, scale_mean,scale_sd, logm_bin_centers, edges,ms, rhos, T, mbins):
    
    H = numpyro.sample("H0", dist.Uniform(50, 80))
    kappa= numpyro.deterministic('kappa', 3.0) # numpyro.sample("kappa", dist.Unifogrm(0.,5.))
    Om0 = numpyro.deterministic('Om0', Om0Planck) # numpyro.sample("Om0", dist.Unifogrm(0.,1.))
    mu = numpyro.sample('mu', dist.Normal(0,5))
    sigma = numpyro.sample('sigma', dist.HalfNormal(1))
    length_scale = numpyro.sample('length_scale', dist.LogNormal(scale_mean,scale_sd))
    
    cov = kernel(logm_bin_centers, logm_bin_centers, jnp.power(sigma,2.0), length_scale, 0.)
    
    logn_tot = numpyro.sample('logn_tot',dist.MultivariateNormal(loc=mu, covariance_matrix=cov))
   
    n_corr = numpyro.deterministic('n_corr',jnp.exp(logn_tot))


    [weights,vts] = jax_compute_weights_vts_op(Samples,H,Om0,mbins,edges,ms,rhos,T,kappa)
    
    N_F_exp = numpyro.deterministic('N_F_exp',jnp.sum(n_corr*vts))

    numpyro.factor('log_likelihood',jnp.sum(jnp.log(jnp.dot(weights,n_corr)))-N_F_exp)
    

def sample_numpyro(Samples, mbins, ms, osnrs, Tobs,z_low,z_high,thinning=100,
        num_warmup=10,
        num_samples=100,
        num_chains=1):
    edges = bin_edges(mbins)
    edges = jnp.array([[[e[0,0],e[0,1],z_low],[e[1,0],e[1,1],z_high]] for e in edges]).astype('float32')
    
    scale_mean,scale_sd, logm_bin_centers = compute_gp_inputs(mbins)
    scale_mean,scale_sd, logm_bin_centers = jnp.asarray(scale_mean),jnp.asarray(scale_sd),jnp.asarray( logm_bin_centers)

    
    RNG = jax.random.PRNGKey(0)
    MCMC_RNG, PRIOR_RNG, _RNG = jax.random.split(RNG, num=3)
    kernel = NUTS(gp_spectral_siren_model_numpyro)
    mcmc = MCMC(
        kernel,
        thinning=thinning,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )

    mcmc.run(PRIOR_RNG,Samples, scale_mean,scale_sd, logm_bin_centers,edges, ms, osnrs, Tobs,mbins)
    
    return mcmc.get_samples()

        
def gp_fixed_cosmo_model_numpyro(weights,vts, scale_mean,scale_sd, logm_bin_centers):
    mu = numpyro.sample('mu', dist.Normal(0,5))
    sigma = numpyro.sample('sigma', dist.HalfNormal(1))
    length_scale = numpyro.sample('length_scale', dist.LogNormal(scale_mean,scale_sd))
    
    cov = kernel(logm_bin_centers, logm_bin_centers, jnp.power(sigma,2.0), length_scale, 0.)
    
    
    logn_tot = numpyro.sample('logn_tot',dist.MultivariateNormal(loc=mu, covariance_matrix=cov))
   
    n_corr = numpyro.deterministic('n_corr',jnp.exp(logn_tot))
    
    N_F_exp = numpyro.deterministic('N_F_exp',jnp.sum(n_corr*vts))

    numpyro.factor('log_likelihood',jnp.sum(jnp.log(jnp.dot(weights,n_corr)))-N_F_exp)

def sample_numpyro_fixed_cosmo(Samples, mbins, ms, osnrs, Tobs,z_low,z_high,thinning=100,
        num_warmup=10,
        num_samples=100,
        num_chains=1):
    edges = bin_edges(mbins)
    edges = jnp.array([[[e[0,0],e[0,1],z_low],[e[1,0],e[1,1],z_high]] for e in edges]).astype('float32')
    
    scale_mean,scale_sd, logm_bin_centers = compute_gp_inputs(mbins)
    scale_mean,scale_sd, logm_bin_centers = jnp.asarray(scale_mean),jnp.asarray(scale_sd),jnp.asarray( logm_bin_centers)

    [weights,vts] = jax_compute_weights_vts_op(Samples,Planck15.H0.value,Planck15.Om0,mbins,edges,ms,osnrs,Tobs,3.0)
    
    RNG = jax.random.PRNGKey(0)
    MCMC_RNG, PRIOR_RNG, _RNG = jax.random.split(RNG, num=3)
    kernel = NUTS( gp_fixed_cosmo_model_numpyro)
    mcmc = MCMC(
        kernel,
        thinning=thinning,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )

    mcmc.run(PRIOR_RNG, weights,vts, scale_mean,scale_sd, logm_bin_centers)
    
    return mcmc.get_samples()
