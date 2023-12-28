__author__="Anarya Ray <anarya.ray@ligo.org>; Ignacio Maga\~na Hernandez <imhernan@andrew.cmu.edu>; Siddharth Mohite <siddharth.mohite@ligo.org>"

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import warnings
import h5py

import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC
from numpyro.infer import NUTS

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from logbesselk.jax import log_bessel_k as log_k

log_k_vec = jax.jit(jax.vmap(log_k,(None,0),0))

jax.config.update("jax_enable_x64", True)

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
def compute_weight_single_ev(samples,pe_prior,H=70.,Om0 = Om0Planck,mbins=10.,kappa=3):
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
    pe_prior = (1+z_samples)**2 if pe_prior is None else pe_prior
    pz_PE = pe_prior* d_samples**2 * ddL_dz  # default PE prior - flat in det frame masses and dL**2 in distance
    pz_weight = pz_pop/pz_PE
 
    weight_mean = jnp.sum(weight.at[sindices,m1_indices,m2_indices].set( pz_weight *(1+z_samples)**2/ (m1d_samples*m2d_samples)),axis=0)
    weight_mean = weight_mean/nsamples
    weight_var = jnp.sum(weight.at[sindices,m1_indices,m2_indices].set( (pz_weight *(1+z_samples)**2/ (m1d_samples*m2d_samples))**2),axis=0)/nsamples**2 - weight_mean**2/nsamples
    return [weight_mean[jnp.tril_indices(len(weight_mean))], jnp.sqrt(weight_var[jnp.tril_indices(len(weight_mean))])]

@jit
def log_prob_spin(sx,sy,sz,m):
    s_max = jnp.where(m<2.5,0.4,0.99)
    return jnp.log(1./(4*np.pi*s_max*(sx**2 + sy**2 + sz**2)))

@jit
def reweight_pinjection(tril_weights):
    return jnp.where((tril_weights!=0),jnp.exp(tril_weights),0)

@jit
def VT_numerical(det_samples, p_draw, mbins, Ndraw, mixture_weights=1.0, H=70., Om0 = Om0Planck, T=1, kappa=3,include_spins = True):
    m1d_samples = det_samples[:,0]
    m2d_samples = det_samples[:,1]
    d_samples = det_samples[:,2]
    s1x, s1y, s1z, s2x, s2y, s2z = det_samples[:,3],det_samples[:,4],det_samples[:,5],det_samples[:,6],det_samples[:,7],det_samples[:,8]
    
    nbins_m = len(mbins)-1
    nsamples = len(m1d_samples)
    sindices = jnp.arange(nsamples)
    vts = jnp.zeros((nsamples,nbins_m,nbins_m))
    
    z_samples = z_of_dL(d_samples,H,Om0=Om0)
    m1s_samples = m1d_samples/(1+z_samples)
    m2s_samples = m2d_samples/(1+z_samples)
    
    pz_pop = T*dV_of_z(z_samples,H,Om0=Om0)*((1.*u.Mpc**3).to(u.Gpc**3).value) *(1+z_samples)**(kappa-1)
    p_pop = pz_pop/m1s_samples/m2s_samples
    
    ddL_dz = ddL_of_z(z_samples,d_samples,H,Om0=Om0)
    jac = (1+z_samples)**2 *  ddL_dz
    
    p_s1s2 = jnp.power(reweight_pinjection(log_prob_spin(s1x,s1y,s1z,m1s_samples)+log_prob_spin(s2x,s2y,s2z,m2s_samples)),int(include_spins))
    
    weight = mixture_weights*p_s1s2*p_pop/p_draw/jac
    
    m1_indices = jnp.clip(jnp.searchsorted(mbins, m1s_samples, side='right')- 1, a_min=0, a_max=nbins_m - 1)
    m2_indices = jnp.clip(jnp.searchsorted(mbins, m2s_samples, side='right')- 1, a_min=0, a_max=nbins_m - 1)
 
    vt_means = jnp.sum(vts.at[sindices,m1_indices,m2_indices].set(weight),axis=0)/(Ndraw)
    
    vt_vars = jnp.sum(vts.at[sindices,m1_indices,m2_indices].set(weight**2),axis=0)/(Ndraw**2)- vt_means**2/Ndraw
    
    return vt_means[jnp.tril_indices(nbins_m)], jnp.sqrt(vt_vars)[jnp.tril_indices(nbins_m)]


@jit
def jax_compute_weights_vts_op(Samples,pe_prior,det_samples,pdraw, Ndraw, H, Om0, mbins, T, kappa):
    compute_weight_single_ev_partial = partial(compute_weight_single_ev, H=H, Om0=Om0, mbins=mbins, kappa=kappa)
    
    
    # Use vmap for vectorizing computations
    compute_weight_results = jnp.asarray(vmap(compute_weight_single_ev_partial)(Samples,pe_prior))
    #VT_bin_results = vmap(VT_bin_partial)(edges)
    VT_numerical_means,VT_numerical_sigmas = VT_numerical(det_samples, pdraw, mbins, Ndraw, H=H, Om0 = Om0, T=T, kappa=kappa)
    
    return [compute_weight_results[0,:,:],compute_weight_results[1,:,:], VT_numerical_means, VT_numerical_sigmas]# VT_bin_results]

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

@jit
def kernel_RBF(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    deltaXsq = jnp.sum(jnp.power((X[jnp.newaxis,:,:]-X[:,jnp.newaxis,:])/length, 2.0),axis=-1)
    k = var * jnp.exp(-0.5 * deltaXsq)
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k

@jit
def kernel_matern(X, Z, alpha, var, length, noise, jitter=1.0e-6, include_noise=True):
    deltaX = jnp.sqrt(2.*alpha*(jnp.sum(jnp.power((X[jnp.newaxis,:,:]-X[:,jnp.newaxis,:])/length, 2.0),axis=-1)+1e-12))
    k = var * (1./(2**(alpha-1.)*jsp.special.gamma(alpha)))* jnp.power(deltaX,alpha)*jnp.exp(log_k_vec(alpha,deltaX.flatten()).reshape(len(deltaX),len(deltaX)))
    
    k = k.at[jnp.diag_indices(len(deltaX),2)].set(var)
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    
    return k

@jit
def kernel_matern_3_by_2(X, Z,var, length, noise, jitter=1.0e-6, include_noise=True):
    m = 0
    alpha = m+1./2.
    deltaX = jnp.sqrt(2.*alpha*(jnp.sum(jnp.power((X[jnp.newaxis,:,:]-X[:,jnp.newaxis,:])/length, 2.0),axis=-1)+1e-12))
    
    
    k = var * jnp.exp(-deltaX) * (1.+deltaX)
    
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    
    return k


@jit
def kernel_matern_int(X, Z,ms, var, length, noise, jitter=1.0e-6, include_noise=True):
    m = len(ms)
    alpha = m+1./2.
    deltaX = jnp.sqrt(2.*alpha*(jnp.sum(jnp.power((X[jnp.newaxis,:,:]-X[:,jnp.newaxis,:])/length, 2.0),axis=-1)+1e-12))
    
    ar=jnp.arange(m)+1
    i = ar[:,jnp.newaxis,jnp.newaxis]
    coeffs = jsp.special.gamma(m+1)/jsp.special.gamma(m-1)/jsp.special.gamma(i)
    
    power = m-i
    k = var * jnp.exp(-deltaX) *(jsp.special.gamma(m+1)/jsp.special.gamma(2*m+1))*jnp.sum(coeffs*jnp.power(2.*deltaX[jnp.newaxis,:,:],power),axis=0)
    
    
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    
    return k

def gp_spectral_siren_model_numpyro(Samples,pe_prior,det_samples,pdraw, Ndraw, scale_mean,scale_sd, logm_bin_centers, T, mbins,kappa_true,sigma_sd,mu_dim):
    H = numpyro.sample("H0", dist.Uniform(58, 72))
    Om0 = numpyro.deterministic('Om0', Om0Planck) # numpyro.sample("Om0", dist.Unifogrm(0.,1.))
    mu = numpyro.sample('mu', dist.Normal(0,5),sample_shape=(mu_dim,))
    sigma = numpyro.sample('sigma', dist.HalfNormal(sigma_sd))
    length_scale = numpyro.sample('length_scale', dist.LogNormal(scale_mean,scale_sd))
    
    cov = kernel_RBF(logm_bin_centers, logm_bin_centers, jnp.power(sigma,2.0), length_scale, 0.)
    
    logn_tot = numpyro.sample('logn_tot',dist.MultivariateNormal(loc=mu, covariance_matrix=cov))
    n_corr = numpyro.deterministic('n_corr',jnp.exp(logn_tot))

    kappa = numpyro.sample('kappa',dist.Uniform(kappa_true-3,kappa_true+3))
    [weights,weight_sigmas,vts,vt_sigmas] = jax_compute_weights_vts_op(Samples,pe_prior,det_samples,pdraw, Ndraw, H, Om0, mbins, T, kappa)
    
    N_F_exp = numpyro.deterministic('N_F_exp',jnp.sum(n_corr*vts))
    numpyro.factor('log_likelihood',jnp.sum(jnp.log(jnp.dot(weights,n_corr)))-N_F_exp)

def sample_numpyro(Samples,pe_prior,det_samples,pdraw, Ndraw, mbins, Tobs, thinning=100,
        num_warmup=10,
        num_samples=100,
        num_chains=1,target_accept_prob=0.9,kappa=3.0,sigma_sd=5,mu_dim=None):
    
    scale_mean,scale_sd, logm_bin_centers = compute_gp_inputs(mbins)
    scale_mean,scale_sd, logm_bin_centers = jnp.asarray(scale_mean),jnp.asarray(scale_sd),jnp.asarray( logm_bin_centers)

    mu_dim = len(logm_bin_centers) if mu_dim is None else 1.
    RNG = jax.random.PRNGKey(0)
    MCMC_RNG, PRIOR_RNG, _RNG = jax.random.split(RNG, num=3)
    kernel = NUTS(gp_spectral_siren_model_numpyro, target_accept_prob= target_accept_prob)
    mcmc = MCMC(
        kernel,
        thinning=thinning,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )

    mcmc.run(PRIOR_RNG,Samples,pe_prior,det_samples,pdraw, Ndraw, scale_mean,scale_sd, logm_bin_centers, Tobs, mbins,kappa,sigma_sd,mu_dim)
    
    return mcmc.get_samples()

        
def gp_fixed_cosmo_model_numpyro(weights,vts, scale_mean,scale_sd, logm_bin_centers):
    mu = numpyro.sample('mu', dist.Normal(0,5))
    sigma = numpyro.sample('sigma', dist.HalfNormal(1))
    length_scale = numpyro.sample('length_scale', dist.LogNormal(scale_mean,scale_sd))
    
    cov = kernel_RBF(logm_bin_centers, logm_bin_centers, jnp.power(sigma,2.0), length_scale, 0.)
    logn_tot = numpyro.sample('logn_tot',dist.MultivariateNormal(loc=mu, covariance_matrix=cov))
    
    n_corr = numpyro.deterministic('n_corr',jnp.exp(logn_tot))
    N_F_exp = numpyro.deterministic('N_F_exp',jnp.sum(n_corr*vts))
    numpyro.factor('log_likelihood',jnp.sum(jnp.log(jnp.dot(weights,n_corr)))-N_F_exp)

def sample_numpyro_fixed_cosmo(Samples,pe_prior,det_samples,pdraw, Ndraw, mbins, Tobs, thinning=100,
        num_warmup=10,
        num_samples=100,
        num_chains=1,target_accept_prob=0.9,kappa=3.0):
    
    scale_mean,scale_sd, logm_bin_centers = compute_gp_inputs(mbins)
    scale_mean,scale_sd, logm_bin_centers = jnp.asarray(scale_mean),jnp.asarray(scale_sd),jnp.asarray( logm_bin_centers)

    [weights,_,vts,_] = jax_compute_weights_vts_op(Samples,pe_prior,det_samples,pdraw, Ndraw, H0Planck, Om0Planck, mbins, Tobs, kappa)
    
    RNG = jax.random.PRNGKey(1000)
    MCMC_RNG, PRIOR_RNG, _RNG = jax.random.split(RNG, num=3)
   
    kernel = NUTS( gp_fixed_cosmo_model_numpyro,target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        kernel,
        thinning=thinning,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )

    mcmc.run(PRIOR_RNG, weights,vts, scale_mean,scale_sd, logm_bin_centers)
    
    return mcmc.get_samples()
