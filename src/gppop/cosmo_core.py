__author__="Anarya Ray <anarya.ray@ligo.org>; Ignacio Maga\~na Hernandez <imhernan@andrew.cmu.edu>; Siddharth Mohite <siddharth.mohite@ligo.org>"

import pymc as pm
import pytensor.tensor as at
import pytensor as ae
import numpy as np
import scipy
import scipy.stats as ss
import matplotlib.pyplot as plt
from pylab import *
from functools import partial

import h5py

import jax

import jax.numpy as jnp

import jax.scipy.stats as jss

from pymc import sampling_jax

from pytensor.link.jax.dispatch import jax_funcify

import arviz as az
from astropy.cosmology import Planck15, FlatLambdaCDM
from astropy import constants, units as u

from jaxinterp2d import interp2d, CartesianGrid
import tqdm
from jax import jit


seed = np.random.randint(1000)
key = jax.random.PRNGKey(seed)


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


#H_true = cosmology.Planck18.H0.value
#@jit
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

    
    #weight[sindices,m1_indices,m2_indices] = pz_weight *(1+z_samples)**2/ (m1d_samples*m2d_samples) 
    weight = jnp.sum(weight.at[sindices,m1_indices,m2_indices].set( pz_weight *(1+z_samples)**2/ (m1d_samples*m2d_samples)),axis=0)
    normalized_weight = weight/nsamples
    return normalized_weight[jnp.tril_indices(len(weight))]

@jit
def draw_thetas(N=10000):
    """Draw `N` random angular factors for the SNR.

    Theta is as defined in [Finn & Chernoff
    (1993)](https://ui.adsabs.harvard.edu/#abs/1993PhRvD..47.2198F/abstract).

    Author: Will Farr
    """

    cos_thetas = jax.random.uniform(key,minval=-1, maxval=1, shape = (N,))
    cos_incs = jax.random.uniform(key,minval=-1, maxval=1, shape=(N,))
    phis = jax.random.uniform(key,minval=0, maxval=2*np.pi, shape=(N,))
    zetas = jax.random.uniform(key,minval=0, maxval=2*np.pi, shape=(N,))

    Fps = 0.5*jnp.cos(2*zetas)*(1 + jnp.square(cos_thetas))*jnp.cos(2*phis) - jnp.sin(2*zetas)*cos_thetas*jnp.sin(2*phis)
    Fxs = 0.5*jnp.sin(2*zetas)*(1 + jnp.square(cos_thetas))*jnp.cos(2*phis) + jnp.cos(2*zetas)*cos_thetas*jnp.sin(2*phis)

    return jnp.sqrt(0.25*jnp.square(Fps)*jnp.square(1 + jnp.square(cos_incs)) + jnp.square(Fxs)*jnp.square(cos_incs))

thetas = draw_thetas()
rns = np.random.randn(10000)

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
    # if dL_Gpc == 0.0:
    #     return 1.0
    if rand_noise:
        noise = rns
    else:
        noise = 0.0
    #Implement Eq. (i)
    return jnp.mean((interp2d(m1det,m2det,ms,ms,rhos)*ref_dist_Gpc/dL_Gpc)[:,:,:,jnp.newaxis]*(thetas[jnp.newaxis,jnp.newaxis,jnp.newaxis,:])+rns[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]>thresh)


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
def jax_compute_weights_vts_op(Samples,H,Om0,mbins,edges,ms,rhos,T,kappa):
  
  return [jnp.array(list(jax.lax.map(partial(compute_weight_single_ev,H=H,Om0=Om0,mbins=mbins,kappa=kappa),Samples))),
          jnp.array(list(jax.lax.map(partial(VT_bin, H=H,Om0=Om0,ms=ms,rhos=rhos,T=T,kappa=kappa), edges)))]

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
        outputs[0][0] = np.array(out[0])
        outputs[1][0] = np.array(out[1])
        
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


def make_gp_spectral_siren_model(Samples, mbins, ms, rhos, T,z_low,z_high):
    
    scale_mean,scale_sd, logm_bin_centers = compute_gp_inputs(mbins)
    scale_mean,scale_sd, logm_bin_centers = np.asarray(scale_mean),np.asarray(scale_sd),np.asarray( logm_bin_centers)
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
        kappa= at.as_tensor_variable(3.0) # pm.Uniform('kappa',0.,5.)
        Om0 = at.as_tensor_variable(Om0Planck) # pm.Uniform('Om0',0.,1.)
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
        
        log_l = pm.Potential('log_l',at.reshape(at.sum(at.log(at.dot(weights,n_corr)))-N_F_exp,(1,)))
        
        
        return model

def sample(model,njobs=1,ndraw=1000,ntune=1000,target_accept = 0.9):
    with model:
        trace = pm.sampling_jax.sample_numpyro_nuts(draws=ndraw,tune=ntune,
                      chains=njobs,
                      target_accept=target_accept)
        
        return trace
        
