__author__="Anarya Ray <anarya.ray@ligo.org>; Ignacio Maga\~na Hernandez <imhernan@andrew.cmu.edu>; Siddharth Mohite <siddharth.mohite@ligo.org>"

import pymc as pm
import pytensor.tensor as at
import pytensor as ae
import numpy as np
import scipy
import scipy.stats as ss
from scipy.interpolate import RectBivariateSpline
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

import jax

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


@jit
def compute_weight_single_ev(samples,H=70.,mbins=10.,kappa=3):
    
    m1d_samples = samples[:,0]
    m2d_samples = samples[:,1]
    d_samples = samples[:,2]
    
    nbins_m = len(mbins)-1
    nsamples = len(samples)
    sindices = jnp.arange(nsamples)
    weight = jnp.zeros((nsamples,nbins_m,nbins_m))
    z_samples = z_of_dL(d_samples,H)
        
    m1_indices = jnp.clip(jnp.searchsorted(mbins, m1d_samples/(1+z_samples), side='right')- 1, a_min=0, a_max=nbins_m - 1)
    m2_indices = jnp.clip(jnp.searchsorted(mbins, m2d_samples/(1+z_samples), side='right')- 1, a_min=0, a_max=nbins_m - 1)
    pz_pop = dV_of_z(z_samples,H) *(1+z_samples)**(kappa-1) #uniform in comoving-volume
    ddL_dz = ddL_of_z(z_samples,d_samples,H) 
    pz_PE = (1+z_samples)**2 * d_samples**2 * ddL_dz # default PE prior - flat in det frame masses and dL**2 in distance
    pz_weight = pz_pop/pz_PE

    
    #weight[sindices,m1_indices,m2_indices] = pz_weight *(1+z_samples)**2/ (m1d_samples*m2d_samples) 
    weight = jnp.sum(weight.at[sindices,m1_indices,m2_indices].set( pz_weight *(1+z_samples)**2/ (m1d_samples*m2d_samples)),axis=0)
    normalized_weight = weight/nsamples
    return normalized_weight#[jnp.tril_indices(len(weight))]

with h5py.File('/home/anarya.ray/gppop-mdc/mock_data_simple/sensitivity/optimal_snr_PSDaLIGODesignSensitivityP1200087.h5', 'r') as f:
    ms = np.array(f['ms'])
    osnrs = np.array(f['SNR'])
    
rbs = RectBivariateSpline(ms, ms, osnrs)
print(osnrs.shape)

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

thetas = draw_thetas()#10000)
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
def VT_bin(edges,H=70.,ms=10.0,rhos=15.0,T=1,ngrid=10,kappa=3):
    m1_low,m1_high,m2_low,m2_high,z_low,z_high = edges[0,0],edges[1,0],edges[0,1],edges[1,1],edges[0,2],edges[1,2]
    z_grid = jnp.linspace(z_low,z_high,ngrid)
    m1_grid = jnp.linspace(m1_low,m1_high,ngrid)
    m2_grid = jnp.linspace(m2_low,m2_high,ngrid)
    
    m1s,m2s,zs = jnp.meshgrid(m1_grid,m2_grid,z_grid,indexing='ij')
    v = dV_of_z(zs,H)*(1+zs)**(kappa-1)*((u.Mpc**3).to(u.Gpc**3))
    ds = dL_of_z(zs,H)
    VT = Pdet_msdet(m1s*(1+zs),m2s*(1+zs),ds,ms,rhos)*v*T/(m1s*m2s)
    
    return jnp.trapz(jnp.trapz(jnp.trapz(VT,z_grid,axis=2),m2_grid,axis=1),m1_grid)/(1.+(m1_low==m2_low))

@jit
def jax_compute_weights_vts_op(Samples,H,mbins,edges,ms,rhos,T,kappa):
  
  return [jnp.array(list(jax.lax.map(partial(compute_weight_single_ev,H=H,mbins=mbins,kappa=kappa),Samples))),
          jnp.array(list(jax.lax.map(partial(VT_bin, H=H,ms=ms,rhos=rhos,T=T,kappa=kappa), edges)))]
