__author__="Anarya Ray <anarya.ray@ligo.org>; Siddharth Mohite <siddharth.mohite@ligo.org>"

from ..utils import bgp as bgputils
from ..utils import cosmo as cutils
import jax.numpy as jnp
from jax import jit
import jax

@jit
def log_prob_spin(sx,sy,sz,m):
    s_max = jnp.where(m<2.5,0.4,0.99)
    return jnp.log(1./(4*jnp.pi*s_max*(sx**2 + sy**2 + sz**2)))

@jit
def reweight_pinjection(tril_weights):
    return jnp.where((tril_weights!=0),jnp.exp(tril_weights),0)

@jit
def dVT_surveyed_powerlaw_1plusredshift(dL,T,kappa,H0,Om0=cutils.Om0Planck):
    z = cutils.z_of_dL(dL,H0,Om0=Om0)
    return T*cutils.dV_of_z(z,H0,Om0=Om0)*(1+z)**(kappa-1)

@jit
def pe_prior_standard(dL):
    return jnp.power(dL,2)

@jit
def pop_m1m2z(m1dm2ddL_samples,T,kappa,H0,Om0=cutils.Om0Planck):
    z_samples = cutils.z_of_dL(m1dm2ddL_samples[:,2],H0,Om0=Om0)
    m1s_samples,m2s_samples = m1dm2ddL_samples[:,0]/(1+z_samples),m1dm2ddL_samples[:,1]/(1+z_samples)
    return dVT_surveyed_powerlaw_1plusredshift(m1dm2ddL_samples[:,-1],T,kappa,H0,Om0=Om0)/m1s_samples/m2s_samples


