__author__="Anarya Ray <anarya.ray@ligo.org>"

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from logbesselk.jax import log_bessel_k as log_k
from jax import jit, vmap
from jax.tree_util import tree_map
from functools import reduce, partial

from numpyro import distributions as dist


log_k_vec = jit(jax.vmap(log_k,(None,0),0))
#vchol = jit(vmap(jnp.linalg.cholesky))
#vkron = jit(vmap(jnp.kron))
vexpand_dims = jit(vmap(partial( jnp.expand_dims,axis=-1)))


jax.config.update("jax_enable_x64", True)

###########################
#### Binning Utilities ####
###########################

def get_tril_indices(mbins,*other_bins):
    nbins_m = len(mbins)-1
    n_other_bins= tuple([len(this_other_bin)-1 for this_other_bin in other_bins])
    
    m1_tril_indices,m2_tril_indices = np.tril_indices(nbins_m)
    
    nbins_m_tril = len(m1_tril_indices)
    
    indices = np.unravel_index(np.arange(int(np.prod(n_other_bins)*nbins_m_tril)),(nbins_m_tril,)+n_other_bins)[1:]
    m1_tril_indices = np.tile(m1_tril_indices,int(len(indices[0])/nbins_m_tril))
    m2_tril_indices = np.tile(m2_tril_indices,int(len(indices[0])/nbins_m_tril))
   
    return (m1_tril_indices,m2_tril_indices,)+indices


@jit
def get_item(array,index=()):
    return array[index]
    

@jit
def binned_weights(samples,weights,tril_indexing,bins,ndraw=None):
    nbins = tuple([len(this_param_bins)-1 for this_param_bins in bins])
    param_samples = tuple([samples[:,j] for j in range(samples.shape[1])])
    
    nsamples = len(samples)
    sindices = jnp.arange(nsamples)
    indices = tuple([jnp.clip(jnp.searchsorted(this_param_bins, this_param_samples, side='right')- 1, a_min=0, a_max=this_param_nbin - 1) for this_param_bins, this_param_samples, this_param_nbin in zip(bins, param_samples, nbins)])
    
    weights_array = jnp.zeros((nsamples,)+nbins)
    weights_array = weights_array.at[(sindices,)+indices].set(weights)
    
    ndraw = nsamples if ndraw is None else ndraw
    
    weights_mean = jnp.sum(weights_array,axis=0)/ndraw
    weights_var = jnp.sum(weights_array**2/ndraw**2 -weights_mean**2/ndraw,axis=0)
    
    return weights_mean[tril_indexing],weights_var[tril_indexing]

@jit
def bin_edges(tril_indexing,*bins):
    vget_item = vmap(partial(get_item,index=tril_indexing))
    lower_edges = jnp.concatenate(tuple(vexpand_dims(vget_item(jnp.asarray(jnp.meshgrid( *tuple([this_param_bins[:-1] for this_param_bins in bins])))))),axis=-1)
    upper_edges = jnp.concatenate(tuple(vexpand_dims(vget_item(jnp.asarray(jnp.meshgrid( *tuple([this_param_bins[1:] for this_param_bins in bins])))))),axis=-1)
    return lower_edges, upper_edges
    
@jit
def generate_bin_centers(tril_indexing,*bins):
    lower_edges, upper_edges = bin_edges(tril_indexing,*bins)
    return(0.5*(lower_edges+upper_edges))

@jit
def generate_bin_volumes(tril_indexing,*bins):
    lower_edges, upper_edges = bin_edges(tril_indexing,*bins)
    return(-lower_edges+upper_edges)

@jit
def get_nbins(mbins,*other_bins):
    nbins_m = (int(len(mbins)*(len(mbins)-1)*0.5),)
    return nbins_m+tuple([len(this_param_bins)-1 for this_param_bins in other_bins])
                                 
############################
####### GP Utilities #######
############################


def get_gp_inputs(bin_centers,nbins):
    reshaped_bin_centers = np.asarray(bin_centers.reshape(nbins+(bin_centers.shape[-1],)))
    ls_mean,ls_sd, sep_bin_centers = [ ], [ ], [ ]
    for i,nbin in enumerate(nbins):
        index = [0 for j in range(len(nbins))]
        index[i]=np.arange(nbin)
        if i==0:
            this_param_bin_center = reshaped_bin_centers[tuple(index)][:,:2]
        else:
            index.append(i+1)
            this_param_bin_center = reshaped_bin_centers[tuple(index)]
            
        sep_bin_centers.append(this_param_bin_center)
       
        dist_array =(this_param_bin_center[:,np.newaxis]-this_param_bin_center[np.newaxis,:])
        dist_array = np.linalg.norm(dist_array,axis=-1) if i==0 else abs(dist_array)
        dist_arrat = dist_array[np.tril_indices(nbin)]
        
        
        scale_min = np.log(np.min(dist_array[dist_array!=0.]))
        scale_max = np.log(np.max(dist_array))
        ls_mean.append(0.5*(scale_min + scale_max)) # chosen to give coverage over the bin-grid
        ls_sd.append((scale_max - scale_min)/4)
    
    return [ls_mean, ls_sd], sep_bin_centers
        
        
        
               
@jit
def kernel_RBF(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    deltaXsq = jnp.sum(jnp.power((X[jnp.newaxis,:]-X[:,jnp.newaxis])/length, 2.0),axis=-1)
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
def kernel_matern_int(X, Z, ms, var, length, noise, jitter=1.0e-6, include_noise=True):
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

@jit
def reduce_kron(iterable):
    return reduce(jnp.kron, iterable)

@jit
def vec_chol(*matrices):
    return tree_map(jnp.linalg.cholesky, matrices)[0]

class MultivariateNormalKron(dist.MultivariateNormal):
    
    def __init__(
        self,
        *kron_covs,
        loc=0.0):
        
        scale_tril = reduce_kron(vec_chol(*kron_covs))
        super(MultivariateNormalKron, self).__init__(loc=loc,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=scale_tril,
        validate_args=None,
    )
    
