__author__="Anarya Ray <anarya.ray@ligo.org>; Siddharth Mohite <siddharth.mohite@ligo.org>"

from ..utils import bgp as bgputils
from functools import partial, reduce

import jax.numpy as jnp
from jax import jit, vmap
from jax.tree_util import tree_map
import jax

import numpyro
from numpyro import distributions as dist


#####################
# n-param gp models #
#####################

vexp = jit(vmap(jnp.exp))

def hierarchical_log_posterior_signficant(weights,vts,prior_model,*prior_args):
    n_corr = prior_model(*prior_args)
    N_F_exp = numpyro.deterministic('N_F_exp',jnp.sum(n_corr*vts))
        
    numpyro.factor('log_likelihood', jnp.sum(jnp.log(jnp.dot(weights,n_corr)))-N_F_exp)

    
def fully_independent_gp_prior( bin_centers, scale_means, scale_sds, sigma_sds, names, mu_dims,n_corr_name=''):
    length_scales = [ ]
    means = [ ]
    sigmas = [ ]
    covs = [ ]
    logn_tots = [ ]

    for i, (bin_center, ls_mu, ls_sd, sigma_sd, name, mu_dim) in enumerate(zip(bin_centers, scale_means, scale_sds, sigma_sds, names, mu_dims)):
        
        length_scales.append(numpyro.sample(f'length_scale_{name}', dist.LogNormal(ls_mu,ls_sd)))
        sigmas.append(numpyro.sample(f'sigma_{name}', dist.HalfNormal(sigma_sd)))
        if mu_dim>0:
            means.append(numpyro.sample(f'mu_{name}', dist.Normal(0,5), sample_shape=(mu_dim,)))
        else:
            means.append(numpyro.deterministic(f'mu_{name}', 0.0))

        covs.append(bgputils.kernel_RBF(bin_center, bin_center, jnp.power(sigmas[i],2.0), length_scales[i], 0.))
        logn_tots.append(numpyro.sample(f'logn_tot_{name}', dist.MultivariateNormal(loc = means[i],covariance_matrix = covs[i])))

    n_corr = numpyro.deterministic(f'n_corr{n_corr_name}', reduce(jnp.outer, tree_map(vexp,logn_tots)).flatten())
    return n_corr


def fully_dependent_gp_prior(bin_centers, scale_means, scale_sds, sigma_sd, names, mu_dim, n_corr_name=''):
        
    length_scales = [ ]
    covs = [ ]
    sigma = numpyro.sample('sigma', dist.HalfNormal(sigma_sd))
    mean = numpyro.sample('mean', dist.Normal(0,10),sample_shape=(mu_dim,))
    ndim = len(bin_centers)

    for i, (bin_center, ls_mu, ls_sd, name) in enumerate(zip(bin_centers, scale_means, scale_sds, names)):
        length_scales.append(numpyro.sample(f'length_scale_{name}', dist.LogNormal(ls_mu,ls_sd)))

        covs.append(bgputils.kernel_RBF(bin_center, bin_center, jnp.power(sigma,2.0/ndim), length_scales[i], 0.))


    logn_tot=numpyro.sample('logn_tot', bgputils.MultivariateNormalKron(tuple(covs), loc = mean))


    n_corr = numpyro.deterministic(f'n_corr{n_corr_name}',jnp.exp(logn_tot))
    return n_corr

def partially_dependent_gp_prior( bin_centers, scale_means, scale_sds, sigma_sds, names, mu_dims, correlated_uptill):
    if correlated_uptill is None or correlated_uptill == 0:
        fully_independent_gps(weights, vts, bin_centers, scale_means, scale_sds, sigma_sds, names, mu_dims)
    elif correlated_uptill == len(bin_centers):
        fully_dependent_gps(weights, vts, bin_centers, scale_means, scale_sds, sigma_sds, names, mu_dims)
    else:
        bin_centers_c, scale_means_c, scale_sds_c, sigma_sd_c, names_c, mu_dim_c = [], [], [], [], [], []
        
        for correlated_param_index in range(correlated_uptill):
            bin_centers_c.append( bin_centers.pop( bin_centers[correlated_param_index]))
            scale_means_c.append( scale_means.pop( scale_means[correlated_param_index]))
            scale_sds_c.append( scale_sds.pop( scale_sds[correlated_param_index]))
            sigma_sd_c.append( sigma_sds.pop( sigma_sds[correlated_param_index]))
            names_c.append( names.pop( names[correlated_param_index]))
            mu_dim_c.append( mu_dims.pop( mu_dims[correlated_param_index]))
        
        n_corr_1 = fully_dependent_gp_prior(bin_centers_c, scale_means_c, scale_sds_c, sigma_sd_c, names_c, mu_dim_c, n_corr_name='1')
        n_corr_2 = fully_independent_gp_prior( bin_centers, scale_means, scale_sds, sigma_sds, names, mu_dims,n_corr_name='2')
        
        n_corr = numpyro.deterministic('n_corr',jnp.outer(n_corr1,n_corr2).flatten())
        
        return n_corr
