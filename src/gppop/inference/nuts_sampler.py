__author__="Anarya Ray <anarya.ray@ligo.org>"

import jax
from jax import jit

from numpyro.infer import MCMC
from numpyro.infer import NUTS

import os

import pickle

def sample_model(model,*args,thinning=100,num_warmup=10,num_samples=100, num_chains=1, target_accept_prob=0.9, storage_backend=None):
    
    RNG = jax.random.PRNGKey(0)
    MCMC_RNG, PRIOR_RNG, _RNG = jax.random.split(RNG, num=3)
    
    if storage_backend is None:
        kernel = NUTS(model, target_accept_prob= target_accept_prob)
        mcmc = MCMC(
            kernel,
            thinning=thinning,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )
        mcmc.run(PRIOR_RNG,*args)
    
    else:
        raise NotImplementedError
    
    return mcmc.get_samples()
'''
        if os.path.exists(storage_backend):
            with open(storage_backend, "rb") as input_file:
                mcmc = pickle.load(input_file)
                mcmc.post_warmup_state = mcmc.last_state
        else:
            kernel = NUTS(model, target_accept_prob= target_accept_prob)
            mcmc = MCMC(
                kernel,
                thinning=thinning,
                num_warmup=num_warmup,
                num_samples=10,
                num_chains=num_chains,
            )
            
  
    else:
        
        kernel = NUTS(model, target_accept_prob= target_accept_prob)
        mcmc = MCMC(
            kernel,
            thinning=thinning,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )
    
'''