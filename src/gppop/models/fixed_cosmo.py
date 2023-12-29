__author__="Anarya Ray <anarya.ray@ligo.org>"

from ..utils import cosmo as cutils, bgp as bgputils, gw as gwutils

from functools import partial, reduce

import jax.numpy as jnp
from jax import jit, vmap
import jax


####################
## m1,m2,z models ##
####################

class mass_redsfhit_models(object):
    
    def __init__(self, mbins,zbins, Samples, det_samples, pdraw, Ndraw, T, pe_prior = None, H0=cutils.H0Planck, Om0=cutils.Om0Planck,include_spins=True):
        
        self.mbins = mbins
        self.zbins = zbins
        self.tril_indexing  = bgputils.get_tril_indices(self.mbins,self.zbins)
        self.bin_centers = bgputils.generate_bin_centers(self.tril_indexing,jnp.log(self.mbins),jnp.log(self.mbins),self.zbins)
        self.bin_volumes = bgputils.generate_bin_volumes(self.tril_indexing,jnp.log(self.mbins),jnp.log(self.mbins),self.zbins)
        self.nbins = bgputils.get_nbins(jnp.log(self.mbins),self.zbins)
        ls_stats, separated_bin_centers = bgputils.get_gp_inputs(self.bin_centers,self.nbins)
        self.ls_stats = ls_stats
        self.separated_bin_centers = separated_bin_centers
        
        self.H0 = H0
        self.Om0 = Om0
        
        if pe_prior is None:
            pe_prior = jnp.asarray(vmap(gwutils.pe_prior_standard)(Samples[:,:,-1]))*jnp.asarray(vmap(partial(cutils.jacobian_detector_by_source,self.H0,Om0=self.Om0))(Samples[:,:,-1]))
        
        self.weights_vts = self.compute_weights_vt_op(Samples,pe_prior,det_samples,pdraw, Ndraw, T, self.H0, self.Om0,include_spins=include_spins)
            
            
    @partial(jit, static_argnums=(0,))
    def compute_weights_vt_op(self,Samples,pe_prior,det_samples,pdraw, Ndraw, T, H, Om0,include_spins=True):
        
        pop_func = vmap(partial(gwutils.pop_m1m2z,T=T,kappa=0,H0=H,Om0=Om0))
        
        weight_at_samples = jnp.asarray(pop_func(Samples))/pe_prior
        posterior_weight_means,posterior_weight_vars = vmap(partial(bgputils.binned_weights, tril_indexing=self.tril_indexing,bins=(self.mbins,self.mbins,self.zbins)))(Samples,weight_at_samples)
        
        if include_spins is not False:
            s1x, s1y, s1z, s2x, s2y, s2z, z = det_samples[:,3],det_samples[:,4],det_samples[:,5],det_samples[:,6],det_samples[:,7],det_samples[:,8],det_samples[:,9]
            spin_prob = gwutils.reweight_pinjection(gwutils.log_prob_spin(s1x,s1y,s1z,det_samples[:,0]/(1+z)) + gwutils.log_prob_spin(s2x,s2y,s2z,det_samples[:,1]/(1+z)))
        
        else:
            spin_porob = jnp.ones(len(det_samples))
        
        #Note if pdraw is in source frame there wont be a jacobian in the selection function
        weight_at_detectable_samples = gwutils.pop_m1m2z(det_samples,T,0,H,Om0=Om0)*spin_prob*cutils.mpc3_to_gpc3/pdraw
        
        vt_mean, vt_var = bgputils.binned_weights(det_samples,weight_at_detectable_samples, self.tril_indexing, (self.mbins,self.mbins,self.zbins),ndraw=Ndraw)
        
        return posterior_weight_means, posterior_weight_vars, vt_mean, vt_var
 