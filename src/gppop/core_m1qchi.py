#!/usr/bin/env python
__author__="Omkar Sridhar <omkar.sridhar@ligo.org>; Anarya Ray <anarya.ray@ligo.org>; Siddharth Mohite <siddharth.mohite@ligo.org>"


import numpy as np
from scipy.stats import multivariate_normal,norm,halfnorm,lognorm
import pymc as pm
import aesara.tensor as tt
import pymc.math as math
from pymc.gp.util import plot_gp_dist
from astropy.cosmology import Planck15,z_at_value
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import warnings
import tqdm

############################
#  Support Functions       #
############################
def log_prob_spin(sx,sy,sz,m):
    '''
    Function that computes the default spin priors
    used to generate spin-parameters of injections.
    
    Author = Siddharth Mohite
    
    Parameters
    ----------
    
    sx    :: float
             x component of spin of the binary component
    
    sy    :: float
             y component of spin of the binary component
    
    sz    :: float
             z component of spin of the binary component
    
    m     :: float
             mass of the binary component
    '''
    s_max = np.where(m<2.5,0.4,0.99)
    return np.log(1./(4*np.pi*s_max*(sx**2 + sy**2 + sz**2)))

def reweight_pinjection(tril_weights):
    '''
    A function that converts log weights of injections to
    weights. Since each injection will have non-zero weights
    in only one bin, the log weights (output of Vt_Utils_spins_with_q.log_reweight_pinjection_mixture)
    are set to zero at all other bins. This function acts as a 
    wrapper around exp such that only non-zero log weights
    are exponentiated.
    
    Author = Siddharth Mohite
    
    Parameters
    ----------
    
    tril_weights    :: numpy.ndarray
                       1d array containing the output of Vt_Utils.log_reweight_pinjection_mixture
                       for one injection.
                       
    Returns
    -------
    
    exponential of tril_weights : numpy.ndarray
                                  1d array containing the exponential of the log
                                  of injection weights.
    '''
    return np.where((tril_weights!=0),np.exp(tril_weights),0)

def construct_arg_mat_out_spins(mbins, qbins, chi_bins):
    '''
    Function that enables removal of unphysical bins in the m1-q space. 
    
    Parameters
    ----------
    mbins :: numpy.ndarray 
                 1d array containing mass bin edges
        
    qbins :: numpy.ndarray 
                 1d array containing mass-ratio bin edges    
        
    chi_bins :: numpy.ndarray
                 1d array containing effective spin bin edges.
    
    '''
    
    nbins_m = len(mbins)-1
    nbins_q = len(qbins) - 1
    nbins_chi = len(chi_bins) - 1
    m1_bin_centers = 0.5 * (mbins[1:] + mbins[:-1])
    qbin_centers = 0.5 * (qbins[1:] + qbins[:-1])
    chi_bin_centers = 0.5 * (chi_bins[1:] + chi_bins[:-1])
    
    M, Q, CHI = np.meshgrid(m1_bin_centers, qbin_centers, chi_bin_centers, indexing='ij')
    
    arg_mat_spin = np.ones((nbins_m, nbins_q, nbins_chi))
    for i in range(1, nbins_m):
        for j in range(nbins_q):
            for k in range(nbins_chi):
                if qbin_centers[j] <= np.min(m1_bin_centers) / M[i, j, k] :
                    arg_mat_spin[i - 1, j, k] = 0
    arg_mat_flat = np.matrix.flatten(arg_mat_spin)    
    
    return arg_mat_spin


class Utils_spins_with_q():
    """
    Utilities for GP rate inference. Contains 
    functions for binning up the m1,q,chieff 
    parameter space, and computing various attributes 
    of bins and posterior weights in bins.
    """
    
    def __init__(self,mbins,qbins,chi_bins,kappa=2.7):
        '''
        Initialize utilities class.
        
        Parameters
        ----------
        
        mbins :: numpy.ndarray 
                 1d array containing mass bin edges
                 
        qbins :: numpy.ndarray 
                 1d array containing mass-ratio bin edges
        
        chi_bins :: numpy.ndarray
                 1d array containing effective spin bin edges.

        kappa   :: float
                   redshift evolution of the merger rate.
        
        '''
        self.mbins = mbins
        self.qbins = qbins
        self.chi_bins = chi_bins
        self.kappa=kappa

    def construct_arg_mat(self):
        '''
        Function that enables removal of unphysical bins in the m1-q space,
        as an attribute of Utils_spin_with_q()
        
        
        '''
    
        nbins_m = len(self.mbins)-1
        nbins_q = len(self.qbins) - 1
        nbins_chi = len(self.chi_bins) - 1
        m1_bin_centers = 0.5 * (self.mbins[1:] + self.mbins[:-1])
        qbin_centers = 0.5 * (self.qbins[1:] + self.qbins[:-1])
        chi_bin_centers = 0.5 * (self.chi_bins[1:] + self.chi_bins[:-1])
        
        M, Q, CHI = np.meshgrid(m1_bin_centers, qbin_centers, chi_bin_centers, indexing='ij')
        
        arg_mat_spin = np.ones((nbins_m, nbins_q, nbins_chi))
        for i in range(1, nbins_m):
            for j in range(nbins_q):
                for k in range(nbins_chi):
                    if qbin_centers[j] <= np.min(m1_bin_centers) / M[i, j, k] :
                        arg_mat_spin[i - 1, j, k] = 0
        arg_mat_flat = np.matrix.flatten(arg_mat_spin)    
        
        return arg_mat_spin
    
    def arraynd_to_tril(self,arr, arg_mat):
        '''
        Function that returns the set of entries 
        (q<=m1_min/m1) of a collection of 2d matrices
        each binned by m1 and q. For the m1,q,chieff inference,
        it returns multiple sets of q-cut (q<=m1_min/m1) 
        entries, one set corresponding to each chieff bin.

        Parameters
        ----------
        arr :: numpy.ndarray
               Input 2d or 3d matrix.

        Returns
        -------
        lower_tri_array : numpy.ndarray
                          Array of lower-triangular entries.
        '''
        arg_mat_flat = np.matrix.flatten(arg_mat)
        args = np.where(arg_mat_flat > 0)[0]
        arr_flat = np.matrix.flatten(arr)
        return arr_flat[args]

    def compute_weights(self,samples,m1m2_given_z_prior=None,chi_prior=None,leftmost_chibin=None,full_prior=None,O4_prior = False):
        '''
        Function to compute the weights needed to reweight
        posterior samples to the population distribution,
        for an event in parameter bins. 

        Parameters
        ----------
        samples            :: numpy.ndarray
                              Array of m1,q,chieff posterior samples.
                              
        m1m2_given_z_prior :: numpy.ndarray
                              if default PE priors were not used then
                              the values of the p(m_1,m_2|z) function used
                              in PE need to be supplied corresponding to
                              each posterior sample.

        chi_prior          :: numpy.ndarray
                              The PE prior on effective spin marginalized over  
                              other spin parameters, (can be conditioned over 
                              masses, see https://arxiv.org/abs/2104.09508)

        left_most_chi_bin  :: float
                              the edge of the smallest effective spin bin. If there are 
                              too many posterior samples outside this bin then this 
                              argument is used to prevent numpy.searchsorted to assign
                              large weights to this bin.


        full_prior         :: numpy.ndarray
                              If instead of individual priors the full p(m1,m2,chi_eff,z)
                              is provided for every sample.

        O4_prior           :: bool
                              If True, it will assume a uniform in co-moving volume 
                              prior on distance/redshift instead of the older dL^2 prior
                              which is assumed if False.
        
        Returns
        -------
        weights : numpy.ndarray
                  The weight matrix of shape(mbins,qbins,chi_bins) for m1,q,chi_eff
                  inference.
        '''
        weights = np.zeros([len(self.mbins)-1,len(self.qbins)-1,len(self.chi_bins)-1])
        wgt_means = np.zeros([len(self.mbins)-1,len(self.qbins)-1,len(self.chi_bins)-1])
        wgt_sigmas = np.zeros([len(self.mbins)-1,len(self.qbins)-1,len(self.chi_bins)-1])

        
        good_idx = np.where((samples[:,0] > self.mbins[0])  * (samples[:,1] > self.qbins[0]) * (samples[:,0] < self.mbins[-1])  * (samples[:,1] < self.qbins[-1]) * (samples[:,3] > self.chi_bins[0])  * (samples[:,3] < self.chi_bins[-1]))[0]
        samp_copy = samples[good_idx]
        #samp_copy = np.delete(samp_copy, bad_idx, 0)
        
        m1_samples = samp_copy[:,0]
        q_samples = samp_copy[:,1]
        z_samples = samp_copy[:,2]
        chi_samples = samp_copy[:,3]
        #uniform in comoving-volume
        dl_values = Planck15.luminosity_distance(z_samples).to(u.Gpc).value
        m1_indices = np.clip(np.searchsorted(self.mbins,m1_samples,side='right') - 1,a_min=0,a_max=len(self.mbins)-2)
        q_indices = np.clip(np.searchsorted(self.qbins,q_samples,side='right') - 1,a_min=0,a_max=len(self.qbins)-2)
        if leftmost_chibin is None:
            chi_indices = np.clip(np.searchsorted(self.chi_bins,chi_samples,side='right') - 1,a_min=0,a_max=len(self.chi_bins)-2)
        else:
            chi_indices = np.clip(np.searchsorted(np.append([leftmost_chibin], self.chi_bins),chi_samples,side='right') - 1,a_min=0,a_max=len(self.chi_bins)-1)
            
        pz_pop = Planck15.differential_comoving_volume(z_samples).to(u.Gpc**3/u.sr).value*((1+z_samples)**(self.kappa-1))
        if full_prior is None:
            ddL_dz = dl_values/(1+z_samples) + (1+z_samples)*Planck15.hubble_distance.to(u.Gpc).value/Planck15.efunc(z_samples)#Jacobian to convert from dL to z 
            m1m2_given_z_prior = m1m2_given_z_prior if m1m2_given_z_prior is not None else (1+z_samples)**2
            if not O4_prior:
                pz_PE = m1m2_given_z_prior * dl_values**2 * ddL_dz # default PE prior - flat in det frame masses and dL**2 in distance
            else : 
                pz_PE = m1m2_given_z_prior * Planck15.differential_comoving_volume(z_samples).to(u.Gpc**3/u.sr).value/(1+z_samples) # m1m2_given_z_prior * dl_values**2 * ddL_dz
            pz_PE*=chi_prior[good_idx]
        else:
            pz_PE=full_prior
        pz_weight = pz_pop/pz_PE
        indices = zip(m1_indices,q_indices,chi_indices)
        if leftmost_chibin is None:
            for i,inds in enumerate(indices):
                    weights[inds[0],inds[1],inds[2]] += pz_weight[i]/(m1_samples[i]*m1_samples[i])
                    wgt_means[inds[0],inds[1],inds[2]] += pz_weight[i]/(m1_samples[i]*m1_samples[i]) / len(samples)
        else:
            for i,inds in enumerate(indices):
                    weights[inds[0],inds[1],inds[2]-1] += float(inds[2]>0)*pz_weight[i]/(m1_samples[i]*m1_samples[i])
                    wgt_means[inds[0],inds[1],inds[2] -1] += float(inds[2]>0)*pz_weight[i]/(m1_samples[i]*m1_samples[i]) / len(samples)
        indices = zip(m1_indices,q_indices,chi_indices)
        if leftmost_chibin is None:
            for i,inds in enumerate(indices):
                    wgt_sigmas[inds[0],inds[1],inds[2]] += ((pz_weight[i]/(m1_samples[i] ** 2)) ** 2 / len(samples) ** 2 - wgt_means[inds[0],inds[1],inds[2]] ** 2 / len(samples) ** 2)
        else:
            for i,inds in enumerate(indices):
                    wgt_sigmas[inds[0],inds[1],inds[2]-1] += float(inds[2]>0)*((pz_weight[i]/(m1_samples[i] ** 2)) ** 2 / len(samples) ** 2 - wgt_means[inds[0],inds[1],inds[2]-1] ** 2 / len(samples) ** 2)
        wgt_sigmas = np.sqrt(wgt_sigmas)
        weights /= sum(sum(sum(weights)))
        return weights, wgt_means, wgt_sigmas

    def deltaLogbin(self):
        '''
        Function that returns the deltaLogbin for each bin.

        Returns
        -------
        deltaLogbin_array : numpy.ndarray
                            n-D array providing deltaLogbin for each bin.
        '''
        m1 = self.mbins
        q = self.qbins
        chi = self.chi_bins
        deltaLogbin_array = np.ones([len(m1)-1,len(q)-1,len(chi)-1])
        for i in range(len(m1)-1):
            for j in range(len(q)-1):
                for k in range(len(chi)-1):
                    if j != i:
                        deltaLogbin_array[i,j,k] = np.log(m1[i+1]/m1[i])*(q[j+1]-q[j])*(chi[k+1]-chi[k])
                    elif j==i:
                        deltaLogbin_array[i,i,k] = np.log(m1[i+1]/m1[i])*(q[i+1]-q[i])*(chi[k+1]-chi[k])
        return deltaLogbin_array
    
    def tril_edges(self):
        '''
        A function that returns the m1,q,chi_eff edges of each bin
        in the form of the output of arraynd_to_tril()
        
        Returns
        -------
        edge_array : numpy.ndarray
                     an array containing upper and lower edges for each 
                     bin.
        '''
        m1 = self.mbins
        q = self.qbins
        chi = self.chi_bins
        edge_array = []
        for i in range(len(m1)-1):
            for j in range(len(q)-1):
                for k in range(len(chi)-1):
                    edge_array.append([[m1[i],q[j],chi[k]],[m1[i+1],q[j+1],chi[k+1]]])
        return np.array(edge_array)

    def generate_log_bin_centers(self):
        '''
        Function that returns n-D bin centers in logm1,q,chi_eff space.

        Returns
        -------
        log_lower_tri_sorted : numpy.ndarray
                               n-D array of the  bin centers in logm space and
                               chi_eff bins in linear space.
        '''
        # zbins = np.log(self.zbins+1.0e-300)
        for k in range(len(self.mbins)-1):
            nbin2 = len(self.chi_bins) - 1
            nbin = len(self.qbins) - 1
            chi_bin_centres = np.asarray([0.5*(self.chi_bins[i+1]+self.chi_bins[i])for i in range(nbin2)])
            q_bin_centres = np.asarray([0.5*(self.qbins[j+1]+self.qbins[j]) for j in range(nbin)])
            l1, l2 = np.meshgrid(chi_bin_centres, q_bin_centres)
            l3 = np.array([[0.5*(np.log(self.mbins[k+1])+np.log(self.mbins[k]))] for i in range(nbin*nbin2)])
            logM = np.concatenate((l3,l2.reshape([nbin*nbin2,1]),l1.reshape([nbin*nbin2,1])),axis=1)
            logM_lower_tri = np.asarray([a for a in logM if a[1]<=a[0]])
            #logM_lower_tri = logM
            logM_lower_tri_sorted = np.asarray([logM_lower_tri[i] for i in np.argsort(logM_lower_tri[:,0],kind='mergesort')])
            if k == 0:
                log_lower_tri_sorted = logM_lower_tri_sorted
            else:
                log_lower_tri_sorted=np.append(log_lower_tri_sorted, logM_lower_tri_sorted,axis =0)
        arg_mat_flat = np.matrix.flatten(self.construct_arg_mat())
        return log_lower_tri_sorted[np.where(arg_mat_flat > 0)[0]]
            
                
    def construct_1dtond_matrix(self,nbins_m, values,nbins_chi, nbins_q, m_min = None, m_max = None, arg_mat = None, tril=True):
        '''
        Inverse of arraynd_to_tril() Returns a n-D
        represenation matrix of a given set of q-cut
        1-D values or multiple sets of q-cut
        1-D values, one set corresponding to 
        each chieff bin.

        Parameters
        ----------
        values : numpy.ndarray
            1-D array of lower triangular entries.
        nbins_m : int
            number of mass bins
        nbins_chi : int
            number of chieff bins
        nbins_q : int
            number of mass-ratio bins
            
        Returns
        -------
        matrix : numpy.ndarray
            n-D symmetric array using values.
        '''
        k=0
        if len(values.shape)>1:
            matrix = np.zeros((nbins_m,nbins_q,nbins_chi)+values.shape[1:])
        else:
            matrix = np.zeros((nbins_m,nbins_q,nbins_chi))
	
        if np.sum(arg_mat) == None:
            arg_mat = self.construct_arg_mat()

        if m_min == None:
            m_min = self.mbins[0]
            m_max = self.mbins[-1]
	
        log_m1_bin_centers = 0.5 * (np.log(self.mbins[1:]) + np.log(self.mbins[:-1]))
        bin_idx = np.arange(len(log_m1_bin_centers))
        idx_arr = bin_idx[(log_m1_bin_centers >= np.log(m_min))&(log_m1_bin_centers <= np.log(m_max))]
        #arg_mat = self.construct_arg_mat()
        #print(idx_arr)
        for i in range(nbins_m):
            for j in range(nbins_q):
                for l in range(nbins_chi):
                    if arg_mat[idx_arr[i],j,l]>0:
                        matrix[i,j,l] = values[k]
                        k+=1
            
        return matrix

    
    def delta_q(self):
        '''
        A function that returns delta q for each bin in the
        lower triangular format of the output of arraynd_to_tril.
        
        
        Returns
        -------
        
        delta_q_array : numpy.ndarray
                        1d array of delta q's
        '''
        delta_q_array = np.zeros([len(self.mbins)-1,len(self.qbins)-1,len(self.chi_bins)-1])
        for i in range(len(self.mbins)-1):
            for j in range(len(self.qbins)-1):
                for k in range(len(self.chi_bins)-1):
                    delta_q_array[i,j,k] = self.qbins[j+1]-self.qbins[j]
        return self.arraynd_to_tril(delta_q_array, self.construct_arg_mat())
    
    def delta_logm1s(self):
        '''
        A function that returns delta log(m1) for each bin in the
        lower triangular format of the output of arraynd_to_tril.
        
        
        Returns
        -------
        
        delta_logm1_array : numpy.ndarray
                            1d array of delta log(m1)'s
        '''
        delta_logm1_array = np.zeros([len(self.mbins)-1,len(self.qbins)-1,len(self.chi_bins)-1])
        for i in range(len(self.mbins)-1):
            for j in range(len(self.qbins)-1):
                for k in range(len(self.chi_bins)-1):
                    delta_logm1_array[i,j,k] = np.log(self.mbins[i+1]/self.mbins[i])

        return self.arraynd_to_tril(delta_logm1_array, self.construct_arg_mat())
    
    def delta_chis(self):
        '''
        A function that returns delta chi_eff for each bin in the
        lower triangular format of the output of arraynd_to_tril.
        
               
        Returns
        -------
        
        delta_chi_array : numpy.ndarray
                          1d array of delta chieff's
        '''
        delta_chi_array = np.zeros([len(self.mbins)-1,len(self.qbins)-1,len(self.chi_bins)-1])
        for i in range(len(self.mbins)-1):
            for j in range(len(self.qbins)-1):
                for k in range(len(self.chi_bins)-1):
                    delta_chi_array[i,j,k] = self.chi_bins[k+1]-self.chi_bins[k]
                    
        return self.arraynd_to_tril(delta_chi_array, self.construct_arg_mat())
    
class Post_Proc_Utils_spins_with_q(Utils_spins_with_q):
    """
    Postprocessing Utilities for GP 
    rate inference. Functions for parsing
    samples of rate densities and computing
    marginal distributions.
    """
    
    def __init__(self,mbins, qbins, chi_bins,kappa=2.7):
        '''
        Initialize post-processing utilities class.
        
        Parameters
        ----------
        
        mbins :: numpy.ndarray 
                 1d array containing mass bin edges
        
        qbins :: numpy.ndarray 
                 1d array containing mass-ratio bin edges
        
        chi_bins :: numpy.ndarray
                 1d array containing effective spin bin edges.
        '''
        
        Utils_spins_with_q.__init__(self,mbins,qbins,chi_bins,kappa=kappa)
    
    def reshape_uncorr(self,n_corr,n_corr_chi):
        '''
        Function for combining uncorrelated mass and 
        effective spin rate densities into combined rate
        densities 
        
        Parameters
        ----------
        
        n_corr   :: numpy.ndarray
                    array containing rate-densities w.r.t. mass bins
        
        n_corr_chi :: numpy.ndarray
                    array containing rate densities w.r.t. effective spin bins
                    
        
        Returns
        -------
        
        n_corr_all : numpy.ndarray
                     array containing combined rate densities
        
        '''
        n_corr_all = np.array([])
        for i in range(len(n_corr_chi)):
            n_corr_all = np.append(n_corr_all,n_corr*n_corr_chi[i])
        return n_corr_all
    
    
    
    def get_Rpm1_corr(self,n_corr,delta_q_array,delta_chi_array,m1_bins,q_bins,log_bin_centers,q_low, q_high, chi_low,chi_high):
        '''
        Function for computing conditional primary mass population: p(m_1|q, chi_eff)
        evaluated at mass-ratios and effective spins belonging to some range
        
        Parameters
        ----------
        
        n_corr                  ::   numpy.ndarray
                                     array containing rate density in each bin
        
        delta_q_array       ::   numpy.ndarray
                                     1d array of delta q's
        
        delta_chi_array       ::   numpy.ndarray
                                     1d array of delta chi_eff's
        
        m1_bins                 ::   numpy.ndarray
                                     1d array containing primary mass bin edges
        
        q_bins                 ::   numpy.ndarray
                                     1d array containing secondary mass-ratio bin edges
                
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        q_low                 ::   float 
                                     lower edge of q bin
                                     
        q_high                 ::   float 
                                     upper edge of q bin
        
        chi_low                 ::   float 
                                     lower edge of chi_eff bin
                                     
        chi_high                 ::   float 
                                     upper edge of chi_eff bin
        
        
        Returns
        -------
        mass1     :   numpy.ndarray
                      1d array of primary masses at which p(m1|q, chi_eff) is evaluated
        Rpm1      :   numpy.ndarray
                      1d array of p(m1|q, chi_eff) evaluated at the above m1 values and
                      at q, chieff belonging to a particular range
        
        '''
        Rpm1 = np.zeros([len(n_corr),1])
        mass1 = np.array([])
        for i in range(len(m1_bins)-1):
                m1_low = m1_bins[i]
                m1_high = m1_bins[i+1]
                m_array = np.linspace(m1_low,m1_high,100)[:-1]
                idx_array = np.arange(len(log_bin_centers))
                bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m1_low))&(log_bin_centers[:,0]<=np.log(m1_high))&
                       (log_bin_centers[:,1]>=q_low)&(log_bin_centers[:,1]<=q_high)&(log_bin_centers[:,2]>=chi_low)&(log_bin_centers[:,2]<=chi_high)]
                rate_density_array = n_corr[:,bin_idx]
                delta_qs = delta_q_array[bin_idx]
                delta_chis= delta_chi_array[bin_idx]
                Rpm1 = np.concatenate((Rpm1,np.sum(rate_density_array*((delta_qs*delta_chis)[None,:]),axis=1)[:,None]/(m_array[None,:])),axis=1)
                mass1 = np.append(mass1,m_array)
        return mass1,Rpm1[:,1:]

    def get_Rpm1_q_corr(self,n_corr,delta_q_array,delta_chi_array,m1_bins,chi_bins,log_bin_centers,q_low,q_high):
        '''
        Function for computing conditional primary mass population: p(m_1|q)
        evaluated at mass-ratios belonging to some range

        Parameters
        ----------

        n_corr                  ::   numpy.ndarray
                                     array containing rate density in each bin

        delta_q_array       ::   numpy.ndarray
                                     1d array of delta q's
                                     
        delta_chi_array       ::   numpy.ndarray
                                     1d array of delta chieff's                             

        m1_bins                 ::   numpy.ndarray
                                     1d array containing primary mass bin edges

        q_bins                 ::   numpy.ndarray
                                     1d array containing mass ratio bin edges

        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin

        q_low                   ::   float 
                                     lower edge of q bin

        q_high                   ::   float 
                                     upper edge of q bin


        Returns
        -------
        mass1     :   numpy.ndarray
                      1d array of primary masses at which p(m1|q) is evaluated
        Rpm1      :   numpy.ndarray
                      1d array of p(m1|q) evaluated at the above m1 values and
                      at mass-ratios belonging to a particular range

        '''
        Rpm1 = np.zeros([len(n_corr),1])
        mass1 = np.array([])
        for i in range(len(m1_bins)-1):
                m1_low = m1_bins[i]
                m1_high = m1_bins[i+1]
                chi_low = chi_bins[0]
                chi_high = chi_bins[-1]
                m_array = np.linspace(m1_low,m1_high,100)[:-1]
                idx_array = np.arange(len(log_bin_centers))
                bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m1_low))&(log_bin_centers[:,0]<=np.log(m1_high))&
                       (log_bin_centers[:,1]>=q_low)&(log_bin_centers[:,1]<=q_high)&(log_bin_centers[:,2]>=chi_low)&(log_bin_centers[:,2]<=chi_high)]
                rate_density_array = n_corr[:,bin_idx]
                delta_qs = delta_q_array[bin_idx]
                delta_chis= delta_chi_array[bin_idx]
                Rpm1 = np.concatenate((Rpm1,np.sum(rate_density_array*((delta_qs*delta_chis)[None,:]),axis=1)[:,None]/(m_array[None,:])),axis=1)
                mass1 = np.append(mass1,m_array)
                #mass1 = mass1.append(m_array)
                #print(mass1)
        print(mass1.shape, Rpm1.shape)
        return mass1,Rpm1[:,1:]
        
    
    
    def get_Rpq_corr(self,n_corr,delta_logm1_array,delta_chi_array,m1_bins,q_bins,log_bin_centers,chi_low,chi_high):
        '''
        Function for computing conditional primary mass population: p(q|chi_eff)
        evaluated at chi_eff belonging to some range
        
        Parameters
        ----------
        
        n_corr                  ::   numpy.ndarray
                                     array containing rate density in each bin
        
        delta_logm1_array       ::   numpy.ndarray
                                     1d array of delta log(m1)'s
        
        delta_chi_array       ::   numpy.ndarray
                                     1d array of delta chieff's
        
        m1_bins                 ::   numpy.ndarray
                                     1d array containing primary mass bin edges
        
        q_bins                 ::   numpy.ndarray
                                     1d array containing secondary mass-ratio bin edges
                
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        chi_low                 ::   float 
                                     lower edge of chi_eff bin
                                     
        chi_high                ::   float 
                                     upper edge of chi_eff bin
        
        
        Returns
        -------
        mrat     :   numpy.ndarray
                      1d array of mass-ratios at which p(q|chi_eff) is evaluated
        Rpq      :   numpy.ndarray
                      1d array of p(q|chi_eff) evaluated at the above q values and
                      at chieff's belonging to a particular range
        
        '''
        Rpq = np.zeros([len(n_corr),1])
        mrat = np.array([])
        for i in range(len(q_bins)-1):
                m1_low = m1_bins[0]
                m1_high = m1_bins[-1]
                q_low = q_bins[i]
                q_high = q_bins[i+1]
                m_array = np.ones(99)
                q_array = np.linspace(q_low,q_high,100)[:-1]
                idx_array = np.arange(len(log_bin_centers))
                bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m1_low))&(log_bin_centers[:,0]<=np.log(m1_high))&
                       (log_bin_centers[:,1]>=q_low)&(log_bin_centers[:,1]<=q_high)&(log_bin_centers[:,2]>=chi_low)&(log_bin_centers[:,2]<=chi_high)]
                rate_density_array = n_corr[:,bin_idx]
                delta_logm1s = delta_logm1_array[bin_idx]
                delta_chis= delta_chi_array[bin_idx]
                Rpq = np.concatenate((Rpq,(np.sum(rate_density_array*((delta_logm1s*delta_chis)[None,:]),axis=1)[:,None]/(m_array[None,:]))),axis=1)
                mrat = np.append(mrat,q_array)
        return mrat,Rpq[:,1:]

    def get_Rpq_corr_m1chi(self, n_corr,delta_logm1_array,delta_chi_array,m1_bins,q_bins,log_bin_centers,m1_low,m1_high,chi_low,chi_high):
        '''
        Function for computing conditional mass-ratio population: p(q|m1, chi_eff)
        evaluated at primary masses and effective spins belonging to some range

        Parameters
        ----------

        n_corr                  ::   numpy.ndarray
                                     array containing rate density in each bin

        delta_logm1_array       ::   numpy.ndarray
                                     1d array of delta log(m1)'s
                                     
        delta_chi_array       ::   numpy.ndarray
                                     1d array of delta chieff's                             

        m1_bins                 ::   numpy.ndarray
                                     1d array containing primary mass bin edges

        q_bins                 ::   numpy.ndarray
                                     1d array containing mass-ratio bin edges

        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin

        m1_low                 ::   float 
                                     lower edge of mass bin
                                     
        m1_high                ::   float 
                                     upper edge of mass bin
                                     
        chi_low                 ::   float 
                                     lower edge of chi_eff bin
                                     
        chi_high                ::   float 
                                     upper edge of chi_eff bin                             


        Returns
        -------
        mrat     :   numpy.ndarray
                      1d array of mass-ratios at which p(q|m1, chi_eff) is evaluated
        Rpq      :   numpy.ndarray
                      1d array of p(q|m1, chi_eff) evaluated at the above q values and
                      at m1,chieff's belonging to a particular range

        '''
        Rpq = np.zeros([len(n_corr),1])
        mrat = np.array([])
        for i in range(len(q_bins)-1):
                #m1_low = m1_bins[0]
                #m1_high = m1_bins[-1]
                q_low = q_bins[i]
                q_high = q_bins[i+1]
                #m_array = np.linspace(m1_low,m1_high,100)[:-1]
                m_array = np.ones(99)
                q_array = np.linspace(q_low,q_high,100)[:-1]
                idx_array = np.arange(len(log_bin_centers))
                bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m1_low))&(log_bin_centers[:,0]<=np.log(m1_high))&
                       (log_bin_centers[:,1]>=q_low)&(log_bin_centers[:,1]<=q_high)&(log_bin_centers[:,2]>=chi_low)&(log_bin_centers[:,2]<=chi_high)]
                rate_density_array = n_corr[:,bin_idx]
                delta_logm1s = delta_logm1_array[bin_idx]
                delta_chis= delta_chi_array[bin_idx]
                Rpq = np.concatenate((Rpq,(np.sum(rate_density_array*((delta_logm1s*delta_chis)[None,:]),axis=1)[:,None]/(m_array[None,:]))),axis=1)

                mrat = np.append(mrat,q_array)
        print(Rpq.shape)
        return mrat,Rpq[:,1:]

    def get_Rpq_corr_m1complement_chi(self, n_corr,delta_logm1_array,delta_chi_array,m1_bins,q_bins,log_bin_centers,m1_low,m1_high,chi_low,chi_high):
        '''
        Function for computing conditional mass-ratio population: p(q|m1, chi_eff)
        evaluated at primary masses and effective spins belonging to some range

        Parameters
        ----------

        n_corr                  ::   numpy.ndarray
                                     array containing rate density in each bin

        delta_logm1_array       ::   numpy.ndarray
                                     1d array of delta log(m1)'s
                                     
        delta_chi_array       ::   numpy.ndarray
                                     1d array of delta chieff's                             

        m1_bins                 ::   numpy.ndarray
                                     1d array containing primary mass bin edges

        q_bins                 ::   numpy.ndarray
                                     1d array containing mass-ratio bin edges

        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin

        m1_low                 ::   float 
                                     lower edge of mass bin
                                     
        m1_high                ::   float 
                                     upper edge of mass bin
                                     
        chi_low                 ::   float 
                                     lower edge of chi_eff bin
                                     
        chi_high                ::   float 
                                     upper edge of chi_eff bin                             


        Returns
        -------
        mrat     :   numpy.ndarray
                      1d array of mass-ratios at which p(q|m1, chi_eff) is evaluated
        Rpq      :   numpy.ndarray
                      1d array of p(q|m1, chi_eff) evaluated at the above q values and
                      at m1,chieff's belonging to a particular range

        '''
        Rpq = np.zeros([len(n_corr),1])
        mrat = np.array([])
        for i in range(len(q_bins)-1):
                #m1_low = m1_bins[0]
                #m1_high = m1_bins[-1]
                q_low = q_bins[i]
                q_high = q_bins[i+1]
                #m_array = np.linspace(m1_low,m1_high,100)[:-1]
                m_array = np.ones(99)
                q_array = np.linspace(q_low,q_high,100)[:-1]
                idx_array = np.arange(len(log_bin_centers))
                bin_idx = idx_array[(~((log_bin_centers[:,0]>=np.log(m1_low))&(log_bin_centers[:,0]<=np.log(m1_high))))&
                       (log_bin_centers[:,1]>=q_low)&(log_bin_centers[:,1]<=q_high)&(log_bin_centers[:,2]>=chi_low)&(log_bin_centers[:,2]<=chi_high)]
                rate_density_array = n_corr[:,bin_idx]
                delta_logm1s = delta_logm1_array[bin_idx]
                delta_chis= delta_chi_array[bin_idx]
                Rpq = np.concatenate((Rpq,(np.sum(rate_density_array*((delta_logm1s*delta_chis)[None,:]),axis=1)[:,None]/(m_array[None,:]))),axis=1)

                mrat = np.append(mrat,q_array)
        print(Rpq.shape)
        return mrat,Rpq[:,1:]

    def get_Rpchi_q(self,log_bin_centers,n_corr_samples,chi_bins,dm1,dq,q_min,q_max):
        '''
        Function for computing p(chi_eff|q) for q values belonging in some range.

        Parameters
        ----------
        
        n_corr_samples          ::   numpy.ndarray
                                     array containing rate density in each bin
        
        dm1                     ::   numpy.ndarray
                                     1d array of delta log(m1)'s
                                     
        dm2                     ::   numpy.ndarray
                                     1d array of delta log(m2)'s
        
        chi_bins                ::   numpy.ndarray
                                     1d array containing chi_eff bin edges
        
                      
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        q_min                   ::   float 
                                     lower edge of the mass-ratio range
                                     
        q_max                   ::   float 
                                     upper edge of the mass-ratio range
        
        
        Returns
        -------
        chi       :   numpy.ndarray
                      1d array of chi_eff at which p(chi_eff|q) is evaluated
        Rp_chi    :   numpy.ndarray
                      1d array of p(chi_eff|q) evaluated at the above chi_eff values
                      and at mass-ratios belonging to a particular range
        '''
        diag_idx = np.where(log_bin_centers[:,0] == log_bin_centers[:,1])[0]
        ones = np.ones(len(dq))
        #ones[diag_idx]*=2.
        nbins_chi = len(chi_bins)-1
        Rp_chi,chi = np.zeros((len(n_corr_samples),1)),np.array([ ])
        for i in range(nbins_chi):
            idx_array = np.arange(len(log_bin_centers))
            bin_idx = idx_array[(log_bin_centers[:,1]<=q_max)&(log_bin_centers[:,1]>=q_min)&(log_bin_centers[:,2]>=chi_bins[i])&(log_bin_centers[:,2]<=chi_bins[i+1])]
            this_Rp_chi = np.sum((n_corr_samples*dm1[None,:]*dq[None,:]*ones[None,:])[:,bin_idx],axis=-1)
            
            this_chi = np.linspace(chi_bins[i],chi_bins[i+1],100)
            chi=np.append(chi,this_chi)
            Rp_chi = np.concatenate((Rp_chi,np.ones(100)[None,:]*this_Rp_chi[:,None]),axis=1)

        return chi, Rp_chi[:,1:]

    def get_Rpchi_m(self,log_bin_centers,n_corr_samples,chi_bins,dm1,dq,m_min,m_max, q_min, q_max):
        diag_idx = np.where(log_bin_centers[:,0] == log_bin_centers[:,1])[0]
        ones = np.ones(len(dq))
        #ones[diag_idx]*=2.
        nbins_chi = len(chi_bins)-1
        Rp_chi,chi = np.zeros((len(n_corr_samples),1)),np.array([ ])

        for i in range(nbins_chi):
            idx_array = np.arange(len(log_bin_centers))
            bin_idx = idx_array[(((log_bin_centers[:,0]<=np.log(m_max))&
                                (log_bin_centers[:,0]>=np.log(m_min))))&(log_bin_centers[:,1]>=q_min)&
                                (log_bin_centers[:,1]<=q_max)&
                                (log_bin_centers[:,2]>=chi_bins[i])&
                                (log_bin_centers[:,2]<=chi_bins[i+1])]
            this_Rp_chi = np.sum((n_corr_samples*dm1[None,:]*dq[None,:]*ones[None,:])[:,bin_idx],axis=-1)
            this_chi = np.linspace(chi_bins[i],chi_bins[i+1],100)
            chi=np.append(chi,this_chi)
            Rp_chi = np.concatenate((Rp_chi,np.ones(100)[None,:]*this_Rp_chi[:,None]),axis=1)

        return chi, Rp_chi[:,1:]

    def get_Rpchi_m_complement(self,log_bin_centers,n_corr_samples,chi_bins,dm1,dq,m_min,m_max, q_min, q_max):
        diag_idx = np.where(log_bin_centers[:,0] == log_bin_centers[:,1])[0]
        ones = np.ones(len(dq))
        #ones[diag_idx]*=2.
        nbins_chi = len(chi_bins)-1
        Rp_chi,chi = np.zeros((len(n_corr_samples),1)),np.array([ ])

        for i in range(nbins_chi):
            idx_array = np.arange(len(log_bin_centers))
            bin_idx = idx_array[(~(((log_bin_centers[:,0]<=np.log(m_max))&
                        (log_bin_centers[:,0]>=np.log(m_min)))))&(log_bin_centers[:,1]>=q_min)&
                        (log_bin_centers[:,1]<=q_max)&
                        (log_bin_centers[:,2]>=chi_bins[i])&
                        (log_bin_centers[:,2]<=chi_bins[i+1])]
            this_Rp_chi = np.sum((n_corr_samples*dm1[None,:]*dq[None,:]*ones[None,:])[:,bin_idx],axis=-1)
            this_chi = np.linspace(chi_bins[i],chi_bins[i+1],100)
            chi=np.append(chi,this_chi)
            Rp_chi = np.concatenate((Rp_chi,np.ones(100)[None,:]*this_Rp_chi[:,None]),axis=1)

        return chi, Rp_chi[:,1:]



    def get_pearson_coeff_mass_range(self, n_corr, dm1, dq, dchi, log_bin_centers, m_min, m_max):
        '''
        Function for computing Pearson correlation coefficient 
        between effective spin and mass-ratio for a given m1 range
        
        Parameters
        ----------
        
        n_corr_samples          ::   numpy.ndarray
                                     array containing rate density in each bin
        
        dm1                     ::   numpy.ndarray
                                     1d array of delta log(m1)'s
                                     
        dq                      ::   numpy.ndarray
                                     1d array of delta q's
                                     
        dchi                    ::   numpy.ndarray
                                     1d array of delta chieff's
                              
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        m_min                   ::   float 
                                     lower edge of the m1 range
                                     
        m_max                   ::   float 
                                     upper edge of the m1 range
        
        
        Returns
        -------
        rho_m1_range    :   numpy.ndarray
                             1d array of Pearson correlation coefficient 
                             between effective spin and mass-ratio
                             for a given m1 range
        '''
        rho_m1_range = []
        
        nbins_m = len(self.mbins)-1
        nbins_q = len(self.qbins) - 1
        nbins_chi = len(self.chi_bins)-1
            
        m1_bin_centers = 0.5 * (self.mbins[1:] + self.mbins[:-1])
        qbin_centers = 0.5 * (self.qbins[1:] + self.qbins[:-1])
        chi_bin_centers = 0.5 * (self.chi_bins[1:] + self.chi_bins[:-1])
        
        q_low = self.qbins[0]
        q_high = self.qbins[-1]
        chi_low = self.chi_bins[0]
        chi_high = self.chi_bins[-1]
        idx_array = np.arange(len(log_bin_centers))
        bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m_min))&(log_bin_centers[:,0]<=np.log(m_max))&
               (log_bin_centers[:,1]>=q_low)&(log_bin_centers[:,1]<=q_high)&(log_bin_centers[:,2]>=chi_low)&(log_bin_centers[:,2]<=chi_high)]
        rate_density_array = n_corr[:,bin_idx]
        delta_ms = dm1[bin_idx]
        delta_qs = dq[bin_idx]
        delta_chis= dchi[bin_idx]
        print(rate_density_array.shape)
        
        log_m1_bin_centers = 0.5 * (np.log(self.mbins[1:]) + np.log(self.mbins[:-1]))
        mbin_idx = np.arange(len(log_m1_bin_centers))
        midx_arr = mbin_idx[(log_m1_bin_centers >= np.log(m_min))&(log_m1_bin_centers <= np.log(m_max))]
        
        mbin_chosen = log_m1_bin_centers[midx_arr]
        
        Q, CHI = np.meshgrid(qbin_centers, chi_bin_centers, indexing='ij')
        arg_mat = self.construct_arg_mat()
        rate_density_4d = np.apply_along_axis(lambda v: self.construct_1dtond_matrix(len(mbin_chosen), v,nbins_chi, nbins_q, m_min, m_max, arg_mat),axis=1,arr=rate_density_array * delta_ms * delta_qs * delta_chis)
        
        P_qchi = rate_density_4d.mean(axis=1)
        P_qchi /= P_qchi.sum(axis=(1,2), keepdims=True)   # (N, B, C)
        
        E_q      = np.sum(Q      * P_qchi, axis=(1,2))
        E_chi    = np.sum(CHI    * P_qchi, axis=(1,2))
        E_q2     = np.sum(Q**2   * P_qchi, axis=(1,2))
        E_chi2   = np.sum(CHI**2 * P_qchi, axis=(1,2))
        E_qchi   = np.sum(Q*CHI  * P_qchi, axis=(1,2))
        
        Var_q    = E_q2 - E_q**2
        Var_chi  = E_chi2 - E_chi**2
        Cov_qchi = E_qchi - E_q * E_chi
        
        rho_m1_range = Cov_qchi / np.sqrt(Var_q * Var_chi)
        
        return rho_m1_range


    def get_pearson_coeff_mass_marg(self, n_corr, dm1, dq, dchi, log_bin_centers):
        '''
        Function for computing Pearson correlation coefficient 
        between effective spin and mass-ratio marginalized 
        across m1 bins
        
        Parameters
        ----------
        
        n_corr_samples          ::   numpy.ndarray
                                     array containing rate density in each bin
        
        dm1                     ::   numpy.ndarray
                                     1d array of delta log(m1)'s
                                     
        dq                      ::   numpy.ndarray
                                     1d array of delta q's
                                     
        dchi                    ::   numpy.ndarray
                                     1d array of delta chieff's
                              
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        
        Returns
        -------
        rho_m1_full    :   numpy.ndarray
                             1d array of Pearson correlation coefficient 
                             between effective spin and mass-ratio
                             marginalized across m1 bins
        '''
        
        
        rho_m1_full = []
        
        nbins_m = len(self.mbins)-1
        nbins_q = len(self.qbins) - 1
        nbins_chi = len(self.chi_bins)-1
            
        m1_bin_centers = 0.5 * (self.mbins[1:] + self.mbins[:-1])
        qbin_centers = 0.5 * (self.qbins[1:] + self.qbins[:-1])
        chi_bin_centers = 0.5 * (self.chi_bins[1:] + self.chi_bins[:-1])

        m_min = self.mbins[0]
        m_max = self.mbins[-1]

        Q, CHI = np.meshgrid(qbin_centers, chi_bin_centers, indexing='ij')
        arg_mat = self.construct_arg_mat()
        rate_density_4d = np.apply_along_axis(lambda v: self.construct_1dtond_matrix(nbins_m,v,nbins_chi, nbins_q, m_min, m_max, arg_mat),axis=1,arr=n_corr * dm1 * dq * dchi)
        
        P_qchi = rate_density_4d.mean(axis=1)
        P_qchi /= P_qchi.sum(axis=(1,2), keepdims=True)   # (N, B, C)
        
        E_q      = np.sum(Q      * P_qchi, axis=(1,2))
        E_chi    = np.sum(CHI    * P_qchi, axis=(1,2))
        E_q2     = np.sum(Q**2   * P_qchi, axis=(1,2))
        E_chi2   = np.sum(CHI**2 * P_qchi, axis=(1,2))
        E_qchi   = np.sum(Q*CHI  * P_qchi, axis=(1,2))
        
        Var_q    = E_q2 - E_q**2
        Var_chi  = E_chi2 - E_chi**2
        Cov_qchi = E_qchi - E_q * E_chi
        
        rho_m1_full = Cov_qchi / np.sqrt(Var_q * Var_chi)
        
        return rho_m1_full

    

    def get_pearson_coeff_chi_range(self, n_corr, dm1, dq, dchi, log_bin_centers, chi_min, chi_max):
        '''
        Function for computing Pearson correlation coefficient 
        between primary mass and mass-ratio for a given chieff range

        Parameters
        ----------
        
        n_corr_samples          ::   numpy.ndarray
                                     array containing rate density in each bin
        
        dm1                     ::   numpy.ndarray
                                     1d array of delta log(m1)'s
                                     
        dq                      ::   numpy.ndarray
                                     1d array of delta q's
                                     
        dchi                    ::   numpy.ndarray
                                     1d array of delta chieff's
                              
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        chi_min                   ::   float 
                                     lower edge of the chieff range
                                     
        chi_max                   ::   float 
                                     upper edge of the chieff range
        
        
        Returns
        -------
        rho_chi_range    :   numpy.ndarray
                      	     1d array of Pearson correlation coefficient 
                      	     between primary mass and mass-ratio
                      	     for a given chieff range
        '''
        
        rho_chi_range = []
        
        nbins_m = len(self.mbins)-1
        nbins_q = len(self.qbins) - 1
        nbins_chi = len(self.chi_bins)-1
            
        m1_bin_centers = 0.5 * (self.mbins[1:] + self.mbins[:-1])
        qbin_centers = 0.5 * (self.qbins[1:] + self.qbins[:-1])
        chi_bin_centers = 0.5 * (self.chi_bins[1:] + self.chi_bins[:-1])
        
        m1_low = self.mbins[0]
        m1_high = self.mbins[-1]
        q_low = self.qbins[0]
        q_high = self.qbins[-1]
        idx_array = np.arange(len(log_bin_centers))
        bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m1_low))&(log_bin_centers[:,0]<=np.log(m1_high))&
               (log_bin_centers[:,1]>=q_low)&(log_bin_centers[:,1]<=q_high)&(log_bin_centers[:,2]>=chi_min)&(log_bin_centers[:,2]<=chi_max)]
        rate_density_array = n_corr[:,bin_idx]
        delta_ms = dm1[bin_idx]
        delta_qs = dq[bin_idx]
        delta_chis= dchi[bin_idx]
        print(rate_density_array.shape)

        chi_bin_idx = np.arange(len(chi_bin_centers))
        chi_idx_arr = chi_bin_idx[(chi_bin_centers >= chi_min)&(chi_bin_centers <= chi_max)]

        chi_bin_chosen = chi_bin_centers[chi_idx_arr]

        m_min = self.mbins[0]
        m_max = self.mbins[-1]

        M, Q = np.meshgrid(m1_bin_centers, qbin_centers, indexing='ij')
            
        arg_mat = self.construct_arg_mat()
        rate_density_4d = np.apply_along_axis(lambda v: self.construct_1dtond_matrix(nbins_m, v,len(chi_bin_chosen), nbins_q, m_min, m_max, arg_mat),axis=1,arr=rate_density_array * delta_ms * delta_qs * delta_chis)
        
        P_mq = rate_density_4d.mean(axis=-1)
        P_mq /= P_mq.sum(axis=(1,2), keepdims=True)   # (N, B, C)
        
        E_m      = np.sum(M      * P_mq, axis=(1,2))
        E_q    = np.sum(Q    * P_mq, axis=(1,2))
        E_m2     = np.sum(M**2   * P_mq, axis=(1,2))
        E_q2   = np.sum(Q**2 * P_mq, axis=(1,2))
        E_mq   = np.sum(M*Q  * P_mq, axis=(1,2))
        
        Var_m    = E_m2 - E_m**2
        Var_q  = E_q2 - E_q**2
        Cov_mq = E_mq - E_m * E_q
        
        rho_chi_range = Cov_mq / np.sqrt(Var_m * Var_q)
        
        return rho_chi_range


    def get_pearson_coeff_chi_marg(self, n_corr, dm1, dq, dchi, log_bin_centers):
        '''
        Function for computing Pearson correlation coefficient 
        between primary mass and mass-ratio 
        marginalized across chieff bins

        Parameters
        ----------
        
        n_corr_samples          ::   numpy.ndarray
                                     array containing rate density in each bin
        
        dm1                     ::   numpy.ndarray
                                     1d array of delta log(m1)'s
                                     
        dq                      ::   numpy.ndarray
                                     1d array of delta q's
                                     
        dchi                    ::   numpy.ndarray
                                     1d array of delta chieff's
                              
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        m_min                   ::   float 
                                     lower edge of the m1 range
                                     
        m_max                   ::   float 
                                     upper edge of the m1 range
        
        
        Returns
        -------
        rho_chi_full    :   numpy.ndarray
                      	     1d array of Pearson correlation coefficient 
                      	     between primary mass and mass-ratio
                      	     marginalized across chieff bins
        '''
        
        rho_chi_full = []
        
        nbins_m = len(self.mbins)-1
        nbins_q = len(self.qbins) - 1
        nbins_chi = len(self.chi_bins)-1
            
        m1_bin_centers = 0.5 * (self.mbins[1:] + self.mbins[:-1])
        qbin_centers = 0.5 * (self.qbins[1:] + self.qbins[:-1])
        chi_bin_centers = 0.5 * (self.chi_bins[1:] + self.chi_bins[:-1])

        m_min = self.mbins[0]
        m_max = self.mbins[-1]

        M, Q = np.meshgrid(m1_bin_centers, qbin_centers, indexing='ij')
            
        arg_mat = self.construct_arg_mat()
        rate_density_4d = np.apply_along_axis(lambda v: self.construct_1dtond_matrix(nbins_m, v,nbins_chi, nbins_q, m_min, m_max, arg_mat),axis=1,arr=n_corr * dm1 * dq * dchi)
        
        P_mq = rate_density_4d.mean(axis=-1)
        P_mq /= P_mq.sum(axis=(1,2), keepdims=True)   # (N, B, C)
        
        E_m      = np.sum(M      * P_mq, axis=(1,2))
        E_q    = np.sum(Q    * P_mq, axis=(1,2))
        E_m2     = np.sum(M**2   * P_mq, axis=(1,2))
        E_q2   = np.sum(Q**2 * P_mq, axis=(1,2))
        E_mq   = np.sum(M*Q  * P_mq, axis=(1,2))
        
        Var_m    = E_m2 - E_m**2
        Var_q  = E_q2 - E_q**2
        Cov_mq = E_mq - E_m * E_q
        
        rho_chi_range = Cov_mq / np.sqrt(Var_m * Var_q)
        
        return rho_chi_range


    def get_two_d_q_chi(self, n_corr, dm1, m_min, m_max, log_bin_centers):
        '''
        Function for computing 2d median merger rate density 
        between effective spin and mass-ratio for a given m1 range

        Parameters
        ----------
        
        n_corr_samples          ::   numpy.ndarray
                                     array containing rate density in each bin
        
        dm1                     ::   numpy.ndarray
                                     1d array of delta log(m1)'s
                              
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        m_min                   ::   float 
                                     lower edge of the m1 range
                                     
        m_max                   ::   float 
                                     upper edge of the m1 range
        
        
        Returns
        -------
        matrix1    :        numpy.ndarray
                      	     1d array of 2d median merger rate density 
                      	     between effective spin and mass-ratio
                      	     for a given m1 range
        '''
        
        nbins_m = len(self.mbins)-1
        nbins_q = len(self.qbins) - 1
        nbins_chi = len(self.chi_bins)-1
        q_low = self.qbins[0]
        q_high = self.qbins[-1]
        chi_low = self.chi_bins[0]
        chi_high = self.chi_bins[-1]
        idx_array = np.arange(len(log_bin_centers))
        bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m_min))&(log_bin_centers[:,0]<=np.log(m_max))&
               (log_bin_centers[:,1]>=q_low)&(log_bin_centers[:,1]<=q_high)&(log_bin_centers[:,2]>=chi_low)&(log_bin_centers[:,2]<=chi_high)]
        rate_density_array = n_corr[:,bin_idx]
        delta_ms = dm1[bin_idx]

        log_m1_bin_centers = 0.5 * (np.log(self.mbins[1:]) + np.log(self.mbins[:-1]))
        mbin_idx = np.arange(len(log_m1_bin_centers))
        midx_arr = mbin_idx[(log_m1_bin_centers >= np.log(m_min))&(log_m1_bin_centers <= np.log(m_max))]

        mbin_chosen = log_m1_bin_centers[midx_arr]

        p_avg_m1qchi = np.mean(rate_density_array, axis = 0)
        #print(np.where(p_avg_m1qchi * delta_ms == 0.0)[0])

        arg_mat = self.construct_arg_mat()
        p_avg = self.construct_1dtond_matrix(len(mbin_chosen), p_avg_m1qchi * delta_ms,nbins_chi, nbins_q,  m_min, m_max, arg_mat)
        #print(p_avg)
        matrix1 = np.sum(p_avg, axis = 0)   
        
        return matrix1


    def get_two_d_m_q(self, n_corr, dchi, chi_min, chi_max, log_bin_centers):
    	
        '''
        Function for computing 2d median merger rate density 
        between effective spin and mass-ratio for a given chieff range

        Parameters
        ----------
        
        n_corr_samples          ::   numpy.ndarray
                                     array containing rate density in each bin
        
        dchi                    ::   numpy.ndarray
                                     1d array of delta chieff's
                              
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        chi_min                   ::   float 
                                     lower edge of the chieff range
                                     
        chi_max                   ::   float 
                                     upper edge of the chieff range
        
        
        Returns
        -------
        matrix1    :        numpy.ndarray
                      	     1d array of 2d median merger rate density 
                      	     between primary mass and mass-ratio
                      	     for a given chieff range
        '''
        
        nbins_m = len(self.mbins)-1
        nbins_q = len(self.qbins) - 1
        nbins_chi = len(self.chi_bins)-1
        
        q_low = self.qbins[0]
        q_high = self.qbins[-1]
        m1_low = self.mbins[0]
        m1_high = self.mbins[-1]
        idx_array = np.arange(len(log_bin_centers))
        bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m1_low))&(log_bin_centers[:,0]<=np.log(m1_high))&
               (log_bin_centers[:,1]>=q_low)&(log_bin_centers[:,1]<=q_high)&(log_bin_centers[:,2]>=chi_min)&(log_bin_centers[:,2]<=chi_max)]
        rate_density_array = n_corr[:,bin_idx]
        delta_chis = dchi[bin_idx]

        chi_bin_centers = 0.5 * (self.chi_bins[1:]+ self.chi_bins[:-1])        
        chi_bin_idx = np.arange(len(chi_bin_centers))
        chi_idx_arr = chi_bin_idx[(chi_bin_centers >= chi_min)&(chi_bin_centers <= chi_max)]

        chi_bin_chosen = chi_bin_centers[chi_idx_arr]

        p_avg_m1qchi = np.mean(rate_density_array, axis = 0)
        #print(np.where(p_avg_m1qchi * delta_ms == 0.0)[0])
        
        p_avg = self.construct_1dtond_matrix(nbins_m, p_avg_m1qchi * delta_chis,nbins_chi=len(chi_bin_chosen), nbins_q = nbins_q)
        #print(p_avg)
        matrix1 = np.sum(p_avg, axis = 2)
        return matrix1

    def get_pm1qchi(self,n_corr,n_sum_bin, m1s,qs,chis,zs,tril_edges):
        '''
        Function for computing p(m1,m2,z) = dN/dm1dm2dz as afunction of
        m1,m2,z. Implements Eq.2 or Eq.8 of https://arxiv.org/pdf/2304.08046.pdf
        
        Parameters
        ----------
        
        n_corr                  ::   numpy.ndarray
                                     2d array containing rate density samples in each bin
                                     of shape (nsamples,nbins)
                                     
        m1s                     ::   numpy.ndarray
                                     1d array containing values of primary mass m1 at which to evalute p(m1,m2,z)
                                     
        m2s                     ::   numpy.ndarray
                                     1d array containing values of secondary mass m2 at which to evalute p(m1,m2,z)
        
        zs                      ::   numpy.ndarray
                                     1d array containing values of redshift z at which to evalute p(m1,m2,z)
        
        tril_edges              ::   numpy.ndarray
                                     array containing values of m1 bin edges in lower triangular format
                                     (output of Utils.tril_edges() function)
        
        Returns
        -------
        
        p_m1m2z   : numpy.ndarray
                    1d array containing p(m1,m2,z) evaluated at the supplied values of m1s, m2s and zs
        '''
        dl_values = Planck15.luminosity_distance(zs).to(u.Gpc).value
        arg_mat = self.construct_arg_mat()
        arg_mat_flat = np.matrix.flatten(arg_mat)
        args = np.where(arg_mat_flat > 0)[0]

        tril_edges = tril_edges[args]

        idx_array = np.arange(len(tril_edges))
        #print(len(tril_edges))
        bin_idx = [idx_array[(tril_edges[:,0,0]<=m1)&(tril_edges[:,1,0]>=m1)&
                   (tril_edges[:,0,1]<=q)&(tril_edges[:,1,1]>=q)&(tril_edges[:,0,2]<=chi)&(tril_edges[:,1,2]>=chi)] for m1,q,chi in zip(m1s,qs,chis)]
        #print(len(bin_idx))
        index_array = np.array([i for i, bi in enumerate(bin_idx) if len(bi)>0])
        bin_idx = np.array([bi[0] for bi in bin_idx  if len(bi)>0])
        #n_corr_at_idx = np.zeros((n_corr.shape[0],len(m1s)))
        n_corr_at_idx = np.zeros((len(m1s)))
    
        n_corr_at_idx[index_array] = n_corr[bin_idx]
        p_m1qchi = n_corr_at_idx/n_sum_bin * (Planck15.differential_comoving_volume(zs).to(u.Gpc**3/u.sr).value*(1+zs) ** (self.kappa - 1))/(m1s ** 2)/len(n_corr)
        return p_m1qchi
    
class Vt_Utils_spins_with_q(Utils_spins_with_q):    
    """
    Utilities for computing selection effects in GP 
    rate inference. Functions for computing the mean and
    std of the volume-time sensitivity during an observing 
    run given a set of simulated sources that were injected
    into detector noise realizations and then found above 
    threshold.
    """
    
    def __init__(self,mbins,qbins,chi_bins,kappa=2.7):
        '''
        Initialize post-processing utilities class.
        
        Parameters
        ----------
        
        mbins               :: numpy.ndarray 
                               1d array containing mass bin edges
        
        qbins :: numpy.ndarray 
                 1d array containing mass-ratio bin edges
        
        chi_bins :: numpy.ndarray
                 1d array containing effective spin bin edges.
        '''
        Utils_spins_with_q.__init__(self,mbins,qbins,chi_bins,kappa=kappa)
        self.arg_mat = construct_arg_mat_out_spins(mbins, qbins, chi_bins)

    def log_reweight_pinjection_mixture(self,m1, m2, z,s1x, s1y, s1z, s2x, s2y, s2z, pdraw, p_draw_chi_given_m1m2, mix_weights,log_p_s1s2):
        '''
        Function for re-weighting an injected event to the 
        binned population model. Evaluates the log of the quantity being
        summed over in Eq. A2 of https://arxiv.org/abs/2304.08046.
        
        Parameters
        ----------
        
        m1           ::  float
                         primary mass of the simulated event
        
        m2           ::  float
                         secondary mass of the simulated event
              
        z            ::  float
                         redshift of the simulated event
        
        s1x          ::  float
                         x-component of the spin of the heavier object 
                         of the simulated event
        
        s1y          ::  float
                         y-component of the spin of the heavier object 
                         of the simulated event
        
        s1z          ::  float
                         z-component of the spin of the heavier object 
                         of the simulated event
        
        s2x          ::  float
                         x-component of the spin of the lighter object 
                         of the simulated event
        
        s2y          ::  float
                         y-component of the spin of the lighter object 
                         of the simulated event
        
        s2z          ::  float
                         z-component of the spin of the lighter object 
                         of the simulated event
                     
        pdraw        ::  float
                         probability with which the simulated event
                         parameters were generated
        
        mix_weights  ::  float
                         mixture-weight associated with this event
                         in the scenario when multiple injection sets
                         are mixed together
        
        Returns
        -------
        
        tril_weights : numpy.ndarray
                       1d array of weights corresponding to each bin in the format
                       of the output of Utils.arraynd_to_tril
                       
        '''
        
        nbins = (len(self.mbins)-1)*(len(self.qbins)-1)*(len(self.chi_bins)-1)
        tril_weights = np.zeros(int(np.sum(self.arg_mat)))
        chi_eff = (m1*s1z+m2*s2z)/(m1+m2)
        q = m2/m1
        if (m1<self.mbins[0])|(q<self.qbins[0])|(m1>self.mbins[-1])|(q>self.qbins[-1])|(chi_eff<self.chi_bins[0])|(chi_eff>self.chi_bins[-1]):
                return tril_weights
        weights = np.zeros([len(self.mbins)-1,len(self.qbins)-1,len(self.chi_bins)-1])    
        m1_idx = np.clip(np.searchsorted(self.mbins,m1,side='right') - 1,a_min=0,a_max=len(self.mbins)-2)
        q_idx = np.clip(np.searchsorted(self.qbins,q,side='right') - 1,a_min=0,a_max=len(self.qbins)-2)
        chi_idx = np.clip(np.searchsorted(self.chi_bins,chi_eff,side='right') - 1,a_min=0,a_max=len(self.chi_bins)-2)
        log_dVdz = np.log(4*np.pi) + np.log(Planck15.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value)
        log_time_dilation = (self.kappa-1.)*np.log1p(z)
        if not log_p_s1s2:
            log_p_s1s2 = log_prob_spin(s1x,s1y,s1z,m1) + log_prob_spin(s2x,s2y,s2z,m2)
        weights[m1_idx,q_idx,chi_idx] = np.log(mix_weights) + log_dVdz  + log_time_dilation +log_p_s1s2 -np.log(p_draw_chi_given_m1m2) - np.log(pdraw) - 2 * np.log(m1)  
        tril_weights = self.arraynd_to_tril(weights, self.arg_mat)

        return tril_weights
    
    def compute_VTs(self,inj_data_set,thresh,key = 'optimal_snr_net',log_p_s1s2=None ):
        '''
        Function that implements calculation of mean 
        and std of emperically estimated volume-time sensitivity.
        
        Parameters
        ----------
        
        inj_data_set   ::   dict
                            a dictionary containing 1d numpy arrays of
                            masses, redshifts, sping, sampling_pdfs, ranking statistic,
                            analysis time, and the total number of injections generated.
        
        thresh         ::   float
                            value of the rankingstatistic threshold. Should match the threshold 
                            value used to select real events used in the analysis.
        
        key            ::   str
                            the key in inj_data_set that corresponds to the rankingstatistic
        
        
        Returns
        -------
        
        vt_means   :    numpy.ndarray
                        1d array containing the mean of the emperically estimated 
                        time volume sensitivity in each bin
        
        vt_sigmas  :    numpy.ndarray
                        1d array containing the std of the emperically estimated 
                        time volume sensitivity in each bin.
                            
        '''
        if type(key) == list:
            assert type(thresh)==list and len(thresh)==len(key)
            
            selector=np.where(np.sum(np.array([inj_data_set[k]>=th for k,th in zip(key,thresh)]),axis=0))[0]
        else:
            selector=np.where(inj_data_set[key]>=thresh)[0]
        
        if log_p_s1s2 is None:
            log_p_s1s2 = np.zeros(len(inj_data_set['mass2_source'])).astype(bool)
        

        n = len(selector)
        mean_weights = np.zeros(len(self.generate_log_bin_centers()))
        var_weights = np.zeros(len(self.generate_log_bin_centers()))
        
        for k, i in enumerate(tqdm.tqdm(selector, total=n)):
            x = self.log_reweight_pinjection_mixture(
                inj_data_set['mass1_source'][i],
                inj_data_set['mass2_source'][i],
                inj_data_set['redshift'][i],
                inj_data_set['spin1x'][i],
                inj_data_set['spin1y'][i],
                inj_data_set['spin1z'][i],
                inj_data_set['spin2x'][i],
                inj_data_set['spin2y'][i],
                inj_data_set['spin2z'][i],
                inj_data_set['sampling_pdf'][i],
                inj_data_set['p_draw_chi_given_m1m2'][i],
                inj_data_set['mixture_weight'][i],
                log_p_s1s2[i],
            )
            
            mean_weights += np.where((x!=0), np.exp(x), 0 )
            var_weights += np.where((x!=0), np.exp(x), 0 )**2
            
        
        vt_means = mean_weights * (inj_data_set['analysis_time_s']/(365.25*24*3600))/inj_data_set['total_generated'] 

        vt_vars = var_weights * (inj_data_set['analysis_time_s']/(365.25*24*3600))**2/inj_data_set['total_generated']**2 - vt_means**2/inj_data_set['total_generated'] 
        vt_sigmas = np.sqrt(vt_vars)
        
        return vt_means, vt_sigmas


    
class Rates_spins_with_q(Utils_spins_with_q):
    """
    Perform GP Rate inference using PyMC. Contains functions
    that create pymc models to sample the posterior distribution
    of rate densities in each bin.
    """
    def __init__(self, mbins,qbins,chi_bins,kappa=2.7):
        '''
        Parameters
        ----------
        
        mbins :: numpy.ndarray 
                 1d array containing mass bin edges
        
        qbins :: numpy.ndarray 
                 1d array containing mass-ratio bin edges
        
        chi_bins :: numpy.ndarray
                 1d array containing effective spin bin edges.
        '''
        Utils_spins_with_q.__init__(self,mbins,qbins,chi_bins,kappa=kappa) 
    

    def make_significant_model_3d_prior_only(self,log_bin_centers,weights,tril_vts,tril_deltaLogbins, ls_mean_m, ls_sd_m,ls_mean_q, ls_sd_q,ls_mean_chi, ls_sd_chi,sigma_sd=1.,mu_dim=None,vt_sigmas=None,vt_accuracy_check=False, wt_means=None, wt_sigmas=None):
        '''
        Function that creates a pymc model that will sample the prior
        for the correlated population model.
                
        Parameters
        ----------
        log_bin_centers                  ::    numpy.ndarray
                                               array containing centers of each bin in log m1, q, chi_eff co-ordinates.
                                               output of Utils.generate_log_bin_centers
        
        weights                          ::    numpy.ndarray
                                               array containing the posterior weights of each event in each bin (shape is 
                                               n_events,nbins).
        
        tril_vts                         ::    numpy.ndarray
                                               array containing mean values of emperically estimated VTs. First output of
                                               Vt_Utils_spins_with_q.compute_VTs 
        
        tril_deltaLogbins                ::    numpy.ndarray
                                               1d array containing delta_log_bin corresponding to each bin in the 
                                               lower triangular format of the output of Utils.arraynd_to_tril
                                               
        ls_mean_m                        ::    float
                                               mean of the mass axis of the lengthscale for the single GP.
                                               
        ls_sd_m                          ::    float
                                               std of the mass axis of the lengthscale for the single GP..
                                               
        ls_mean_q                        ::    float
                                               mean of the mass-ratio axis of the lengthscale for the single GP.
                                               
        ls_sd_q                          ::    float
                                               std of the mass-ratio axis of the lengthscale for the single GP.
        
        ls_mean_chi                        ::    float
                                               mean of the chi_eff axis of the lengthscale for the single GP.
                                               
        ls_sd_chi                          ::    float
                                               std of the chi_eff axis of the lengthscale for the single GP.
        
        sigma_sd                         ::    float
                                               std of the sigma for GP. Default is 10
        
        mu_dim                         ::    int
                                               number of mean functions for the GP. Can be 1
                                               or None. Default is None which corresponds to mu_dim = 
                                               number of bins.
        
        vt_sigmas                        ::    numpy.ndarray
                                               1d array containing std values of emperically estimated
                                               VTs. Second output of Vt_Utils_spins_with_q.compute_VTs. Default is 
                                               None (Should not be None if vt_accuracy_check=True)
        
        wt_means                        ::    numpy.ndarray
                                               array containing mean values of posterior weights.
                                               Second output of Utils_spins_with_q.compute_weights. Default is 
                                               None (shape is n_events,nbins).
                                               
        wt_sigmas                        ::    numpy.ndarray
                                               array containing std values of posterior weights.
                                               Third output of Utils_spins_with_q.compute_weights. Default is 
                                               None (shape is n_events,nbins).
        
        vt_accuracy_check                ::    bool
                                               Whether or not to implement marginalization of Monte 
                                               Carlo uncertainties in VT estimation. If True,
                                               samples from the posterior. If False 
                                               (default), samples from the posterior.
        
                                               
        
        Returns
        -------
        
        gp_model  : pymc.Model object.
                    model object for sampling the rate densities posterior.
        '''
        tril_vts = tril_vts*tril_deltaLogbins
        arg = tril_vts>0.
        if(len(np.where(~arg)[0])>0):
            tril_vts = tril_vts[np.where(arg)[0]]
            weights = weights[:,np.where(arg)[0]]
            wt_means = wt_means[:,np.where(arg)[0]]
            wt_sigmas = wt_sigmas[:,np.where(arg)[0]]
            weights/=np.sum(weights,axis=1).reshape(weights.shape[0],1)
        
        if vt_accuracy_check :
            assert vt_sigmas is not None
            vt_sigmas*=tril_deltaLogbins
            n_eff = tt.as_tensor(tril_vts**2/vt_sigmas[np.where(arg)[0]]**2)
        
        else:
            n_eff = 1
        
        if mu_dim is None:
            mu_dim=len(log_bin_centers)
        assert mu_dim==1 or mu_dim==len(log_bin_centers)
        
        nchi= len(self.chi_bins)-1
        # nm = len(self.mbins)-1
        # nq = len(self.qbins)-1
        nm = int(len(log_bin_centers)/(nchi))
        assert nm == len(log_bin_centers)/nchi
        #assert nm == len(log_bin_centers)/nchi
        bin_centers_chi = log_bin_centers[0:nchi,2][:,None]
        log_bin_centers_m = log_bin_centers[0::nchi, :2]
        with pm.Model() as gp_model:
            mu = pm.TruncatedNormal('mu', mu=0, sigma=10, lower=-8.0, upper=5.0, shape=mu_dim)
            sigma = pm.HalfNormal('sigma',sigma=sigma_sd)
            length_scale_m = pm.Lognormal('length_scale_m',mu=ls_mean_m,sigma=ls_sd_m)
            length_scale_q = pm.Lognormal('length_scale_q',mu=ls_mean_q,sigma=ls_sd_q)
            length_scale_chi = pm.Lognormal('length_scale_chi',mu=ls_mean_chi,sigma=ls_sd_chi)
            covariance_m = sigma * pm.gp.cov.ExpQuad(input_dim=2,ls=[length_scale_m, length_scale_q])
            covariance_chi = sigma * pm.gp.cov.ExpQuad(input_dim=1,ls=length_scale_chi)
            gp = pm.gp.LatentKron(cov_funcs=[covariance_m, covariance_chi]) 
            logn_corr = gp.prior('logn_corr',Xs=[log_bin_centers_m, bin_centers_chi])
            logn_tot = pm.Deterministic('logn_tot', mu+logn_corr)
            n_corr = pm.Deterministic('n_corr',tt.exp(logn_tot))
            n_corr_physical = pm.Deterministic('n_corr_physical',n_corr[arg])
            n_f_exp = n_corr_physical*tril_vts
            N_F_exp = pm.Deterministic('N_F_exp',tt.sum(n_f_exp*(1.-0.5*int(vt_accuracy_check)*n_f_exp/n_eff)))
            
            
        return gp_model
    
    def make_significant_model_3d_n_eff_opt(self,log_bin_centers,weights,tril_vts,tril_deltaLogbins, ls_mean_m, ls_sd_m,ls_mean_q, ls_sd_q,ls_mean_chi, ls_sd_chi,sigma_sd=1.,mu_dim=None,vt_sigmas=None, variance_cut=False, wt_means=None, wt_sigmas=None, exponent = -30, maximum_uncertainty = 1.0):
        '''
        Function that creates a pymc model that will sample the posterior 
        for the correlated population model, with an additional likelihood 
        penalty imposed to improve Monte Carlo convergence.
                
        Parameters
        ----------
        log_bin_centers                  ::    numpy.ndarray
                                               array containing centers of each bin in log m1, q, chi_eff co-ordinates.
                                               output of Utils.generate_log_bin_centers
        
        weights                          ::    numpy.ndarray
                                               array containing the posterior weights of each event in each bin (shape is 
                                               n_events,nbins).
        
        tril_vts                         ::    numpy.ndarray
                                               array containing mean values of emperically estimated VTs. First output of
                                               Vt_Utils_spins_with_q.compute_VTs 
        
        tril_deltaLogbins                ::    numpy.ndarray
                                               1d array containing delta_log_bin corresponding to each bin in the 
                                               lower triangular format of the output of Utils.arraynd_to_tril
                                               
        ls_mean_m                        ::    float
                                               mean of the mass axis of the lengthscale for the single GP.
                                               
        ls_sd_m                          ::    float
                                               std of the mass axis of the lengthscale for the single GP..
                                               
        ls_mean_q                        ::    float
                                               mean of the mass-ratio axis of the lengthscale for the single GP.
                                               
        ls_sd_q                          ::    float
                                               std of the mass-ratio axis of the lengthscale for the single GP.
        
        ls_mean_chi                        ::    float
                                               mean of the chi_eff axis of the lengthscale for the single GP.
                                               
        ls_sd_chi                          ::    float
                                               std of the chi_eff axis of the lengthscale for the single GP.
        
        sigma_sd                         ::    float
                                               std of the sigma for GP. Default is 10
        
        mu_dim                         ::    int
                                               number of mean functions for the GP. Can be 1
                                               or None. Default is None which corresponds to mu_dim = 
                                               number of bins.
        
        vt_sigmas                        ::    numpy.ndarray
                                               1d array containing std values of emperically estimated
                                               VTs. Second output of Vt_Utils_spins_with_q.compute_VTs. Default is 
                                               None (Should not be None if vt_accuracy_check=True)
        
        wt_means                        ::    numpy.ndarray
                                               array containing mean values of posterior weights.
                                               Second output of Utils_spins_with_q.compute_weights. Default is 
                                               None (shape is n_events,nbins).
                                               
        wt_sigmas                        ::    numpy.ndarray
                                               1d array containing std values of posterior weights.
                                               Third output of Utils_spins_with_q.compute_weights. Default is 
                                               None (shape is n_events,nbins).                                              
        
        variance_cut                     ::    bool
                                               Whether or not to implement variance cut for monitoring Monte Carlo uncertainties.

        maximum_uncertainty              ::    float
                                               The maximum variance acceptable
        
        exponent                         ::    int
        					 Exponent determining the steepness of
        					 the step function for likelihood penalty. Default is -30.
                                               
        
        Returns
        -------
        
        gp_model  : pymc.Model object.
                    model object for sampling the rate densities posterior.
        '''
        tril_vts = tril_vts*tril_deltaLogbins
        arg = tril_vts>0.
        if(len(np.where(~arg)[0])>0):
            tril_vts = tril_vts[np.where(arg)[0]]
            weights = weights[:,np.where(arg)[0]]
          
        
        if variance_cut :
            assert vt_sigmas is not None and wt_sigmas is not None and wt_means is not None
            wt_means = wt_means[:,np.where(arg)[0]]
            wt_sigmas = wt_sigmas[:,np.where(arg)[0]]
            vt_sigmas*=tril_deltaLogbins
            vt_sigmas = vt_sigmas[np.where(arg)[0]]
        
        else:
            wt_sigmas = np.zeros_like(weights)
            wt_means = weights
            vt_sigmas = np.zeros_like(tril_vts)
        
        if mu_dim is None:
            mu_dim=len(log_bin_centers)
        assert mu_dim==1 or mu_dim==len(log_bin_centers)
        
        nchi= len(self.chi_bins)-1
        nm = int(len(log_bin_centers)/(nchi))
        bin_centers_chi = log_bin_centers[0:nchi,2][:,None]
        log_bin_centers_m = log_bin_centers[0::nchi, :2]
        
        with pm.Model() as gp_model:
            mu = pm.TruncatedNormal('mu', mu=0, sigma=10, lower=-8.0, upper=5.0, shape=mu_dim)
            sigma = pm.HalfNormal('sigma',sigma=sigma_sd)
            length_scale_m = pm.Lognormal('length_scale_m',mu=ls_mean_m,sigma=ls_sd_m)
            length_scale_q = pm.Lognormal('length_scale_q',mu=ls_mean_q,sigma=ls_sd_q)
            length_scale_chi = pm.Lognormal('length_scale_chi',mu=ls_mean_chi,sigma=ls_sd_chi)
            covariance_m = sigma * pm.gp.cov.ExpQuad(input_dim=2,ls=[length_scale_m, length_scale_q])
            covariance_chi = sigma * pm.gp.cov.ExpQuad(input_dim=1,ls=length_scale_chi)
            gp = pm.gp.LatentKron(cov_funcs=[covariance_m, covariance_chi]) 
            logn_corr = gp.prior('logn_corr',Xs=[log_bin_centers_m, bin_centers_chi])
            logn_tot = pm.Deterministic('logn_tot', mu+logn_corr)
            n_corr = pm.Deterministic('n_corr',tt.exp(logn_tot))
            n_corr_physical = pm.Deterministic('n_corr_physical',n_corr[arg])
            n_f_exp = n_corr_physical*tril_vts
            N_F_exp = pm.Deterministic('N_F_exp',tt.sum(n_f_exp))
            log_l = pm.Potential('log_l',tt.sum(tt.log(tt.dot(weights,n_corr_physical))) - N_F_exp)
            
            
            #Variance due to PE samples
            numerator =  tt.sum((wt_sigmas*n_corr_physical)**2, axis = 1)
            denominator = tt.dot(wt_means, n_corr_physical)**2
            variance_pe = pm.Deterministic('var_pe', tt.sum(numerator/denominator))

            #Variance due to selection samples
            variance_selection = pm.Deterministic("var_n_det", tt.sum((n_corr_physical * vt_sigmas)**2))

            #log likelihood variance
            variance_log_l = pm.Deterministic("var_log_L", variance_pe+variance_selection+1e-10)

            #Penalty
            variance_penalty = pm.Potential('variance_cut', - int(variance_cut) * tt.log1p((maximum_uncertainty**2/variance_log_l) ** (exponent))) 
            
        return gp_model

    

