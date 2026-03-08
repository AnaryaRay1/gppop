#!/usr/bin/env python
__author__="Anarya Ray <anarya.ray@ligo.org>; Omkar Sridhar <omkar.sridhar@ligo.org>; Siddharth Mohite <siddharth.mohite@ligo.org>"


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

class Utils_spins():
    """
    Utilities for GP rate inference. Contains 
    functions for binning up the m1,m2,z or m1,m2 only
    parameter spaces, and computing various attributes 
    of bins and posterior weights in bins.
    """
    
    def __init__(self,mbins,chi_bins,kappa=2.7):
        '''
        Initialize utilities class.
        
        Parameters
        ----------
        
        mbins :: numpy.ndarray 
                 1d array containing mass bin edges
        
        chi_bins :: numpy.ndarray
                 1d array containing effective spin bin edges.

        kappa   :: float
                   redshift evolution of the merger rate.
        
        '''
        self.mbins = mbins
        self.chi_bins = chi_bins
        self.kappa=kappa
    
    def arraynd_to_tril(self,arr):
        '''
        Function that returns the set of lower-triangular
        entries (m2<=m1) of a collection of 2d matrices
        each binned by m1 and m2. For the m1,m2,z inference,
        it returns multiple sets of lower triangular (m2<=m1) 
        entries, one set corresponding to each effective spin bin.
        Uses numpy's tril_indices function.

        Parameters
        ----------
        arr :: numpy.ndarray
               Input 2d or 3d matrix.

        Returns
        -------
        lower_tri_array : numpy.ndarray
                          Array of lower-triangular entries.
        '''
        array = np.array([])
        for i in range(len(self.chi_bins)-1):
            lower_tri_indices = np.tril_indices(len(arr[:,:,i]))
            array=np.append(array,arr[:,:,i][lower_tri_indices])
        return array

    def compute_weights(self,samples,m1m2_given_z_prior=None,chi_prior=None,leftmost_chibin=None,full_prior=None,O4_prior = False):
        '''
        Function to compute the weights needed to reweight
        posterior samples to the population distribution,
        for an event in parameter bins. 

        Parameters
        ----------
        samples            :: numpy.ndarray
                              Array of m1,m2,z posterior samples.
                              
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
                  The weight matrix of shape(mbins,mbins) for m1,m2 only 
                  inference and of shape(mbins,mbins,chi_bins) for m1,m2,chi_eff
                  inference.
        '''
        weights = np.zeros([len(self.mbins)-1,len(self.mbins)-1,len(self.chi_bins)-1])
        wgt_means = np.zeros([len(self.mbins)-1,len(self.mbins)-1,len(self.chi_bins)-1])
        wgt_sigmas = np.zeros([len(self.mbins)-1,len(self.mbins)-1,len(self.chi_bins)-1])
        m1_samples = samples[:,0]
        m2_samples = samples[:,1]
        z_samples = samples[:,2]
        chi_samples = samples[:,3]
        #uniform in comoving-volume
        dl_values = Planck15.luminosity_distance(z_samples).to(u.Gpc).value
        m1_indices = np.clip(np.searchsorted(self.mbins,m1_samples,side='right') - 1,a_min=0,a_max=len(self.mbins)-2)
        m2_indices = np.clip(np.searchsorted(self.mbins,m2_samples,side='right') - 1,a_min=0,a_max=len(self.mbins)-2)
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
            pz_PE*=chi_prior
        else:
            pz_PE=full_prior
        pz_weight = pz_pop/pz_PE
        indices = zip(m1_indices,m2_indices,chi_indices)
        if leftmost_chibin is None:
            for i,inds in enumerate(indices):
                    weights[inds[0],inds[1],inds[2]] += pz_weight[i]/(m1_samples[i]*m2_samples[i])
                    wgt_means[inds[0],inds[1],inds[2]] += pz_weight[i]/(m1_samples[i]*m2_samples[i]) / len(samples)
        else:
            for i,inds in enumerate(indices):
                    weights[inds[0],inds[1],inds[2]-1] += float(inds[2]>0)*pz_weight[i]/(m1_samples[i]*m2_samples[i])
                    wgt_means[inds[0],inds[1],inds[2]-1] +=  float(inds[2]>0)*pz_weight[i]/(m1_samples[i]*m2_samples[i]) / len(samples)
        indices = zip(m1_indices,m2_indices,chi_indices)            
        if leftmost_chibin is None:
            for i,inds in enumerate(indices):
                    wgt_sigmas[inds[0],inds[1],inds[2]] += ((pz_weight[i]/(m1_samples[i] * m2_samples[i])) ** 2 / len(samples) ** 2 - wgt_means[inds[0],inds[1],inds[2]] ** 2 / len(samples) ** 2)
        else:
            for i,inds in enumerate(indices):
                    wgt_sigmas[inds[0],inds[1],inds[2]-1] +=  float(inds[2]>0)*((pz_weight[i]/(m1_samples[i] * m2_samples[i])) ** 2 / len(samples) ** 2 - wgt_means[inds[0],inds[1],inds[2]] ** 2 / len(samples) ** 2)
                    
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
        m2 = self.mbins
        chi = self.chi_bins
        deltaLogbin_array = np.ones([len(m1)-1,len(m2)-1,len(chi)-1])
        for k in range(len(chi)-1):
            for i in range(len(m1)-1):
                for j in range(len(m2)-1):
                    if j != i:
                        deltaLogbin_array[i,j,k] = np.log(m1[i+1]/m1[i])*np.log(m2[j+1]/m2[j])*(chi[k+1]-chi[k])
                    elif j==i:
                        deltaLogbin_array[i,i,k] = 0.5*np.log(m1[i+1]/m1[i])*np.log(m2[j+1]/m2[j])*(chi[k+1]-chi[k])
        return deltaLogbin_array
    
    def tril_edges(self):
        '''
        A function that returns the m1,m2,chi_eff edges of each bin
        in the form of the output of arraynd_to_tril()
        
        Returns
        -------
        edge_array : numpy.ndarray
                     an array containing upper and lower edges for each 
                     bin.
        '''
        m1 = self.mbins
        m2 = self.mbins
        chi = self.chi_bins
        edge_array = []
        for k in range(len(chi)-1):
            for i in range(len(m1)-1):
                for j in range(len(m2)-1):
                    if(m2[j]>m1[i]):
                        continue
                    edge_array.append([[m1[i],m2[j],chi[k]],[m1[i+1],m2[j+1],chi[k+1]]])
        return np.array(edge_array)

    def generate_log_bin_centers(self):
        '''
        Function that returns n-D bin centers in logm1,logm2,chi_eff space.

        Returns
        -------
        log_lower_tri_sorted : numpy.ndarray
                               n-D array of the  bin centers in logm space and
                               chi_eff bins in linear space.
        '''
        # zbins = np.log(self.zbins+1.0e-300)
        for k in range(len(self.chi_bins)-1):
            log_m1 = np.log(self.mbins)
            log_m2 = np.log(self.mbins)
            nbin = len(log_m1) - 1
            logm1_bin_centres = np.asarray([0.5*(log_m1[i+1]+log_m1[i])for i in range(nbin)])
            logm2_bin_centres = np.asarray([0.5*(log_m2[i+1]+log_m2[i])for i in range(nbin)])
            l1,l2 = np.meshgrid(logm1_bin_centres,logm2_bin_centres)
            l3 = np.array([[0.5*(self.chi_bins[k+1]+self.chi_bins[k])] for i in range(nbin*nbin)])
            logM = np.concatenate((l1.reshape([nbin*nbin,1]),l2.reshape([nbin*nbin,1]),l3),axis=1)
            logM_lower_tri = np.asarray([a for a in logM if a[1]<=a[0]])
            logM_lower_tri_sorted = np.asarray([logM_lower_tri[i] for i in np.argsort(logM_lower_tri[:,0],kind='mergesort')])
            if k == 0:
                log_lower_tri_sorted = logM_lower_tri_sorted
            else:
                log_lower_tri_sorted=np.append(log_lower_tri_sorted, logM_lower_tri_sorted,axis =0)
        return log_lower_tri_sorted
            
                
    def construct_1dtond_matrix(self,nbins_m,values,nbins_chi, tril=True):
        '''
        Inverse of arraynd_to_tril() Returns a n-D
        represenation matrix of a given set of the lower
        triangular 1-D values or multiple sets of lower 
        triangular 1D values, one set corresponding to 
        each effective spin bin.

        Parameters
        ----------
        values : numpy.ndarray
            1-D array of lower triangular entries.
        nbins_m : int
            number of mass bins
        nbins_chi : int
            number of effective spin bins
            
        Returns
        -------
        matrix : numpy.ndarray
            n-D symmetric array using values.
        '''
        k=0
        if len(values.shape)>1:
            matrix = np.zeros((nbins_m,nbins_m,nbins_chi)+values.shape[1:])
        else:
            matrix = np.zeros((nbins_m,nbins_m,nbins_chi))
        for l in range(nbins_chi):
            for i in range(nbins_m):
                for j in range(i+1 if tril else nbins_m ):
                    matrix[i,j,l] = values[k]
                    k+=1
            
        return matrix

    def delta_logm2s(self,mbins):
        '''
        A function that returns delta log(m2) for each bin in the
        lower triangular format of the output of arraynd_to_tril.
        
        Parameters
        ----------
        mbins : numpy array of mass bin edges
        
        Returns
        -------
        
        delta_logm2_array : numpy.ndarray
                        array of delta log(m2)'s
        '''
        delta_logm2_array = np.zeros([len(self.mbins)-1,len(self.mbins)-1,len(self.chi_bins)-1])
        for k in range(len(self.chi_bins)-1):
            for i in range(len(self.mbins)-1):
                for j in range(len(self.mbins)-1):
                    if j != i:
                        delta_logm2_array[i,j,k] = np.log(self.mbins[j+1]/self.mbins[j])
                    elif j==i:
                        delta_logm2_array[i,j,k] = 0.5*np.log(self.mbins[j+1]/self.mbins[j])
        return self.arraynd_to_tril(delta_logm2_array)
    
    def delta_logm1s(self,mbins):
        '''
        A function that returns delta log(m1) for each bin in the
        lower triangular format of the output of arraynd_to_tril.
        
        Parameters
        ----------
        mbins : numpy array of mass bin edges
        
        Returns
        -------
        
        delta_logm1_array : numpy.ndarray
                            1d array of delta log(m1)'s
        '''
        delta_logm1_array = np.zeros([len(self.mbins)-1,len(self.mbins)-1,len(self.chi_bins)-1])
        for k in range(len(self.chi_bins)-1):
            for i in range(len(self.mbins)-1):
                for j in range(len(self.mbins)-1):
                    if j != i:
                        delta_logm1_array[i,j,k] = np.log(self.mbins[i+1]/self.mbins[i])
                    elif j==i:
                        delta_logm1_array[i,j,k] = 0.5*np.log(self.mbins[i+1]/self.mbins[i])
        return self.arraynd_to_tril(delta_logm1_array)
    
    def delta_chis(self):
        '''
        A function that returns delta chi_eff for each bin in the
        lower triangular format of the output of arraynd_to_tril.
        
               
        Returns
        -------
        
        delta_logm2_array : numpy.ndarray
                        array of delta log(m2)'s
        '''
        delta_chi_array = np.zeros([len(self.mbins)-1,len(self.mbins)-1,len(self.chi_bins)-1])
        for k in range(len(self.chi_bins)-1):
            for i in range(len(self.mbins)-1):
                for j in range(len(self.mbins)-1):
                    delta_chi_array[i,j,k] = self.chi_bins[k+1]-self.chi_bins[k]
                    
        return self.arraynd_to_tril(delta_chi_array)
    
class Post_Proc_Utils_spins(Utils_spins):
    """
    Postprocessing Utilities for GP 
    rate inference. Functions for parsing
    samples of rate densities and computing
    marginal distributions.
    """
    
    def __init__(self,mbins, chi_bins,kappa=2.7):
        '''
        Initialize post-processing utilities class.
        
        Parameters
        ----------
        
        mbins :: numpy.ndarray 
                 1d array containing mass bin edges
        
        chi_bins :: numpy.ndarray
                 1d array containing effective spin bin edges.
        '''
        
        Utils_spins.__init__(self,mbins,chi_bins,kappa=kappa)
    
    def reshape_uncorr(self,n_corr,n_corr_chi):
        '''
        Function for combining uncorrelated mass and 
        redshift rate densities into combined rate
        densities (Eq. .
        
        Parameters
        ----------
        
        n_corr   :: numpy.ndarray
                    array containing rate-densities w.r.t. mass bins
        
        n_corr_chi :: numpy.ndarray
                    array containing rate densities w.r.t. redshift bins
                    
        
        Returns
        -------
        
        n_corr_all : numpy.ndarray
                     array containing combined rate densities
        
        '''
        n_corr_all = np.array([])
        for i in range(len(n_corr_chi)):
            n_corr_all = np.append(n_corr_all,n_corr*n_corr_chi[i])
        return n_corr_all
    
    def get_Rpm1(self,n_corr,delta_logm2_array,m1_bins,m2_bins,chi_bins,log_bin_centers):
        '''
        Function for computing marginal primary mass population: dR/dm1
        (obtained by integrating dR/dm1dm2dchi_eff over chi_eff and m2)
        
        Parameters
        ----------
        
        n_corr                  ::   numpy.ndarray
                                     array containing rate density in each bin
        
        delta_logm2_array       ::   numpy.ndarray
                                     1d array of delta log(m2)'s
        
        m1_bins                 ::   numpy.ndarray
                                     1d array containing primary mass bin edges
        
        m2_bins                 ::   numpy.ndarray
                                     1d array containing secondary mass bin edges
        
        chi_bins                ::   numpy.ndarray
                                     1d array containing chi_eff bin edges
        
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        
        
        Returns
        -------
        mass1     :   numpy.ndarray
                      1d array of primary masses at which dRdm1 is evaluated
        Rpm1      :   numpy.ndarray
                      1d array of dR/dm1 evaluated at the above m1 values
        
        '''
        Rpm1 = Rpm1 = np.zeros([len(n_corr),1])
        mass1 = np.array([])
        for i in range(len(m1_bins)-1):
            m1_low = m1_bins[i]
            m1_high = m1_bins[i+1]
            m2_low = m2_bins[0]
            m2_high = m2_bins[-1]
            chi_high = chi_bins[-1]
            chi_low = chi_bins[0]
            m_array = np.linspace(m1_low,m1_high,100)[:-1]
            idx_array = np.arange(len(log_bin_centers))
            bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m1_low))&(log_bin_centers[:,0]<=np.log(m1_high))&
                   (log_bin_centers[:,1]>=np.log(m2_low))&(log_bin_centers[:,1]<=np.log(m2_high))&(log_bin_centers[:,2]>=chi_low)&(log_bin_centers[:,2]<=chi_high)]
            rate_density_array = n_corr[:,bin_idx]
            delta_logm2s = delta_logm2_array[bin_idx]
            Rpm1 = np.concatenate((Rpm1,np.sum(rate_density_array*delta_logm2s[None,:],axis=1)[:,None]/(m_array[None,:])),axis=1)
            mass1 = np.append(mass1,m_array)
        return mass1,Rpm1[:,1:]
    
    def get_Rpm1_corr(self,n_corr,delta_logm2_array,delta_chi_array,m1_bins,m2_bins,log_bin_centers,chi_low,chi_high):
        '''
        Function for computing conditional primary mass population: p(m_1|chi_eff)
        evaluated at effective spin belonging to some range
        
        Parameters
        ----------
        
        n_corr                  ::   numpy.ndarray
                                     array containing rate density in each bin
        
        delta_logm2_array       ::   numpy.ndarray
                                     1d array of delta log(m2)'s
        
        m1_bins                 ::   numpy.ndarray
                                     1d array containing primary mass bin edges
        
        m2_bins                 ::   numpy.ndarray
                                     1d array containing secondary mass bin edges
                
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        chi_low                 ::   float 
                                     upper edge of chi_eff bin
                                     
        chi_low                 ::   float 
                                     upper edge of chi_eff bin
        
        
        Returns
        -------
        mass1     :   numpy.ndarray
                      1d array of primary masses at which p(m1|chi_eff) is evaluated
        Rpm1      :   numpy.ndarray
                      1d array of p(m1|chi_eff) evaluated at the above m1 values and
                      at effective spins belonging to a particular range
        
        '''
        Rpm1 = np.zeros([len(n_corr),1])
        mass1 = np.array([])
        for i in range(len(m1_bins)-1):
                m1_low = m1_bins[i]
                m1_high = m1_bins[i+1]
                m2_low = m2_bins[0]
                m2_high = m2_bins[-1]
                m_array = np.linspace(m1_low,m1_high,100)[:-1]
                idx_array = np.arange(len(log_bin_centers))
                bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m1_low))&(log_bin_centers[:,0]<=np.log(m1_high))&
                       (log_bin_centers[:,1]>=np.log(m2_low))&(log_bin_centers[:,1]<=np.log(m2_high))&(log_bin_centers[:,2]>=chi_low)&(log_bin_centers[:,2]<=chi_high)]
                rate_density_array = n_corr[:,bin_idx]
                delta_logm2s = delta_logm2_array[bin_idx]
                delta_chis= delta_chi_array[bin_idx]
                Rpm1 = np.concatenate((Rpm1,np.sum(rate_density_array*((delta_logm2s*delta_chis)[None,:]),axis=1)[:,None]/(m_array[None,:])),axis=1)
                mass1 = np.append(mass1,m_array)
        return mass1,Rpm1[:,1:]
        
    def get_Rpm2(self,n_corr,delta_logm1_array,m1_bins,m2_bins,chi_bins,log_bin_centers):
        '''
        Function for computing marginal primary mass population: dR/dm2
        (obtained by integrating dR/dm1dm2dchi_eff over chi_eff and m1)
        
        Parameters
        ----------
        
        n_corr                  ::   numpy.ndarray
                                     array containing rate density in each bin
        
        delta_logm2_array       ::   numpy.ndarray
                                     1d array of delta log(m2)'s
        
        m1_bins                 ::   numpy.ndarray
                                     1d array containing primary mass bin edges
        
        m2_bins                 ::   numpy.ndarray
                                     1d array containing secondary mass bin edges
        
        chi_bins                ::   numpy.ndarray
                                     1d array containing chi_eff bin edges
        
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        
        
        Returns
        -------
        mass1     :   numpy.ndarray
                      1d array of primary masses at which dR/dm2 is evaluated
        Rpm1      :   numpy.ndarray
                      1d array of dR/dm2 evaluated at the above m2 values
        
        '''
        Rpm1 = np.zeros([len(n_corr),1])
        mass1 = np.array([])
        for i in range(len(m2_bins)-1):
            m2_low = m2_bins[i]
            m2_high = m2_bins[i+1]
            m1_low = m1_bins[0]
            m1_high = m1_bins[-1]
            chi_high = chi_bins[-1]
            chi_low = chi_bins[0]
            m_array = np.linspace(m2_low,m2_high,100)[:-1]
            idx_array = np.arange(len(log_bin_centers))
            bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m1_low))&(log_bin_centers[:,0]<=np.log(m1_high))&
                   (log_bin_centers[:,1]>=np.log(m2_low))&(log_bin_centers[:,1]<=np.log(m2_high))&(log_bin_centers[:,2]>=chi_low)&(log_bin_centers[:,2]<=chi_high)]
            rate_density_array = n_corr[:,bin_idx]
            delta_logm1s = delta_logm1_array[bin_idx]
            Rpm1 = np.concatenate((Rpm1,np.sum(rate_density_array*delta_logm1s[None,:],axis=1)[:,None]/(m_array[None,:])),axis=1)
            #np.append(Rpm1,[np.sum(rate_density_array*delta_logm1s)/m for m in m_array])
            mass1 = np.append(mass1,m_array)
        return mass1,Rpm1[:,1:]
    
    def get_Rpm2_corr(self,n_corr,delta_logm1_array,delta_chi_array,m1_bins,m2_bins,log_bin_centers,chi_low,chi_high):
        '''
        Function for computing conditional primary mass population: p(m_2|chi_eff)
        evaluated at chi_eff belonging to some range
        
        Parameters
        ----------
        
        n_corr                  ::   numpy.ndarray
                                     array containing rate density in each bin
        
        delta_logm2_array       ::   numpy.ndarray
                                     1d array of delta log(m2)'s
        
        m1_bins                 ::   numpy.ndarray
                                     1d array containing primary mass bin edges
        
        m2_bins                 ::   numpy.ndarray
                                     1d array containing secondary mass bin edges
                
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        chi_low                 ::   float 
                                     lower edge of chi_eff bin
                                     
        chi_high                ::   float 
                                     upper edge of chi_eff bin
        
        
        Returns
        -------
        mass2     :   numpy.ndarray
                      1d array of primary masses at which p(m2|chi_eff) is evaluated
        Rpm2      :   numpy.ndarray
                      1d array of p(m2|chi_eff) evaluated at the above m1 values and
                      at effective spins belonging to a particular range
        
        '''
        Rpm2 = np.zeros([len(n_corr),1])
        mass2 = np.array([])
        for i in range(len(m2_bins)-1):
                m1_low = m1_bins[0]
                m1_high = m1_bins[-1]
                m2_low = m2_bins[i]
                m2_high = m2_bins[i+1]
                m_array = np.linspace(m2_low,m2_high,100)[:-1]
                idx_array = np.arange(len(log_bin_centers))
                bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m1_low))&(log_bin_centers[:,0]<=np.log(m1_high))&
                       (log_bin_centers[:,1]>=np.log(m2_low))&(log_bin_centers[:,1]<=np.log(m2_high))&(log_bin_centers[:,2]>=chi_low)&(log_bin_centers[:,2]<=chi_high)]
                rate_density_array = n_corr[:,bin_idx]
                delta_logm1s = delta_logm1_array[bin_idx]
                delta_chis= delta_chi_array[bin_idx]
                Rpm2 = np.concatenate((Rpm2,(np.sum(rate_density_array*((delta_logm1s*delta_chis)[None,:]),axis=1)[:,None]/(m_array[None,:]))),axis=1)
                mass2 = np.append(mass2,m_array)
        return mass2,Rpm2[:,1:]

    def get_Rpchi_q(self,log_bin_centers,n_corr_samples,chi_bins,dm1,dm2,q_min,q_max):
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
        ones = np.ones(len(dm2))
        ones[diag_idx]*=2.
        nbins_chi = len(chi_bins)-1
        Rp_chi,chi = np.zeros((len(n_corr_samples),1)),np.array([ ])
        for i in range(nbins_chi):
            idx_array = np.arange(len(log_bin_centers))
            bin_idx = idx_array[(log_bin_centers[:,1]<=log_bin_centers[:,0]+np.log(q_max))&(log_bin_centers[:,1]>log_bin_centers[:,0]+np.log(q_min))&(log_bin_centers[:,2]>=chi_bins[i])&(log_bin_centers[:,2]<=chi_bins[i+1])]
            this_Rp_chi = np.sum((n_corr_samples*dm1[None,:]*dm2[None,:]*ones[None,:])[:,bin_idx],axis=-1)
            
            this_chi = np.linspace(chi_bins[i],chi_bins[i+1],100)
            chi=np.append(chi,this_chi)
            Rp_chi = np.concatenate((Rp_chi,np.ones(100)[None,:]*this_Rp_chi[:,None]),axis=1)

        return chi, Rp_chi[:,1:]

    def get_Rpchi_m(self,log_bin_centers,n_corr_samples,chi_bins,dm1,dm2,m_min,m_max):
        '''
        Function for computing p(chi_eff|m1,m2) for either m1 or m2 values belonging in 
        some range.

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
        
        m_min                   ::   float 
                                     lower edge of the mass range
                                     
        m_max                   ::   float 
                                     upper edge of the mass range
        
        
        Returns
        -------
        chi       :   numpy.ndarray
                      1d array of chi_eff at which p(chi_eff|m1,m2) is evaluated
        Rpm2      :   numpy.ndarray
                      1d array of p(chi_eff|m1,m2) evaluated at the above chi_eff values
                      and at masses belonging to a particular range
        '''
        diag_idx = np.where(log_bin_centers[:,0] == log_bin_centers[:,1])[0]
        ones = np.ones(len(dm2))
        ones[diag_idx]*=2.
        nbins_chi = len(chi_bins)-1
        Rp_chi,chi = np.zeros((len(n_corr_samples),1)),np.array([ ])

        for i in range(nbins_chi):
            idx_array = np.arange(len(log_bin_centers))
            bin_idx = idx_array[(((log_bin_centers[:,0]<=np.log(m_max))&
                                (log_bin_centers[:,0]>=np.log(m_min)))|
                                 ((log_bin_centers[:,1]<=np.log(m_max))&
                                (log_bin_centers[:,1]>=np.log(m_min))))&
                                (log_bin_centers[:,2]>=chi_bins[i])&
                                (log_bin_centers[:,2]<=chi_bins[i+1])]
            this_Rp_chi = np.sum((n_corr_samples*dm1[None,:]*dm2[None,:]*ones[None,:])[:,bin_idx],axis=-1)
            this_chi = np.linspace(chi_bins[i],chi_bins[i+1],100)
            chi=np.append(chi,this_chi)
            Rp_chi = np.concatenate((Rp_chi,np.ones(100)[None,:]*this_Rp_chi[:,None]),axis=1)

        return chi, Rp_chi[:,1:]


    def get_Rpchi_m_both(self,log_bin_centers, n_corr_samples,chi_bins,dm1,dm2,m_min,m_max):
        '''
        Function for computing p(chi_eff|m1,m2) for m1 and m2 values belonging in 
        some range.

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
        
        m_min                   ::   float 
                                     lower edge of the mass range
                                     
        m_max                   ::   float 
                                     upper edge of the mass range
        
        
        Returns
        -------
        chi       :   numpy.ndarray
                      1d array of chi_eff at which p(chi_eff|m1,m2) is evaluated
        Rpm2      :   numpy.ndarray
                      1d array of p(chi_eff|m1,m2) evaluated at the above chi_eff values
                      and at masses belonging to a particular range
        '''
        diag_idx = np.where(log_bin_centers[:,0] == log_bin_centers[:,1])[0]
        ones = np.ones(len(dm2))
        ones[diag_idx]*=2.
        nbins_chi = len(chi_bins)-1
        Rp_chi,chi = np.zeros((len(n_corr_samples),1)),np.array([ ])

        for i in range(nbins_chi):
            idx_array = np.arange(len(log_bin_centers))
            bin_idx = idx_array[(((log_bin_centers[:,0]<=np.log(m_max))&
                                (log_bin_centers[:,0]>=np.log(m_min)))&
                                 ((log_bin_centers[:,1]<=np.log(m_max))&
                                (log_bin_centers[:,1]>=np.log(m_min))))&
                                (log_bin_centers[:,2]>=chi_bins[i])&
                                (log_bin_centers[:,2]<=chi_bins[i+1])]
            this_Rp_chi = np.sum((n_corr_samples*dm1[None,:]*dm2[None,:]*ones[None,:])[:,bin_idx],axis=-1)
            this_chi = np.linspace(chi_bins[i],chi_bins[i+1],100)
            chi=np.append(chi,this_chi)
            Rp_chi = np.concatenate((Rp_chi,np.ones(100)[None,:]*this_Rp_chi[:,None]),axis=1)

        return chi, Rp_chi[:,1:]

    def get_Rpchi_m_complement(self,log_bin_centers,n_corr_samples,chi_bins,dm1,dm2,m_min,m_max):
        '''
        Function for computing p(chi_eff|m1,m2) for both m1 and m2 values not within
        some range.

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
        
        m_min                   ::   float 
                                     lower edge of the mass range
                                     
        m_max                   ::   float 
                                     upper edge of the mass range
        
        
        Returns
        -------
        chi       :   numpy.ndarray
                      1d array of chi_eff at which p(chi_eff|m1,m2) is evaluated
        Rp_chi    :   numpy.ndarray
                      1d array of p(chi_eff|m1,m2) evaluated at the above chi_eff values
                      and at masses not belonging to a particular range
        '''
        diag_idx = np.where(log_bin_centers[:,0] == log_bin_centers[:,1])[0]
        ones = np.ones(len(dm2))
        ones[diag_idx]*=2.
        nbins_chi = len(chi_bins)-1
        Rp_chi,chi = np.zeros((len(n_corr_samples),1)),np.array([ ])

        for i in range(nbins_chi):
            idx_array = np.arange(len(log_bin_centers))
            bin_idx = idx_array[(~(((log_bin_centers[:,0]<=np.log(m_max))&
                        (log_bin_centers[:,0]>=np.log(m_min)))|
                         ((log_bin_centers[:,1]<=np.log(m_max))&
                        (log_bin_centers[:,1]>=np.log(m_min)))))&
                        (log_bin_centers[:,2]>=chi_bins[i])&
                        (log_bin_centers[:,2]<=chi_bins[i+1])]
            this_Rp_chi = np.sum((n_corr_samples*dm1[None,:]*dm2[None,:]*ones[None,:])[:,bin_idx],axis=-1)
            this_chi = np.linspace(chi_bins[i],chi_bins[i+1],100)
            chi=np.append(chi,this_chi)
            Rp_chi = np.concatenate((Rp_chi,np.ones(100)[None,:]*this_Rp_chi[:,None]),axis=1)

        return chi, Rp_chi[:,1:]

    def get_Rp_chi(self,chi_bins,mbins,dm1,dm2,n_corr_chi_samples,n_corr_samples,n_corr_m_samples,log_bin_centers):
        '''
        Function for computing p(chi_eff|m1,m2) for both m1 and m2 values not within
        some range.

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
        
        m_min                   ::   float 
                                     lower edge of the mass range
                                     
        m_max                   ::   float 
                                     upper edge of the mass range
        
        
        Returns
        -------
        chi       :   numpy.ndarray
                      1d array of chi_eff at which p(chi_eff|m1,m2) is evaluated
        Rpm2      :   numpy.ndarray
                      1d array of p(chi_eff|m1,m2) evaluated at the above chi_eff values
                      and at masses not belonging to a particular range
        '''
        nbins_m = int(len(mbins)*(len(mbins)-1)*0.5)
        n_corr_samples_m_only = np.zeros_like(n_corr_samples)
        n_corr_samples_m_only[:,:nbins_m] = n_corr_m_samples

        diag_idx = np.where(log_bin_centers[:,0] == log_bin_centers[:,1])[0]
        ones = np.ones(len(dm2))
        ones[diag_idx]*=2.
        nbins_chi = len(chi_bins)-1
        
        Rp_chi,chi = np.zeros((len(n_corr_samples),100)),np.array([ ])
        for i in range(nbins_chi):
            
            this_Rp_chi = n_corr_chi_samples[:,i]*np.sum(n_corr_samples_m_only*dm1*dm2*ones,axis=-1)
            
            this_chi = np.linspace(chi_bins[i],chi_bins[i+1],100)
            chi=np.append(chi,this_chi)
            Rp_chi = np.concatenate((Rp_chi,np.ones(100)[None,:]*this_Rp_chi[:,None]),axis=1)
    
        return chi, Rp_chi[:,100:]

    def get_pm1m2chi(self,n_corr,m1s,m2s,chis,zs,tril_edges,kappa=2.9):
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
                                     array containing values of m1 bin edges in lower 
                                     triangular format (output of Utils.tril_edges())
        
        Returns
        -------
        
        p_m1m2z   : numpy.ndarray
                    1d array containing p(m1,m2,z) evaluated at the supplied values of m1s, m2s and zs
        '''
        
        idx_array = np.arange(len(tril_edges))
        bin_idx = [idx_array[(tril_edges[:,0,0]<=m1)&(tril_edges[:,1,0]>=m1)&
                   (tril_edges[:,0,1]<=m2)&(tril_edges[:,1,1]>=m2)&(tril_edges[:,0,2]<=chi)&(tril_edges[:,1,2]>=chi)] for m1,m2,chi in zip(m1s,m2s,chis)]
       
        idx_array = np.array([(True if len(bi)>0 else False) for bi in bin_idx])
        bin_idx = np.array([bi[0] for bi in bin_idx  if len(bi)>0])
        n_corr_at_idx = np.zeros((n_corr.shape[0],len(m1s)))
        n_corr_at_idx[:,idx_array] = n_corr[:,bin_idx]
        p_m1m2chi = n_corr_at_idx * (Planck15.differential_comoving_volume(zs).to(u.Gpc**3/u.sr).value*(1+zs)**(kappa-1))/m1s/m2s
        return p_m1m2chi
    
class Vt_Utils_spins(Utils_spins):    
    """
    Utilities for computing selection effects in GP 
    rate inference. Functions for computing the mean and
    std of the volume-time sensitivity during an observing 
    run given a set of simulated sources that were injected
    into detector noise realizations and then found above 
    threshold.
    """
    
    def __init__(self,mbins,chi_bins,kappa=2.7):
        '''
        Initialize post-processing utilities class.
        
        Parameters
        ----------
        
        mbins               :: numpy.ndarray 
                               1d array containing mass bin edges
        
        chi_bins               :: numpy.ndarray
                               1d array containing chi_eff bin edges.
                            
        include_spins       :: bool
                               whether or not to reweight spin distributions
        '''
        Utils_spins.__init__(self,mbins,chi_bins,kappa=kappa)

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
        
        nbins = int(len(self.mbins)*(len(self.mbins)-1)/2)*(len(self.chi_bins)-1)
        tril_weights = np.zeros(nbins)
        chi_eff = (m1*s1z+m2*s2z)/(m1+m2)
        if (m1<self.mbins[0])|(m2<self.mbins[0])|(m1>self.mbins[-1])|(m2>self.mbins[-1])|(chi_eff<self.chi_bins[0])|(chi_eff>self.chi_bins[-1]):
                return tril_weights
        weights = np.zeros([len(self.mbins)-1,len(self.mbins)-1,len(self.chi_bins)-1])    
        m1_idx = np.clip(np.searchsorted(self.mbins,m1,side='right') - 1,a_min=0,a_max=len(self.mbins)-2)
        m2_idx = np.clip(np.searchsorted(self.mbins,m2,side='right') - 1,a_min=0,a_max=len(self.mbins)-2)
        chi_idx = np.clip(np.searchsorted(self.chi_bins,chi_eff,side='right') - 1,a_min=0,a_max=len(self.chi_bins)-2)
        log_dVdz = np.log(4*np.pi) + np.log(Planck15.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value)
        log_time_dilation = (self.kappa-1.)*np.log1p(z)
        if not log_p_s1s2:
            log_p_s1s2 = log_prob_spin(s1x,s1y,s1z,m1) + log_prob_spin(s2x,s2y,s2z,m2)
        weights[m1_idx,m2_idx,chi_idx] = np.log(mix_weights) + log_dVdz  + log_time_dilation +log_p_s1s2 -np.log(p_draw_chi_given_m1m2) - np.log(pdraw) - np.log(m1*m2)  
        tril_weights = self.arraynd_to_tril(weights)

        return tril_weights
    
    def compute_VTs(self,inj_data_set,thresh,key = 'optimal_snr_net',log_p_s1s2=None ):
        '''
        Function that implements Eqs. B7 and B8 of https://arxiv.org/abs/2304.08046
        to calculate mean and std of emperically estimated volume-time sensitivity.
        
        Parameters
        ----------
        
        inj_data_set   ::   dict
                            a dictionary containing 1d numpy arrays of
                            masses, effective spins, sping, sampling_pdfs, ranking statistic,
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
        log_pinjs = np.array(list(map(self.log_reweight_pinjection_mixture,inj_data_set['mass1_source'][selector],inj_data_set['mass2_source'][selector],inj_data_set['redshift'][selector], inj_data_set['spin1x'][selector],inj_data_set['spin1y'][selector],inj_data_set['spin1z'][selector], inj_data_set['spin2x'][selector],inj_data_set['spin2y'][selector], inj_data_set['spin2z'][selector],inj_data_set['sampling_pdf'][selector], inj_data_set['p_draw_chi_given_m1m2'][selector], inj_data_set['mixture_weight'][selector],log_p_s1s2[selector])))
        
        vt_means = np.sum(np.array(list(map(reweight_pinjection,log_pinjs))),axis=0)*(inj_data_set['analysis_time_s']/(365.25*24*3600))/inj_data_set['total_generated']
        
        vt_vars = np.sum(np.array(list(map(reweight_pinjection,log_pinjs)))**2,axis=0)*(inj_data_set['analysis_time_s']/(365.25*24*3600))**2/inj_data_set['total_generated']**2 - vt_means**2/inj_data_set['total_generated']
        
        vt_sigmas = np.sqrt(vt_vars)
        
        return vt_means, vt_sigmas
        
    
class Rates_spins(Utils_spins):
    """
    Perform GP Rate inference using PyMC. Contains functions
    that create pymc models to sample the posterior distribution
    of rate densities in each bin.
    """
    def __init__(self, mbins,chi_bins,kappa=2.7):
        '''
        Parameters
        ----------
        
        mbins :: numpy.ndarray 
                 1d array containing mass bin edges
        
        zbins :: numpy.ndarray
        '''
        Utils_spins.__init__(self,mbins,chi_bins,kappa=kappa)
        
    def make_significant_model_3d_evolution_only(self,log_bin_centers,weights,tril_vts,tril_deltaLogbins, ls_mean_m, ls_sd_m,ls_mean_chi, ls_sd_chi,sigma_sd=1.,mu_chi_dim=None, vt_sigmas=None,vt_accuracy_check=None):
        '''
        Function that creates a pymc model that will sample the posterior in 
        Eq. A6 (or B11 if vt_accuracy_check=True) of https://arxiv.org/abs/2304.08046
        for the un-correlated population model in Eq. 8 and the GP priors in Eqs. 9,10.
                
        Parameters
        ----------
        log_bin_centers                  ::    numpy.ndarray
                                               array containing centers of each bin in log m1, log m2, z co-ordinates.
                                               output of Utils.generate_log_bin_centers
        
        weights                          ::    numpy.ndarray
                                               array containing the posterior weights of each event in each bin (shape is 
                                               n_events,nbins). The weight for each ev
        
        tril_vts                         ::    numpy.ndarray
                                               array containing mean values of emperically estimated VTs. First output of
                                               Vt_Utils.compute_vts 
        
        tril_deltaLogbins                ::    numpy.ndarray
                                               1d array containing delta_log_bin corresponding to each bin in the 
                                               lower triangular format of the output of Utils.arraynd_to_tril
                                               
        ls_mean_m                        ::    float
                                               mean of the lengthscale for the GP corresponding to masses
                                               
        ls_sd_m                          ::    float
                                               std of the lengthscale for the GP corresponding to masses.
                                               
        ls_mean_chi                        ::    float
                                               mean of the lengthscale for the GP corresponding to effective spin
                                               
        ls_sd_chi                          ::    float
                                               std of the lengthscale for the GP corresponding to effective spin
        
        sigma_sd                         ::    float
                                               std of the sigma for GP corresponding to masses. Default is 1
        
        mu_chi_dim                         ::    int
                                               number of mean functions for the GP corresponding to effective spin. Can be 1
                                               or None. Default is None which corresponds to mu_dim = number of
                                               chi_eff bins.
        
        vt_sigmas                        ::    numpy.ndarray
                                               1d array containing std values of emperically estimated VTs. Second output of
                                               Vt_Utils.compute_vts. Default is None (Should not be None if vt_accuracy_check=True)
        
        vt_accuracy_check                ::    bool
                                               Whether or not to implement marginalization of Monte Carlo uncertainties in VT 
                                               estimation. If True, samples from the posterior on Eq. B11. If False (default),
                                               samples from the posterior in Eq. A6.
                                               
        
        Returns
        -------
        
        gp_model  : pymc.Model object.
                    model object for sampling the rate densities posterior.
        '''
        nchi= len(self.chi_bins)-1
        nm = int(len(log_bin_centers)/nchi)
        assert nm == len(log_bin_centers)/nchi
        chi_bin_centers = log_bin_centers[0::nm,2][:,None]
        logm_bin_centers = log_bin_centers[:nm,:2]
        vts = (tril_vts*tril_deltaLogbins).reshape((nchi,nm)).T
        N_ev = len(weights)
        weights = np.array([weights[i].reshape((nchi,nm)).T for i in range(len(weights))])
        
        if mu_chi_dim is None:
            mu_chi_dim=nchi
        assert mu_chi_dim ==1 or mu_chi_dim == nchi
        
        if vt_accuracy_check :
            assert vt_sigmas is not None
            vt_sigmas = (vt_sigmas*tril_deltaLogbins).reshape((nchi,nm)).T
        else:
            vt_sigmas = np.zeros((nchi,nm)).T
            
        with pm.Model() as gp_model:
            mu = pm.Normal('mu',mu=0,sigma=10,shape=nm)
            mu_chi = pm.Normal('mu_chi',mu=0,sigma=1,shape=mu_chi_dim)
            sigma = pm.HalfNormal('sigma',sigma=sigma_sd)
            sigma_chi = 1.
            length_scale = pm.Lognormal('length_scale_m',mu=ls_mean_m,sigma=ls_sd_m)
            length_scale_chi = pm.Lognormal('length_scale_chi',mu=ls_mean_chi,sigma=ls_sd_chi)
            covariance = sigma**2*pm.gp.cov.ExpQuad(input_dim=2,ls=length_scale)
            covariance_chi = sigma_chi**2*pm.gp.cov.ExpQuad(1,ls=length_scale_chi)
            gp = pm.gp.Latent(cov_func=covariance)
            gp_chi = pm.gp.Latent(cov_func=covariance_chi)
            logn_corr = gp.prior('logn_corr',X=logm_bin_centers)
            logn_corr_chi = gp_chi.prior('logn_corr_chi',X=chi_bin_centers)
            logn_tot = pm.Deterministic('logn_tot', mu+logn_corr)
            logn_tot_chi = pm.Deterministic('logn_tot_chi', mu_chi+logn_corr_chi)
            n_corr = pm.Deterministic('n_corr',tt.exp(logn_tot))
            n_corr_chi = pm.Deterministic('n_corr_chi',tt.exp(logn_tot_chi))
            N_F_exp = pm.Deterministic('N_F_exp',tt.sum(tt.exp(logn_tot+tt.log(tt.sum(vts*n_corr_chi,axis = 1)))))
            #N_F_exp_var = pm.Deterministic('N_F_exp_var', tt.sum(tt.exp(2.*logn_tot+ 2.*tt.log(tt.sum( vt_sigmas*n_corr_chi,axis = 1)))) if vt_accuracy_check else pm.math.constant(0.,dtype=float))
            log_l = pm.Potential('log_l',tt.sum(tt.log(tt.dot(tt.sum(weights*n_corr_chi,axis=2),n_corr))) - N_F_exp)#+0.5*N_F_exp_var)
            #n_eff_potential = pm.Potential('n_eff_potential', pm.math.switch(pm.math.le((tt.exp(logn_tot+tt.log(tt.sum(vts*n_corr_chi,axis = 1)))*(int(vt_accuracy_check))-2*tt.exp(2.*logn_tot+ 2.*tt.log(tt.sum( vt_sigmas*n_corr_chi,axis = 1)))).max(),0.),0.,-100))
            
        return gp_model
    
    
    def make_gp_prior_model_3d_evolution_only(self,log_bin_centers, ls_mean_m, ls_sd_m,ls_mean_chi, ls_sd_chi, sigma_sd=1.,mu_chi_dim=None):
        '''
        Function that creates a pymc model for sampling rate-densities
        from the GP priors in Eqs. 9,10.
                
        Parameters
        ----------
        log_bin_centers                  ::    numpy.ndarray
                                               array containing centers of each bin in log m1, log m2, z co-ordinates.
                                               output of Utils.generate_log_bin_centers
                                             
        ls_mean_m                        ::    float
                                               mean of the lengthscale for the GP corresponding to masses
                                               
        ls_sd_m                          ::    float
                                               std of the lengthscale for the GP corresponding to masses.
                                               
        ls_mean_chi                        ::    float
                                               mean of the lengthscale for the GP corresponding to effective spin
                                               
        ls_sd_chi                          ::    float
                                               std of the lengthscale for the GP corresponding to effective spin
        
        sigma_sd                         ::    float
                                               std of the sigma for GP corresponding to masses. Default is 1
        
        mu_chi_dim                         ::    int
                                               number of mean functions for the GP corresponding to effective spin. Can be 1
                                               or None. Default is None which corresponds to mu_dim = number of
                                               chi_eff bins.
        
        Returns
        -------
        
        gp_model  : pymc.Model object.
                    model object for sampling the rate densities prior.
        
        '''
        nz= len(self.zbins)-1
        nm = int(len(log_bin_centers)/nz)
        assert nm == len(log_bin_centers)/nz
        z_bin_centers = log_bin_centers[0::nm,2][:,None]
        logm_bin_centers = log_bin_centers[:nm,:2]
        if mu_z_dim is None:
            mu_z_dim=nz
        assert mu_z_dim ==1 or mu_z_dim == nz
        with pm.Model() as gp_model:
            mu = pm.Normal('mu',mu=0,sigma=10,shape=nm)
            mu_z = pm.Normal('mu_z',mu=0,sigma=1,shape=mu_z_dim)
            sigma = pm.HalfNormal('sigma',sigma=sigma_sd)
            sigma_z = 1.
            length_scale = pm.Lognormal('length_scale_m',mu=ls_mean_m,sigma=ls_sd_m)
            length_scale_z = pm.Lognormal('length_scale_z',mu=ls_mean_z,sigma=ls_sd_z)
            covariance = sigma**2*pm.gp.cov.ExpQuad(input_dim=2,ls=length_scale)
            covariance_z = sigma_z**2*pm.gp.cov.ExpQuad(1,ls=length_scale_z)
            gp = pm.gp.Latent(cov_func=covariance)
            gp_z = pm.gp.Latent(cov_func=covariance_z)
            logn_corr = gp.prior('logn_corr',X=logm_bin_centers)
            logn_corr_z = gp_z.prior('logn_corr_z',X=z_bin_centers)
            logn_tot = pm.Deterministic('logn_tot', mu+logn_corr)
            logn_tot_z = pm.Deterministic('logn_tot_z', mu_z+logn_corr_z)
            n_corr = pm.Deterministic('n_corr',tt.exp(logn_tot))
            n_corr_z = pm.Deterministic('n_corr_z',tt.exp(logn_tot_z))
            
        return gp_model
    

    def make_significant_model_3d_n_eff_opt(self,log_bin_centers,weights,tril_vts,tril_deltaLogbins, ls_mean_m, ls_sd_m,ls_mean_chi, ls_sd_chi,sigma_sd=1.,mu_dim=None,vt_sigmas=None,mc_convergence_check=True, wt_means=None, wt_sigmas=None, exponent = -30):
        '''
        Function that creates a pymc model that will sample the posterior 
        for the correlated population model, with an additional likelihood 
        penalty imposed to improve Monte Carlo convergence.
                
        Parameters
        ----------
        log_bin_centers                  ::    numpy.ndarray
                                               array containing centers of each bin in log m1, log m2, z co-ordinates.
                                               output of Utils.generate_log_bin_centers
        
        weights                          ::    numpy.ndarray
                                               array containing the posterior weights of each event in each bin (shape is 
                                               n_events,nbins).
        
        tril_vts                         ::    numpy.ndarray
                                               array containing mean values of emperically estimated VTs. First output of
                                               Vt_Utils.compute_vts 
        
        tril_deltaLogbins                ::    numpy.ndarray
                                               1d array containing delta_log_bin corresponding to each bin in the 
                                               lower triangular format of the output of Utils.arraynd_to_tril
                                               
        ls_mean_m                        ::    float
                                               mean of the mass axis of the lengthscale for the single GP.
                                               
        ls_sd_m                          ::    float
                                               std of the mass axis of the lengthscale for the single GP..
                                               
        ls_mean_chi                        ::    float
                                               mean of the lengthscale for the GP corresponding to effective spin
                                               
        ls_sd_chi                          ::    float
                                               std of the lengthscale for the GP corresponding to effective spin
        
        sigma_sd                         ::    float
                                               std of the sigma for GP corresponding to masses. Default is 1
        
        mu_chi_dim                         ::    int
                                               number of mean functions for the GP corresponding to effective spin. Can be 1
                                               or None. Default is None which corresponds to mu_dim = number of
                                               chi_eff bins.
        
        vt_sigmas                        ::    numpy.ndarray
                                               1d array containing std values of emperically estimated
                                               VTs. Second output of Vt_Utils.compute_vts. Default is 
                                               None
        
        wt_means                        ::    numpy.ndarray
                                               array containing mean values of posterior weights.
                                               Second output of Utils_spins.compute_weights. Default is 
                                               None (shape is n_events,nbins).
                                               
        wt_sigmas                        ::    numpy.ndarray
                                               1d array containing std values of posterior weights.
                                               Third output of Utils_spins.compute_weights. Default is 
                                               None (shape is n_events,nbins).                                              
        
        mc_convergence_check             ::    bool
                                               Whether or not to implement Monte Carlo
                                               convergence penalty. If True (default),
                                               implements the likelihood penalty. If False,
                                               implements the standard sampling procedure.
        
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
            wt_means = wt_means[:,np.where(arg)[0]]
            wt_sigmas = wt_sigmas[:,np.where(arg)[0]]
            weights/=np.sum(weights,axis=1).reshape(weights.shape[0],1)
        
        assert vt_sigmas is not None
        vt_sigmas*=tril_deltaLogbins
        n_eff = tt.as_tensor(tril_vts**2/vt_sigmas[np.where(arg)[0]]**2)
        
        if mu_dim is None:
            mu_dim=len(log_bin_centers)
        assert mu_dim==1 or mu_dim==len(log_bin_centers)
        
        nchi= len(self.chi_bins)-1
        nm = int(len(log_bin_centers)/nchi)
        assert nm == len(log_bin_centers)/nchi
        bin_centers_chi = log_bin_centers[0::nm,2][:,None]
        log_bin_centers_m = log_bin_centers[:nm,:2]
        with pm.Model() as gp_model:
            mu = pm.TruncatedNormal('mu', mu=0, sigma=10, lower=-8.0, upper=5.0, shape=mu_dim)
            sigma = pm.HalfNormal('sigma',sigma=sigma_sd)
            length_scale_m = pm.Lognormal('length_scale_m',mu=ls_mean_m,sigma=ls_sd_m)
            length_scale_chi = pm.Lognormal('length_scale_chi',mu=ls_mean_chi,sigma=ls_sd_chi)
            covariance_m = sigma*pm.gp.cov.ExpQuad(input_dim=2,ls=length_scale_m)
            covariance_chi = sigma*pm.gp.cov.ExpQuad(1,ls=length_scale_chi)
            gp = pm.gp.LatentKron(cov_funcs=[covariance_chi, covariance_m]) 
            logn_corr = gp.prior('logn_corr',Xs=[bin_centers_chi, log_bin_centers_m])
            logn_tot = pm.Deterministic('logn_tot', mu+logn_corr)
            n_corr = pm.Deterministic('n_corr',tt.exp(logn_tot))
            n_corr_physical = pm.Deterministic('n_corr_physical',n_corr[arg])
            n_f_exp = n_corr_physical*tril_vts
            N_F_exp = pm.Deterministic('N_F_exp',tt.sum(n_f_exp))
            log_l = pm.Potential('log_l',tt.sum(tt.log(tt.dot(weights,n_corr_physical))) - N_F_exp)
            Ndet =  pm.Deterministic('Ndet', tt.sum(n_f_exp, axis = -1))  #np.dot(n_corr_physical, tril_vts)
            denominator =  tt.sum((wt_sigmas*n_corr_physical)**2, axis = 1)
            numerator = tt.dot(wt_means, n_corr_physical)**2
            N_eff_samp = pm.Deterministic('N_eff_samp', tt.min(numerator/denominator))
            n_eff_potential = pm.Potential('n_eff_potential', (-tt.log1p((n_eff/(2 * Ndet)) ** (exponent)) -tt.log1p((tt.log10(N_eff_samp)/0.6) ** (exponent))) * int(mc_convergence_check))
            
        return gp_model
    
    def make_significant_model_3d_prior(self,log_bin_centers,tril_deltaLogbins, ls_mean_m, ls_sd_m,ls_mean_chi, ls_sd_chi,sigma_sd=1.,mu_dim=None):
        '''
        Function that creates a pymc model that will sample the posterior in 
        Eq. A6 (or B11 if vt_accuracy_check=True) of https://arxiv.org/abs/2304.08046
        for the correlated population model in Eq. 2 and the GP prior in Eq. 5.
                
        Parameters
        ----------
        log_bin_centers                  ::    numpy.ndarray
                                               array containing centers of each bin in log m1, log m2, z co-ordinates.
                                               output of Utils.generate_log_bin_centers
        
        weights                          ::    numpy.ndarray
                                               array containing the posterior weights of each event in each bin (shape is 
                                               n_events,nbins).
        
        tril_vts                         ::    numpy.ndarray
                                               array containing mean values of emperically estimated VTs. First output of
                                               Vt_Utils.compute_vts 
        
        tril_deltaLogbins                ::    numpy.ndarray
                                               1d array containing delta_log_bin corresponding to each bin in the 
                                               lower triangular format of the output of Utils.arraynd_to_tril
                                               
        ls_mean_m                        ::    float
                                               mean of the mass axis of the lengthscale for the single GP.
                                               
        ls_sd_m                          ::    float
                                               std of the mass axis of the lengthscale for the single GP..
                                               
        ls_mean_chi                        ::    float
                                               mean of the lengthscale for the GP corresponding to effective spin
                                               
        ls_sd_chi                          ::    float
                                               std of the lengthscale for the GP corresponding to effective spin
        
        sigma_sd                         ::    float
                                               std of the sigma for GP corresponding to masses. Default is 1
        
        mu_chi_dim                         ::    int
                                               number of mean functions for the GP corresponding to effective spin. Can be 1
                                               or None. Default is None which corresponds to mu_dim = number of
                                               chi_eff bins.
        
        vt_sigmas                        ::    numpy.ndarray
                                               1d array containing std values of emperically estimated
                                               VTs. Second output of Vt_Utils.compute_vts. Default is 
                                               None (Should not be None if vt_accuracy_check=True)
        
        vt_accuracy_check                ::    bool
                                               Whether or not to implement marginalization of Monte 
                                               Carlo uncertainties in VT estimation. If True,
                                               samples from the posterior on Eq. B11. If False 
                                               (default), samples from the posterior in Eq. A6.
        
                                               
        
        Returns
        -------
        
        gp_model  : pymc.Model object.
                    model object for sampling the rate densities posterior.
        '''
        
        if mu_dim is None:
            mu_dim=len(log_bin_centers)
        assert mu_dim==1 or mu_dim==len(log_bin_centers)
        
        nchi= len(self.chi_bins)-1
        nm = int(len(log_bin_centers)/nchi)
        assert nm == len(log_bin_centers)/nchi
        bin_centers_chi = log_bin_centers[0::nm,2][:,None]
        log_bin_centers_m = log_bin_centers[:nm,:2]
        with pm.Model() as gp_model:
            mu = pm.TruncatedNormal('mu', mu=0, sigma=10, lower=-8.0, upper=5.0, shape=mu_dim)
            sigma = pm.HalfNormal('sigma',sigma=sigma_sd)
            length_scale_m = pm.Lognormal('length_scale_m',mu=ls_mean_m,sigma=ls_sd_m)
            length_scale_chi = pm.Lognormal('length_scale_chi',mu=ls_mean_chi,sigma=ls_sd_chi)
            covariance_m = sigma*pm.gp.cov.ExpQuad(input_dim=2,ls=length_scale_m)
            covariance_chi = sigma*pm.gp.cov.ExpQuad(1,ls=length_scale_chi)
            gp = pm.gp.LatentKron(cov_funcs=[covariance_chi, covariance_m])
            Lt = pm.gp
            logn_corr = gp.prior('logn_corr',Xs=[bin_centers_chi, log_bin_centers_m])
            logn_tot = pm.Deterministic('logn_tot', mu+logn_corr)
            n_corr = pm.Deterministic('n_corr',tt.exp(logn_tot))
            
        return gp_model

    
