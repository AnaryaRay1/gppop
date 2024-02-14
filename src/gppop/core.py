#!/usr/bin/env python
__author__="Anarya Ray <anarya.ray@ligo.org>; Siddharth Mohite <siddharth.mohite@ligo.org>"


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
    in only one bin, the log weights (output of Vt_Utils.log_reweight_pinjection_mixture)
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

class Utils():
    """
    Utilities for GP rate inference. Contains 
    functions for binning up the m1,m2,z or m1,m2 only
    parameter spaces, and computing various attributes 
    of bins and posterior weights in bins.
    """
    
    def __init__(self,mbins,zbins):
        '''
        Initialize utilities class.
        
        Parameters
        ----------
        
        mbins :: numpy.ndarray 
                 1d array containing mass bin edges
        
        zbins :: numpy.ndarray
                 1d array containing redshift bin edges.
        
        '''
        self.mbins = mbins
        self.zbins = zbins
    
    def arraynd_to_tril(self,arr):
        '''
        Function that returns the set of lower-triangular
        entries (m2<=m1) of a collection of 2d matrices
        each binned by m1 and m2. For the m1,m2,z inference,
        it returns multiple sets of lower triangular (m2<=m1) 
        entries, one set corresponding to each redshift bin.
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
        for i in range(len(self.zbins)-1):
            lower_tri_indices = np.tril_indices(len(arr[:,:,i]))
            array=np.append(array,arr[:,:,i][lower_tri_indices])
        return array

    def compute_weights(self,samples,m1m2_given_z_prior=None):
        '''
        Function to compute the weights needed to reweight
        posterior samples to the population distribution,
        for an event in parameter bins. This implements Equation
        A.1 of https://arxiv.org/pdf/2304.08046.pdf.

        Parameters
        ----------
        samples            :: numpy.ndarray
                              Array of m1,m2,z posterior samples.
                              
        m1m2_given_z_prior :: numpy.ndarray
                              if default PE priors were not used then
                              the values of the p(m_1,m_2|z) function used
                              in PE need to be supplied corresponding to
                              each posterior sample.
        
        Returns
        -------
        weights : numpy.ndarray
                  The weight matrix of shape(mbins,mbins) for m1,m2 only 
                  inference and of shape(mbins,mbins,zbins) for m1,m2,z
                  inference.
        '''
        weights = np.zeros([len(self.mbins)-1,len(self.mbins)-1,len(self.zbins)-1])
        m1_samples = samples[:,0]
        m2_samples = samples[:,1]
        z_samples = samples[:,2]
        #uniform in comoving-volume
        dl_values = Planck15.luminosity_distance(z_samples).to(u.Gpc).value
        m1_indices = np.clip(np.searchsorted(self.mbins,m1_samples,side='right') - 1,a_min=0,a_max=len(self.mbins)-2)
        m2_indices = np.clip(np.searchsorted(self.mbins,m2_samples,side='right') - 1,a_min=0,a_max=len(self.mbins)-2)
        z_indices = np.clip(np.searchsorted(self.zbins,z_samples,side='right') - 1,a_min=0,a_max=len(self.zbins)-2)
        pz_pop = Planck15.differential_comoving_volume(z_samples).to(u.Gpc**3/u.sr).value/(1+z_samples)
        ddL_dz = dl_values/(1+z_samples) + (1+z_samples)*Planck15.hubble_distance.to(u.Gpc).value/Planck15.efunc(z_samples)#Jacobian to convert from dL to z 
        if m1m2_given_z_prior is None:
            pz_PE = (1+z_samples)**2 * dl_values**2 * ddL_dz # default PE prior - flat in det frame masses and dL**2 in distance
        else : 
            pz_PE = m1m2_given_z_prior * dl_values**2 * ddL_dz
        pz_weight = pz_pop/pz_PE
        indices = zip(m1_indices,m2_indices,z_indices)
        for i,inds in enumerate(indices):
                weights[inds[0],inds[1],inds[2]] += pz_weight[i]/(m1_samples[i]*m2_samples[i])
        weights /= sum(sum(sum(weights)))
        return weights

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
        z = self.zbins
        deltaLogbin_array = np.ones([len(m1)-1,len(m2)-1,len(z)-1])
        for k in range(len(z)-1):
            for i in range(len(m1)-1):
                for j in range(len(m2)-1):
                    if j != i:
                        deltaLogbin_array[i,j,k] = np.log(m1[i+1]/m1[i])*np.log(m2[j+1]/m2[j])*(z[k+1]-z[k])
                    elif j==i:
                        deltaLogbin_array[i,i,k] = 0.5*np.log(m1[i+1]/m1[i])*np.log(m2[j+1]/m2[j])*(z[k+1]-z[k])
        return deltaLogbin_array
    
    def tril_edges(self):
        '''
        A function that returns the m1,m2,z edges of each bin
        (or m1,m2 edges for mass-only inference) in the form of 
        the output of arraynd_to_tril()
        
        Returns
        -------
        edge_array : numpy.ndarray
                     an array containing upper and lower edges for each 
                     bin.
        '''
        m1 = self.mbins
        m2 = self.mbins
        z = self.zbins
        edge_array = []
        for k in range(len(z)-1):
            for i in range(len(m1)-1):
                for j in range(len(m2)-1):
                    if(m2[j]>m1[i]):
                        continue
                    edge_array.append([[m1[i],m2[j],z[k]],[m1[i+1],m2[j+1],z[k+1]]])
        return np.array(edge_array)

    def generate_log_bin_centers(self):
        '''
        Function that returns n-D bin centers in logm1,logm2,z space.

        Returns
        -------
        log_lower_tri_sorted : numpy.ndarray
                               n-D array of the  bin centers in logm1 space and
                               redshift bins in linear space.
        '''
        zbins = np.log(self.zbins+1.0e-300)
        for k in range(len(self.zbins)-1):
            log_m1 = np.log(self.mbins)
            log_m2 = np.log(self.mbins)
            nbin = len(log_m1) - 1
            logm1_bin_centres = np.asarray([0.5*(log_m1[i+1]+log_m1[i])for i in range(nbin)])
            logm2_bin_centres = np.asarray([0.5*(log_m2[i+1]+log_m2[i])for i in range(nbin)])
            l1,l2 = np.meshgrid(logm1_bin_centres,logm2_bin_centres)
            l3 = np.array([[0.5*(np.exp(zbins[k+1])+np.exp(zbins[k]))] for i in range(nbin*nbin)])
            logM = np.concatenate((l1.reshape([nbin*nbin,1]),l2.reshape([nbin*nbin,1]),l3),axis=1)
            logM_lower_tri = np.asarray([a for a in logM if a[1]<=a[0]])
            logM_lower_tri_sorted = np.asarray([logM_lower_tri[i] for i in np.argsort(logM_lower_tri[:,0],kind='mergesort')])
            if k == 0:
                log_lower_tri_sorted = logM_lower_tri_sorted
            else:
                log_lower_tri_sorted=np.append(log_lower_tri_sorted, logM_lower_tri_sorted,axis =0)
        return log_lower_tri_sorted
            
                
    def construct_1dtond_matrix(self,nbins_m,values,nbins_z, tril=True):
        '''
        Inverse of arraynd_to_tril() Returns a n-D
        represenation matrix of a given set of the lower
        triangular 1-D values or multiple sets of lower 
        triangular 1D values, one set corresponding to 
        each redshift bin.

        Parameters
        ----------
        values : numpy.ndarray
            1-D array of lower triangular entries.
        nbins_m : int
            number of mass bins
        nbins_z : int
            number of redshift bins
            
        Returns
        -------
        matrix : numpy.ndarray
            n-D symmetric array using values.
        '''
        k=0
        if len(values.shape)>1:
            matrix = np.zeros((nbins_m,nbins_m,nbins_z)+values.shape[1:])
        else:
            matrix = np.zeros((nbins_m,nbins_m,nbins_z))
        for l in range(nbins_z):
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
        delta_logm2_array = np.zeros([len(self.mbins)-1,len(self.mbins)-1,len(self.zbins)-1])
        for k in range(len(self.zbins)-1):
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
        delta_logm1_array = np.zeros([len(self.mbins)-1,len(self.mbins)-1,len(self.zbins)-1])
        for k in range(len(self.zbins)-1):
            for i in range(len(self.mbins)-1):
                for j in range(len(self.mbins)-1):
                    if j != i:
                        delta_logm1_array[i,j,k] = np.log(self.mbins[i+1]/self.mbins[i])
                    elif j==i:
                        delta_logm1_array[i,j,k] = 0.5*np.log(self.mbins[i+1]/self.mbins[i])
        return self.arraynd_to_tril(delta_logm1_array)
    
    def delta_Vc(self):
        '''
        A function that returns the comoving volume in source frame
        contained  within the redshift bin edges of each bin, in the
        lower triangular format of the output of arraynd_to_tril.
        
        
        
        Returns
        -------
        
        delta_logz_array : numpy.ndarray
                        1d array of comoving volumes
        '''
        delta_logz_array = np.zeros([len(self.mbins)-1,len(self.mbins)-1,len(self.zbins)-1])
        for k in range(len(self.zbins)-1):
            z_array = np.linspace(self.zbins[k],self.zbins[k+1],100)
            integrand = Planck15.differential_comoving_volume(z_array).to(u.Gpc**3/u.sr).value/(1+z_array)
            dz = np.trapz(integrand,z_array)
            for i in range(len(self.mbins)-1):
                for j in range(len(self.mbins)-1):

                    delta_logz_array[i,j,k] = dz

        return self.arraynd_to_tril(delta_logz_array)
    
class Post_Proc_Utils(Utils):
    """
    Postprocessing Utilities for GP 
    rate inference. Functions for parsing
    samples of rate densities and computing
    marginal distributions.
    """
    
    def __init__(self,mbins, zbins):
        '''
        Initialize post-processing utilities class.
        
        Parameters
        ----------
        
        mbins :: numpy.ndarray 
                 1d array containing mass bin edges
        
        zbins :: numpy.ndarray
                 1d array containing redshift bin edges.
        '''
        
        Utils.__init__(self,mbins,zbins)
    
    def reshape_uncorr(self,n_corr,n_corr_z):
        '''
        Function for combining uncorrelated mass and 
        redshift rate densities into combined rate
        densities (Eq. .
        
        Parameters
        ----------
        
        n_corr   :: numpy.ndarray
                    array containing rate-densities w.r.t. mass bins
        
        n_corr_z :: numpy.ndarray
                    array containing rate densities w.r.t. redshift bins
                    
        
        Returns
        -------
        
        n_corr_all : numpy.ndarray
                     array containing combined rate densities
        
        '''
        n_corr_all = np.array([])
        for i in range(len(n_corr_z)):
            n_corr_all = np.append(n_corr_all,n_corr*n_corr_z[i])
        return n_corr_all
    
    def get_Rpm1(self,n_corr,delta_logm2_array,m1_bins,m2_bins,zbins,log_bin_centers):
        '''
        Function for computing marginal primary mass population: dR/dm1
        (obtained by integrating dR/dm1dm2 over z and m2)
        
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
        
        zbins                   ::   numpy.ndarray
                                     1d array containing redshift bin edges
        
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        
        
        Returns
        -------
        mass1     :   numpy.ndarray
                      1d array of primary masses at which dRdm1 is evaluated
        Rpm1      :   numpy.ndarray
                      1d array of dR/dm1 evaluated at the above m1 values
        
        '''
        Rpm1 = np.array([])
        mass1 = np.array([])
        for i in range(len(m1_bins)-1):
            m1_low = m1_bins[i]
            m1_high = m1_bins[i+1]
            m2_low = m2_bins[0]
            m2_high = m2_bins[-1]
            z_high = zbins[-1]
            z_low = zbins[0]
            m_array = np.linspace(m1_low,m1_high,100)[:-1]
            idx_array = np.arange(len(log_bin_centers))
            bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m1_low))&(log_bin_centers[:,0]<=np.log(m1_high))&
                   (log_bin_centers[:,1]>=np.log(m2_low))&(log_bin_centers[:,1]<=np.log(m2_high))&(log_bin_centers[:,2]>=z_low)&(log_bin_centers[:,2]<=z_high)]
            rate_density_array = n_corr[bin_idx]
            delta_logm2s = delta_logm2_array[bin_idx]
            Rpm1 = np.append(Rpm1,[np.sum(rate_density_array*delta_logm2s)/m for m in m_array])
            mass1 = np.append(mass1,m_array)
        return mass1,Rpm1
    
    def get_Rpm1_corr(self,n_corr,delta_logm2_array,m1_bins,m2_bins,log_bin_centers,z_low,z_high):
        '''
        Function for computing conditional primary mass population: p(m_1|z)
        evaluated at redshifts belonging to some range
        
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
        
        z_low                   ::   float 
                                     upper edge of redshift bin
                                     
        z_low                   ::   float 
                                     upper edge of redshift bin
        
        
        Returns
        -------
        mass1     :   numpy.ndarray
                      1d array of primary masses at which p(m1|z) is evaluated
        Rpm1      :   numpy.ndarray
                      1d array of p(m1|z) evaluated at the above m1 values and
                      at redshifts belonging to a particular range
        
        '''
        Rpm1 = np.array([])
        mass1 = np.array([])
        for i in range(len(m1_bins)-1):
                m1_low = m1_bins[i]
                m1_high = m1_bins[i+1]
                m2_low = m2_bins[0]
                m2_high = m2_bins[-1]
                m_array = np.linspace(m1_low,m1_high,100)[:-1]
                idx_array = np.arange(len(log_bin_centers))
                bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m1_low))&(log_bin_centers[:,0]<=np.log(m1_high))&
                       (log_bin_centers[:,1]>=np.log(m2_low))&(log_bin_centers[:,1]<=np.log(m2_high))&(log_bin_centers[:,2]>=z_low)&(log_bin_centers[:,2]<=z_high)]
                rate_density_array = n_corr[bin_idx]
                delta_logm2s = delta_logm2_array[bin_idx]
                Rpm1 = np.append(Rpm1,[np.sum(rate_density_array*delta_logm2s)/m for m in m_array])
                mass1 = np.append(mass1,m_array)
        return mass1,Rpm1
        
    def get_Rpm2(self,n_corr,delta_logm1_array,m2_bins,zbins,log_bin_centers,m1_bounds = None):
        '''
        Function for computing marginal secondary mass population: dR/dm2
        (obtained by integrating dR/dm1dm2 over z and m1)
        
        Parameters
        ----------
        
        n_corr                  ::   numpy.ndarray
                                     array containing rate density in each bin
        
        delta_logm1_array       ::   numpy.ndarray
                                     1d array of delta log(m1)'s
              
        m2_bins                 ::   numpy.ndarray
                                     1d array containing secondary mass bin edges
        
        zbins                   ::   numpy.ndarray
                                     1d array containing redshift bin edges
        
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        
        
        Returns
        -------
        mass2     :   numpy.ndarray
                      1d array of secondary masses at which dRdm2 is evaluated
        Rpm2      :   numpy.ndarray
                      1d array of dR/dm2 evaluated at the above m2 values
        
        '''
        Rpm2 = np.array([])
        mass2 = np.array([])
        for i in range(len(m2_bins)-1):
            m2_low = m2_bins[i]
            m2_high = m2_bins[i+1]
            z_high = zbins[-1]
            z_low = zbins[0]
            m_array = np.linspace(m2_low,m2_high,100)[:-1]
            idx_array = np.arange(len(log_bin_centers))
            if m1_bounds is None:
                bin_idx = idx_array[(log_bin_centers[:,1]>=np.log(m2_low))&(log_bin_centers[:,1]<=np.log(m2_high))&(log_bin_centers[:,2]<=z_high)&(log_bin_centers[:,2]>=z_low)]
            else:
                bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m1_bounds[0]))&(log_bin_centers[:,0]<=np.log(m1_bounds[1]))&(log_bin_centers[:,1]>=np.log(m2_low))&(log_bin_centers[:,1]<=np.log(m2_high))&(log_bin_centers[:,2]<=z_high)&(log_bin_centers[:,2]>=z_low)]
            rate_density_array = n_corr[bin_idx]
            delta_logm1s = delta_logm1_array[bin_idx]
            Rpm2 = np.append(Rpm2,[np.sum(rate_density_array*delta_logm1s)/m for m in m_array])
            mass2 = np.append(mass2,m_array)
        return mass2,Rpm2
    
    def get_Rpm2_corr(self,n_corr,delta_logm1_array,m2_bins,log_bin_centers,z_low,z_high):
        '''
        Function for computing conditional secondary mass population: p(m_2|z)
        evaluated at redshifts belonging to some range
        
        Parameters
        ----------
        
        n_corr                  ::   numpy.ndarray
                                     array containing rate density in each bin
        
        delta_logm1_array       ::   numpy.ndarray
                                     1d array of delta log(m1)'s
        
        m1_bins                 ::   numpy.ndarray
                                     1d array containing primary mass bin edges
        
        m2_bins                 ::   numpy.ndarray
                                     1d array containing secondary mass bin edges
                
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        z_low                   ::   float 
                                     upper edge of redshift bin
                                     
        z_low                   ::   float 
                                     upper edge of redshift bin
        
        
        Returns
        -------
        mass2     :   numpy.ndarray
                      1d array of secondary masses at which p(m2|z) is evaluated
        Rpm2      :   numpy.ndarray
                      1d array of p(m2|z) evaluated at the above m2 values and
                      at redshifts belonging to a particular range
        
        '''
        Rpm2 = np.array([])
        mass2 = np.array([])
        for i in range(len(m2_bins)-1):
                m2_low = m2_bins[i]
                m2_high = m2_bins[i+1]
                m_array = np.linspace(m2_low,m2_high,100)[:-1]
                idx_array = np.arange(len(log_bin_centers))
                bin_idx = idx_array[(log_bin_centers[:,1]>=np.log(m2_low))&(log_bin_centers[:,1]<=np.log(m2_high))&(log_bin_centers[:,2]<=z_high)&(log_bin_centers[:,2]>=z_low)]
                rate_density_array = n_corr[bin_idx]
                delta_logm1s = delta_logm1_array[bin_idx]
                Rpm2 = np.append(Rpm2,[np.sum(rate_density_array*delta_logm1s)/m for m in m_array])
                mass2 = np.append(mass2,m_array)
        return mass2,Rpm2
    
    def get_Rz(self,n_corr,delta_logm2_array,delta_logm1_array,m1_bins,m2_bins,z_bins,log_bin_centers):
        '''
        Function for computing the redshift evolution of the merger rate R(z)
        (obtained by integrating dR/dm1dm2 over m1 and m2)
        
        Parameters
        ----------
        
        n_corr                  ::   numpy.ndarray
                                     array containing rate density in each bin
        
        delta_logm2_array       ::   numpy.ndarray
                                     1d array of delta log(m2)'s
        
        delta_logm1_array       ::   numpy.ndarray
                                     1d array of delta log(m1)'s
        
        m1_bins                 ::   numpy.ndarray
                                     1d array containing primary mass bin edges
              
        m2_bins                 ::   numpy.ndarray
                                     1d array containing secondary mass bin edges
        
        zbins                   ::   numpy.ndarray
                                     1d array containing redshift bin edges
        
        log_bin_centers         ::   numpy.ndarray
                                     array containing log of the centers of each bin
        
        
        
        Returns
        -------
        Z         :   numpy.ndarray
                      1d array of redshifts at which R(z) is evaluated
        R         :   numpy.ndarray
                      1d array of R(z) evaluated at the above z values
        
        '''
        Rz = np.array([])
        Z = np.array([])
        diag_idx = np.where(log_bin_centers[:,0] == log_bin_centers[:,1])[0]
        ones = np.ones(len(delta_logm2_array))
        ones[diag_idx]*=2.
        for i in range(len(z_bins)-1):
            z_low = z_bins[i]
            z_high = z_bins[i+1]
            m2_low = m2_bins[0]
            m2_high = m2_bins[-1]
            m1_low = m1_bins[0]
            m1_high = m1_bins[-1]
            idx_array = np.arange(len(log_bin_centers))
            z_array = np.linspace(z_low,z_high,100)
            bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m1_low))&(log_bin_centers[:,0]<=np.log(m1_high))&
                       (log_bin_centers[:,1]>=np.log(m2_low))&(log_bin_centers[:,1]<=np.log(m2_high))&(log_bin_centers[:,2]>=z_low)&(log_bin_centers[:,2]<=z_high)]
            rate_density_array = n_corr[bin_idx]
            delta_logm2s = (delta_logm2_array*ones)[bin_idx]
            delta_logm1s = delta_logm1_array[bin_idx]
            
            Rz = np.append(Rz,[np.sum(rate_density_array*delta_logm1s*delta_logm2s) for z in z_array])
            Z = np.append(Z, z_array)
        return Z,Rz
    
    def get_Rq(self,n_corr,qs,tril_edges,mmin=None,mmax=None):
        '''
        Function for computing the marginalized dR/dq
        
        Parameters
        ----------
        
        n_corr                  ::   numpy.ndarray
                                     1d array containing a rate density sample in each bin
                                     of shape (nbins,)
                                     
        Qs                      ::   numpy.ndarray
                                     1d array containing values of q at which to evalute dR/dq
                                     
        tril_edges              ::   numpy.ndarray
                                     array containing values of m1 bin edges in lower triangular format
                                     (output of Utils.tril_edges() function)
        
        mmin                    ::    float
                                      minimum value of m1 used when marginalizing over m1. Default is None in which case
                                      the lowest m1 bin edge is used
        
        mmax                    ::    float
                                      maximum value of m1 used when marginalizing over m1. Default is None in which case
                                      the highest m1 bin edge is used
        
        Returns
        -------
        
        dR/dq   : numpy.ndarray
                    1d array containing dR/dq evaluated at the supplied values of Qs
        '''
        
        mmin = min(self.mbins) if mmin is None else mmin
        mmax = max(self.mbins) if mmax is None else mmax
        ones = np.ones_like(qs)[:,np.newaxis]
        m1_max = np.minimum(tril_edges[np.newaxis,:,1,0]*ones,tril_edges[np.newaxis,:,1,1]/(qs[:,np.newaxis]))
        m1_min = np.maximum(tril_edges[np.newaxis,:,0,0]*ones,tril_edges[np.newaxis,:,0,1]/(qs[:,np.newaxis]))
        imask1 = m1_min<m1_max
        mask2 = np.where((tril_edges[np.newaxis,:,0,0]*ones>mmin)*(tril_edges[np.newaxis,:,1,0]*ones<mmax))
        n_corr_at_idx = np.repeat(n_corr[np.newaxis,:],len(qs),axis=0)
        n_corr_at_idx[imask1] *= np.log(m1_max[imask1])-np.log(m1_min[imask1])
        n_corr_at_idx[~imask1] = 0
        n_corr_at_idx[mask2] = 0
        return np.sum(n_corr_at_idx/(qs[:,np.newaxis]),axis=1)

    def get_pm1m2z(self,n_corr,m1s,m2s,zs,tril_edges):
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
        idx_array = np.arange(len(tril_edges))
        
        bin_idx = [idx_array[(tril_edges[:,0,0]<=m1)&(tril_edges[:,1,0]>=m1)&
                   (tril_edges[:,0,1]<=m2)&(tril_edges[:,1,1]>=m2)&(tril_edges[:,0,2]<=z)&(tril_edges[:,1,2]>=z)] for m1,m2,z in zip(m1s,m2s,zs)]
        
        sample_idx = np.array([(len(bi)>0)*(m1>=m2) for m1,m2,bi in zip(m1s,
                                                                        m2s,bin_idx)])
        bin_idx = np.array([bi[0] for m1,m2,
                            bi in zip(m1s,m2s,bin_idx)  if len(bi)>0 and m1>=m2])
        n_corr_at_idx = np.zeros((n_corr.shape[0],len(m1s)))
        n_corr_at_idx[:,sample_idx] = n_corr[:,bin_idx]
        p_m1m2z = n_corr_at_idx * (Planck15.differential_comoving_volume(zs).to(u.Gpc**3/u.sr).value/(1+zs))/m1s/m2s
        return p_m1m2z
    
class Vt_Utils(Utils):    
    """
    Utilities for computing selection effects in GP 
    rate inference. Functions for computing the mean and
    std of the volume-time sensitivity during an observing 
    run given a set of simulated sources that were injected
    into detector noise realizations and then found above 
    threshold.
    """
    
    def __init__(self,mbins,zbins,include_spins=True):
        '''
        Initialize post-processing utilities class.
        
        Parameters
        ----------
        
        mbins               :: numpy.ndarray 
                               1d array containing mass bin edges
        
        zbins               :: numpy.ndarray
                               1d array containing redshift bin edges.
                            
        include_spins       :: bool
                               whether or not to reweight spin distributions
        '''
        Utils.__init__(self,mbins,zbins)
        self.include_spins = include_spins

    def log_reweight_pinjection_mixture(self,m1, m2, z,s1x, s1y, s1z, s2x, s2y, s2z, pdraw, mix_weights):
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
        if(not self.include_spins):
            s1x=s2x=s1y=s2y=s1z=s2z=1.

        nbins = int(len(self.mbins)*(len(self.mbins)-1)/2)*(len(self.zbins)-1)
        tril_weights = np.zeros(nbins)
        if (m1<self.mbins[0])|(m2<self.mbins[0])|(m1>self.mbins[-1])|(m2>self.mbins[-1])|(z<self.zbins[0])|(z>self.zbins[-1]):
                return tril_weights
        weights = np.zeros([len(self.mbins)-1,len(self.mbins)-1,len(self.zbins)-1])    
        m1_idx = np.clip(np.searchsorted(self.mbins,m1,side='right') - 1,a_min=0,a_max=len(self.mbins)-2)
        m2_idx = np.clip(np.searchsorted(self.mbins,m2,side='right') - 1,a_min=0,a_max=len(self.mbins)-2)
        z_idx = np.clip(np.searchsorted(self.zbins,z,side='right') - 1,a_min=0,a_max=len(self.zbins)-2)
        log_dVdz = np.log(4*np.pi) + np.log(Planck15.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value)
        log_time_dilation = -np.log1p(z)
        log_p_s1s2 = log_prob_spin(s1x,s1y,s1z,m1) + log_prob_spin(s2x,s2y,s2z,m2)
        weights[m1_idx,m2_idx,z_idx] = np.log(mix_weights) + log_dVdz  + log_time_dilation + int(self.include_spins)*log_p_s1s2 - np.log(pdraw) - np.log(m1*m2)  
        tril_weights = self.arraynd_to_tril(weights)

        return tril_weights
    
    def compute_VTs(self,inj_data_set,thresh,key = 'optimal_snr_net' ):
        '''
        Function that implements Eqs. B7 and B8 of https://arxiv.org/abs/2304.08046
        to calculate mean and std of emperically estimated volume-time sensitivity.
        
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
        
        log_pinjs = np.array(list(map(self.log_reweight_pinjection_mixture,inj_data_set['mass1_source'][selector],inj_data_set['mass2_source'][selector],inj_data_set['redshift'][selector], inj_data_set['spin1x'][selector],inj_data_set['spin1y'][selector],inj_data_set['spin1z'][selector], inj_data_set['spin2x'][selector],inj_data_set['spin2y'][selector], inj_data_set['spin2z'][selector],inj_data_set['sampling_pdf'][selector], inj_data_set['mixture_weight'][selector])))
        
        vt_means = np.sum(np.array(list(map(reweight_pinjection,log_pinjs))),axis=0)*(inj_data_set['analysis_time_s']/(365.25*24*3600))/inj_data_set['total_generated']
        
        vt_vars = np.sum(np.array(list(map(reweight_pinjection,log_pinjs)))**2,axis=0)*(inj_data_set['analysis_time_s']/(365.25*24*3600))**2/inj_data_set['total_generated']**2 - vt_means**2/inj_data_set['total_generated']
        
        vt_sigmas = np.sqrt(vt_vars)
        
        return vt_means, vt_sigmas
        
        
class Rates(Utils):
    """
    Perform GP Rate inference using PyMC. Contains functions
    that create pymc models to sample the posterior distribution
    of rate densities in each bin.
    """
    def __init__(self, mbins,zbins):
        '''
        Parameters
        ----------
        
        mbins :: numpy.ndarray 
                 1d array containing mass bin edges
        
        zbins :: numpy.ndarray
        '''
        Utils.__init__(self,mbins,zbins)
        
    def make_significant_model_3d_evolution_only(self,log_bin_centers,weights,tril_vts,tril_deltaLogbins, ls_mean_m, ls_sd_m,ls_mean_z, ls_sd_z,sigma_sd=1.,mu_z_dim=None, vt_sigmas=None,vt_accuracy_check=None):
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
                                               
        ls_mean_z                        ::    float
                                               mean of the lengthscale for the GP corresponding to redshift
                                               
        ls_sd_z                          ::    float
                                               std of the lengthscale for the GP corresponding to redshift.
        
        sigma_sd                         ::    float
                                               std of the sigma for GP corresponding to masses. Default is 1
        
        mu_z_dim                         ::    int
                                               number of mean functions for the GP corresponding to redshift. Can be 1
                                               or None. Default is None which corresponds to mu_dim = number of
                                               redshift bins.
        
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
        nz= len(self.zbins)-1
        nm = int(len(log_bin_centers)/nz)
        assert nm == len(log_bin_centers)/nz
        z_bin_centers = log_bin_centers[0::nm,2][:,None]
        logm_bin_centers = log_bin_centers[:nm,:2]
        vts = (tril_vts*tril_deltaLogbins).reshape((nz,nm)).T
        N_ev = len(weights)
        weights = np.array([weights[i].reshape((nz,nm)).T for i in range(len(weights))])
        
        if mu_z_dim is None:
            mu_z_dim=nz
        assert mu_z_dim ==1 or mu_z_dim == nz
        
        if vt_accuracy_check :
            assert vt_sigmas is not None
            vt_sigmas = (vt_sigmas*tril_deltaLogbins).reshape((nz,nm)).T
        else:
            vt_sigmas = np.zeros((nz,nm)).T
            
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
            N_F_exp = pm.Deterministic('N_F_exp',tt.sum(tt.exp(logn_tot+tt.log(tt.sum(vts*n_corr_z,axis = 1)))))
            N_F_exp_var = pm.Deterministic('N_F_exp_var', tt.sum(tt.exp(2.*logn_tot+ 2.*tt.log(tt.sum( vt_sigmas*n_corr_z,axis = 1)))) if vt_accuracy_check else pm.math.constant(0.,dtype=float))
            log_l = pm.Potential('log_l',tt.sum(tt.log(tt.dot(tt.sum(weights*n_corr_z,axis=2),n_corr))) - N_F_exp+0.5*N_F_exp_var)
            n_eff_potential = pm.Potential('n_eff_potential', pm.math.switch(pm.math.le((tt.exp(logn_tot+tt.log(tt.sum(vts*n_corr_z,axis = 1)))*(int(vt_accuracy_check))-2*tt.exp(2.*logn_tot+ 2.*tt.log(tt.sum( vt_sigmas*n_corr_z,axis = 1)))).max(),0.),0.,-100))
            
        return gp_model
    
    
    def make_gp_prior_model_3d_evolution_only(self,log_bin_centers, ls_mean_m, ls_sd_m,ls_mean_z, ls_sd_z, sigma_sd=1.,mu_z_dim=None):
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
                                               
        ls_mean_z                        ::    float
                                               mean of the lengthscale for the GP corresponding to redshift
                                               
        ls_sd_z                          ::    float
                                               std of the lengthscale for the GP corresponding to redshift.
        
        sigma_sd                         ::    float
                                               std of the sigma for GP corresponding to masses. Default is 1
        
        mu_z_dim                         ::    int
                                               number of mean functions for the GP corresponding to redshift. Can be 1
                                               or None. Default is None which corresponds to mu_dim = number of
                                               redshift bins.
        
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
    
    def make_significant_model_3d(self,log_bin_centers,weights,tril_vts,tril_deltaLogbins, ls_mean_m, ls_sd_m,ls_mean_z, ls_sd_z,sigma_sd=10,mu_dim=None,vt_sigmas=None,vt_accuracy_check=False):
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
                                               
        ls_mean_z                        ::    float
                                               mean of the redshift axis of the lengthscale for the single GP.
                                               
        ls_sd_z                          ::    float
                                               std of the redshift axis of the lengthscale for the single GP.
        
        sigma_sd                         ::    float
                                               std of the sigma for GP. Default is 10
        
        mu_z_dim                         ::    int
                                               number of mean functions for the GP. Can be 1
                                               or None. Default is None which corresponds to mu_dim = 
                                               number of bins.
        
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
        tril_vts = tril_vts*tril_deltaLogbins
        arg = tril_vts>0.
        if(len(np.where(~arg)[0])>0):
            tril_vts = tril_vts[np.where(arg)[0]]
            weights = weights[:,np.where(arg)[0]]
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
        
        nbins_m = int(len(self.mbins)*(len(self.mbins)-1)*0.5)
        log_bin_centers_m = log_bin_centers[:nbins_m,:2]
        log_bin_centers_z = log_bin_centers[0::nbins_m,2][:,None]
        with pm.Model() as gp_model:
            mu = pm.Normal('mu',mu=0,sigma=10,shape=mu_dim)
            sigma = pm.HalfNormal('sigma',sigma=sigma_sd)
            length_scale_m = pm.Lognormal('length_scale_m',mu=ls_mean_m,sigma=ls_sd_m)
            length_scale_z = pm.Lognormal('length_scale_z',mu=ls_mean_z,sigma=ls_sd_z)
            covariance_m = sigma*pm.gp.cov.ExpQuad(input_dim=2,ls=[length_scale_m,length_scale_m])
            covariance_z = sigma*pm.gp.cov.ExpQuad(input_dim=1,ls=[length_scale_z])
            gp = pm.gp.LatentKron(cov_funcs=[covariance_z, covariance_m]) 
            logn_corr = gp.prior('logn_corr',Xs=[log_bin_centers_z,log_bin_centers_m])
            logn_tot = pm.Deterministic('logn_tot', mu+logn_corr)
            n_corr = pm.Deterministic('n_corr',tt.exp(logn_tot))
            n_corr_physical = pm.Deterministic('n_corr_physical',n_corr[arg])
            n_f_exp = n_corr_physical*tril_vts
            N_F_exp = pm.Deterministic('N_F_exp',tt.sum(n_f_exp*(1.-0.5*int(vt_accuracy_check)*n_f_exp/n_eff)))
            log_l = pm.Potential('log_l',tt.sum(tt.log(tt.dot(weights,n_corr_physical))) - N_F_exp)
            n_eff_potential = pm.Potential('n_eff_potential', pm.math.switch(pm.math.le((int(vt_accuracy_check)*n_f_exp-2*n_eff).max(),0.),0.,-100))
            
        return gp_model
    
    def make_gp_prior_model_3d(self,log_bin_centers, ls_mean_m, ls_sd_m,ls_mean_z, ls_sd_z,sigma_sd=10,mu_dim=None):
        '''
        Function that creates a pymc model for sampling rate-densities
        from the GP prior in Eqs. 5.
        
        Parameters
        ----------
        log_bin_centers                  ::    numpy.ndarray
                                               array containing centers of each bin in log m1, log m2, z co-ordinates.
                                               output of Utils.generate_log_bin_centers
                                             
        ls_mean_m                        ::    float
                                               mean of the mass axis of the lengthscale for the single GP.
                                               
        ls_sd_m                          ::    float
                                               std of the mass axis of the lengthscale for the single GP..
                                               
        ls_mean_z                        ::    float
                                               mean of the redshift axis of the lengthscale for the single GP.
                                               
        ls_sd_z                          ::    float
                                               std of the redshift axis of the lengthscale for the single GP.
        
        sigma_sd                         ::    float
                                               std of the sigma for GP. Default is 10
        
        mu_z_dim                         ::    int
                                               number of mean functions for the GP. Can be 1
                                               or None. Default is None which corresponds to mu_dim = 
                                               number of bins.
        
        
        Returns
        -------
        
        gp_model  : pymc.Model object.
                    model object for sampling the rate densities prior.
        
        '''
        if mu_dim is None:
            mu_dim=len(log_bin_centers)
        assert mu_dim==1 or mu_dim==len(log_bin_centers)
        nbins_m = int(len(self.mbins)*(len(self.mbins)-1)*0.5)
        log_bin_centers_m = log_bin_centers[:nbins_m,:2]
        log_bin_centers_z = log_bin_centers[0::nbins_m,2][:,None]
        with pm.Model() as gp_model:
            mu = pm.Normal('mu',mu=0,sigma=10,shape=mu_dim)
            sigma = pm.HalfNormal('sigma',sigma=sigma_sd)
            length_scale_m = pm.Lognormal('length_scale_m',mu=ls_mean_m,sigma=ls_sd_m)
            length_scale_z = pm.Lognormal('length_scale_z',mu=ls_mean_z,sigma=ls_sd_z)
            covariance_m = sigma*pm.gp.cov.ExpQuad(input_dim=2,ls=[length_scale_m,length_scale_m])
            covariance_z = sigma*pm.gp.cov.ExpQuad(input_dim=1,ls=[length_scale_z])
            gp = pm.gp.LatentKron(cov_funcs=[covariance_z, covariance_m]) 
            logn_corr = gp.prior('logn_corr',Xs=[log_bin_centers_z,log_bin_centers_m])
            logn_tot = pm.Deterministic('logn_tot', mu+logn_corr)
            n_corr = pm.Deterministic('n_corr',tt.exp(logn_tot))
        
        return gp_model
    