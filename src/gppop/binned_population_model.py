#!/usr/bin/env python
"""
Utilities for reweighting data to a binned population model constructing marginal distributions in post_processing.
"""

__author__="Anarya Ray <anarya.ray@ligo.org>; Claude (Anthropic)"

from .gwutils import *
import numpy as np
from scipy.stats import multivariate_normal,norm,halfnorm,lognorm
from astropy.cosmology import Planck15,z_at_value
from astropy import units as u
from scipy.interpolate import interp1d
import warnings
import tqdm

############################
#  Support Functions       #
############################


class Utils():
    """
    Utilities for Gaussian-process rate inference.

    Provides methods for constructing the physical bin mask, converting
    between flattened physical-bin arrays and four-dimensional binned arrays,
    computing posterior-sample weights, and calculating bin widths, edges, and
    centers in the ``m1``, ``q``, ``x``, and ``y`` parameter space.
    """
    
    def __init__(self,mbins,qbins,xbins, ybins, kappa=None):
        '''
        Initialize the utilities class.

        Parameters
        ----------
        mbins :: numpy.ndarray
                 One-dimensional array containing primary-mass bin edges.

        qbins :: numpy.ndarray
                 One-dimensional array containing mass-ratio bin edges.

        xbins :: numpy.ndarray
                 One-dimensional array containing bin edges for the first
                 additional parameter.

        ybins :: numpy.ndarray
                 One-dimensional array containing bin edges for the second
                 additional parameter.

        kappa :: float, optional
                 Redshift-evolution index of the merger rate.
        '''
        self.mbins = mbins
        self.qbins = qbins
        self.x_bins = xbins
        self.y_bins = ybins
        self.kappa=kappa

    def construct_arg_mat(self):
        '''
        Construct the mask identifying physical bins in the four-dimensional
        ``m1``, ``q``, ``x``, and ``y`` parameter space.

        Returns
        -------
        arg_mat : numpy.ndarray
                  Array whose nonzero entries identify physical bins.
        '''
    
        
        
        return  construct_arg_mat_out_spins(self.mbins, self.qbins, self.x_bins, self.y_bins)
    
    def arraynd_to_tril(self,arr, arg_mat):
        '''
        Extract entries corresponding to physical bins from an n-dimensional
        binned array.

        The input array is flattened, and entries for which ``arg_mat`` is
        nonzero are returned in their flattened order.

        Parameters
        ----------
        arr :: numpy.ndarray
               Input array containing values in all bins.

        arg_mat :: numpy.ndarray
                   Mask with the same flattened bin ordering as ``arr``.

        Returns
        -------
        lower_tri_array : numpy.ndarray
                          One-dimensional array containing values from physical
                          bins.
        '''
        arg_mat_flat = np.matrix.flatten(arg_mat)
        args = np.where(arg_mat_flat > 0)[0]
        arr_flat = np.matrix.flatten(arr)
        
        return arr_flat[args]

    def compute_weights(self,samples,m1m2_given_z_prior=None,xy_prior=None,full_prior=None,O4_prior = False):
        '''
        Compute posterior-sample weights in the binned population model.

        Parameters
        ----------
        samples :: numpy.ndarray
                   Array of posterior samples. Columns are interpreted as
                   primary mass, mass ratio, redshift, ``x``, and ``y``.

        m1m2_given_z_prior :: numpy.ndarray, optional
                              Values of the parameter-estimation
                              ``p(m1, m2 | z)`` prior for each sample. If not
                              supplied, the default detector-frame mass prior
                              is used.

        xy_prior :: numpy.ndarray, optional
                     Values of the parameter-estimation prior on the xy
                     variables for each sample.

       

        full_prior :: numpy.ndarray, optional
                      Full parameter-estimation prior evaluated for each
                      sample. When supplied, it is used instead of the
                      individual prior factors.

        O4_prior :: bool, optional
                    If True, use a prior uniform in comoving volume and source
                    time. Otherwise, use the older luminosity-distance-squared
                    prior.

        Returns
        -------
        weights : numpy.ndarray
                  Normalized four-dimensional weight array.

        wgt_means : numpy.ndarray
                    Empirical mean contribution in each bin.

        wgt_sigmas : numpy.ndarray
                     Empirical standard deviation of the contribution in each
                     bin.
        '''
        weights = np.zeros([len(self.mbins)-1,len(self.qbins)-1,len(self.x_bins)-1,len(self.y_bins)-1])
        wgt_means = np.zeros([len(self.mbins)-1,len(self.qbins)-1,len(self.x_bins)-1,len(self.y_bins)-1])
        wgt_sigmas = np.zeros([len(self.mbins)-1,len(self.qbins)-1,len(self.x_bins)-1,len(self.y_bins)-1])

        
        good_idx = np.where((samples[:,0] > self.mbins[0])  * (samples[:,1] > self.qbins[0]) * (samples[:,0] < self.mbins[-1])  * (samples[:,1] < self.qbins[-1]) * (samples[:,3] > self.x_bins[0])  * (samples[:,3] < self.x_bins[-1])* (samples[:,4] > self.y_bins[0])  * (samples[:,4] < self.y_bins[-1]))[0]
        samp_copy = samples[good_idx]
        #samp_copy = np.delete(samp_copy, bad_idx, 0)
        
        m1_samples = samp_copy[:,0]
        q_samples = samp_copy[:,1]
        z_samples = samp_copy[:,2]
        
        x_samples = samp_copy[:,3]
        y_samples = z_samples if self.kappa is None else samp_copy[:,4]
        #uniform in comoving-volume
        dl_values = Planck15.luminosity_distance(z_samples).to(u.Gpc).value
        m1_indices = np.clip(np.searchsorted(self.mbins,m1_samples,side='right') - 1,a_min=0,a_max=len(self.mbins)-2)
        q_indices = np.clip(np.searchsorted(self.qbins,q_samples,side='right') - 1,a_min=0,a_max=len(self.qbins)-2)
        x_indices = np.clip(np.searchsorted(self.x_bins,x_samples,side='right') - 1,a_min=0,a_max=len(self.x_bins)-2)
        y_indices = np.clip(np.searchsorted(self.y_bins,y_samples,side='right') - 1,a_min=0,a_max=len(self.y_bins)-2)
        
            
        pz_pop = Planck15.differential_comoving_volume(z_samples).to(u.Gpc**3/u.sr).value*((1+z_samples)**(self.kappa-1))
        if full_prior is None:
            ddL_dz = dl_values/(1+z_samples) + (1+z_samples)*Planck15.hubble_distance.to(u.Gpc).value/Planck15.efunc(z_samples)#Jacobian to convert from dL to z 
            m1m2_given_z_prior = m1m2_given_z_prior if m1m2_given_z_prior is not None else (1+z_samples)**2
            if not O4_prior:
                pz_PE = m1m2_given_z_prior * dl_values**2 * ddL_dz # default PE prior - flat in det frame masses and dL**2 in distance
            else : 
                pz_PE = m1m2_given_z_prior * Planck15.differential_comoving_volume(z_samples).to(u.Gpc**3/u.sr).value/(1+z_samples) # m1m2_given_z_prior * dl_values**2 * ddL_dz
            pz_PE*=xy_prior[good_idx]
            
        else:
            pz_PE=full_prior
        pz_weight = pz_pop/pz_PE
        indices = zip(m1_indices,q_indices,x_indices, y_indices)
        
        for i,inds in enumerate(indices):
            weights[inds[0],inds[1],inds[2], inds[3]] += pz_weight[i]/(m1_samples[i]*m1_samples[i])
            wgt_means[inds[0],inds[1],inds[2], inds[3]] += pz_weight[i]/(m1_samples[i]*m1_samples[i]) / len(samples)
        
        indices = zip(m1_indices,q_indices,x_indices, y_indices)
        
        for i,inds in enumerate(indices):
            wgt_sigmas[inds[0],inds[1],inds[2], inds[3]] += ((pz_weight[i]/(m1_samples[i] ** 2))** 2 / len(samples)** 2) 
        
        wgt_sigmas = np.sqrt(wgt_sigmas-wgt_means**2/len(samples))
        
        weights /= sum(sum(sum(sum(weights))))
        return weights, wgt_means, wgt_sigmas

    def deltaLogbin(self):
        '''
        Compute the four-dimensional bin-volume factor for every bin.

        The factor is the product of the logarithmic primary-mass width and
        the linear widths in mass ratio, ``x``, and ``y``.

        Returns
        -------
        deltaLogbin_array : numpy.ndarray
                            Four-dimensional array containing the bin-volume
                            factor for each bin.
        '''
        m1 = self.mbins
        q = self.qbins
        x = self.x_bins
        y = self.y_bins
        deltaLogbin_array = np.ones([len(m1)-1,len(q)-1,len(x)-1, len(y)-1])
        for i in range(len(m1)-1):
            for j in range(len(q)-1):
                for k in range(len(x)-1):
                    for l in range(len(y)-1):
                        if j != i:
                            deltaLogbin_array[i,j,k,l] = np.log(m1[i+1]/m1[i])*(q[j+1]-q[j])*(x[k+1]-x[k])*(y[l+1]-y[l])
                        elif j==i:
                            deltaLogbin_array[i,j,k,l] = np.log(m1[i+1]/m1[i])*(q[j+1]-q[j])*(x[k+1]-x[k])*(y[l+1]-y[l])
        return deltaLogbin_array
    
    def tril_edges(self):
        '''
        Return the lower and upper edges of every four-dimensional bin.

        Returns
        -------
        edge_array : numpy.ndarray
                     Array containing the lower and upper ``m1``, ``q``, ``x``,
                     and ``y`` edges of each bin.
        '''
        m1 = self.mbins
        q = self.qbins
        x = self.x_bins
        y = self.y_bins
        edge_array = []
        for i in range(len(m1)-1):
            for j in range(len(q)-1):
                for k in range(len(x)-1):
                    for l in range(len(y)-1):
                        edge_array.append([[m1[i],q[j],x[k]], y[l],[m1[i+1],q[j+1],x[k+1], y[l+1]]])
        return np.array(edge_array)

    def generate_log_bin_centers(self):
        '''
        Generate centers of the physical bins.

        Primary-mass centers are computed in logarithmic space, while the
        ``q``, ``x``, and ``y`` centers are computed in linear space.

        Returns
        -------
        log_lower_tri_sorted : numpy.ndarray
                               Array of shape ``(n_physical_bins, 4)`` containing
                               the ``log(m1)``, ``q``, ``x``, and ``y`` bin
                               centers.
        '''
        
        arg_mat_flat = np.matrix.flatten(self.construct_arg_mat())
        log_lower_tri_sorted = [ ]
        m1 = self.mbins
        q = self.qbins
        x = self.x_bins
        y = self.y_bins
        for i in range(len(m1)-1):
            for j in range(len(q)-1):
                for k in range(len(x)-1):
                    for l in range(len(y)-1):
                        log_lower_tri_sorted.append([(np.log(m1[i])+np.log(m1[i+1]))*0.5,
                                                     (q[j]+q[j+1])*0.5,
                                                     (x[k]+x[k+1])*0.5,
                                                     (y[l]+y[l+1])*0.5])
        
        log_lower_tri_sorted = np.array(log_lower_tri_sorted)
        return log_lower_tri_sorted[np.where(arg_mat_flat > 0)[0]]
            
                
    def construct_1dtond_matrix(self,nbins_m, values,nbins_y, nbins_x, nbins_q, m_min = None, m_max = None, arg_mat = None, tril=True):
        '''
        Map flattened physical-bin values to a four-dimensional binned array.

        Parameters
        ----------
        nbins_m :: int
                   Number of primary-mass bins in the output.

        values :: numpy.ndarray
                  Values ordered according to the flattened physical-bin mask.
                  Additional trailing dimensions are preserved.

        nbins_y :: int
                   Number of ``y`` bins.

        nbins_x :: int
                   Number of ``x`` bins.

        nbins_q :: int
                   Number of mass-ratio bins.

        m_min :: float, optional
                 Minimum primary mass included in the output.

        m_max :: float, optional
                 Maximum primary mass included in the output.

        arg_mat :: numpy.ndarray, optional
                   Mask identifying physical bins.

        tril :: bool, optional
                Retained for interface compatibility.

        Returns
        -------
        matrix : numpy.ndarray
                 Four-dimensional array, with any trailing dimensions from
                 ``values`` appended.
        '''
        k=0
        if len(values.shape)>1:
            matrix = np.zeros((nbins_m,nbins_q,nbins_x, nbins_y)+values.shape[1:])
        else:
            matrix = np.zeros((nbins_m,nbins_q,nbins_x, nbins_y))
	
        if np.sum(arg_mat) == None:
            arg_mat = self.construct_arg_mat()

        if m_min == None:
            m_min = self.mbins[0]
            m_max = self.mbins[-1]
	
        log_m1_bin_centers = 0.5 * (np.log(self.mbins[1:]) + np.log(self.mbins[:-1]))
        bin_idx = np.arange(len(log_m1_bin_centers))
        idx_arr = bin_idx[(log_m1_bin_centers >= np.log(m_min))&(log_m1_bin_centers <= np.log(m_max))]
        
        for i in range(nbins_m):
            for j in range(nbins_q):
                for l in range(nbins_x):
                    for m in range(nbins_y):
                        if arg_mat[idx_arr[i],j,l,m]>0:
                            matrix[i,j,l,m] = values[k]
                            k+=1
            
        return matrix

    
    def delta_qs(self):
        '''
        Return mass-ratio bin widths for the physical bins.

        Returns
        -------
        delta_q_array : numpy.ndarray
                        One-dimensional array of ``q`` bin widths ordered
                        according to the physical-bin mask.
        '''
        delta_q_array = np.zeros([len(self.mbins)-1,len(self.qbins)-1,len(self.x_bins)-1, len(self.y_bins)-1])
        for i in range(len(self.mbins)-1):
            for j in range(len(self.qbins)-1):
                for k in range(len(self.x_bins)-1):
                    for l in range(len(self.y_bins)-1):
                        delta_q_array[i,j,k,l] = self.qbins[j+1]-self.qbins[j]
        return self.arraynd_to_tril(delta_q_array, self.construct_arg_mat())
    
    def delta_logm1s(self):
        '''
        Return logarithmic primary-mass bin widths for the physical bins.

        Returns
        -------
        delta_logm1_array : numpy.ndarray
                            One-dimensional array of ``log(m1)`` bin widths
                            ordered according to the physical-bin mask.
        '''
        delta_logm1_array = np.zeros([len(self.mbins)-1,len(self.qbins)-1,len(self.x_bins)-1, len(self.y_bins)-1])
        for i in range(len(self.mbins)-1):
            for j in range(len(self.qbins)-1):
                for k in range(len(self.x_bins)-1):
                    for l in range(len(self.y_bins)-1):
                        delta_logm1_array[i,j,k,l] = np.log(self.mbins[i+1]/self.mbins[i])

        return self.arraynd_to_tril(delta_logm1_array, self.construct_arg_mat())
    
    def delta_xs(self):
        '''
        Return ``x`` bin widths for the physical bins.

        Returns
        -------
        delta_x_array : numpy.ndarray
                        One-dimensional array of ``x`` bin widths ordered
                        according to the physical-bin mask.
        '''
        delta_chi_array = np.zeros([len(self.mbins)-1,len(self.qbins)-1,len(self.x_bins)-1, len(self.y_bins)-1])
        for i in range(len(self.mbins)-1):
            for j in range(len(self.qbins)-1):
                for k in range(len(self.x_bins)-1):
                    for l in range(len(self.y_bins)-1):
                        delta_chi_array[i,j,k,l] = self.x_bins[k+1]-self.x_bins[k]
                    
        return self.arraynd_to_tril(delta_chi_array, self.construct_arg_mat())
    def delta_ys(self):
        '''
        Return ``y`` bin widths for the physical bins.

        Returns
        -------
        delta_y_array : numpy.ndarray
                        One-dimensional array of ``y`` bin widths ordered
                        according to the physical-bin mask.
        '''
        delta_chi_array = np.zeros([len(self.mbins)-1,len(self.qbins)-1,len(self.x_bins)-1, len(self.y_bins)-1])
        for i in range(len(self.mbins)-1):
            for j in range(len(self.qbins)-1):
                for k in range(len(self.x_bins)-1):
                    for l in range(len(self.y_bins)-1):
                        delta_chi_array[i,j,k,l] = self.y_bins[l+1]-self.y_bins[l]
                    
        return self.arraynd_to_tril(delta_chi_array, self.construct_arg_mat())
    
class Post_Proc_Utils(Utils):
    """
    Post-processing utilities for Gaussian-process rate inference.

    Provides methods for selecting bins and computing one- and two-dimensional
    marginal rate distributions from samples of binned rate densities.
    """
    
    def __init__(self,mbins,qbins,xbins, ybins, kappa=None):
        '''
        Initialize the post-processing utilities class.

        Parameters
        ----------
        mbins :: numpy.ndarray
                 One-dimensional array containing primary-mass bin edges.

        qbins :: numpy.ndarray
                 One-dimensional array containing mass-ratio bin edges.

        xbins :: numpy.ndarray
                 One-dimensional array containing ``x`` bin edges.

        ybins :: numpy.ndarray
                 One-dimensional array containing ``y`` bin edges.

        kappa :: float, optional
                 Redshift-evolution index of the merger rate.
        '''
        
        Utils.__init__(self,mbins,qbins,x_bins, y_bins, kappa=kappa)
        self.named_bins = {"m1":mbins, "q":qbins, "x":x_bins, "y":y_bins}
        self.named_dbins = {"m1":self.delta_logm1s(), "q":self.delta_qs(), "x":self.delta_xs(), "y":self.delta_ys()}
        self.names = ["m1", "q", "x", "y"]
    
    def get_bin_idx(self, log_bin_centers, m1_low, m1_high, q_low, q_high, x_low, x_high, y_low, y_high):
        '''
        Select bins whose centers lie within specified parameter bounds.

        Parameters
        ----------
        log_bin_centers :: numpy.ndarray
                           Array containing ``log(m1)``, ``q``, ``x``, and ``y``
                           bin centers.

        m1_low, m1_high :: float
                           Lower and upper primary-mass bounds.

        q_low, q_high :: float
                         Lower and upper mass-ratio bounds.

        x_low, x_high :: float
                         Lower and upper ``x`` bounds.

        y_low, y_high :: float
                         Lower and upper ``y`` bounds.

        Returns
        -------
        bin_idx : numpy.ndarray
                  Indices of bins satisfying all bounds.
        '''
        idx_array = np.arange(len(log_bin_centers))
        bin_idx = idx_array[(log_bin_centers[:,0]>=np.log(m1_low))&(log_bin_centers[:,0]<=np.log(m1_high))&(log_bin_centers[:,1]>=q_low)&(log_bin_centers[:,1]<=q_high)&(log_bin_centers[:,2]>=x_low)&(log_bin_centers[:,2]<=x_high)&(log_bin_centers[:,3]>=y_low)&(log_bin_centers[:,3]<=y_high)]

        return bin_idx

    def get_Rp1D(self, log_bin_centers, n_corr, lows, highs, umarg_name):
        '''
        Compute a one-dimensional marginal rate distribution.

        Parameters
        ----------
        log_bin_centers :: numpy.ndarray
                           Array containing the centers of the physical bins.

        n_corr :: numpy.ndarray
                  Rate-density samples, with bins along the final axis.

        lows :: dict
                Lower integration bounds keyed by parameter name.

        highs :: dict
                 Upper integration bounds keyed by parameter name.

        umarg_name :: str
                      Name of the parameter retained in the one-dimensional
                      distribution.

        Returns
        -------
        X : numpy.ndarray
            Evaluation points for the retained parameter.

        Rp : numpy.ndarray
             Marginal rate distribution for each input rate-density sample.
        '''
        dbins = [self.named_dbins[k] for k in self.names if k!= unmarg_name]
        arg_dict = {"log_bin_centers":log_bin_centers}
        for key, low in lows.items():
            arg_dict[f"key_{low}"]=low
            arg_dict[f"key_{high}"]=highs[key]
        Rp = np.zeros([len(n_corr),1])
        X = np.array([])
        unmarg_bins = self.named_bins[unmarg_name]
        for i in range(len(unmarg_bins)-1):
                low = unmarg_bins[i]
                high = unmarg_bins[i+1]
                x_array = np.linspace(low,high,100)[:-1]
                jac = x_array if unmarg_name=="m1" else np.ones_like(x_array)
                arg_dict[f"{unmarg_name}_low"] = low
                arg_dict[f"{unmarg_name}_low"] = high
                bin_idx = self.get_bin_idx(**arg_dict)
                rate_density_array = n_corr[:,bin_idx]
                dbins = [val[bin_idex] for val in dbins.values()]
                Rp= np.concatenate((Rp,np.sum(rate_density_array*(math.prod(dbins)[None,:]),axis=1)[:,None]/(jax[None,:])),axis=1)
                X = np.append(X,x_array)
        return X,Rp[:,1:]
        
        
    
    def get_R_twod(self, log_bin_centers,n_corr_med, lows, highs, marg_axis1, marg_axis2):
        '''
        Compute a normalized two-dimensional marginal rate distribution.

        Parameters
        ----------
        log_bin_centers :: numpy.ndarray
                           Array containing the centers of the physical bins.

        n_corr_med :: numpy.ndarray
                      One-dimensional array of rate densities in physical-bin
                      order.

        lows :: array-like
                Lower bounds for the parameters that are integrated over.

        highs :: array-like
                 Upper bounds for the parameters that are integrated over.

        marg_axis1 :: str
                      Name of the first parameter retained in the output.

        marg_axis2 :: str
                      Name of the second parameter retained in the output.

        Returns
        -------
        grid1 : numpy.ndarray
                Bin edges for the first retained parameter.

        grid2 : numpy.ndarray
                Bin edges for the second retained parameter.

        rate_2d : numpy.ndarray
                  Normalized two-dimensional marginal rate distribution.
        '''
        dbin1  = math.prod([self.named_dbins[k] for k in self.names if (k!=marg_axis1 or k!=marg_axis2)])
        dbin2 = math.prod([self.named_dbins[k]  for k in self.names if (k==marg_axis1 or k==marg_axis2)])
        grids = [self.named_bins[k] for k in self.names if (k==marg_axis1 or k==marg_axis2)]
        
        edges = [ ]
        nbins_m = len(self.mbins)-1
        nbins_q = len(self.qbins)-1
        nbins_x = len(self.x_bins)-1
        nbins_y = len(self.y_bins)-1

        axes = [ ]
        for i,n in enumerate(self.names):
            if n!=marg_axis1 and n!=marg_axis2:
                
                edges.append(min(self.named_bins[n]))
                edges.append(max(self.named_bins[n]))
            else:
                axess.append[i]
                edges.append(lows[i])
                edges.append(highs[i])
                
        bin_idx = self.get_bin_idx(log_bin_centers, *tuple(edges))
        ones = np.zeros_like(n_corr_med)
        ones[bin_idx] = 1.0
        rate = n_corr_med*dbin2*ones
        arg_mat = self.construct_arg_mat()
        rate_4d = self.construct_1dtond_matrix(nbins_m, rate, nbins_y, nbins_x, nbins_q, arg_mat = arg_mat, tril=True).sum((axes[0], axes[1]))
        rate_4d_norm = self.construct_1dtond_matrix(nbins_m, rate*dbin1, nbins_y, nbins_x, nbins_q, arg_mat = arg_mat, tril=True).sum()
        return grids[0], grids[1], rate_4d/rate_4d_norm 

    def get_R_twod_vec(self, log_bin_centers,n_corr_med, lows, highs, marg_axis1, marg_axis2):
        '''
        Compute normalized two-dimensional marginal rate distributions for
        multiple rate-density samples.

        Parameters
        ----------
        log_bin_centers :: numpy.ndarray
                           Array containing the centers of the physical bins.

        n_corr_med :: numpy.ndarray
                      Two-dimensional array of rate-density samples, with bins
                      along the final axis.

        lows :: array-like
                Lower bounds for the parameters that are integrated over.

        highs :: array-like
                 Upper bounds for the parameters that are integrated over.

        marg_axis1 :: str
                      Name of the first parameter retained in the output.

        marg_axis2 :: str
                      Name of the second parameter retained in the output.

        Returns
        -------
        grid1 : numpy.ndarray
                Bin edges for the first retained parameter.

        grid2 : numpy.ndarray
                Bin edges for the second retained parameter.

        rate_2d : numpy.ndarray
                  Normalized two-dimensional marginal rate distribution for
                  each input sample.
        '''
        dbin1  = math.prod([self.named_dbins[k] for k in self.names if (k!=marg_axis1 or k!=marg_axis2)])
        dbin2 = math.prod([self.named_dbins[k]  for k in self.names if (k==marg_axis1 or k==marg_axis2)])
        grids = [self.named_bins[k] for k in self.names if (k==marg_axis1 or k==marg_axis2)]
        
        edges = [ ]
        nbins_m = len(self.mbins)-1
        nbins_q = len(self.qbins)-1
        nbins_x = len(self.x_bins)-1
        nbins_y = len(self.y_bins)-1

        axes = [ ]
        for i,n in enumerate(self.names):
            if n!=marg_axis1 and n!=marg_axis2:
                
                edges.append(min(self.named_bins[n]))
                edges.append(max(self.named_bins[n]))
            else:
                axess.append[i]
                edges.append(lows[i])
                edges.append(highs[i])
                
        bin_idx = self.get_bin_idx(log_bin_centers, *tuple(edges))
        ones = np.zeros_like(n_corr_med)
        ones[:,bin_idx] = 1.0
        rate = n_corr_med*dbin2*ones
        arg_mat = self.construct_arg_mat()
        rate_4d = np.apply_along_axis(lambda v: self.construct_1dtond_matrix(nbins_m, v, nbins_y, nbins_x, nbins_q, arg_mat = arg_mat, tril=True).sum((axes[0], axes[1])),axis=1,arr=rate.copy())
        rate_4d_norm = np.apply_along_axis(lambda v: self.construct_1dtond_matrix(nbins_m, v, nbins_y, nbins_x, nbins_q, arg_mat = arg_mat, tril=True).sum(),axis=1,arr=rate*dbin1)
        return grids[0], grids[1], rate_4d/rate_4d_norm 
        

    
    
class Vt_Utils(Utils):    
    """
    Utilities for computing selection effects in Gaussian-process rate
    inference.

    These methods reweight simulated injections to the binned population model
    and estimate the mean and standard deviation of the volume-time
    sensitivity in each physical bin.
    """
    
    def __init__(self,mbins,qbins,x_bins, y_bins,kappa=2.7):
        '''
        Initialize the selection-effect utilities class.

        Parameters
        ----------
        mbins :: numpy.ndarray
                 One-dimensional array containing primary-mass bin edges.

        qbins :: numpy.ndarray
                 One-dimensional array containing mass-ratio bin edges.

        x_bins :: numpy.ndarray
                  One-dimensional array containing ``x`` bin edges.

        y_bins :: numpy.ndarray
                  One-dimensional array containing ``y`` bin edges.

        kappa :: float, optional
                 Redshift-evolution index of the merger rate.
        '''
        Utils.__init__(self,mbins,qbins,x_bins, y_bins, kappa=kappa)
        self.arg_mat = self.construct_arg_mat()

    def log_reweight_pinjection_mixture(self,m1, m2, z,s1x, s1y, s1z, s2x, s2y, s2z, pdraw, p_draw_xy_given_m1q, mix_weights,log_p_s1s2, compute_qxy ):
        '''
        Compute the log injection-reweighting contribution for one event.

        The nonzero entry is the logarithm of the quantity summed in Eq. A2 of
        https://arxiv.org/abs/2304.08046.

        Parameters
        ----------
        m1 :: float
              Source-frame primary mass of the simulated event.

        m2 :: float
              Source-frame secondary mass of the simulated event.

        z :: float
             Redshift of the simulated event.

        s1x, s1y, s1z :: float
                         Cartesian spin components of the primary object.

        s2x, s2y, s2z :: float
                         Cartesian spin components of the secondary object.

        pdraw :: float
                 Probability density with which the event parameters were
                 generated.

        p_draw_xy_given_m1q :: float
                               Conditional draw density for ``x`` and ``y``
                               given primary mass and mass ratio.

        mix_weights :: float
                       Mixture weight associated with the injection set.

        log_p_s1s2 :: float or bool
                      Log spin probability for the event. If false, it is
                      computed from the component spins and masses.

        compute_qxy :: callable
                       Function that computes ``q``, ``x``, and ``y`` from the
                       event parameters.

        Returns
        -------
        tril_weights : numpy.ndarray
                       One-dimensional array of log weights in physical-bin
                       order. All bins except the event's bin are zero.
        '''
        
        nbins = (len(self.mbins)-1)*(len(self.qbins)-1)*(len(self.x_bins)-1)*(len(self.y_bins)-1)
        tril_weights = np.zeros(int(np.sum(self.arg_mat)))
        q, x, y = compute_qxy(m1, m2, z,s1x, s1y, s1z, s2x, s2y, s2z)

        if (m1<self.mbins[0])|(q<self.qbins[0])|(m1>self.mbins[-1])|(q>self.qbins[-1])|(x<self.x_bins[0])|(x>self.x_bins[-1])|(y<self.y_bins[0])|(y>self.y_bins[-1]):
                return tril_weights
        weights = np.zeros([len(self.mbins)-1,len(self.qbins)-1,len(self.x_bins)-1,len(self.y_bins)-1])    
        m1_idx = np.clip(np.searchsorted(self.mbins,m1,side='right') - 1,a_min=0,a_max=len(self.mbins)-2)
        q_idx = np.clip(np.searchsorted(self.qbins,q,side='right') - 1,a_min=0,a_max=len(self.qbins)-2)
        x_idx = np.clip(np.searchsorted(self.x_bins,x,side='right') - 1,a_min=0,a_max=len(self.x_bins)-2)
        y_idx = np.clip(np.searchsorted(self.y_bins,y,side='right') - 1,a_min=0,a_max=len(self.y_bins)-2)
        log_dVdz = np.log(4*np.pi) + np.log(Planck15.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value)
        log_time_dilation = (self.kappa-1.)*np.log1p(z)
        if not log_p_s1s2:
            log_p_s1s2 = log_prob_spin(s1x,s1y,s1z,m1) + log_prob_spin(s2x,s2y,s2z,m2)
        weights[m1_idx,q_idx,x_idx, y_idx] = np.log(mix_weights) + log_dVdz  + log_time_dilation +log_p_s1s2 -np.log(p_draw_xy_given_m1q) - np.log(pdraw) - 2 * np.log(m1)  
        tril_weights = self.arraynd_to_tril(weights, self.arg_mat)

        return tril_weights
    
    def compute_VTs(self,inj_data_set,thresh,key = 'optimal_snr_net',log_p_s1s2=None ):
        '''
        Estimate the volume-time sensitivity in each physical bin.

        Parameters
        ----------
        inj_data_set :: dict
                        Dictionary containing injection parameters, sampling
                        densities, ranking statistics, analysis time, and the
                        total number of generated injections.

        thresh :: float or list
                  Ranking-statistic threshold, or one threshold per key when
                  ``key`` is a list.

        key :: str or list, optional
               Key, or list of keys, in ``inj_data_set`` corresponding to the
               ranking statistic used to select found injections.

        log_p_s1s2 :: numpy.ndarray, optional
                      Log spin probabilities for the injections. If omitted,
                      they are computed during reweighting.

        Returns
        -------
        vt_means : numpy.ndarray
                   Mean empirically estimated volume-time sensitivity in each
                   physical bin.

        vt_sigmas : numpy.ndarray
                    Standard deviation of the empirically estimated
                    volume-time sensitivity in each physical bin.
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
        conv = compute_q_chieff_chip if self.kappa is not None else compute_q_chieff_z
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
                conv,
            )
            
            mean_weights += np.where((x!=0), np.exp(x), 0 )
            var_weights += np.where((x!=0), np.exp(x), 0 )**2
            
        
        vt_means = mean_weights * (inj_data_set['analysis_time_s']/(365.25*24*3600))/inj_data_set['total_generated'] 

        vt_vars = var_weights * (inj_data_set['analysis_time_s']/(365.25*24*3600))**2/inj_data_set['total_generated']**2 - vt_means**2/inj_data_set['total_generated'] 
        vt_sigmas = np.sqrt(vt_vars)
        
        return vt_means, vt_sigmas