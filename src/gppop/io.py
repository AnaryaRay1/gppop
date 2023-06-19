#!/usr/bin/env python
__author__="Anarya Ray"

from .gppop import Vt_Utils, Post_Proc_Utils
import numpy as np
import arviz as az
from astropy.cosmology import Planck15,z_at_value
from pesummary.io import read
from scipy.interpolate import interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import h5py

############################
#  Support Functions       #
############################
def read_posteriors_pesummary(event_list,nsamples):
    '''
    Function that reads pesummary files and extracts posterior
    samples of m1,m2 and z from it. See https://lscsoft.docs.ligo.org/pesummary/
    for details.
    
    Parameters
    ----------
    
    event_list      :: dict
                       dictionary containing paths to pesummary files
                       for each event, keyed by the waveform-approximant
                       corresponding to which the posterior samples
                       are to be chosen.
    
    nsamples        :: int
                       number of poterior samples to down-sample to.
    
    Returns
    -------
    
    posterior_samples_allevents :: numpy.ndarray
                                   array of shape (N_events,nsamples,3) containing
                                   m1,m2,z posterior samples for all events.
    '''
    parameters = ['mass_1_source', 'mass_2_source','redshift']
    posteriors = list()
    for event,key in event_list.items():
        _posterior = pd.DataFrame()
        ff = read(event)
        samples_dict = ff.samples_dict
        posterior_samples = samples_dict[key]
        for parameter in parameters:
            _posterior[parameter] = posterior_samples[parameter]
        posteriors.append(_posterior)
    
    N_events = len(posteriors)
    posterior_samples_allevents = np.zeros([Nevents,n_samples,3])    
    rng = np.random.default_rng()
    
    for i,_posterior in enumerate(posteriors):
        samples = rng.choice(_posterior.loc[:,['mass_1_source','mass_2_source','redshift']].values,replace=False,size=nsamples)
        posterior_samples_allevents[i] = samples
    
    return posterior_samples_allevents

def read_posteriors_h5py(event_list,nsamples,parameter_translator = dict(
    mass_1_det='m1_detector_frame_Msun',
    mass_2_det='m2_detector_frame_Msun',
    luminosity_distance='luminosity_distance_Mpc')):
    
    '''
    Function that reads h5py files and extracts posterior
    samples of detector frame masses and luminosity distances
    from them and converts the samples to that of source frame
    masses and redshifts. Required for O1O2 events.
    
    Parameters
    ----------
    
    event_list           :: dict
                            dictionary containing paths to pesummary files
                            for each event, keyed either by the waveform-approximant
                            corresponding to which the posterior samples are to be chosen.
                       
    nsamples             :: int
                            number of poterior samples to down-sample to.
                       
    parameter_translator :: dict
                            dictionary converting key's of parameters in file
                            to keys the ones expected by this function.
    
    Returns
    -------
    
    posterior_samples_allevents :: numpy.ndarray
                                   array of shape (N_events,nsamples,3) containing
                                   m1,m2,z posterior samples for all events.
                            
    '''
    Zs = np.linspace(0.,10,1000)
    DLs = Planck15.luminosity_distance(Zs).value
    z_interp = interp1d(Zs,DLS)
    
    posteriors = list()
    for event,key in event_list.items():
        _posterior = pd.DataFrame()
        __posterior = pd.DataFrame()
        with h5py.File(event,'r') as ff:
            posterior_samples = ff[key]
            for parameter, param_key in parameter_translator.items():
                __posterior[parameter] = posterior_samples[parameter_key]
            z_samples = z_interp(__posterior['luminosity_distance'])
            _posterior['redshift'] = z_samples
            _posterior['mass_1_source'] = __posterior['m1_detector_frame_Msun']/(1+z_samples)
            _posterior['mass_2_source'] = __posterior['m2_detector_frame_Msun']/(1+z_samples)
        posteriors.append(_posterior)
    
    N_events = len(posteriors)
    posterior_samples_allevents = np.zeros([Nevents,n_samples,3])    
    rng = np.random.default_rng()
    
    for i,_posterior in enumerate(posteriors):
        samples = rng.choice(_posterior.loc[:,['mass_1_source','mass_2_source','redshift']].values,replace=False,size=nsamples)
        posterior_samples_allevents[i] = samples
    
    return posterior_samples_allevents
        
    
    
class output_data_products(Post_Proc_Utils):
    """
    Class for producing output data products
    according to RnP recommended guidelines
    for a population analysis.
    """
    def __init__(self,mbins,zbins,trace_file,uncor=False):
        '''
        Initialize output_data_products class.
        
        Parameters
        ----------
        
        mbins         :: numpy.ndarray
                         1d array containing mass bin edges
        
        trace_file    :: str
                         Path to netcdf file containing trace yielded by sampling
        
        zbins         :: numpy.ndarray
                         1d array containing redshift bin edges
                         
        uncor         :: bool
                         Whether or not to treat samples as results of an uncorrelated
                         mass-redshift inference. Default is False
                         
        '''
        Post_Proc_Utils.__init__(self,mbins,zbins)
        
        self.nbins_m =int((len(mbins)*(len(mbins)-1)*0.5))
        self.nbins_z = int(len(zbins)-1) 
        self.nbins = int((len(mbins)*(len(mbins)-1)*0.5*(len(zbins)-1 if zbins is not None else 1)))
        
        self.trace = az.from_netcdf(trace_file)['posterior']
        self.N_samples = int(len(self.trace['draw'])*len(self.trace['chain']))
        self.uncor = uncor
        
        if not uncor:
            n_corr = self.trace['n_corr'].to_numpy()
            self.n_corr_samples = np.array([n_corr[:,:,i].reshape((self.N_samples,)) for i in range(self.nbins)]).T
        else:
            n_corr_m_samples = np.array([self.trace['n_corr'].to_numpy()[:,:,i].reshape((self.N_samples,)) for i in range(self.nbins_m)]).T
            n_corr_z_samples = np.array([self.trace['n_corr_z'].to_numpy()[:,:,i].reshape((self.N_samples,)) for i in range(self.nbins_z)]).T
            self.n_corr_samples = np.array([self.reshape_uncorr(n_corr_m,n_corr_z) for n_corr_m, n_corr_z in zip(n_corr_m_samples,n_corr_z_samples)])
            
        self.n_corr_mean = np.mean(self.n_corr_samples,axis=0)
    
    def reweight_pe_samples(self,m1m2z_samples,n_corr_sample=None, m1m2z_prior=None,size=None):
        '''
        Function for re-weighting posterior samples of masses and
        redshifts for particular event using the binned population model.
        
        Parameters
        ----------
        
        m1m2z_samples      :: numpy.ndarray
                              array of shape (Nsamples,3) containing
                              posterior samples of masses and redshifts.
                              
        n_corr_sample      :: numpy.ndarray
                              1d array containing rate densities in each bin,
                              corresponding to which the population model for
                              re-weighting is to be constructed. Default is None
                              which implies the mean of the inferred rate density
                              samples i.e. the best fit population, will be used.
                              
        m1m2z_prior        :: numpy.ndarray
                              array of same shape as m1m2z_samples containing values of
                              the prior distribtion used in PE. Default is None which
                              implies the default PE prior of uniform in detector frame masses
                              and luminosity_distance cubed will be used
                              
        size               :: int
                              Number of re-weighted posterior samples to draw. Default is None
                              which implies the same number of samples as in m1m2z_samples will
                              be drawn.
                              
        
        Returns
        -------
        
        m1m2z_samples  : numpy.ndarray
                         array containing re-weighted posterior samples.
        '''
        n_corr = self.n_corr_mean if n_corr_sample is None else n_corr_sample
        if m1m2z_prior is not None:
            assert m1m2z_samples.shape[0] == len(m1m2z_prior)
        else:
            z_samples = m1m2z_samples[:,2]
            pz_pop = Planck15.differential_comoving_volume(z_samples).to(u.Gpc**3/u.sr).value/(1+z_samples)
            ddL_dz = dl_values/(1+z_samples) + (1+z_samples)*Planck15.hubble_distance.to(u.Gpc).value/Planck15.efunc(z_samples)#Jacobian to convert from dL to z 
            m1m2z_prior = (1+z_samples)**2 * dl_values**2 * ddL_dz # default PE prior
        
        p_m1m2z_pop = self.get_pm1m2z(n_corr,m1m2z_samples[:,0],m1m2z_samples[:,1],m1m2z_samples[:,2],self.tril_edges())
        weights = p_m1m2z_pop/m1m2z_prior
        weights /= np.sum(weights) 
        return m1m2z_samples[np.random.choice(len(weights),p=weights,size=size),:]
    
    def posterior_predictive_samples(self,n_corr_sample=None, size=1000):
        '''
        Function for drawing samples of m1,m2,z from the binned
        population model.
        
        Parameters
        ----------
        n_corr_sample      :: numpy.ndarray
                              1d array containing rate densities in each bin,
                              corresponding to which the population model for
                              re-weighting is to be constructed. Default is None
                              which implies the mean of the inferred rate density
                              samples i.e. the best fit population, will be used.
                              
        size               :: int
                              Number of m1,m2,z samples to draw. Default is 1000.
        
        Returns
        -------
        
        m1     : numpy.ndarray
               1d array containing samples of primary mass drawn from the binned population.
               
        m2     : numpy.ndarray
               1d array containing samples of secondary mass drawn from the binned population.
               
        z     : numpy.ndarray
                1d array containing samples of redshift drawn from the binned population.                     
        '''
        n_corr = self.n_corr_mean if n_corr_sample is None else n_corr_sample
        assert size>=1000
        
        m1s = np.random.uniform(min(mbins),max(mbins),size=size*50)
        m2s = np.random.uniform(min(mbins),m1s,size=size*50)
        zs = np.random.uniform(min(zbins),max(zbins),size=size*50)
        
        p_m1m2z_pop = self.get_pm1m2z(n_corr_mean,m1s,
                                      m2s,zs,self.tril_edges())
        weights = p_m1m2z_pop/sum(p_m1m2z_pop)
        indices = np.random.choice(size*50,p=weights,size=size)
        
        return m1s[indices],m2s[indices],zs[indices]
    
    
    def rate_samples(self):
        '''
        Function for converting the inferred samples of rate densities
        into samples of the total merger rate.
        
        Returns
        -------
        
        rate_samples : numpy.ndarray
                       1d array containing rate samples
        '''
        log_bin_centers = self.generate_log_bin_centers()
        
        diag_idx = np.where(log_bin_centers[:,0] == log_bin_centers[:,1])[0]
        ones = np.ones(len(delta_logm2_array))
        ones[diag_idx]*=2.
        
        dm1,dm2 = self.delta_logm1s(self.mbins), self.delta_logm1s(self.mbins)
        
        rate_samples = np.sum(self.n_corr_samples*dm1[np.newaxis,:]*dm2[np.newaxis,:]*ones[np.newaxis,:],axis=1)
        
        return rate_samples
    
    def marginal_distributions_grid(self):
        '''
        Function for computing marginal m1, m2 and z population distributions
        on a grid corresponding to each hyper-parameter draw.
        
        Returns
        -------
        
        Z       : numpy.ndarray
                  1d array containing redshift grid points
        
        R_z     : numpy.ndarray
                  array containing the total marger rate at 
                  grid points.
        
        mass1   : numpy.ndarray
                  1d array containing primary mass grid points
        
        R_pm1   : numpy.ndarray
                  array containing the dR/dm1 at 
                  grid points.
        
        mass2   : numpy.ndarray
                  1d array containing secondary mass grid points
        
        R_pm2   : numpy.ndarray
                  array containing the dR/dm2 at 
                  grid points.
        '''
        dm1 = self.delta_logm1s(self.mbins)
        dm2 = self.delta_logm2s(self.mbins)
        log_bin_centers = self.generate_log_bin_centers()
        
        R_z,R_pm1,R_pm2= [ ] , [ ], [ ]
        for i,n_corr_sample in enumerate(self.n_corr_samples):
            Z,Rz = self.get_Rz(n_corr_sample,dm1,dm2,self.mbins,self.mbins,
                            self.zbins,log_bin_centers)
            R_z.append(Rz)
            
            mass1,Rpm1 = self.get_Rpm1(n_corr_sample,dm2,self.mbins,self.mbins,self.zbins,log_bin_centers)
            R_pm1.append(Rpm1)
            
            mass2, Rpm2 = self.get_Rpm2(n_corr_sample,dm1,self.mbins,self.zbins,log_bin_centers)
            R_pm2.append(Rpm2)
        
        return np.array(Z), np.array(R_z), np.array(mass1), np.array(R_pm1), np.array(mass2), np.array(R_pm2)
    
    def conditional_distributions_grid(self):
        '''
        Function for computing conditional distributions p(m1|z) and p(m2|z)
        on a grid, as function of redshif bins, corresponding to each 
        hyper-parameter draw. To be used only for correlated mass-redshift
        inference.
        
        Returns
        -------
        
        mass1   : numpy.ndarray
                  1d array containing primary mass grid points
        
        R_pm1   : numpy.ndarray
                  array containing the p(m1|z) at 
                  grid points.
        
        mass2   : numpy.ndarray
                  1d array containing secondary mass grid points
        
        R_pm2   : numpy.ndarray
                  array containing the p(m2|z) at 
                  grid points.
        
        '''
        assert not self.uncor
        
        dm1 = self.delta_logm1s(self.mbins)
        dm2 = self.delta_logm2s(self.mbins)
        log_bin_centers = self.generate_log_bin_centers()
        
        R_pm1, R_pm2 = [], []
        for i, n_corr_sample in enumerate(self.n_corr_samples):
            Rpm1_z, Rpm2_z = [ ], [ ]
            for j in range(self.nbins_z):
                mass1,p = self.get_Rpm1_corr(n_corr_sample, dm2,self.mbins, self.mbins, log_bin_centers,self.zbins[j],self.zbins[j+1])
                Rpm1_z.append(p)
                
                mass2,p = self.get_Rpm2_corr(n_corr_sample,dm2,self.mbins,log_bin_centers,self.zbins[j], self.zbins[j+1])
                Rpm2_z.append(p)
            R_pm1.append(Rpm1_z)
            R_pm2.append(Rpm2_z)
        
        return np.array(mass1), np.array(R_pm1), np.array(mass2), np.array(R_pm2)
    
    def reweight_injections(self,n_corr_sample,nsamples,inj_dataset,thresh,key = 'optimal_snr_net'):
        '''
        FIX ME, to be added soon
        '''
        # if type(key) == list:
        #     assert type(thresh)==list and len(thresh)==len(key)

        #     selector=np.where(np.sum(np.array([inj_data_set[k]>=th for k,th in zip(key,thresh)]),axis=0))[0]
        # else:
        #     selector=np.where(inj_data_set[key]>=thresh)[0]

        # m1s,m2s,zs = inj_dataset['mass1'][selector],inj_dataset['mass2'][selector],inj_dataset['redshift'][selector]
        pass
        
    
    
class parse_input(Vt_Utils):
    """
    Class for parsing input data products
    into quantities to be used in GP inference that
    are directly loadable by the driver script.
    """
    def __init__(self,mbins,zbins,posterior_samples,m1m2_given_z_prior,injection_dataset, thresh,key,include_spins = True):
        '''
        Initialize class for parsing inputs.
        
        Parameters
        ----------
        
        mbins                :: numpy.ndarray
                                1d array containing mass bin edges
        
        posterior_samples    :: numpy.ndarray
                                array of shape N_events,nsamples,3 containing
                                posterior samples of events (outputs of read_posteriors_pesummary
                                or read_posteriors_h5py)
        m1m2_given_z_prior   :: numpy.ndarray
                                if default PE priors were not used then
                                the values of the p(m_1,m_2|z) function used
                                in PE need to be supplied corresponding to
                                each posterior sample for each event.
        
        inj_data_set         :: dict
                                a dictionary containing 1d numpy arrays of
                                masses, redshifts, sping, sampling_pdfs, ranking statistic,
                                analysis time, and the total number of injections generated.
        
        thresh               :: float
                                value of the rankingstatistic threshold. Should match the threshold 
                                value used to select real events used in the analysis.
        
        key                  :: str or list
                                the key(s) in inj_data_set that corresponds to the rankingstatistics.
        
        zbins                :: numpy.ndarray
                                1d array containing redshift bin edges
                                default is None which is for m1,m2 only
                                inference.
                         
        include_spins        :: bool
                                whether or not to reweight spin distributions of 
                                injections               
        '''
        Vt_Utils.__init__(self,mbins,zbins,include_spins=include_spins)
        self.posterior_samples = posterior_samples
        self.injection_dataset = injection_dataset
        self.thresh = thresh
        self.key = key
        self.N_events = posterior_samples.shape[0]
        self.nbins = int((len(mbins)*(len(mbins)-1)*0.5*(len(zbins)-1 if zbins is not None else 1)))
        self.m1m2_given_z_prior = m1m2_given_z_prior
        self.tril_deltaLogbin = self.arraynd_to_tril(self.deltaLogbin())
        
    def save_weights(self,filename):
        '''
        Function for computing and saving posterior weights for all events.
        
        Parameters
        ----------
        
        filename  :: str
                     path to text file where posterior weights shall be saved
        '''
        weights = np.zeros([self.N_events,self.nbins])
        for i in range(self.N_events):
            weights[i]= self.arraynd_to_tril(self.compute_weights(samples=self.posterior_samples[i],m1m2_given_z_prior = self.m1m2_given_z_prior[i] if self.m1m2_given_z_prior is not None else None))
        self.weights = weights
        np.savetxt(filename,weights)
        
    
    def interpolate_vts(self,tril_vts,arg=None):
        '''
        Function for interpolating VTs over bins that have no injections
        using a GP regressor.
        
        Parameters
        ----------
        
        tril_vts  :: numpy.ndarray
                     1d array containing VT meands or stds to be interpolated.
        
        arg       :: numpy.ndarray
                     1d array containing bin indices at which interpolated
                     VTs need to be calculated.
                     
        Returns
        -------
        
        new_vts  : numpy.ndarray
                   interpolated VTs.
        '''
        if(all(tril_vts>0) or arg is None):
            return tril_vts
        
        vts = tril_vts/self.tril_deltaLogbin
        X = self.generate_log_bin_centers()[vts!=0]
        y = np.log((vts)[vts!=0])
        kernel = RBF(length_scale=0.3,length_scale_bounds=[0.1,2])
        gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=5).fit(X,y)
        y_pred = gp.predict(self.generate_log_bin_centers()[arg.astype(int)])
        vt_pred = np.exp(y_pred)
        new_vts = vts.copy()
        new_vts[arg.astype(int)] = vt_pred
        return new_vts*self.tril_deltaLogbin
    
    def save_vts(self,filename,interp_args=None):
        '''
        Function for computing and saving means and stds of emperically estimated VTs.
        
        Parameters
        ----------
        
        filename  :: str
                     path to text file where VTs shall be saved
        '''
        vt_means, vt_sigmas = self.compute_VTs(self.injection_dataset,self.thresh,key = self.key )
        if interp_args is not None:
            vt_means = self.interpolate_vts(vt_means,arg=interp_args)
            vt_sigmas = self.interpolate_vts(vt_sigmas,arg=interp_args)
        self.vt_means = vt_means
        self.vt_sigmas = vt_sigmas
        np.savetxt(filename,np.c_[vt_means/self.tril_deltaLogbin,vt_sigmas/self.tril_deltaLogbin])
        
