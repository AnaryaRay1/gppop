#!/usr/bin/env python
__author__="Anarya Ray"

from .gppop import Vt_Utils, Post_Proc_Utils, log_prob_spin
import numpy as np
import arviz as az
from astropy.cosmology import Planck15,z_at_value
from astropy import units as u
from pesummary.io import read
from scipy.interpolate import interp1d
from scipy.special import logsumexp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import h5py
from popsummary import popresult
import pandas as pd

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
    
    event_list      :: list
                       (Nevents, 1) shaped list containing waveform approximant and paths
                       to pesummary files of the GW events to be analyzed
    
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
    for [key,event] in event_list:
        print(f'reading {key} samples from {event}')
        _posterior = pd.DataFrame()
        ff = read(event)
        samples_dict = ff.samples_dict
        posterior_samples = samples_dict[key]
        for parameter in parameters:
            _posterior[parameter] = posterior_samples[parameter]
        posteriors.append(_posterior)
    
    N_events = len(posteriors)
    posterior_samples_allevents = np.zeros([N_events,nsamples,3])    
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
    
    event_list           :: list
                            (Nevents, 1) shaped list containing waveform approximant and
                            paths to pesummary files of the GW events to be analyzed
                       
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
    z_interp = interp1d(DLs,Zs)
    
    posteriors = list()
    for [key,event] in event_list:
        print(f'reading {key} samples from {event}')
        _posterior = pd.DataFrame()
        __posterior = pd.DataFrame()
        with h5py.File(event,'r') as ff:
            posterior_samples = ff[key][()]
            for parameter, param_key in parameter_translator.items():
                __posterior[parameter] = posterior_samples[param_key]
            z_samples = z_interp(__posterior['luminosity_distance'])
            _posterior['redshift'] = z_samples
            _posterior['mass_1_source'] = __posterior['mass_1_det']/(1+z_samples)
            _posterior['mass_2_source'] = __posterior['mass_2_det']/(1+z_samples)
        posteriors.append(_posterior)
    
    N_events = len(posteriors)
    posterior_samples_allevents = np.zeros([N_events,nsamples,3])    
    rng = np.random.default_rng()
    
    for i,_posterior in enumerate(posteriors):
        samples = rng.choice(_posterior.loc[:,['mass_1_source','mass_2_source','redshift']].values,replace=False,size=nsamples)
        posterior_samples_allevents[i] = samples
    
    return posterior_samples_allevents


def create_metafile(mbins,zbins,metafilename, n_pe_samples, injection_filename, injection_keys, thresh, thresh_keys, include_spins = True, event_dict = None, parameter_translator_dict = None, pe_summary_event_dict = None, m1m2_given_z_prior = None, analysis_type = 'uncor'):
        """
        Function for creating popsummary compatible meta-file containing all gppop-inputs
        
        Parameters
        ----------
        
        mbins                :: numpy.ndarray
                                1d array containing mass bin edges
        
        zbins                :: numpy.ndarray
                                1d array containing redshift bin edges
                                default is None which is for m1,m2 only
                                inference.
        
        metafilename         :: str
                                name of /path to the meta file

        n_pe_samples         :: int
                                numeber of PE samples to choose from each event
        
        injection_filename   :: str
                                path to file containing injections for VT estimation.
                                
        injection_keys       :: list
                                list of the keys of paramters to load from the injection file
        
        thresh               :: float
                                value of the rankingstatistic threshold. Should match the 
                                threshold 
                                value used to select real events used in the analysis.
        
        key                  :: str
                                the key in inj_data_set that corresponds to the 
                                rankingstatistic
                
        include_spins        :: bool
                                whether or not to reweight spin distributions
        
        event_dict           :: dict
                                dictionary containing information about O1O2 like events (ie 
                                whose pe runs are not fully pesumamry compatable), keyed by 
                                the event name. Each entry should be a list containing the 
                                name of the waveform approximant and the path to the file 
                                containing posterior samples of event parameters. Example:
                                {'GW150914':["IMRPhenomPv2", '/path/to
                                /GW150914_posterior_samples.h5], 'GW170817':[...],..}
                                default is None which means no o1o2 like events were used.
        
        
        parameter_translator_dict :: dict
                                     dictionary converting key's of parameters in the o1o2 event 
                                     files the ones expected by the read_posteriors_h5py
                                     function.
        
        pe_summary_event_dict :: dict
                                dictionary containing information about O3 like events (ie 
                                whose pe runs are pesumamry compatible), keyed by 
                                the event name. Each entry should be a list containing the 
                                name of the waveform approximant and the path to the file 
                                containing posterior samples of event parameters. Example:
                                {'GW150914':["IMRPhenomPv2", '/path/to
                                /GW150914_posterior_samples.h5'], 'GW170817':[...],..}
                                default is None which means no o1o2 like events were used.
        
        m1m2_given_z_prior  :: dict
                               a dictionary keyed by event names containig information regarding 
                               the prior used in PE. Each value should be either None or a 
                               callable. If None then it is assumed that the default pe prior was 
                               used for that event. If callable then should bea function that 
                               takes a (nsamples,3) shaped array of m1,m2,z samples and returns
                               a (nsamples,) sized array of p(m1,m2|z) evaluated at those samples
                               . Default is None which meand all events will be assumed to have a 
                               default pe prior.
        
        analysis_type      :: str
                              one of 'uncor' or 'cor'. Determines whether to carryout an 
                              uncorrelated mass-redshift analysis or a correlated one. Default is 
                              'uncor'
        
        """
        assert (event_dict is not None) or (pe_summary_event_dict is not None)
        
        event_list , waveform_list, path_list, posterior_samples  = [ ], [ ], [ ], None
        if event_dict is not None:
            for event, content in event_dict.items():
                event_list.append(event)
                waveform_list.append(content[0])
                path_list.append(content[1])
            
            posterior_samples = read_posteriors_h5py([[w,p] for [w,p] in event_dict.values()], n_pe_samples, parameter_translator_dict = parameter_translator_dict) if parameter_translator_dict is not None else read_posteriors_h5py([[w,p] for [w,p] in event_dict.values()],n_pe_samples)
        
        if pe_summary_event_dict is not None:
            for event, content in pe_summary_event_dict.items():
                event_list.append(event)
                waveform_list.append(content[0])
                path_list.append(content[1])

            pe_summary_event_samples = read_posteriors_pesummary([[w,p] for [w,p] in pe_summary_event_dict.values()],n_pe_samples)
            posterior_samples = np.concatenate((posterior_samples,pe_summary_event_samples), axis = 0) if posterior_samples is not None else pe_summary_event_samples
            
        popsummary = popresult.PopulationResult(fname = metafilename, events = event_list, event_waveforms = waveform_list, event_sample_IDs = np.arange(len(event_list)), event_parameters = ['m1','m2','z'],model_names = [f'gppop_{analysis_type}'], references = ['https://arxiv.org/abs/2304.08046'])
        
        gppop_metadata = {}
        gppop_metadata['mbins'] = mbins
        gppop_metadata['zbins'] = zbins
        gppop_metadata['unweighted_event_samples']= posterior_samples
        pe_prior = np.ones(posterior_samples.shape[:-1])
            
        if m1m2_given_z_prior is not None:
            assert type(m1m2_given_z_prior) == dict
            for i, prior_func in enumerate(m1m2_given_z_prior.values()):
                assert callabe(prior_func) or prior_func is None
                if prior_func is not None:
                    pe_perior[i] = prior_func(posterior_samples[i,:,:])
        gppop_metadata['m1m2_given_z_prior'] = pe_prior
        
        inj_dataset = {}
        with h5py.File(injection_filename,'r') as hf:
            inj_dataset['analysis_time_s'] = hf.attrs['analysis_time_s'] # years
            inj_dataset['total_generated'] = hf.attrs['total_generated']
            for param,key in injection_keys.items():
                    inj_dataset[param] = hf[key][()]
        formats = ['f8' for i in range(len(inj_dataset.keys()))][2:]
        dtypes = dict(names = list(inj_dataset.keys())[2:], formats = formats)
        inj_dataset_array = np.zeros(len(list(inj_dataset.values())[2]), dtype=dtypes)
        for key in list(inj_dataset.keys())[2:]:
            inj_dataset_array[key] = inj_dataset[key]
        gppop_metadata['inj_dataset'] = inj_dataset_array
        gppop_metadata['thresh']=thresh
        gppop_metadata['thresh_keys']=thresh_keys
        gppop_metadata['analysis_time_s'] = inj_dataset['analysis_time_s']
        gppop_metadata['total_generated'] = inj_dataset['total_generated']
        
        gp_inputs = parse_input(mbins,zbins,posterior_samples,
                           m1m2_given_z_prior,inj_dataset,
                           thresh,thresh_keys)
        
        gppop_metadata['posterior_weights'] = gp_inputs.weights
        gppop_metadata['vt_means'] = gp_inputs.vt_means
        gppop_metadata['vt_sigmas'] = gp_inputs.vt_sigmas
        
        with h5py.File(metafilename, 'a') as f:
            if 'gppop_metadata' in list(f.keys()):
                    del f['gppop_metadata']
                    gpmd = f.create_group('gppop_metadata')
            else:
                    gpmd = f.create_group('gppop_metadata')
            for key,val in gppop_metadata.items():
                gpmd.create_dataset(key,data = val)

    
def write_results_to_metafile(metafilename,trace_file_posterior,trace_file_prior, n_draw_pe, n_draw_inj, n_draw_pred, overwrite = False, include_spins = True):
    """
    Function for generating output data products and writing them to the 
    same meta-file created by create_metafile.
    
    Parameters
    ----------
    metafilename         :: str
                            name of /path to the meta file
    
    trace_file_posterior :: str
                            path to .nc file containing posterior samples
                            of rate densities
    
    trace_file_prior     :: str
                            path to .nc file containing prior samples
                            of rate densities
    
    n_draw_pe            :: int
                            number of populoation re-weighted pe samples to draw for each
                            event
    
    n_draw_inj           :: int
                            number of populoation re-weighted detectable injections to draw
    
    n_draw_pred          :: int
                            number of fair population draws.
    
    overwrite            :: bool
                            Whether or not to overwrite output data products in case they already exist in 
                            meta-file. Default is False.
    
    include_spins        :: bool
                            Whether or not to include spins while re-weighting injections. Default is True
    """
    popsummary = popresult.PopulationResult(fname = metafilename)
    analysis_type = popsummary.get_metadata('model_names')[0][6:]
    
    with h5py.File(metafilename,'r') as hf:
                pe_samples = hf['gppop_metadata']['unweighted_event_samples'][()]
                pe_prior = hf['gppop_metadata']['m1m2_given_z_prior'][()]
                injections = hf['gppop_metadata']['inj_dataset'][()]
                thresh = hf['gppop_metadata']['thresh'][()]
                keys = hf['gppop_metadata']['thresh_keys'][()]
                keys = [k.decode('utf-8') for k in keys]
                mbins = hf['gppop_metadata']['mbins'][()]
                zbins = hf['gppop_metadata']['zbins'][()]
    
    gp_outputs = {"posterior":output_data_products(mbins,zbins,trace_file_posterior,uncor = (analysis_type == 'uncor')), "prior" : output_data_products(mbins,zbins,trace_file_prior,uncor = (analysis_type == 'uncor'))}
    
    if len(gp_outputs['posterior'].mu_samples.shape)==1:
        hyperparameter_names = np.array([f'$\lambda_m$',f'$\lambda_z$',f'$\sigma$',f'$\mu$'])
    else:
        hyperparameter_names = np.array([f'$\lambda_m$',f'$\lambda_z$',f'$\sigma$'])
        hyperparameter_names = np.append(hyperparameter_names,np.array([f'$\mu_{i}$' for i in range(gp_outputs['posterior'].nbins_m)]))
    hyperparameter_names = np.append(hyperparameter_names,np.array([f'$n_{i}$' for i in range(gp_outputs['posterior'].nbins)])).tolist()
    popsummary.set_metadata('hyperparameters',hyperparameter_names,overwrite=overwrite)
    popsummary.set_metadata('hyperparameter_latex_labels', hyperparameter_names,overwrite=overwrite)
    
    for group, gp_output in gp_outputs.items():
            
            if len(gp_output.mu_samples.shape)==1:
                hyper_samples = np.array([gp_output.lambda_m_samples, gp_output.lambda_z_samples, gp_output.sigma_samples, gp_output.mu_samples]).T
            else:
                hyper_samples = np.concatenate((np.array([gp_output.lambda_m_samples, gp_output.lambda_z_samples, gp_output.sigma_samples]).T, gp_output.mu_samples), axis = 1)
            hyper_samples = np.concatenate((hyper_samples,gp_output.n_corr_samples),axis = 1)
            popsummary.set_hyperparameter_samples(hyper_samples,overwrite = overwrite, group = group)
                
            reweighted_pe_samples = np.zeros([pe_samples.shape[0], n_draw_pe, gp_output.N_samples, pe_samples.shape[-1]])
            for i in range(pe_samples.shape[0]):
                print(f"re-weighting {i}th event's pe samples using hyper-{group}")
                reweighted_pe_samples[i,:,:,:] = gp_output.reweight_pe_samples(pe_samples[i], n_corr_sample=gp_output.n_corr_samples, m1m2z_prior=None if all(pe_prior[i]== 1.0) else pe_prior[i], size=n_draw_pe)
            
            popsummary.set_reweighted_event_samples(reweighted_pe_samples,overwrite = overwrite, group = group)
            
            reweighted_injections = np.zeros([n_draw_inj,1, gp_output.N_samples, 9])
            print(f'reweighting injections using hyper-{group}')
            reweighted_injections[:,0,:,:] = gp_output.reweight_injections(injections, list(thresh), key = list(keys), include_spins = include_spins, n_corr_sample= gp_output.n_corr_samples, size=n_draw_inj)
            popsummary.set_reweighted_injections(reweighted_injections, overwrite=overwrite, group=group)
            
            print(f'generating fair population draws from the hyper-{group}')
            fair_draws = np.zeros([n_draw_pred, gp_output.N_samples, 3])
            fair_draws = gp_output.posterior_predictive_samples(n_corr_sample=gp_output.n_corr_samples, size=n_draw_pred)
            popsummary.set_fair_population_draws(fair_draws, overwrite=overwrite, group=group)
            
            
            if analysis_type == 'uncor':
                print(f'computing marginal rate densities on grid for hyper-{group}')
                Z,Rz,m1,Rpm1,m2,Rpm2 = gp_output.marginal_distributions_grid()
                popsummary.set_rates_on_grids('redshift', 'z', Z.reshape((len(Z),1)), Rz, overwrite=overwrite, group=group)
                popsummary.set_rates_on_grids('primary_mass', 'm1', m1.reshape((len(m1),1)), Rpm1, overwrite=overwrite, group=group)
                popsummary.set_rates_on_grids('secondary_mass', 'm2', m2.reshape((len(m2),1)), Rpm2, overwrite=overwrite, group=group)
            
            else:
                print(f'computing conditional rate densities on grid for hyper-{group}')
                m1,Rpm1z,m2,Rpm2z = gp_output.conditional_distributions_grid()
                popsummary.set_rates_on_grids('primary_mass', ['m1','z'], m1.reshape((len(m1),1)), Rpm1z, overwrite=overwrite, group=group)
                popsummary.set_rates_on_grids('secondary_mass', 'm2', m2.reshape((len(m2),1)), Rpm2z, overwrite=overwrite, group=group)

                    
            
    
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
        
        trace = az.from_netcdf(trace_file)['posterior']
        self.N_samples = int(len(trace['draw'])*len(trace['chain']))
        self.uncor = uncor
        
        if not uncor:
            n_corr = trace['n_corr'].to_numpy()
            self.n_corr_samples = np.array([n_corr[:,:,i].reshape((self.N_samples,)) for i in range(self.nbins)]).T
            mu = trace['mu'].to_numpy()
            self.mu_samples = np.array([mu[:,:,i].reshape((self.N_samples,)) for i in range(len(trace['mu_dim_0']))]).T

        else:
            n_corr_m_samples = np.array([trace['n_corr'].to_numpy()[:,:,i].reshape((self.N_samples,)) for i in range(self.nbins_m)]).T
            n_corr_z_samples = np.array([trace['n_corr_z'].to_numpy()[:,:,i].reshape((self.N_samples,)) for i in range(self.nbins_z)]).T
            self.n_corr_samples = np.array([self.reshape_uncorr(n_corr_m,n_corr_z) for n_corr_m, n_corr_z in zip(n_corr_m_samples,n_corr_z_samples)])
            mu_samples_m = np.array([trace['mu'].to_numpy()[:,:,i].reshape(self.N_samples) for i in range(self.nbins_m)]).T
            mu_samples_z = np.array([trace['mu_z'].to_numpy()[:,:,i].reshape(self.N_samples) for i in range(len(trace['mu_z_dim_0']))]).T
            self.mu_samples = np.concatenate((mu_samples_m,mu_samples_z), axis = 1)
            self.n_corr_z_tot = np.sum(n_corr_z_samples,axis=1)
        self.lambda_m_samples = trace['length_scale_m'].to_numpy().reshape(self.N_samples)
        self.lambda_z_samples = trace['length_scale_z'].to_numpy().reshape(self.N_samples)
        self.sigma_samples = trace['sigma'].to_numpy().reshape(self.N_samples)
        self.n_corr_mean = np.mean(self.n_corr_samples,axis=0)[np.newaxis,:]
    
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
                              2d array of shape (nsamples, nbins) containing 
                              rate density samples for each bin, 
                              corresponding to which the population models for
                              re-weighting are to be constructed. Default is None
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
        size = len(m1m2z_samples) if size is None else size
        z_samples = m1m2z_samples[:,2]
        dl_values = Planck15.luminosity_distance(z_samples).to(u.Gpc).value
        ddL_dz = dl_values/(1+z_samples) + (1+z_samples)*Planck15.hubble_distance.to(u.Gpc).value/Planck15.efunc(z_samples)#Jacobian to convert from dL to z 
        if m1m2z_prior is not None:
            assert m1m2z_samples.shape[0] == len(m1m2z_prior)
            m1m2z_prior = m1m2z_prior * dl_values**2 * ddL_dz
        else:
            m1m2z_prior = (1+z_samples)**2 * dl_values**2 * ddL_dz # default PE prior
        
        p_m1m2z_pop = self.get_pm1m2z(n_corr,m1m2z_samples[:,0],m1m2z_samples[:,1],m1m2z_samples[:,2],self.tril_edges())
        weights = p_m1m2z_pop/m1m2z_prior
        weights /= np.sum(weights,axis=1)[:,np.newaxis]
        return m1m2z_samples[np.array([np.random.choice(weights.shape[1],p=weights[i,:],size=size) for i in range(len(weights))]).T,:]
    
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
               
        z      : numpy.ndarray
                 1d array containing samples of redshift drawn from the binned population.                     
        '''
        n_corr = self.n_corr_mean if n_corr_sample is None else n_corr_sample
        
        m1s = np.random.uniform(min(self.mbins),max(self.mbins),size=size*50)
        m2s = np.random.uniform(min(self.mbins),m1s,size=size*50)
        zs = np.random.uniform(min(self.zbins),max(self.zbins),size=size*50)
        
        m1m2z_samples = np.array([m1s,m2s,zs]).T
        p_m1m2z_pop = self.get_pm1m2z(n_corr_sample,m1s,
                                      m2s,zs,self.tril_edges())
        weights = p_m1m2z_pop/np.sum(p_m1m2z_pop,axis=1)[:,np.newaxis]
        indices = np.array([np.random.choice(size*50,p=weights[i,:],size=size) for i in range(weights.shape[0])]).T
        
        return m1m2z_samples[indices,:]
    
    
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
        
        n_corr_samples_m = self.n_corr_samples.copy()
        if self.uncor:
            n_corr_samples_m[:,self.nbins_m:] = 0
            n_corr_samples_m/= self.n_corr_z_tot[:,np.newaxis]
                
        
        R_z,R_pm1,R_pm2= [ ] , [ ], [ ]
        for i,(n_corr_sample,n_corr_sample_m) in enumerate(zip(self.n_corr_samples,n_corr_samples_m)):
            Z,Rz = self.get_Rz(n_corr_sample,dm1,dm2,self.mbins,self.mbins,
                            self.zbins,log_bin_centers)
            R_z.append(Rz)
            
            mass1,Rpm1 = self.get_Rpm1(n_corr_sample_m, dm2,self.mbins,self.mbins,self.zbins,log_bin_centers)
            R_pm1.append(Rpm1)
            
            mass2, Rpm2 = self.get_Rpm2(n_corr_sample_m,dm1,self.mbins,self.zbins,log_bin_centers)
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
    
    def reweight_injections(self, inj_dataset,thresh,key = 'optimal_snr_net',include_spins = False, n_corr_sample=None,size=None):
        '''
        Function for re-weighting found injections using the binned population model.
        
        Parameters
        ----------
        
        inj_data_set       ::   dict
                                a dictionary containing 1d numpy arrays of
                                masses, redshifts, sping, sampling_pdfs and ranking statistic,
        
        thresh             ::   float
                                value of the rankingstatistic threshold. Should match the 
                                threshold 
                                value used to select real events used in the analysis.
        
        key                ::   str
                                the key in inj_data_set that corresponds to the rankingstatistic
                
        include_spins      :: bool
                              whether or not to reweight spin distributions
        
        n_corr_sample      :: numpy.ndarray
                              1d array containing rate densities in each bin,
                              corresponding to which the population model for
                              re-weighting is to be constructed. Default is None
                              which implies the mean of the inferred rate density
                              samples i.e. the best fit population, will be used.
                              
        size               :: int
                              Number of re-weighted posterior samples to draw. Default is None
                              which implies the same number of samples as in m1m2z_samples will
                              be drawn.
                              
        
        
        Returns
        -------
        
        output  : numpy.ndarray
                  array containing parameters of re-weighted detectable injections.
        '''
        if type(key) == list:
            assert type(thresh)==list and len(thresh)==len(key)

            selector=np.where(np.sum(np.array([inj_dataset[k]>=th for k,th in zip(key,thresh)]),axis=0))[0]
        else:
            selector=np.where(inj_dataset[key]>=thresh)[0]
        
        if(not include_spins):
            s1xs=s2xs=s1ys=s2ys=s1zs=s2zs=np.ones(len(selector))
        else:
            s1xs,s1ys,s1zs,s2xs,s2ys,s2zs=inj_dataset['spin1x'][selector],inj_dataset['spin1y'][selector],inj_dataset['spin1z'][selector], inj_dataset['spin2x'][selector],inj_dataset['spin2y'][selector], inj_dataset['spin2z'][selector]
        
        m1s,m2s,zs, logp_draw, mix_weights = inj_dataset['mass1_source'][selector],inj_dataset['mass2_source'][selector],inj_dataset['redshift'][selector],inj_dataset['sampling_pdf'][selector], inj_dataset['mixture_weight'][selector]
        
        log_p_s1s2 = log_prob_spin(s1xs,s1ys,s1zs,m1s) + log_prob_spin(s2xs,s2ys,s2zs,m2s)
        
        logp_m1m2z_pop = np.log(self.get_pm1m2z(n_corr_sample,m1s,
                                      m2s,zs,self.tril_edges())+1e-310)
        
        log_weights = (np.log(mix_weights) + log_p_s1s2*int(include_spins) - logp_draw)[np.newaxis,:]
        log_weights = logp_m1m2z_pop + log_weights
        log_weights -= logsumexp(log_weights,axis=1)[:,np.newaxis]
        
        size = len(log_weights) if size is None else size
        
        idx = np.array([np.random.choice(log_weights.shape[1],p=np.exp(log_weights[i,:]),size=size) for i in range(len(log_weights))]).T
        output = np.array([ m1s,m2s,zs,s1xs,s1ys,s1zs,s2zs,s2ys,s2zs]).T 
        
        return output[idx,:]
    
    
class parse_input(Vt_Utils):
    """
    Class for parsing input data products
    into quantities to be used in GP inference that
    are directly loadable by the driver script.
    """
    def __init__(self,mbins,zbins,posterior_samples,m1m2_given_z_prior,injection_dataset, thresh, key ,include_spins = True):
        '''
        Initialize class for parsing inputs.
        
        Parameters
        ----------
        
        mbins                :: numpy.ndarray
                                1d array containing mass bin edges
        
        zbins                :: numpy.ndarray
                                1d array containing redshift bin edges
                                default is None which is for m1,m2 only
                                inference.
                         
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
                                value of the rankingstatistic threshold. Should match the 
                                threshold 
                                value used to select real events used in the analysis.
        
        key                  :: str or list
                                the key(s) in inj_data_set that corresponds to the
                                rankingstatistics.
        
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
        self.n_mbins = int(len(mbins)*(len(mbins)-1)*0.5)
        self.m1m2_given_z_prior = m1m2_given_z_prior
        self.tril_deltaLogbin = self.arraynd_to_tril(self.deltaLogbin())
        
        self.compute_all_weights()
        self.compute_all_vts()
        
    def compute_all_weights(self):
        '''
        Function for computing posterior weights for all events.
        '''
        weights = np.zeros([self.N_events,self.nbins])
        for i in range(self.N_events):
            if self.m1m2_given_z_prior is None:
                m1m2_given_z_prior = None
            else:
                m1m2_given_z_prior = self.m1m2_given_z_prior[i] if (not all(self.m1m2_given_z_prior[i]==1.0)) else None
            weights[i]= self.arraynd_to_tril(self.compute_weights(samples=self.posterior_samples[i],m1m2_given_z_prior = self.m1m2_given_z_prior[i] if self.m1m2_given_z_prior is not None else None))
        self.weights = weights
        
        
    
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
    
    def compute_all_vts(self,interp_args=None):
        '''
        Function for computing means and stds of emperically estimated VTs.
        
        Parameters
        ----------
        
        interp_args  :: numpy.ndarray
                        1d array containing bin indices at which vt is to be intepolated
        '''
        vt_means, vt_sigmas = self.compute_VTs(self.injection_dataset,self.thresh,key = self.key )

        if interp_args is not None:
            vt_means = self.interpolate_vts(vt_means,arg=interp_args)
            vt_sigmas = self.interpolate_vts(vt_sigmas,arg=interp_args)
        else:
            for i in range(len(self.zbins)-1):
                if i==0:
                    vt_mean_summed = vt_means.copy()[i*self.n_mbins:(i+1)*self.n_mbins]
                else:
                    vt_mean_summed+= vt_means[i*self.n_mbins:(i+1)*self.n_mbins]
            interp_arg_m = np.where(vt_mean_summed==0)[0]
            if len(interp_arg_m)==0:
                pass
            else:
                interp_args = np.array([])
                for i in range(len(self.zbins)-1):
                    interp_args = np.append(interp_args,interp_arg_m+i*self.n_mbins)
                vt_means = self.interpolate_vts(vt_means,arg=interp_args)
                vt_sigmas = self.interpolate_vts(vt_sigmas,arg=interp_args)
        self.vt_means = vt_means/self.tril_deltaLogbin
        self.vt_sigmas = vt_sigmas/self.tril_deltaLogbin
            