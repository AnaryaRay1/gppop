#!/usr/bin/env python

import sys

import numpy as np
import h5py




import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import yaml
from astropy.cosmology import Planck15
from scipy.interpolate import interp1d




import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

import seaborn as sns
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('colorblind')
colors=sns.color_palette('colorblind')
fs=80




import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

mpl.rcParams['font.family'] = 'DejaVu Sans'

mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams['mathtext.default'] = 'regular'

mpl.rcParams['text.usetex'] = False
















from gppop.binned_population_model import Utils, Vt_Utils
from gppop.gwutils import prior_chieff_chip_isotropic as spin_prior 




with open(sys.argv[1], "r") as stream:
    config = yaml.full_load(stream)



mbins = np.array(config["mbins"])
chi_bins = np.array(config["chieff_bins"])
chip_bins = np.array(config["chip_bins"])
qbins = np.array(config["qbins"])




utils = Utils(mbins,qbins,chi_bins,chip_bins, kappa=2.7)
arg_mat_spin = utils.construct_arg_mat()
log_bin_centers=utils.generate_log_bin_centers()
tril_deltaLogbin = utils.arraynd_to_tril(utils.deltaLogbin(), utils.construct_arg_mat())





Zs = np.linspace(0.,10,1000)
DLs = Planck15.luminosity_distance(Zs).value
z_interp = interp1d(DLs,Zs)















N=1e6
import tqdm
print("start finding smallest number of pe samples")
for filename in tqdm.tqdm(list(config['pe_summary_event_dict'].values())):
    
    with h5py.File(filename[1],'r') as hf:
        posterior = hf[filename[0]]['posterior_samples'][()]
        
        posterior = hf[filename[0]]['posterior_samples'][()]
        m1 = posterior['mass_1_source']
        if N>len(m1):
            N=len(m1)
        

N_samples = N
np.random.seed(2**31-78)

spin_priors = [ ]

print("Loading PE samples and computing spin priors")
posterior_samples_o4b = np.zeros((len(config['pe_summary_event_dict'].keys()),N_samples,5))
for i,filename in enumerate(tqdm.tqdm(list(config['pe_summary_event_dict'].values()))):
    
   
    
    with h5py.File(filename[1],'r') as hf:
        
        posterior = hf[filename[0]]['posterior_samples'][()]
        this_N_samples = len(posterior['mass_1_source'])
        indices = np.random.choice(this_N_samples,size=N_samples,replace=False)
        z = posterior['redshift'][indices]
        m1 = posterior['mass_1_source'][indices]
        m2 = posterior['mass_2_source'][indices]
        q = posterior['mass_ratio'][()][indices]
        chi_eff = posterior['chi_eff'][()][indices]
        chi_p = posterior['chi_p'][()][indices]
        posterior_samples_o4b[i,:,:] = np.array([m1,q,z,chi_eff, chi_p]).T
        spin_priors.append(spin_prior(chi_eff, chi_p, q,amax=0.99))
        
posterior_samples_all = posterior_samples_o4b.copy()
posterior_samples_o4b = [ ]
spin_priors = np.array(spin_priors)


spin_priors = np.array(spin_priors)
spin_priors[np.where(np.isnan(spin_priors))]=np.nanmin(spin_priors)
spin_priors[spin_priors<=0] = np.min(spin_priors[spin_priors>0])




print("Computing Weights")

weights = [ ]
wt_means = [ ]
wt_sigmas = [ ]
for i,posterior_samples_this_event in enumerate(tqdm.tqdm(posterior_samples_all)):
    weight_set = utils.compute_weights(posterior_samples_this_event,
                                       xy_prior=spin_priors[i], 
                                       O4_prior = True)
    weights.append(utils.arraynd_to_tril(weight_set[0], arg_mat_spin))
    wt_means.append(utils.arraynd_to_tril(weight_set[1], arg_mat_spin))
    wt_sigmas.append(utils.arraynd_to_tril(weight_set[2], arg_mat_spin))
    


weights = np.array(weights)
wt_means = np.array(wt_means)
wt_sigmas = np.array(wt_sigmas)




weights_norm = np.sum(weights, axis = 1)
weights_cut = (weights.T / weights_norm).T
weights_cut.shape



wts_sum = np.sum(weights_cut,axis=0)



w= utils.construct_1dtond_matrix(len(mbins)-1,wts_sum,nbins_y=len(chip_bins)-1,nbins_x=len(chi_bins)-1, nbins_q = len(qbins) - 1)


fig,axes = plt.subplots(len(chi_bins)-1, len(chip_bins)-1, figsize=(8*5*(len(chip_bins)-1)/(len(chi_bins)-1),40))


print("Plotting Weights")
for i in range(len(chi_bins)-1):
    for j in range(len(chip_bins)-1):
            ax = axes[i,j]
            ax.set_title(r'$\chi_{eff}$ = '+f"{0.5*(chi_bins[i+1]+chi_bins[i]):.2f}"+r"$\chi_p$= "+f"{0.5*(chip_bins[j+1]+chip_bins[j]):.2f}")
            matrix1 = w[:,:,i,j]
            if(matrix1.min()==matrix1.max()):
                continue
    
            pc = ax.pcolor(mbins,qbins,matrix1.T,norm=LogNorm(vmin=matrix1[matrix1!=0].min(),vmax=matrix1.max()),cmap='viridis')
            ax.set_xscale('log')
            if j==0:
                ax.set_ylabel(r'$q$')
            if i==(len(chi_bins)-1):
                ax.set_xlabel(r'$m_1$')
            

fig.tight_layout()
fig.savefig("__run__/w.png")



vt_utils = Vt_Utils(mbins,qbins,chi_bins,chip_bins,kappa=2.7)


print("Loading Injections and computing spin priors")
inj_dataset = {}
with h5py.File(config['injection_file'],'r') as hf:

    
    m2 = hf["events"][()]["mass2_source"]
    inj_dataset['analysis_time_s'] = hf.attrs['total_analysis_time'] # years
    inj_dataset['total_generated'] = hf.attrs['total_generated']
    mix_weights = hf['events'][()]['weights']
    for param,key in config['injection_keys'].items():
            inj_dataset[param] = hf['events'][()][key]

chi_eff = (inj_dataset['mass1_source']*inj_dataset['spin1z']+
           inj_dataset['mass2_source']*inj_dataset['spin2z'])/(inj_dataset['mass1_source']+
                                                                        inj_dataset['mass2_source'])
q = (inj_dataset["mass2_source"]/inj_dataset["mass1_source"])
chi1p = np.sqrt(inj_dataset["spin1x"]**2+inj_dataset["spin1y"]**2)
chi2p = np.sqrt(inj_dataset["spin2x"]**2+inj_dataset["spin2y"]**2)
chi_p = np.maximum(chi1p, (4 * q + 3) / (3 * q + 4) * q * chi2p)

spin_priors = spin_prior(chi_eff, chi_p, q,amax=0.99)

spin_priors[np.where(np.isnan(spin_priors))]=np.nanmin(spin_priors)

spin_priors[spin_priors<=0] = np.min(spin_priors[spin_priors>0])
inj_dataset['p_draw_chi_given_m1m2'] = spin_priors
thresh = config['threshold']
thresh_keys = config['threshold_keys']
for key in thresh_keys:
    if 'o1o2' in key:
        continue
    inj_dataset[key] = 1./inj_dataset[key] 
inj_dataset["mixture_weight"] = mix_weights
inj_dataset["sampling_pdf"] = np.exp(inj_dataset["sampling_pdf"])




for i, (key, th) in enumerate(zip(thresh_keys, thresh)):
    if i == 0:
        arg = inj_dataset[key]>=th
    else:
        arg += inj_dataset[key]>=th



print("Computing VTs")

vt_means,vt_sigmas = vt_utils.compute_VTs(inj_dataset,thresh,key = thresh_keys )





vts_nd = utils.construct_1dtond_matrix(len(mbins)-1, vt_means, nbins_chip=len(chip_bins)-1, nbins_chieff=len(chi_bins)-1, nbins_q = len(qbins) - 1)





wbyv = wts_sum/(vt_means + 1e-30)





nbins_m = len(mbins)-1
nbins_q = len(qbins) - 1 
nbins_chi = len(chi_bins)-1
nbins_chip = len(chip_bins)-1





interp_vts = vt_means.copy()



log_bin_centers[np.where(args),:]








print("Plotting VTs")


ivts_nd = utils.construct_1dtond_matrix(len(mbins)-1,interp_vts,nbins_chip=len(chip_bins)-1 ,nbins_chieff=len(chi_bins)-1, nbins_q = len(qbins) - 1)
wbyvfinal = w/(ivts_nd + 1e-30)





fig,axes = plt.subplots(len(chi_bins)-1, len(chip_bins)-1, figsize=(8*5*(len(chip_bins)-1)/(len(chi_bins)-1),40))



for i in range(len(chi_bins)-1):
    for j in range(len(chip_bins)-1):
            ax = axes[i,j]
            ax.set_title(r'$\chi_{eff}$ = '+f"{0.5*(chi_bins[i+1]+chi_bins[i]):.2f}"+r"$\chi_p$= "+f"{0.5*(chip_bins[j+1]+chip_bins[j]):.2f}")
            matrix1 = ivts_nd[:,:,i,j]
            if(matrix1.min()==matrix1.max()):
                continue
    
            pc = ax.pcolor(mbins,qbins,matrix1.T,norm=LogNorm(vmin=matrix1[matrix1!=0].min(),vmax=matrix1.max()),cmap='viridis')
            ax.set_xscale('log')
            if j==0:
                ax.set_ylabel(r'$q$')
            if i==(len(chi_bins)-1):
                ax.set_xlabel(r'$m_1$')

fig.tight_layout()
fig.savefig("__run__/vt.png")

print("Computing Lengthscale Priors")
chi_bin_centers = 0.5 * (chi_bins[1:] + chi_bins[:-1])
q_bin_centers = 0.5 * (qbins[1:] + qbins[:-1])
chip_bin_centers = 0.5 * (chip_bins[1:] + chip_bins[:-1]) 
logm_bin_centers = 0.5 * (np.log(mbins[1:]) + np.log(mbins[:-1]))

dist_array = np.zeros(int(nbins_m*(nbins_m+1)))
k=0
for i in range(len(logm_bin_centers)):
    for j in range(i+1):
        dist_array[k] = np.linalg.norm(logm_bin_centers[i]-logm_bin_centers[j])
        k+=1

scale_min = np.log(np.min(dist_array[dist_array!=0.]))
scale_max = np.log(np.max(dist_array))
scale_mean_m = 0.5*(scale_min + scale_max) # chosen to give coverage over the bin-grid
scale_sd_m = (scale_max - scale_min)/4 # fix 3-sigma difference to the sd of the length scale dist

dist_array = np.zeros(int(nbins_q*(nbins_q+1)))
k=0
for i in range(len(q_bin_centers)):
    for j in range(i+1):
        dist_array[k] = np.linalg.norm(q_bin_centers[i]-q_bin_centers[j])
        k+=1

scale_min = np.log(np.min(dist_array[dist_array!=0.]))
scale_max = np.log(np.max(dist_array))
scale_mean_q = 0.5*(scale_min + scale_max) # chosen to give coverage over the bin-grid
scale_sd_q = (scale_max - scale_min)/4 # fix 3-sigma difference to the sd of the length scale dist

dist_array = np.zeros(int(nbins_chi*(nbins_chi+1)))
k=0
for i in range(len(chi_bin_centers)):
    for j in range(i+1):
        dist_array[k] = np.linalg.norm(chi_bin_centers[i]-chi_bin_centers[j])
        k+=1

scale_min = np.log(np.min(dist_array[dist_array!=0.]))
scale_max = np.log(np.max(dist_array))
scale_mean_chi = 0.5*(scale_min + scale_max) # chosen to give coverage over the bin-grid
scale_sd_chi = (scale_max - scale_min)/4 # fix 3-sigma difference to the sd of the length scale dist

dist_array = np.zeros(int(nbins_chip*(nbins_chip+1)))
k=0
for i in range(len(chip_bin_centers)):
    for j in range(i+1):
        dist_array[k] = np.linalg.norm(chip_bin_centers[i]-chip_bin_centers[j])
        k+=1

scale_min = np.log(np.min(dist_array[dist_array!=0.]))
scale_max = np.log(np.max(dist_array))
scale_mean_chip = 0.5*(scale_min + scale_max) # chosen to give coverage over the bin-grid
scale_sd_chip = (scale_max - scale_min)/4 # fix 3-sigma difference to the sd of the length scale dist




scales_m = np.array([scale_mean_m,scale_sd_m])
scales_q = np.array([scale_mean_q,scale_sd_q])
scales_chi = np.array([scale_mean_chi,scale_sd_chi])
scales_chip = np.array([scale_mean_chip,scale_sd_chip])

print("Saving metadata")


with h5py.File(f"__run__/{config['metadat']}", "w") as f:  
    gppop_data = f.create_group('gppop_metadata')
    gppop_data.create_dataset("mbins", data=mbins)
    gppop_data.create_dataset("qbins", data=qbins)
    gppop_data.create_dataset("chi_bins", data=chi_bins)
    gppop_data.create_dataset("chip_bins", data=chip_bins)
    gppop_data.create_dataset("scales_m", data=scales_m)
    gppop_data.create_dataset("scales_q", data=scales_q)
    gppop_data.create_dataset("scales_chi", data=scales_chi) 
    gppop_data.create_dataset("scales_chip", data=scales_chip) 
    gppop_data.create_dataset("posterior_weights", data=weights_cut)
    gppop_data.create_dataset("vts", data=interp_vts/tril_deltaLogbin)
    gppop_data.create_dataset("vt_means", data=vt_means)
    gppop_data.create_dataset("vt_sigmas", data=vt_sigmas)
    gppop_data.create_dataset("wt_means", data=wt_means)
    gppop_data.create_dataset("wt_sigmas", data=wt_sigmas)
