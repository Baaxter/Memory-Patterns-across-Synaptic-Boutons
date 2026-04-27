#%% Base
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import io
import time
import importlib.resources as pkg_resources  # Importing package resource handler
from IBM_functions_package import activity_model_functions as mdl
from IBM_functions_package import plotting_related_functions

# Load the .npz parameter file from the package folder into memory
with pkg_resources.files("IBM_functions_package").joinpath("model_fit_normfac_3.885.npz").open("rb") as f:
    fitparams = np.load(io.BytesIO(f.read()))  # Read entire file into memory
    
# Define save path for plots
save_path = "/home/baxter/IBM/presentations/figures/two_comaprtments"
os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
#%% Parameters

# Global parameters  
nKCs       = int(fitparams['restparams'][0])  # Number of Kenyon Cells 
dt         = 0.01                             # Time step for simulation (seconds) [0.01 => Qick Plotting, 0.001 thorough simulation]  
rng        = np.random.default_rng(420)       # Random number generator with a fixed seed for comperability. 
bline      = fitparams['fittedpars_KD'][4]    # Baseline calcium concentration [Now 0.0; 0.02 before fitting]  

# Input scale  
prr        = fitparams['restparams'][1]       # Rate of reliable responders 
pur        = fitparams['restparams'][2]       # Rate of unreliable responders  

# Null parameters  
tauinp     = fitparams['fittedpars_KD'][1]    # Input time constant (seconds) [Now 0.149; 0.2 before fitting]  
tauKCdec   = fitparams['fittedpars_KD'][0]    # Calcium decay time constant (seconds) [Now 0.44; 1.5 before fitting]  
n_str      = fitparams['fittedpars_KD'][5]    # Standard deviation of OU-process [Now 0.020; 0.01 before fitting]  

# Adaptation parameters  
tauadapt   = fitparams['fittedpars_KD'][2]    # Time constant for adaptation (seconds) [Now 2.68; 2.0 before fitting]  
adaptscale = fitparams['fittedpars_KD'][3]    # Strength of adaptation [Now 0.58; 1.0 before fitting]  

# Inhibition parameters  
tauinh     = fitparams['fittedpars_WT'][0]    # Time constant for lateral inhibition (seconds) [Now 1.004; 1.5 before fitting]  
inhfactor  = fitparams['fittedpars_WT'][1]    # Inhibition scale factor [Now 15.53; 10 for sparse input, 1 for uniform input before fitting]  
infp       = fitparams['fittedpars_WT'][2]    # Midpoint of the sigmoidal function [Now 1.28; 0.5 before fitting]  
slf        = fitparams['fittedpars_WT'][3]    # Slope factor of the sigmoidal function [Now 0.16; 0.03 before fitting]  

#%% Preallocations

# Generate tarr, starr (time array, stimulus array)
burnin_time = 2  # Burn period in seconds
ston = 5  # Odor onset in seconds
stdur = 5  # Odor duration in seconds
stoff = ston + stdur # Stimulus offset
tend = stoff + 10 # Simulation termination
tarr, starr = mdl.generate_step_stimulus(tend, ston, stoff, dt, 1 - bline)  # Run a 5s long stimulus

# Generate odor / input scale
nrs = rng.poisson(prr * nKCs)                      # Number of reliable responders
nurs = rng.poisson(pur * nKCs)                     # Number of unreliable responders
inpscale = np.zeros(nKCs)                          # Sensitivity of KCs to input stimulus
rs = rng.choice(np.arange(nKCs), nrs, replace=False)               # Indices of reliable responders
remaining_indices = list(set(np.arange(nKCs)) - set(rs))           # Remaining indices after picking reliable responders
us = rng.choice(np.array(remaining_indices), nurs, replace=False)  # Indices of unreliable responders
inpscale[rs] = rng.uniform(0.5, 1.0, nrs)          # Reliable responders
inpscale[us] = rng.uniform(0, 0.5, nurs)           # Unreliable responders
responders = np.concatenate([rs, us])              # Responding boutons

# Generate Inhibition Matrix and Scale
inhscale = np.ones([nKCs, nKCs])               # Lateral inhibition connectivity matrix, first dimension inhibited KC, second dimension is the inhibiting KC
inhscale[np.diag_indices(nKCs)] = 0            # Remove self-inhibition
inhscale /= np.sum(inhscale, axis=1)[:, None]  # Normalize so each KC receives equal inhibition from all others
inhscale *= inhfactor                          # Ramp up inhibition to showcase the effect

# Generate noise
n_str = n_str * np.sqrt(2 / tauKCdec * dt)
noise = rng.standard_normal([len(tarr) - 1, nKCs])

# Generate Simulation arrays
CaKCcal = np.zeros([len(tarr), nKCs])       # Calyx calcium array
CaKClobe_ave = np.zeros([len(tarr), nKCs])  # Aversive compartment calcium array
CaKClobe_app = np.zeros([len(tarr), nKCs])  # Appetitive compartment calcium array
CaKCcal[0] = bline                          # Calyx baseline 
CaKClobe_ave[0] = bline                     # Aversive compartment baseline
CaKClobe_app[0] = bline                     # Appetetive compartment baseline
adapt = np.zeros([len(tarr), nKCs])         # Adaptation array - sheared by all compartmetns since calyx gernated
inh_ave = np.zeros([len(tarr), nKCs])       # Lateral inhibition array aversive compartment
inh_app = np.zeros([len(tarr), nKCs])       # Lateral inhibition array appetitive compartment
#%% Simulations

for i in range(len(tarr) - 1):
    # Null dynamics
    CaKCcal[i + 1] = CaKCcal[i] + ((inpscale * starr[i]) / tauinp - (CaKCcal[i] - bline) / tauKCdec) * dt
    CaKClobe_ave[i + 1] = CaKClobe_ave[i] + ((inpscale * starr[i]) / tauinp - (CaKClobe_ave[i] - bline) / tauKCdec) * dt
    CaKClobe_app[i + 1] = CaKClobe_app[i] + ((inpscale * starr[i]) / tauinp - (CaKClobe_app[i] - bline) / tauKCdec) * dt

    # Add noise
    CaKCcal[i + 1] += noise[i] * n_str
    CaKClobe_ave[i + 1] += noise[i] * n_str
    CaKClobe_app[i + 1] += noise[i] * n_str

    # Adaptation
    adapt[i + 1] = mdl.adaptation_dynamics(adapt[i], CaKCcal[i], tauadapt, adaptscale, dt)
    CaKCcal[i + 1] -= (adapt[i] * CaKCcal[i]) * (1 / tauKCdec) * dt
    CaKClobe_ave[i + 1] -= (adapt[i] * CaKClobe_ave[i]) * (1 / tauKCdec) * dt
    CaKClobe_app[i + 1] -= (adapt[i] * CaKClobe_app[i]) * (1 / tauKCdec) * dt

    # Inhibition 
    inh_ave[i + 1] = mdl.inhibition_dynamics(inh_ave[i], CaKClobe_ave[i], tauinh, inhscale, dt) \
                     * mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_ave[i], infp, slf)
    inh_app[i + 1] = mdl.inhibition_dynamics(inh_app[i], CaKClobe_ave[i], tauinh, inhscale, dt) \
                     * mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_ave[i], infp, slf)
    CaKClobe_ave[i + 1] -= (inh_ave[i]) * (1 / tauKCdec) * dt
    CaKClobe_app[i + 1] -= (inh_ave[i]) * (1 / tauKCdec) * dt

    # Make sure everything is non-negative
    CaKCcal[i + 1] = np.clip(CaKCcal[i + 1], 0, None)
    CaKClobe_ave[i + 1] = np.clip(CaKClobe_ave[i + 1], 0, None)
    CaKClobe_app[i + 1] = np.clip(CaKClobe_app[i + 1], 0, None)

#%% Plotting Calyx and Aversive

# Prepare data
burnin_index = int(burnin_time / dt)
plot_tarr = tarr[burnin_index:]
plot_CaKCcal = CaKCcal[burnin_index:]
plot_CaKClobe_ave = CaKClobe_ave[burnin_index:]
plot_starr = starr[burnin_index:]

# Plotting
fig, ax = plt.subplots(figsize=(16, 10)) # Inches, put in dimensions for putting into project
fig.canvas.manager.set_window_title("Calyx and Aversive Compartment")

ax.plot(plot_tarr, np.mean(plot_CaKCcal[:, responders], axis=1), 'g', lw=2.5, label='Calyx')
ax.plot(plot_tarr, np.mean(plot_CaKClobe_ave[:, responders], axis=1), 'k', lw=2.5, label='Aversive Compartment')
ax.plot(plot_tarr[plot_starr > 0], np.ones(len(plot_tarr[plot_starr > 0])) * -0.01, color='black', lw=5)

ax.set_xlabel('t [s]')
ax.set_ylabel('Activity (a.u.)')
ax.legend(loc='upper right', frameon=False)

fig.subplots_adjust(
    top=0.955,
    bottom=0.09,
    left=0.07,
    right=0.98,
    hspace=0.2,
    wspace=0.2
)

plt.show()

# Save in both PNG and SVG formats
fig.savefig(os.path.join(save_path, "calyx_aversive.png"))
fig.savefig(os.path.join(save_path, "calyx_aversive.svg"))
#%% Modified Plotting: Side-by-side Aversive and Appetitive Compartments

plot_CaKClobe_app = CaKClobe_app[burnin_index:]

fig, axs = plt.subplots(1, 2, figsize=(18, 10))
fig.canvas.manager.set_window_title("Calyx, Aversive and Appetetive Compartments")

# Aversive compartment
axs[0].plot(plot_tarr, np.mean(plot_CaKCcal[:, responders], axis=1), 'g', lw=2.5, label='Calyx')
axs[0].plot(plot_tarr, np.mean(plot_CaKClobe_ave[:, responders], axis=1), 'k', lw=2.5, label='Aversive Compartment')
axs[0].plot(plot_tarr[plot_starr > 0], np.ones(len(plot_tarr[plot_starr > 0])) * -0.01, color='black', lw=5)
axs[0].set_ylabel('Activity (a.u.)')

# Appetitive compartment
axs[1].plot(plot_tarr, np.mean(plot_CaKCcal[:, responders], axis=1), 'g', lw=2.5, label='Calyx')
axs[1].plot(plot_tarr, np.mean(plot_CaKClobe_app[:, responders], axis=1), 'k', lw=2.5, label='Appetitive Compartment')
axs[1].plot(plot_tarr[plot_starr > 0], np.ones(len(plot_tarr[plot_starr > 0])) * -0.01, color='black', lw=5)

# Turn off tick labels in second subplot
axs[1].set_yticklabels([])

axs[0].legend(loc='upper left', bbox_to_anchor=(0.48, 1.02), frameon=False)
axs[1].legend(loc='upper left', bbox_to_anchor=(0.48, 1.02), frameon=False)

# Set common x-label at the figure level
fig.text(0.5, 0.024, 't [s]', ha='center')

fig.subplots_adjust(
    top=0.964,
    bottom=0.076,
    left=0.074,
    right=0.881,
    hspace=0.2,
    wspace=0.305
    )

plt.show()
# Save in both PNG and SVG formats
fig.savefig(os.path.join(save_path, "calyx_aversive_appetetive_compartments.png"))
fig.savefig(os.path.join(save_path, "calyx_aversive_appetetive_compartments.svg"))

