#%% Base
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import io
import importlib.resources as pkg_resources  # Importing package resource handler
from IBM_functions_package import activity_model_functions as mdl
from IBM_functions_package import plotting_related_functions

# Load the .npz parameter file from the package folder into memory
with pkg_resources.files("IBM_functions_package").joinpath("model_fit_normfac_3.885.npz").open("rb") as f:
    fitparams = np.load(io.BytesIO(f.read()))  # Read entire file into memory


# Set random seed for reproducibility 420 for example of better model behavior
rng = np.random.default_rng(420) 

# Global parameters  
nKCs       = int(fitparams['restparams'][0])  # Number of Kenyon Cells [Now 700; 700 before fitting]  
dt         = 0.01                             # Time step for simulation (seconds) [Now 0.01; 0.001 before fitting]  
rng        = np.random.default_rng(420)       # Random number generator with a fixed seed [Now 420; 666 before fitting]  
bline      = fitparams['fittedpars_KD'][4]    # Baseline calcium concentration [Now 0.0; 0.02 before fitting]  

# Input scale  
prr        = fitparams['restparams'][1]       # Rate of reliable responders [Now 0.05; 0.05 before fitting]  
pur        = fitparams['restparams'][2]       # Rate of unreliable responders [Now 0.15; 0.15 before fitting]  

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


# Generate tarr, starr (stimulus and time arry)
burnin_time = 2  # Burn-in period in seconds
ston = 5 # odor onset in seconds
stdur = 5 # odor duration in seconds
stoff = ston + stdur
tend = stoff + 5
tarr, starr = mdl.generate_step_stimulus(tend, ston, stoff, dt, 1 - bline) # Run a 5s long stimulus

# Generate odor / input scale
nrs = rng.poisson(prr * nKCs)                    # Number of reliable responders
nurs = rng.poisson(pur * nKCs)                   # Number of unreliable responders
inpscale = np.zeros(nKCs)                       # Sensitivity of KCs to input stimulus
rs = rng.choice(np.arange(nKCs), nrs, replace=False)  # Indices of reliable responders
remaining_indices = list(set(np.arange(nKCs)) - set(rs))    # Remaining indices after picking reliable responders
us = rng.choice(np.array(remaining_indices), nurs, replace=False)  # Indices of unreliable responders
inpscale[rs] = rng.uniform(0.5, 1.0, nrs)          # Reliable responders
inpscale[us] = rng.uniform(0, 0.5, nurs)           # Unreliable responders
responders = np.where(inpscale > 0)[0]             # List of responing KCs 

# Genrate Ihnibition Martix and Scale
inhscale = np.ones([nKCs, nKCs])     # Lateral inhibition connectivity matrix, first dimension is the KC inhibited, second dimension is the KC inhibiting.
inhscale[np.diag_indices(nKCs)] = 0  # Remove self-inhibition
inhscale /= np.sum(inhscale, axis=1)[:, None]  # Normalize so that each KC receives the same amount of inhibition from all other KCs.
inhscale *= inhfactor  # Ramp up the degree of inhibition just to showcase the effect.

# Generate noise
n_str = n_str * np.sqrt(2 / tauKCdec * dt)
noise = rng.standard_normal([len(tarr) - 1, nKCs])

# Generate Simulation arrays
CaKCcal = np.zeros([len(tarr), nKCs])  # Calix calcium transient array preallocated. Shape is time x KCs
CaKClobe = np.zeros([len(tarr), nKCs])  # MB lobe for inhibition condition with activity-dependent modulation
CaKCcal[0] = bline  # Initial calcium concentration
CaKClobe[0] = bline  # Initial calcium concentration
adapt = np.zeros([len(tarr), nKCs])  # Adaptation array - shared by all since calyx-only
inh = np.zeros([len(tarr), nKCs])  # Lateral inhibition array


for i in range(len(tarr) - 1):
    # Null
    CaKCcal[i + 1] = CaKCcal[i] + ((inpscale * starr[i]) / tauinp - (CaKCcal[i] - bline) / tauKCdec) * dt
    CaKClobe[i + 1] = CaKClobe[i] + ((inpscale * starr[i]) / tauinp - (CaKClobe[i] - bline) / tauKCdec) * dt

    # ADD NOISE
    CaKCcal[i + 1] += noise[i] * n_str
    CaKClobe[i + 1] += noise[i] * n_str

    # Adaptation
    adapt[i + 1] = mdl.adaptation_dynamics(adapt[i], CaKCcal[i], tauadapt, adaptscale, dt)
    CaKCcal[i + 1] -= (adapt[i] * CaKCcal[i]) * (1 / tauKCdec) * dt
    CaKClobe[i + 1] -= (adapt[i] * CaKClobe[i]) * (1 / tauKCdec) * dt

    # Inhibition 
    inh[i + 1] = mdl.inhibition_dynamics(inh[i], CaKClobe[i], tauinh, inhscale, dt) \
                 * mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe[i], infp, slf)  # modulated inhibition
    CaKClobe[i + 1] -= (inh[i]) * (1 / tauKCdec) * dt

    # Make sure everything is nonzero
    CaKCcal[i + 1] = np.clip(CaKCcal[i + 1], 0, None)
    CaKClobe[i + 1] = np.clip(CaKClobe[i + 1], 0, None)
#%% Plotting

# Exclude the burn-in time from the plotting data
burnin_index = int(burnin_time / dt)  # Convert burn-in time to index
plot_tarr = tarr[burnin_index:]
plot_CaKCcal = CaKCcal[burnin_index:]
plot_CaKClobe = CaKClobe[burnin_index:]
plot_starr = starr[burnin_index:]

# Generate the plot with enhanced styling
fig, ax = plt.subplots()  # High-resolution

fig.canvas.manager.set_window_title("Calyx Lobe Activity")

# Plot with improved style
ax.plot(plot_tarr, np.mean(plot_CaKCcal[:, responders], axis=1),'g', lw=2, label=r'Calyx')
ax.plot(plot_tarr, np.mean(plot_CaKClobe[:, responders], axis=1), 'k', lw=2, label=r'Lobe')
ax.plot(plot_tarr[plot_starr > 0], np.ones(len(plot_tarr[plot_starr > 0])) * -0.01, color='black', lw=5)

# Add title and adjust styling
ax.set_xlabel('Time (s)')
ax.set_ylabel('Activity (a.u.)')
ax.legend(loc='upper right')

# Display the plot without interactive window settings for consistent figure rendering
plt.show()

