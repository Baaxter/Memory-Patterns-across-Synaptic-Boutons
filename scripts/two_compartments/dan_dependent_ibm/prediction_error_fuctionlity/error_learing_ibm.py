#%% Base
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import io
import importlib.resources as pkg_resources
from sympy import nsolve
from sympy.abc import x, y, z
from IBM_functions_package import activity_model_functions as mdl
from IBM_functions_package import plotting_related_functions

# Load the .npz parameter file from the package folder into memory
with pkg_resources.files("IBM_functions_package").joinpath("model_fit_normfac_3.885.npz").open("rb") as f:
    fitparams = np.load(io.BytesIO(f.read()))  # Read entire file into memory

# Load the MBON tuning file from the same package folder
with pkg_resources.files("IBM_functions_package").joinpath("MBON_sigmoid_tuning_noisy.npz").open("rb") as f:
    mbon_tuning_data = np.load(io.BytesIO(f.read()), allow_pickle=True)

#%% Parameters and Preallocations


# Global parameters  
nKCs       = int(fitparams['restparams'][0])  # Number of Kenyon Cells  
dt         = 0.01                             # Time step for simulation (seconds) [0.01 => Qick Plotting, 0.001 thorough simulation]    
rng        = np.random.default_rng(666)       # Random number generator with a fixed seed for comperability.
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
infp       = fitparams['fittedpars_WT'][2]    # Midpoint of the sigmoidal function for modulation of inhibition [Now 1.28; 0.5 before fitting]  
slf        = fitparams['fittedpars_WT'][3]    # Slope factor of the sigmoidal function for modulation of inhibition [Now 0.16; 0.03 before fitting]

#leraning parameters
learnrate = 0.10            # fixed learning rate
taudan = 2.0                # DAN time constant
alpha = 0.2                 # Stenght of IBM -> This needs can be used to fit the model to the prediction error lering cemincs, 
                            # especially the DAN activity, since IBM indicetly contoldres the calium activity in the lobes it has influence on MBON output and therefore DAN activity

# Parameters for MBON sigmoid fitting
sup  = 1.5   # Desired upper asymptote of the final sigmoid
mbub = 1     # Target output value at x = 1 (used as a constraint)
#%% Preallocations

maxmbs = np.array([mbon_tuning_data['inoff'].mean()])
minmbs = np.array([mbon_tuning_data['inon'].mean()])

# Generate odor stimulus
ston = 10  # Odor onset in seconds
stdur = 60 # Odor duration in seconds
stoff = 70 # Odor offset
tend = 80  # Recording offset
tarr, starr = mdl.generate_step_stimulus(tend, ston, stoff, dt, 1 - bline) # Simulate a 60s long stimulus

# Gernerate shock stimuls
nshocks = 12
shdur   = 1.25
shint   = 3.75
shstr   = 0.5 # fit to match last plots resulting from a pulsed based simulation 
__, sharr = mdl.generate_stimulus_stream(ston + shint/2, nshocks, shdur, shint, shstr, 10 + shint/2 + dt, dt)


# Responder populations and inutscale
nrs =  rng.poisson(prr * nKCs)                  # Number of reliable responders
nurs = rng.poisson(pur * nKCs)                  # Number of unreliable responders
inpscale = np.zeros(nKCs)                       # Sensitivity of KCs to input stimulus
rs = rng.choice(np.arange(nKCs), nrs, replace=False)  # Indices of reliable responders
remaining_indices = list(set(np.arange(nKCs)) - set(rs))    # Remaining indices after picking reliable responders
us = rng.choice(np.array(remaining_indices), nurs, replace=False)  # Indices of unreliable responders
inpscale[rs] = rng.uniform(0.5, 1.0, nrs)          # Reliable responders
inpscale[us] = rng.uniform(0, 0.5, nurs)           # Unreliable responders
responders = np.concatenate([rs, us])
indx_active_kc = np.where(inpscale > 0)[0] # Active cells indices

# Genrate Ihnibition Martix and Scale
inhscale = np.ones([nKCs, nKCs])     # Lateral inhibition connectivity matrix, first dimension is the KC inhibited, second dimension is the KC inhibiting.
inhscale[np.diag_indices(nKCs)] = 0  # Remove self-inhibition
inhscale /= np.sum(inhscale, axis=1)[:, None]  # Normalize so that each KC receives the same amount of inhibition from all other KCs.
inhscale *= inhfactor  # Ramp up the degree of inhibition just to showcase the effect.

# Generate noise
n_str = n_str * np.sqrt(2 / tauKCdec * dt)
noise = rng.standard_normal([len(tarr) - 1, nKCs])
#%% MBON Calculations (Taken form learning_rate_screening_mbon_normalized.py)

#Compute average ON and OFF responses across samples
maxmbs = np.array([mbon_tuning_data['inoff'].mean()])   # Mean OFF response (upper range)
minmbs = np.array([mbon_tuning_data['inon'].mean()])    # Mean ON response  (lower range)

sols = nsolve(
    [
        x + (sup - x) / (1 + np.e**(y / z)),                  # Asymptote constraint
        x - 0.5 + (sup - x) / (1 + np.e**((y - 0.5) / z)),    # Midpoint = 0.5
        x - mbub + (sup - x) / (1 + np.e**((y - 1) / z))      # Output = mbub at x=1
    ],
    [x, y, z],                                                # Unknowns
    [0, 0.5, 0.09]                                            # Initial guess
)

# Extract solved parameters
mnmbon, infpmbon, slfmbon = [float(s) for s in sols]
#%% Preallocate simulation arrays

# --- Calcium dynamics arrays ---
CaKC_cal          = np.zeros([len(tarr), nKCs])   # Calcium in calyx compartment
CaKC_lobe_ave     = np.zeros([len(tarr), nKCs])   # Calcium in aversive lobe
CaKC_lobe_app     = CaKC_lobe_ave.copy()          # Calcium in appetitive lobe (same init as aversive)

# Initial conditions
CaKC_cal[0]       = bline
CaKC_lobe_ave[0]  = bline
CaKC_lobe_app[0]  = bline

# --- Adaptation & inhibition ---
adapt   = np.zeros([len(tarr), nKCs])             # Adaptation array
inh_ave = np.zeros([len(tarr), nKCs])             # Lateral inhibition (aversive)
inh_app = inh_ave.copy()                           # Lateral inhibition (appetitive)

# --- Synaptic weights (time x KCs) ---
synw_ave = np.ones((len(tarr), nKCs))             # Synaptic weights aversive compartment
synw_app = np.ones((len(tarr), nKCs))             # Synaptic weights appetitive compartment

# --- Dopaminergic activity ---
dan_ave  = np.zeros(len(tarr))                    # Negative-valence DAN
dan_app  = np.zeros(len(tarr))                    # Positive-valence DAN

# --- MBON outputs & predicted valence ---
mbon_out_ave_arr = np.zeros(len(tarr))            # MBON output (aversive)
mbon_out_app_arr = np.zeros(len(tarr))            # MBON output (appetitive)

valpred_ave_arr  = np.zeros(len(tarr))            # Predicted valence (aversive)
valpred_app_arr  = np.zeros(len(tarr))            # Predicted valence (appetitive)

# Generate Bouton Scale Array
butscale_ave = np.ones([nKCs, len(tarr)])  # Norm Array as original bouton sensitivity scale (Number of KCs x simulation timestepps)
butscale_app = butscale_ave[:,0]           # Vector of untrained bouton scale [nKCs X 1] as appetetive traing effect
#%% Simulation Loop
for i in range(len(tarr) - 1):

    # --- Calcium dynamics with noise (Euler forward) ---
    CaKC_cal[i + 1] = (CaKC_cal[i] +
                       ((inpscale * starr[i]) / tauinp - (CaKC_cal[i] - bline) / tauKCdec) * dt
                       + n_str * noise[i])

    CaKC_lobe_ave[i + 1] = (CaKC_lobe_ave[i] +
                            ((inpscale * butscale_ave[:, i] * starr[i]) / tauinp - (CaKC_lobe_ave[i] - bline) / tauKCdec) * dt
                            + n_str * noise[i])

    CaKC_lobe_app[i + 1] = (CaKC_lobe_app[i] +
                            ((inpscale * butscale_app * starr[i]) / tauinp - (CaKC_lobe_app[i] - bline) / tauKCdec) * dt
                            + n_str * noise[i])

    # --- Adaptation dynamics (returns adaptation at i+1) ---
    adapt[i + 1] = mdl.adaptation_dynamics(adapt[i], CaKC_cal[i], tauadapt, adaptscale, dt)

    # apply adaptation effect to the newly computed Ca values
    CaKC_cal[i + 1]      -= (adapt[i] * CaKC_cal[i])      * (1 / tauKCdec) * dt
    CaKC_lobe_ave[i + 1] -= (adapt[i] * CaKC_lobe_ave[i]) * (1 / tauKCdec) * dt
    CaKC_lobe_app[i + 1] -= (adapt[i] * CaKC_lobe_app[i]) * (1 / tauKCdec) * dt

    # --- Lateral inhibition dynamics ---
    inh_ave[i + 1] = (
        mdl.inhibition_dynamics(inh_ave[i], CaKC_lobe_ave[i], tauinh, inhscale, dt)
        * mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKC_lobe_ave[i], infp, slf)
    )

    inh_app[i + 1] = (
        mdl.inhibition_dynamics(inh_app[i], CaKC_lobe_app[i], tauinh, inhscale, dt)
        * mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKC_lobe_app[i], infp, slf)
    )

    # apply inhibition using the inhibition computed for this timestep (i+1)
    CaKC_lobe_ave[i + 1] -= inh_ave[i + 1] * (1 / tauKCdec) * dt
    CaKC_lobe_app[i + 1] -= inh_app[i + 1] * (1 / tauKCdec) * dt

    # --- Prevent negative calcium ---
    CaKC_cal[i + 1]      = np.clip(CaKC_cal[i + 1],      0, None)
    CaKC_lobe_ave[i + 1] = np.clip(CaKC_lobe_ave[i + 1], 0, None)
    CaKC_lobe_app[i + 1] = np.clip(CaKC_lobe_app[i + 1], 0, None)

    # --- MBON output (aversive & appetitive) ---
    # each synw_*[i] is a KC->MBON weight vector for time i
    mbon_in_ave = (CaKC_lobe_ave[i] @ synw_ave[i])
    mbon_in_app = (CaKC_lobe_app[i] @ synw_app[i])

    # normalize to sigmoid input range (keeps original form)
    mbon_out_ave = mdl.sigmoidal_func(
        (mbon_in_ave - minmbs[0]) / (maxmbs[0] - minmbs[0]),
        slfmbon, infpmbon, mnmbon, sup - mnmbon
    )
    mbon_out_app = mdl.sigmoidal_func(
        (mbon_in_app - minmbs[0]) / (maxmbs[0] - minmbs[0]),
        slfmbon, infpmbon, mnmbon, sup - mnmbon
    )

    mbon_out_ave_arr[i] = mbon_out_ave
    mbon_out_app_arr[i] = mbon_out_app

    # --- Predicted valence ---
    # keep your original sign choices but be explicit
    valpred_ave = -(mbon_out_app - mbon_out_ave)   # aversive-prediction signal
    valpred_app = -(mbon_out_ave - mbon_out_app)   # appetitive-prediction signal

    valpred_ave_arr[i] = valpred_ave
    valpred_app_arr[i] = valpred_app

    # --- DAN update ---
    # Use the stimulus and the appropriate valence for each DAN
    dan_ave[i + 1] = mdl.DAN_dynamics(dan_ave[i], taudan, dt, sharr[i], -(valpred_ave))
    dan_app[i + 1] = mdl.DAN_dynamics(dan_app[i], taudan, dt, sharr[0], valpred_app)

    # --- Synaptic plasticity ---
    synw_ave[i + 1] = mdl.KC_MBON_coincidence_based_weight_change(
        synw_ave[i], CaKC_lobe_ave[i], dan_ave[i], dt, learnrate
    )
    synw_app[i + 1] = mdl.KC_MBON_coincidence_based_weight_change(
        synw_app[i], CaKC_lobe_app[i], dan_app[i], dt, learnrate
    )

    # Update bouton sensitivtiy scale in aversive compartment response to aversive singaling
    butscale_ave[responders, i + 1] = butscale_ave[responders, i] - (CaKC_lobe_ave[i, responders] * dan_ave[i] * alpha) * dt

# Final valence report
predicted_valence = valpred_ave
print(f"Final predicted valence: {predicted_valence:.4f}")
#%% Control / Conditioning Activity Lineplot with Shock Indicators

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(16,11))

# Name the figure 
fig.canvas.manager.set_window_title("Inter Conditioning Activity Precition Error with IBM Line Plot")

# Set y-axis limits for both plots
axs[0].set_ylim(-0.0068, 2.3)  # Adjust as needed

# --- Shock timing ---
shock_starts = ston + shint/2 + np.arange(nshocks) * (shdur + shint)
shock_ends   = shock_starts + shdur

# --- Aversive Compartment ---
axs[0].plot(tarr, CaKC_lobe_ave[:, rs], color='r', alpha=0.3)                                         # Aversive compartment reliable (red)
axs[0].plot(tarr, CaKC_lobe_ave[:, us], color='darkorange', alpha=0.25)                               # Aversive compartment unreliable (orange)
axs[0].plot(tarr, np.mean(CaKC_lobe_ave[:, responders], axis=1), color='black', lw=4)                 # Mean activity
axs[0].hlines(y=0.006, xmin=ston, xmax=stoff, colors='gray', linestyles='-', lw=5, label='Stimulus')  # Gray stimulus line

# Shock indicators (red bars)
for s, e in zip(shock_starts, shock_ends):
    axs[0].hlines(y=0.006, xmin=s, xmax=e, colors='maroon', lw=5)

axs[0].set_title('Aversive Compartment')

# --- Appetitive Compartment ---
axs[1].plot(tarr, CaKC_lobe_app[:, rs], color='r', alpha=0.3)                                         # Appetitive compartment reliable (red)
axs[1].plot(tarr, CaKC_lobe_app[:, us], color='darkorange', alpha=0.3)                                 # Appetitive compartment unreliable (orange)
axs[1].plot(tarr, np.mean(CaKC_lobe_app[:, responders], axis=1), color='black', lw=4)                 # Mean activity
axs[1].hlines(y=0.006, xmin=ston, xmax=stoff, colors='gray', linestyles='-', lw=5)                     # Gray stimulus line

# Shock indicators (red bars)
for s, e in zip(shock_starts, shock_ends):
    axs[1].hlines(y=0.006, xmin=s, xmax=e, colors='maroon', lw=5)

axs[1].set_title('Appetitive Compartment')

# --- Empty plots for custom legend ---
axs[1].plot([], [], color='r', lw=2.5, label='Reliable')             # Empty plot for legend 
axs[1].plot([], [], color='darkorange', lw=2.5, label='Unreliable')  # Empty plot for legend
axs[1].plot([], [], color='black', lw=4, label='Mean Activity')      # Empty plot for legend

axs[1].legend(loc='upper right', frameon=False)  # Legend on right plot

# --- Labels ---
fig.text(0.528, 0.02, 't [s]', ha='center')   # Centered x-label
fig.supylabel('Ca Activity (a.u)')            # Super y-axis label for the entire figure

# --- Adjust layout ---
fig.subplots_adjust(
    top=0.935,
    bottom=0.09,
    left=0.095,
    right=0.98,
    hspace=0.2,
    wspace=0.2
)

plt.show()
#%% Aversive / Appetitive Interconditioning Activity with Stimulus and Shock Indicators

# Create figure and subplots with shared x/y axes
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(16,11))

# Name the figure 
fig.canvas.manager.set_window_title("Inter Conditioning Activity Precition Error with IBM")

# Set y-limits
axs[0].set_ylim([-0.0065, 2.2])

# --- Shock timing ---
shock_starts = ston + shint/2 + np.arange(nshocks) * (shdur + shint)
shock_ends   = shock_starts + shdur

# --- Compute mean and standard deviation for aversive compartment ---
mean_rs_ave = np.mean(CaKC_lobe_ave[:, rs], axis=1)
std_rs_ave = np.std(CaKC_lobe_ave[:, rs], axis=1)
mean_us_ave = np.mean(CaKC_lobe_ave[:, us], axis=1)
std_us_ave = np.std(CaKC_lobe_ave[:, us], axis=1)

# Reliable responders mean and shading (Aversive)
axs[0].plot(tarr, mean_rs_ave, 'r', label='Reliable', lw=2)
axs[0].fill_between(tarr, 
                    np.clip(mean_rs_ave - std_rs_ave, 0, None), 
                    mean_rs_ave + std_rs_ave, 
                    color='r', alpha=0.25)

# Unreliable responders mean and shading (Aversive)
axs[0].plot(tarr, mean_us_ave, color='darkorange', label='Unreliable', lw=2)
axs[0].fill_between(tarr, 
                    np.clip(mean_us_ave - std_us_ave, 0, None), 
                    mean_us_ave + std_us_ave, 
                    color='darkorange', alpha=0.25)

# Gray stimulus indicator
axs[0].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * 0.004, '-', lw=5, color='gray')

# Shock indicators (maroon bars above stimulus line)
for s, e in zip(shock_starts, shock_ends):
    axs[0].hlines(y=0.001, xmin=s, xmax=e, colors='maroon', lw=5)

axs[0].set_ylabel(r'Activity [a.u.]')
axs[0].set_title('Aversive Compartment')

# --- Compute mean and standard deviation for appetitive compartment ---
mean_rs_app = np.mean(CaKC_lobe_app[:, rs], axis=1)
std_rs_app = np.std(CaKC_lobe_app[:, rs], axis=1)
mean_us_app = np.mean(CaKC_lobe_app[:, us], axis=1)
std_us_app = np.std(CaKC_lobe_app[:, us], axis=1)

# Reliable responders mean and shading (Appetitive)
axs[1].plot(tarr, mean_rs_app, 'r', label='Reliable', lw=2)
axs[1].fill_between(tarr, 
                    np.clip(mean_rs_app - std_rs_app, 0, None), 
                    mean_rs_app + std_rs_app, 
                    color='r', alpha=0.25)

# Unreliable responders mean and shading (Appetitive)
axs[1].plot(tarr, mean_us_app, color='darkorange', label='Unreliable', lw=2)
axs[1].fill_between(tarr, 
                    np.clip(mean_us_app - std_us_app, 0, None), 
                    mean_us_app + std_us_app, 
                    color='darkorange', alpha=0.25)

# Gray stimulus indicator
axs[1].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * 0.004, '-', lw=5, color='gray')

# Shock indicators (maroon bars)
for s, e in zip(shock_starts, shock_ends):
    axs[1].hlines(y=0.001, xmin=s, xmax=e, colors='maroon', lw=5)

axs[1].set_title('Appetitive Compartment')
axs[1].legend(loc='upper right')

# --- Common x-label ---
fig.text(0.525, 0.024, 't [s]', ha='center')

# --- Adjust layout ---
fig.subplots_adjust(
    top=0.95,
    bottom=0.09,
    left=0.08,
    right=0.98,
    hspace=0.2,
    wspace=0.1
)

plt.show()

#%% Improved Calcium, DAN, Weight, Valence plotting

shock_starts = ston + shint/2 + np.arange(nshocks) * (shdur + shint)
shock_ends   = shock_starts + shdur

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
fig.canvas.manager.set_window_title("Prediction Error Learning mecanics with IBM")
plt.subplots_adjust(hspace=0.35, left=0.12, right=0.85)   # More space

# ==== 1. MBON activity ====
axes[0].plot(tarr, mbon_out_app_arr, color='gold', lw=2, label='Avoidance')
axes[0].plot(tarr, mbon_out_ave_arr, color='teal', lw=2, label='Approach')
axes[0].set_ylabel('MBON activity')
# --- Gray line on TOP of the axis (using axis-coordinates) ---
axes[0].hlines(
    y=0.037,                             # 1.0 = top of axes
    xmin=ston, xmax=stoff,
    colors='gray', linestyles='-', lw=5,
    transform=axes[0].get_xaxis_transform(),   # <-- key line!
)
# --- Shock indicators (red bars above gray line) ---
for s, e in zip(shock_starts, shock_ends):
    axes[0].hlines(
        y=-0.6,         # slightly above 0.037 gray bar
        xmin=s, xmax=e,
        colors='maroon', lw=5
    )
axes[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# ==== 2. DAN activity ====
axes[1].plot(tarr, dan_app, color='gold', lw=2, label='+ Valence')
axes[1].plot(tarr, dan_ave, color='teal', lw=2, label='- Valence')
axes[1].set_ylabel('DAN activity')
# --- Gray line on TOP of the axis (same style as MBON) ---
axes[1].hlines(
    y=0.039,                           # matching MBON axis-line height
    xmin=ston, xmax=stoff,
    colors='gray', linestyles='-', lw=5,
    transform=axes[1].get_xaxis_transform(),
)
for s, e in zip(shock_starts, shock_ends):
    axes[1].hlines(
        y=0.0001,         # slightly above the gray line position 0.033
        xmin=s, xmax=e,
        colors='maroon', lw=5
    )
axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# ==== 3. Synaptic weights ====
axes[2].plot(tarr, np.mean(synw_app, axis=1), color='gold', lw=2, label='Aversive Comp.')
axes[2].plot(tarr, np.mean(synw_ave, axis=1), color='teal', lw=2, label='Appetetiv Comp.')
axes[2].set_ylabel('Syn weight')
# --- Gray line on TOP of the axis (same style as MBON & DAN) ---
axes[2].hlines(
    y=0.965,                             # same axis-relative vertical position
    xmin=ston, xmax=stoff,
    colors='gray', linestyles='-', lw=5,
)
for s, e in zip(shock_starts, shock_ends):
    axes[2].hlines(
        y=0.965,         # slightly above the gray bar at 0.03959
        xmin=s, xmax=e,
        colors='maroon', lw=5
    )
axes[2].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# ==== 4. Predicted valence ====
axes[3].plot(tarr, valpred_ave_arr, color='teal', lw=2, label='CS+')
axes[3].set_ylabel('Valence')
axes[3].set_xlabel('t [s]')
# --- Gray line on TOP of the axis (same as all others) ---
axes[3].hlines(
    y=0.037,                             # same axis-relative position
    xmin=ston, xmax=stoff,
    colors='gray', linestyles='-', lw=5,
    transform=axes[3].get_xaxis_transform(),
)
for s, e in zip(shock_starts, shock_ends):
    axes[3].hlines(
        y=-0.92,         # slightly above 0.037 gray line
        xmin=s, xmax=e,
        colors='maroon', lw=5
    )
axes[3].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

plt.tight_layout()
plt.subplots_adjust(
    left=0.092,
    bottom=0.133,
    right=0.794,
    top=0.878,
    wspace=0.202,
    hspace=0.343,
)

plt.show()