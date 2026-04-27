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

# Define save path for plots
save_path = "/home/baxter/IBM/presentations/figures/two_compartments"
os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
#%% Parameters

# Global parameters  
nKCs       = int(fitparams['restparams'][0])  # Number of Kenyon Cells  
dt         = 0.01                             # Time step for simulation (seconds) [0.01 => Qick Plotting, 0.001 thorough simulation]    
rng        = np.random.default_rng(420)       # Random number generator with a fixed seed for comperability.
bline      = fitparams['fittedpars_KD'][4]    # Baseline calcium concentration [Now 0.0; 0.02 before fitting]

# Input scale  
prr        = fitparams['restparams'][1]       # Rate of reliable responders to test odor
pur        = fitparams['restparams'][2]       # Rate of unreliable responders to test odor  

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

# DAN pulse timing intesety settings
npules = 12                # Number of DAN pulses
pulsdur = 1.25             # Pulse duration in seconds (on time)
pulsint = 3.75             # Inter-pulse interval in seconds (off time)
danon =  15                # DAN onset time at 15 seconds
danoff = 75                # DAN offset time at 75 seconds
dan_0 = 0.2                # Initial pulse magnitude (default from thesis 0.5)
r = 0.8                    # Reduction factor, will be raised for current pulse index
alpha = 0.5                # Learning rate

# Generate tarr, starr (stimulus and time array) (AVERSIVE CONDTIOING SETTING)
ston = 10  # Odor onset in seconds
stdur = 60 # Odor duration in seconds
stoff = 70 # Odor offset
tend = 80  # Recording offset
tarr, starr = mdl.generate_step_stimulus(tend, ston, stoff, dt, 1 - bline) # Simulate a 60s long stimulus

# Generate odor / input scale
nrs =  rng.poisson(prr * nKCs)                    # Number of reliable responders
nurs = rng.poisson(pur * nKCs)                    # Number of unreliable responders
inpscale = np.zeros(nKCs)                         # Sensitivity of KCs to input stimulus
rs = rng.choice(np.arange(nKCs), nrs, replace=False)  # Indices of reliable responders
remaining_indices = list(set(np.arange(nKCs)) - set(rs))           # Remaining indices after picking reliable responders
us = rng.choice(np.array(remaining_indices), nurs, replace=False)  # Indices of unreliable responders
inpscale[rs] = rng.uniform(0.5, 1.0, nrs)          # Reliable responders
inpscale[us] = rng.uniform(0, 0.5, nurs)           # Unreliable responders
responders = np.concatenate([rs, us])              # Responding boutons

# Generate Inhibition Matrix and Scale
inhscale = np.ones([nKCs, nKCs])              # Lateral inhibition connectivity matrix [inhibited KC X inhibiting KC]
inhscale[np.diag_indices(nKCs)] = 0           # Remove self-inhibition
inhscale /= np.sum(inhscale, axis=1)[:, None] # Normalize so that each KC receives the same amount of inhibition from all other KCs.
inhscale *= inhfactor                         # Control inhibition magnitude 

# Generate noise
n_str = n_str * np.sqrt(2 / tauKCdec * dt)
noise = rng.standard_normal([len(tarr) - 1, nKCs])

# Generate DAN-Array
__, danarr = mdl.generate_step_stimulus(tend, danon, danoff, dt) # Generates an array with 1s between danon and danoff
dan_start_idx = int(danon / dt)     # 15 s -> tarr index 1500
dan_end_idx = int(danoff / dt)      # 75 s -> tarr index 7500
pulse_duration_in_steps = int(pulsdur / dt) + 1               # 1.25 s "on" period (126 steps)
interval_duration_in_steps = int((pulsdur + pulsint) / dt)    # 5 s total per pulse (500 steps)

# Apply pulse pattern (1.25s on, 3.75s off) to the DAN array, with decreasing magnitude
for pulse_idx in range(npules):
    # Calculate the magnitude for a given pulse with uesung reduction factor r
    current_danmag = dan_0 * (r ** pulse_idx)
    
    # Calculate start and end indices for each pulse in the DAN array
    pulse_start_idx = dan_start_idx + pulse_idx * interval_duration_in_steps
    pulse_end_idx = pulse_start_idx + pulse_duration_in_steps

    # Set the "on" period to the current pulse magnitude
    danarr[pulse_start_idx:pulse_end_idx] = current_danmag  
    # Set the subsequent "off" period to 0
    danarr[pulse_end_idx:pulse_start_idx + interval_duration_in_steps] = 0  

# Ensure that all time points from the DAN offset (75 s, index 7500) onward are set to 0
danarr[dan_end_idx:] = 0

# Generate Simulation arrays
CaKCcal   = np.zeros([len(tarr), nKCs])     # Calyx calcium array
CaKClobe_ave = np.zeros([len(tarr), nKCs])  # Aversive compartment calcium array
CaKClobe_app =  np.zeros([len(tarr), nKCs]) # Appetitive compartment calcium array
CaKCcal[0] = bline                          # Calyx baseline 
CaKClobe_ave[0] = bline                     # Aversive compartment baseline
CaKClobe_app[0] = bline                     # Appetetive compartment baseline
adapt = np.zeros([len(tarr), nKCs])         # Adaptation array - sheared by all compartmetns since calyx gernated
inh_ave = np.zeros([len(tarr), nKCs])       # Lateral inhibition array aversive compartment
inh_app = np.zeros([len(tarr), nKCs])       # Lateral inhibition array appetitive compartment

# Generate Bouton Scale Array
butscale_ave = np.ones([nKCs, len(tarr)])  # Norm Array as original bouton sensitivity scale (Number of KCs x simulation timestepps)
butscale_app = butscale_ave[:,0]           # Vector of untrained bouton scale [nKCs X 1] as appetetive traing effect
#%% Simulations

# Simulate calcium arrays useing Euler's method
for i in range(len(tarr) - 1):
    # Calyx activity
    CaKCcal[i + 1] = CaKCcal[i] + ((inpscale * starr[i]) / tauinp - (CaKCcal[i] - bline) / tauKCdec) * dt
    CaKClobe_ave[i + 1] = CaKClobe_ave[i] + ((inpscale * butscale_ave[:, i] * starr[i]) / tauinp - (CaKClobe_ave[i] - bline) / tauKCdec) * dt
    CaKClobe_app[i + 1] = CaKClobe_app[i] + ((inpscale * butscale_app * starr[i]) / tauinp - (CaKClobe_app[i] - bline) / tauKCdec) * dt
    
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
    inh_ave[i + 1] = mdl.inhibition_dynamics(inh_ave[i], CaKClobe_ave[i], tauinh, inhscale, dt) * \
                     mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_ave[i], infp, slf)
    inh_app[i + 1] = mdl.inhibition_dynamics(inh_app[i], CaKClobe_app[i], tauinh, inhscale, dt) * \
                     mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_app[i], infp, slf)
    CaKClobe_ave[i + 1] -= (inh_ave[i]) * (1 / tauKCdec) * dt
    CaKClobe_app[i + 1] -= (inh_app[i]) * (1 / tauKCdec) * dt

    # Ensure non-zero activity values
    CaKCcal[i + 1] = np.clip(CaKCcal[i + 1], 0, None)
    CaKClobe_ave[i + 1] = np.clip(CaKClobe_ave[i + 1], 0, None)
    CaKClobe_app[i + 1] = np.clip(CaKClobe_app[i + 1], 0, None)

    # Update bouton sensitivtiy scale in untrained compartment in response to aversive singaling
    butscale_ave[responders, i + 1] = butscale_ave[responders, i] - (CaKClobe_ave[i, responders] * danarr[i] * alpha) * dt
#%% Control / Conditioning Activity Lineplot 

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(16,11))

# Name the figure 
fig.canvas.manager.set_window_title("Inter Conditioning Activity Appetitive vs. Aversive Compartment Line Plot")

# Set y-axis limits for both plots
axs[0].set_ylim(-0.0068, 2.3)  # Adjust as needed

# Aversive Compartment
axs[0].plot(tarr, CaKClobe_ave[:, rs], color='r', alpha=0.3)                                         # Aversive compartment reliable (red)
axs[0].plot(tarr, CaKClobe_ave[:, us], color='darkorange', alpha=0.25)                               # Aversive compartment unreliable (orange)
axs[0].plot(tarr, np.mean(CaKClobe_ave[:, responders], axis=1), color='black', lw=4)                 # Mean activity
axs[0].hlines(y=0.006, xmin=ston, xmax=stoff, colors='gray', linestyles='-', lw=5, label='Stimulus') # Thick black stimulus line

# Plot the dopamine pulses (DAN pulses) as red lines above the stimulus line
pulse_starts = danon + np.arange(npules) * (pulsdur + pulsint)  # Start times of the pulses
pulse_ends = pulse_starts + pulsdur  # End times of the pulses
for start, end in zip(pulse_starts, pulse_ends):
    axs[0].hlines(y=0.0056, xmin=start, xmax=end, colors='maroon', linestyles='-', lw=5.)  # Red DAN pulse line
axs[0].set_title('Aversive Compartment')  # Title for aversive compartment

# Appetitive Compartment
axs[1].plot(tarr, CaKClobe_app[:, rs], color='r', alpha=0.3)  # Green for reliable
axs[1].plot(tarr, CaKClobe_app[:, us], color='darkorange', alpha=0.3)  # Purple for unreliable
axs[1].plot(tarr, np.mean(CaKClobe_app[:, responders], axis=1), color='black', lw=4)  # Mean activity
axs[1].hlines(y=0.006, xmin=ston, xmax=stoff, colors='gray', linestyles='-', lw=5)  # Thick gray stimulus line
axs[1].set_title('Appetitive Compartment')  # Corrected title spelling

# Empty plots for adding custom text to the legend
axs[1].plot([], [], color='r', lw=2.5, label='Reliable')             # Empty plot for legend 
axs[1].plot([], [], color='darkorange', lw=2.5, label='Unreliable')  # Empty plot for legend
axs[1].plot([], [], color='black', lw=4, label='Mean Activity')      # Empty plot for legend

axs[1].legend(loc='upper right', frameon=False)  # Adjusted legend position
# Add centered x-label
fig.text(0.528, 0.02, 't [s]', ha='center')


fig.supylabel('Ca Activity (a.u)')  # Super y-axis label for the entire figure

fig.subplots_adjust(
    top=0.935,
    bottom=0.09,
    left=0.095,
    right=0.98,
    hspace=0.2,
    wspace=0.2
    )

plt.show()

# Save in both PNG and SVG formats
fig.savefig(os.path.join(save_path, "inter_conditioning_activity_appet_aver_line_plot.png"))
fig.savefig(os.path.join(save_path, "inter_conditioning_activity_appet_aver_line_plot.svg"))
#%% Aversive / Appetitive Interconditioning Activity

# Create figure and subplots with shared x/y axes
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(16,11))

# Name the figure 
fig.canvas.manager.set_window_title("Inter Conditioning Activity Appetitive vs. Aversive Compartment")

# Set y-lim
axs[0].set_ylim([-0.0065, 2.2])

# Compute mean and standard deviation for aversive compartment
mean_rs_ave = np.mean(CaKClobe_ave[:, rs], axis=1)
std_rs_ave = np.std(CaKClobe_ave[:, rs], axis=1)
mean_us_ave = np.mean(CaKClobe_ave[:, us], axis=1)
std_us_ave = np.std(CaKClobe_ave[:, us], axis=1)

# Reliable responders mean and shade (Aversive)
axs[0].plot(tarr, mean_rs_ave, 'r', label='Reliable', lw=2)  # Mean reliable
axs[0].fill_between(tarr, 
                    np.clip(mean_rs_ave - std_rs_ave, 0, None), 
                    mean_rs_ave + std_rs_ave, 
                    color='r', alpha=0.25)  # ±1 standard deviation shading reliable

# Unreliable responders mean and shade (Conditioning)
axs[0].plot(tarr, mean_us_ave, color='darkorange', label='Unreliable', lw=2)  # Mean unreliable
axs[0].fill_between(tarr, 
                    np.clip(mean_us_ave - std_us_ave, 0, None), 
                    mean_us_ave + std_us_ave, 
                    color='darkorange', alpha=0.25)  # ±1 standard deviation shading unreliable

# Stimulus indicator, subplot title, legend
axs[0].set_ylabel(r'Activity [a.u.]')
axs[0].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * 0.004, '-', lw=5, color='gray')
pulse_starts = danon + np.arange(npules) * (pulsdur + pulsint)  # Start times of the pulses
pulse_ends = pulse_starts + pulsdur  # End times of the pulses
for start, end in zip(pulse_starts, pulse_ends):
    axs[0].hlines(y=0.004, xmin=start, xmax=end, colors='maroon', linestyles='-', lw=5)  # Red DAN pulse line
axs[0].set_title('Aversive Compartment')

# Compute mean and standard deviation for appetitive compartment
mean_rs_app = np.mean(CaKClobe_app[:, rs], axis=1)
std_rs_app = np.std(CaKClobe_app[:, rs], axis=1)
mean_us_app = np.mean(CaKClobe_app[:, us], axis=1)
std_us_app = np.std(CaKClobe_app[:, us], axis=1)

# Plot Reliable responders mean and shade (Appetitive)
axs[1].plot(tarr, mean_rs_app, 'r', label='Reliable', lw=2)  # Mean reliable
axs[1].fill_between(tarr, 
                    np.clip(mean_rs_app - std_rs_app, 0, None), 
                    mean_rs_app + std_rs_app, color='r', alpha=0.25) # ±1 standard deviation shading reliable

# Unreliable responders mean and shade (Control)
axs[1].plot(tarr, mean_us_app, color='darkorange', label='Unreliable', lw=2)  # Mean unreliable
axs[1].fill_between(tarr, 
                    np.clip(mean_us_app - std_us_app, 0, None), 
                    mean_us_app + std_us_app, 
                    color='darkorange', alpha=0.25)  # ±1 standard deviation shading unreliable

# Labels, stimulus indicator, subplot title
axs[1].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * 0.004, '-', lw=5, color='gray')
axs[1].set_title('Appetitive Compartment')
axs[1].legend(loc='upper right')  # Adjusted legend position

# Set common x-label at the figure level
fig.text(0.525, 0.024, 't [s]', ha='center')

fig.subplots_adjust(
    top=0.95,
    bottom=0.09,
    left=0.08,
    right=0.98,
    hspace=0.2,
    wspace=0.1
    )

plt.show()
# Save in both PNG and SVG formats
fig.savefig(os.path.join(save_path, "inter_conditioning_activity_appet_aver_shade.png"))
fig.savefig(os.path.join(save_path, "inter_conditioning_activity_appet_aver_shade.svg"))
#%% Aversive / Appetetive Inter conditoning  Activity Colormatix

# Sort responders by their input scale
sorted_indices = np.argsort(inpscale[responders])[::-1]  # Sort in descending order
responders_sorted = responders[sorted_indices]  # Reorder responders

# Find the index where input scale is closest to 0.5
threshold_index = np.argmax(inpscale[responders_sorted] < 0.5)  # First KC below 0.5

# Convert threshold index to the correct y-position in imshow
threshold_y_position = len(responders_sorted) - threshold_index  # Flip because of invert_yaxis()

# Apply sorting to the full tarr time array
CaKClobe_app_sorted = CaKClobe_app[:, responders_sorted]
CaKClobe_ave_sorted = CaKClobe_ave[:, responders_sorted]

# Shift time axis so that ston = 0s and stoff = 60s
tarr_shifted = tarr - ston  # Now `ston` is at 0s

# Compute DAN pulse onsets in shifted time (for subplot)
dan_pulse_times = np.arange(danon, danoff, pulsint + pulsdur) - ston  # Shift times to match x-axis

# Define color scale limits across both plots
vmin = min(CaKClobe_app_sorted.min(), CaKClobe_ave_sorted.min())
vmax = max(CaKClobe_app_sorted.max(), CaKClobe_ave_sorted.max())

# Create figure and subplots, increase right margin to fit color bar
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(18,11))
fig.canvas.manager.set_window_title("Interconditioning Activity Aversive vs. Appetitive Color Matrix")

# Reduce space between subplots
plt.subplots_adjust(wspace=0.09)

# Aversive subplot - Sorted by Input Scale
img1 = axs[0].imshow(CaKClobe_ave_sorted.T, aspect='auto', cmap='viridis', interpolation='none', 
                     vmin=vmin, vmax=vmax, extent=[tarr_shifted[0], tarr_shifted[-1], 0, len(responders_sorted)])

# Horizontal threshold line
axs[0].hlines(y=threshold_y_position, xmin=tarr_shifted[0], xmax=tarr_shifted[-1], color='white', linestyles='dashed', lw=2)

# Stimulus ON/OFF lines
axs[0].axvline(x=0, color='red', linestyle='-', linewidth=2)
axs[0].axvline(x=60, color='red', linestyle='-', linewidth=2)

# DAN pulses
for pulse_time in dan_pulse_times:
    axs[0].axvline(x=pulse_time, color='gray', linestyle='-', linewidth=1)

# Title and formatting
axs[0].set_title('Aversive Compartment')
axs[0].invert_yaxis()
axs[0].set_xlim(left=-10, right=tarr_shifted[-1])  # <<<<<< Extend x-axis to -10s

# Appetitive subplot - Sorted by Input Scale
img2 = axs[1].imshow(CaKClobe_app_sorted.T, aspect='auto', cmap='viridis', interpolation='none', 
                     vmin=vmin, vmax=vmax, extent=[tarr_shifted[0], tarr_shifted[-1], 0, len(responders_sorted)])

# Threshold line
axs[1].hlines(y=threshold_y_position, xmin=tarr_shifted[0], xmax=tarr_shifted[-1], color='white', linestyles='dashed', lw=2)

# Stimulus lines
axs[1].axvline(x=0, color='red', linestyle='-', linewidth=2)
axs[1].axvline(x=60, color='red', linestyle='-', linewidth=2)

# Title and formatting
axs[1].set_title('Appetitive Compartment')
axs[1].invert_yaxis()
axs[1].set_xlim(left=-10, right=tarr_shifted[-1])  # <<<<<< Optional: Match x-axis range to aversive


# X-axis ticks for both subplots
tick_positions = np.arange(-10, np.ceil(tarr_shifted[-1] / 5) * 5 + 1, 10)
tick_labels = [f"{int(t)}" for t in tick_positions]

axs[0].set_xticks(tick_positions)
axs[0].set_xticklabels(tick_labels)

axs[1].set_xticks(tick_positions)
axs[1].set_xticklabels(tick_labels)

# Remove y-axis ticks
axs[0].set_yticks([])
axs[1].set_yticks([])

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.105, 0.03*0.5, 0.85])
cbar = fig.colorbar(img1, cax=cbar_ax, orientation='vertical')
cbar.ax.tick_params(labelsize=17)
cbar.set_label('Activity [a.u.]', fontsize=30)

# Shared labels
fig.text(0.475, 0.03, 't [s]', ha='center', fontsize=32)
fig.text(0.0223, 0.23, 'Reliable', ha='center', va='center', rotation=90, fontsize=32)
fig.text(0.0223, 0.63, 'Unreliable', ha='center', va='center', rotation=90, fontsize=32)

# Final layout adjustment
fig.subplots_adjust(
    top=0.955,
    bottom=0.11,
    left=0.04,
    right=0.91,
    hspace=0.205,
    wspace=0.12
    )


plt.show()
# Save in both PNG and SVG formats
fig.savefig(os.path.join(save_path, "inter_conditioning_activity_appet_aver_color_matrix.png"))
fig.savefig(os.path.join(save_path, "inter_conditioning_activity_appet_aver_color_matrix.svg"))
#%% Bouton Scale Aversive Compartment Plot

fig, ax = plt.subplots(figsize=(16,11))

# Name the figure 
fig.canvas.manager.set_window_title("Conditioning Induced Bouton Scale Reduction Aversive Compartment")

# Plot bouton scale for reliable and unreliable responders
ax.plot(tarr, butscale_ave[rs, :].T, color='r', alpha=0.4)  
ax.plot(tarr, butscale_ave[us, :].T, color='darkorange', alpha=0.4)

# Stimulus and Dopamine
ax.hlines(y=0.403, xmin=ston, xmax=stoff, colors='gray', linestyles='-', lw=5)
pulse_starts = danon + np.arange(npules) * (pulsdur + pulsint)  # Start times of the pulses
pulse_ends = pulse_starts + pulsdur  # End times of the pulses
for start, end in zip(pulse_starts, pulse_ends):
    ax.hlines(y=0.403, xmin=start, xmax=end, colors='maroon', linestyles='-', lw=5)

# Empty plots for legend 
ax.plot([], [], color='r', label='Reliable')  
ax.plot([], [], color='darkorange', label='Unreliable ')

# Labels and Formatting
ax.set_xlabel('t [s]')
ax.set_ylabel('Bouton Sensitivity')
ax.set_ylim(0.4)  # Set y-axis limits
ax.legend(loc='best')  # Automatically position legend


# Final layout adjustment
fig.subplots_adjust(
    top=0.975,
    bottom=0.09,
    left=0.07,
    right=0.975,
    hspace=0.2,
    wspace=0.2
    )

plt.show()

fig.savefig(os.path.join(save_path, "conditioning_induced_bouton_scale_reduction_aversive.png"))
fig.savefig(os.path.join(save_path, "conditioning_induced_bouton_scale_reduction_aversive.svg"))
#%% DAN Array Over Time

# Create a figure and an axes object
fig, ax = plt.subplots(figsize=(16,9))

# Set the window title (if supported by your backend)
fig.canvas.manager.set_window_title("DAN Activity")

# Plot DAN activity
ax.plot(tarr, danarr, label='DAN Activity', lw=2)

# Set axis labels
ax.set_xlabel("t [s]")
ax.set_ylabel("DAN Activity Level [a.u.]")

# Plot stimulus indicator (gray horizontal line from ston to stoff)
ax.hlines(y=0, xmin=ston, xmax=stoff, colors='gray', linestyles='-', lw=4)

# Calculate pulse start and end times for the DAN pulses
pulse_starts = danon + np.arange(npules) * (pulsdur + pulsint)  # Start times of the pulses
pulse_ends = pulse_starts + pulsdur  # End times of the pulses

# Plot DAN pulse indicators (maroon horizontal lines)
for start, end in zip(pulse_starts, pulse_ends):
    ax.hlines(y=0, xmin=start, xmax=end, colors='maroon', linestyles='-', lw=4)

# Set the y-axis limits
ax.set_ylim([-0.001, 0.26])

# Add a legend
ax.legend()

# Final layout adjustment
fig.subplots_adjust(
    top=0.975,
    bottom=0.095,
    left=0.09,
    right=0.975,
    hspace=0.2,
    wspace=0.2
    )

# Display the plot
plt.show()

fig.savefig(os.path.join(save_path, "dan_activity_aversive_compartment.png"))
fig.savefig(os.path.join(save_path, "dan_activity_aversive_compartment.svg"))
#%% DAN-Activity and Bouton Sensitivity Together

# Generate pulse start and end times
pulse_starts = [danon + i * (pulsdur + pulsint) for i in range(npules)]
pulse_ends = [start + pulsdur for start in pulse_starts]

# Create a figure with 2 subplots, sharing the x-axis
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(16,11))

# Name the figure 
fig.canvas.manager.set_window_title("DAN Activity and Bouton Scale Aversive Compartment")

# --- Subplot 1: DAN Activity ---
axs[0].plot(tarr, danarr, label='DAN', lw=2)

# Plot stimulus indicator (gray horizontal line from ston to stoff)
axs[0].hlines(y=0.0008, xmin=ston, xmax=stoff, colors='gray', linestyles='-', lw=5)

# Plot DAN pulse indicators (maroon horizontal lines)
for start, end in zip(pulse_starts, pulse_ends):
    axs[0].hlines(y=0.0008, xmin=start, xmax=end, colors='maroon', linestyles='-', lw=5)

# Axis labels and formatting
axs[0].set_ylabel("Activity [a.u.]")
axs[0].set_ylim([-0.002, 0.26])
axs[0].legend(loc='upper left')

# --- Subplot 2: Bouton Scale Reduction ---
axs[1].plot(tarr, butscale_ave[rs, :].T, color='r', alpha=0.4)  
axs[1].plot(tarr, butscale_ave[us, :].T, color='darkorange', alpha=0.4)

# Empty plots for legend
axs[1].plot([], color='r', label='Reliable')
axs[1].plot([], color='darkorange', label='Unreliable')

# Plot stimulus and dopamine indicators
axs[1].hlines(y=0.4055, xmin=ston, xmax=stoff, colors='gray', linestyles='-', lw=5)
for start, end in zip(pulse_starts, pulse_ends):
    axs[1].hlines(y=0.4055, xmin=start, xmax=end, colors='maroon', linestyles='-', lw=5)

# Labels and formatting
axs[1].set_ylabel("b [a.u.]")
axs[1].set_xlabel("t [s]")
axs[1].set_ylim([0.4, 1.0])  # Adjust if needed
axs[1].legend(loc='upper left')

# Adjust layout with manually extracted parameters
fig.subplots_adjust(
    top=0.965,
    bottom=0.087,
    left=0.074,
    right=0.99,
    hspace=0.165,
    wspace=0.2
)

# Show plot
plt.show()

fig.savefig(os.path.join(save_path, "dan_activity_bouton_scale_aversive_comaprtment.png"))
fig.savefig(os.path.join(save_path, "dan_activity_bouton_scale_aversive_comaprtment.svg"))

#%% Save inpscale array and responder indices 
butscale_app = butscale_ave[:, 0]
butscale_ave = butscale_ave[:, -1]

# Define the absolute path to the save directory
save_path = '/home/baxter/IBM/code_repo/IBM_functions_package'

# Define a dictionary of arrays to save
data_to_save = {
    'butscale_appetetive': butscale_app,
    'butscale_aversive': butscale_ave,
    'inpscale': inpscale,
    'rs': rs,
    'us': us,
    'dan_0': dan_0,
    'r': r,
    'responders': responders
}

# Save all arrays into one .npz file
np.savez(os.path.join(save_path, 'dan_conditioning_results.npz'), **data_to_save)
