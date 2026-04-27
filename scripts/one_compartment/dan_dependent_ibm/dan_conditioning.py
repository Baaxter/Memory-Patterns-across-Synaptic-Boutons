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

# DAN pulse timing and settings
npules = 12                # Number of DAN pulses
pulsdur = 1.25             # Pulse duration in seconds (on time)
pulsint = 3.75             # Inter-pulse interval in seconds (off time)
danon =  15                # DAN onset time at 15 seconds
danoff = 75                # DAN offset time at 75 seconds
dan_0 = 0.2                # Initial pulse magnitude (default form thesis 0.5)
r = 0.8                    # Reduction factor, will be raised current puls index
alpha = 0.5                # Leraing Rate

# Generate tarr, starr (stimulus and time arry)
ston = 10  # odor onset in seconds
stdur = 60 # odor duration in seconds
stoff = 70 # odor offest
tend = 80 # recoring offset
tarr, starr = mdl.generate_step_stimulus(tend, ston, stoff, dt, 1 - bline) # Run a 5s long stimulus

# Generate odor / input scale
nrs =  rng.poisson(prr * nKCs)                    # Number of reliable responders
nurs = rng.poisson(pur * nKCs)                   # Number of unreliable responders
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

# Generate DAN-Array
__, danarr = mdl.generate_step_stimulus(tend, danon, danoff, dt)  # Generates an array with 1s between danon and danoff
dan_start_idx = int(danon / dt)     # 15 s -> index 1500
dan_end_idx = int(danoff / dt)      # 75 s -> index 7500
pulse_duration_in_steps = int(pulsdur / dt) + 1             # 1.25 s "on" period (126 steps)
interval_duration_in_steps = int((pulsdur + pulsint) / dt)    # 5 s total per pulse (500 steps)

# Apply pulse pattern (1.25s on, 3.75s off) to the DAN array, with decreasing magnitude
for pulse_idx in range(npules):
    # Calculate the magnitude for the current pulse with the reduction factor
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


# Generate Bouton Scale Array
butscale = np.ones([nKCs, len(tarr)]) # Array of ones as oringial state of bouon scale 
butscale_ut = butscale[:,0]           # Vector of untrained bouton scale to simulate controle 

# Generate Simulation arrays
CaKCcal   = np.zeros([len(tarr), nKCs])  # Calcium calix
CaKClobe  = np.zeros([len(tarr), nKCs])  # Calcium trained compartment
CaKCcal[0] = bline                       # Initial calcium calix
CaKClobe[0] = bline                      # Initial calcium trained compartment
adapt     = np.zeros([len(tarr), nKCs])  # Adaptation array
inh       = np.zeros([len(tarr), nKCs])  # Lateral inhibition array

# Dopamine unaffected arrays for comparison
CaKClobe_ut = CaKClobe.copy()  # Untrained compartment 
inh_ut      = inh.copy()       # Untrained lateral inhibition state


for i in range(len(tarr) - 1):
    # Update calcium dynamics (Null dynamics)
    CaKCcal[i + 1] = CaKCcal[i] + ((inpscale * starr[i]) / tauinp - (CaKCcal[i] - bline) / tauKCdec) * dt
    CaKClobe[i + 1] = CaKClobe[i] + ((inpscale * butscale[:, i] * starr[i]) / tauinp - (CaKClobe[i] - bline) / tauKCdec) * dt
    CaKClobe_ut[i + 1] = CaKClobe_ut[i] + ((inpscale * butscale_ut * starr[i]) / tauinp - (CaKClobe_ut[i] - bline) / tauKCdec) * dt
    
    # Add noise
    CaKCcal[i + 1] += noise[i] * n_str
    CaKClobe[i + 1] += noise[i] * n_str
    CaKClobe_ut[i + 1] += noise[i] * n_str

    # Adaptation
    adapt[i + 1] = mdl.adaptation_dynamics(adapt[i], CaKCcal[i], tauadapt, adaptscale, dt)
    CaKCcal[i + 1] -= (adapt[i] * CaKCcal[i]) * (1 / tauKCdec) * dt
    CaKClobe[i + 1] -= (adapt[i] * CaKClobe[i]) * (1 / tauKCdec) * dt
    CaKClobe_ut[i + 1] -= (adapt[i] * CaKClobe_ut[i]) * (1 / tauKCdec) * dt

    # Inhibition
    inh[i + 1] = mdl.inhibition_dynamics(inh[i], CaKClobe[i], tauinh, inhscale, dt) * \
                     mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe[i], infp, slf)
    inh_ut[i + 1] = mdl.inhibition_dynamics(inh_ut[i], CaKClobe_ut[i], tauinh, inhscale, dt) * \
                        mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_ut[i], infp, slf)
    
    CaKClobe[i + 1] -= (inh[i]) * (1 / tauKCdec) * dt
    CaKClobe_ut[i + 1] -= (inh_ut[i]) * (1 / tauKCdec) * dt

    # Ensure non-negative values
    CaKCcal[i + 1] = np.clip(CaKCcal[i + 1], 0, None)
    CaKClobe[i + 1] = np.clip(CaKClobe[i + 1], 0, None)
    CaKClobe_ut[i + 1] = np.clip(CaKClobe_ut[i + 1], 0, None)

    # Update bouton scale to simulate dopaminergic effect (applied only to the trained compartment)
    butscale[indx_active_kc, i + 1] = butscale[indx_active_kc, i] - (CaKClobe[i, indx_active_kc] * danarr[i] * alpha) * dt
#%% Control/Conditioning Activity Lineplot my style 

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

# Name the figure 
fig.canvas.manager.set_window_title("KC Activity Control vs. Conditioning Lines")

# Control Compartment
axs[0].plot(tarr, CaKClobe_ut[:, rs], color='#2ca02c', alpha=0.3)  # Green for reliable
axs[0].plot(tarr, CaKClobe_ut[:, us], color='#9467bd', alpha=0.3)  # Purple for unreliable
axs[0].plot(tarr, np.mean(CaKClobe_ut[:, responders], axis=1), color='#A3144B', lw=4)  # Mean activity
axs[0].hlines(y=-0.01, xmin=ston, xmax=stoff, colors='k', linestyles='-', lw=5)  # Thick black stimulus line
axs[0].set_title('Control', fontsize=32)  # Increased title font size

# Conditioning Compartment
axs[1].plot(tarr, CaKClobe[:, rs], color='#2ca02c', alpha=0.3)  # Green for reliable
axs[1].plot(tarr, CaKClobe[:, us], color='#9467bd', alpha=0.25)  # Purple for unreliable
axs[1].plot(tarr, np.mean(CaKClobe[:, responders], axis=1), color='#A3144B', lw=4)  # Mean activity

# Plot the dopamine pulses (DAN pulses) as red lines above the stimulus line
pulse_starts = danon + np.arange(npules) * (pulsdur + pulsint)  # Start times of the pulses
pulse_ends = pulse_starts + pulsdur  # End times of the pulses
for start, end in zip(pulse_starts, pulse_ends):
    axs[1].hlines(y=0.005, xmin=start, xmax=end, colors='r', linestyles='-', lw=6.5)  # Red DAN pulse line

# Empty plots for adding custom text to the legend
axs[1].plot([], [], color='#2ca02c', lw=2.5, label='Reliable Responders')  # Empty plot for legend 
axs[1].plot([], [], color='#9467bd', lw=2.5, label='Unreliable Responders')  # Empty plot for legend
axs[1].plot([], [], color='#A3144B', lw=3.5, label='Mean Activity')  # Mean activity
axs[1].plot([], [], 'r', label='Shock', lw=6)  # Empty plot for legend
axs[1].hlines(y=-0.01, xmin=ston, xmax=stoff, colors='k', linestyles='-', lw=5, label='Stimulus')  # Thick black stimulus line
axs[1].legend(fontsize=18, loc='upper right', frameon=True, bbox_to_anchor=(1.0, 0.9), edgecolor='black')  # Adjusted legend position
axs[1].set_title('Conditioning', fontsize=32)  # Increased title font size

# Super labels
fig.supxlabel('Time (s)')  # Super x-axis label for the entire figure
fig.supylabel('Ca Activity (a.u)')  # Super y-axis label for the entire figure

plt.show()

#%% Control/Conditioning Activity Lineplot ibo style

# Create figure and subplots with shared x/y axes
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

# Name the figure 
fig.canvas.manager.set_window_title("KC Activity Control vs. Conditioning")

# Set y-lim
axs[0].set_ylim([-0.007, 2.2])

# Compute mean and standard deviation for Control
mean_rs_ut = np.mean(CaKClobe_ut[:, rs], axis=1)
std_rs_ut = np.std(CaKClobe_ut[:, rs], axis=1)
mean_us_ut = np.mean(CaKClobe_ut[:, us], axis=1)
std_us_ut = np.std(CaKClobe_ut[:, us], axis=1)

# Reliable responders mean and shade (Control)
axs[0].plot(tarr, mean_rs_ut, 'r', label='Reliable', lw=2)  # Mean reliable
axs[0].fill_between(tarr, 
                    np.clip(mean_rs_ut - std_rs_ut, 0, None), 
                    mean_rs_ut + std_rs_ut, 
                    color='r', alpha=0.25)  # ±1 standard deviation shading reliable

# Unreliable responders mean and shade (Control)
axs[0].plot(tarr, mean_us_ut, color='darkorange', label='Unreliable', lw=2)  # Mean unreliable
axs[0].fill_between(tarr, 
                    np.clip(mean_us_ut - std_us_ut, 0, None), 
                    mean_us_ut + std_us_ut, 
                    color='darkorange', alpha=0.25)  # ±1 standard deviation shading unreliable

# Labels, stimulus indicator, subplot title
axs[0].set_ylabel(r'Activity [a.u.]')
axs[0].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * 0.004, '-', lw=5, color='gray')
axs[0].set_title('Control')


# Compute mean and standard deviation for Conditioning
mean_rs = np.mean(CaKClobe[:, rs], axis=1)
std_rs = np.std(CaKClobe[:, rs], axis=1)
mean_us = np.mean(CaKClobe[:, us], axis=1)
std_us = np.std(CaKClobe[:, us], axis=1)

# Reliable responders mean and shade (Conditioning)
axs[1].plot(tarr, mean_rs, 'r', label='Reliable', lw=2)  # Mean reliable
axs[1].fill_between(tarr, 
                    np.clip(mean_rs - std_rs, 0, None), 
                    mean_rs + std_rs, 
                    color='r', alpha=0.25)  # ±1 standard deviation shading reliable

# Unreliable responders mean and shade (Conditioning)
axs[1].plot(tarr, mean_us, color='darkorange', label='Unreliable', lw=2)  # Mean unreliable
axs[1].fill_between(tarr, 
                    np.clip(mean_us - std_us, 0, None), 
                    mean_us + std_us, 
                    color='darkorange', alpha=0.25)  # ±1 standard deviation shading unreliable

# Stimulus indicator, subplot title, legend
axs[1].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * 0.004, '-', lw=5, color='gray')
pulse_starts = danon + np.arange(npules) * (pulsdur + pulsint)  # Start times of the pulses
pulse_ends = pulse_starts + pulsdur  # End times of the pulses
for start, end in zip(pulse_starts, pulse_ends):
    axs[1].hlines(y=0.004, xmin=start, xmax=end, colors='maroon', linestyles='-', lw=5)  # Red DAN pulse line
axs[1].set_title('Conditioning')
axs[1].legend(loc='upper right')  # Adjusted legend position

# Set common x-label at the figure level
fig.text(0.525, 0.024, 't [s]', ha='center')


plt.show()
#%% Control/Conditioning Activity Colormatix

# Sort responders by their input scale
sorted_indices = np.argsort(inpscale[responders])[::-1]  # Sort in descending order
responders_sorted = responders[sorted_indices]  # Reorder responders

# Find the index where input scale is closest to 0.5
threshold_index = np.argmax(inpscale[responders_sorted] < 0.5)  # First KC below 0.5

# **Convert threshold index to the correct y-position in imshow**
threshold_y_position = len(responders_sorted) - threshold_index  # Flip because of invert_yaxis()

# **Apply sorting to the full tarr time array**
CaKClobe_ut_sorted = CaKClobe_ut[:, responders_sorted]
CaKClobe_sorted = CaKClobe[:, responders_sorted]

# **Shift time axis so that ston = 0s and stoff = 60s**
tarr_shifted = tarr - ston  # Now `ston` is at 0s

# **Compute DAN pulse onsets in shifted time (for second subplot)**
dan_pulse_times = np.arange(danon, danoff, pulsint + pulsdur) - ston  # Shift times to match x-axis

# Define color scale limits across both plots
vmin = min(CaKClobe_ut_sorted.min(), CaKClobe_sorted.min())
vmax = max(CaKClobe_ut_sorted.max(), CaKClobe_sorted.max())

# Create figure and subplots, increase right margin to fit color bar
fig, axs = plt.subplots(1, 2, sharey=True)
fig.canvas.manager.set_window_title("SP2_KC Activity Control vs. Conditioning Color Matrix (Ordered by Input Scale)")

# **Reduce space between subplots**
plt.subplots_adjust(wspace=0.09)  # Adjust horizontal spacing

# Control subplot - Sorted by Input Scale
img1 = axs[0].imshow(CaKClobe_ut_sorted.T, aspect='auto', cmap='viridis', interpolation='none', 
                      vmin=vmin, vmax=vmax, extent=[tarr_shifted[0], tarr_shifted[-1], 0, len(responders_sorted)])

# **Fix horizontal line at correct input scale**
axs[0].hlines(y=threshold_y_position, xmin=tarr_shifted[0], xmax=tarr_shifted[-1], color='white', linestyles='dashed', lw=2)

# **Add red vertical lines at stimulus onset (0s) and offset (60s)**
axs[0].axvline(x=0, color='red', linestyle='-', linewidth=2)    # Stimulus ON (0s)
axs[0].axvline(x=60, color='red', linestyle='-', linewidth=2)   # Stimulus OFF (60s)

# Control subplot formatting
axs[0].set_title('Control', fontsize = 32)

# Adjust x-axis ticks to be every 5s
tick_positions = np.arange(np.floor(tarr_shifted[0] / 5) * 5, np.ceil(tarr_shifted[-1] / 5) * 5 + 1, 10)
axs[0].set_xticks(tick_positions)
axs[0].set_xticklabels([f"{t:.0f}" for t in tick_positions])  # Convert to integers

# Remove y-axis ticks and labels for both subplots
axs[0].set_yticks([])
axs[1].set_yticks([])

axs[0].invert_yaxis()  # Ensure correct order (largest input scale at bottom)

# Conditioning subplot - Sorted by Input Scale
img2 = axs[1].imshow(CaKClobe_sorted.T, aspect='auto', cmap='viridis', interpolation='none', 
                      vmin=vmin, vmax=vmax, extent=[tarr_shifted[0], tarr_shifted[-1], 0, len(responders_sorted)])

# **Fix horizontal line at correct input scale**
axs[1].hlines(y=threshold_y_position, xmin=tarr_shifted[0], xmax=tarr_shifted[-1], color='white', linestyles='dashed', lw=2)

# **Add red vertical lines at stimulus onset (0s) and offset (60s)**
axs[1].axvline(x=0, color='red', linestyle='-', linewidth=2)    # Stimulus ON (0s)
axs[1].axvline(x=60, color='red', linestyle='-', linewidth=2)   # Stimulus OFF (60s)

# **Add gray vertical lines for DAN pulses (shocks) in the second subplot only**
for pulse_time in dan_pulse_times:
    axs[1].axvline(x=pulse_time, color='gray', linestyle='-', linewidth=1)  # Thin gray lines for shocks

# Conditioning subplot formatting
axs[1].set_title('Conditioning', fontsize = 32)

# Apply the same tick fix with 5s interval
axs[1].set_xticks(tick_positions)
axs[1].set_xticklabels([f"{t:.0f}" for t in tick_positions])  # Convert to integers

axs[1].invert_yaxis()

# **Create color bar to the right of the second subplot**
# **Create color bar to the right of the second subplot**
cbar_ax = fig.add_axes([0.92, 0.105, 0.03*0.5, 0.85])  # [left, bottom, width, height]
cbar = fig.colorbar(img1, cax=cbar_ax, orientation='vertical')
cbar.ax.tick_params(labelsize=17)  # Adjust font size
cbar.set_label('Activity [a.u.]', fontsize = 30)

# **Add a single x-axis label for both subplots**
fig.text(0.475, 0.03, 't [s]', ha='center', fontsize = 32)
fig.text(0.0223, 0.23, 'Reliable', ha='center', va='center', rotation=90, fontsize = 32)
fig.text(0.0223, 0.63,  'Unreliable', ha='center', va='center', rotation=90, fontsize = 32)

fig.subplots_adjust(
    top=0.955,
    bottom=0.11,
    left=0.04,
    right=0.91,
    hspace=0.2,
    wspace=0.1
    )

plt.show()
#%% Bouton Scale Plot

fig, ax = plt.subplots()

# Name the figure 
fig.canvas.manager.set_window_title("Condtioning Induced Bouton Scale Reduction")

# Plot bouton scale for reliable and unreliable responders
ax.plot(tarr, butscale[rs, :].T, color='r', alpha=0.4)  
ax.plot(tarr, butscale[us, :].T, color='darkorange', alpha=0.4)

# Stimulus and Dopamine
ax.hlines(y=0.404, xmin=ston, xmax=stoff, colors='gray', linestyles='-', lw=5)
pulse_starts = danon + np.arange(npules) * (pulsdur + pulsint)  # Start times of the pulses
pulse_ends = pulse_starts + pulsdur  # End times of the pulses
for start, end in zip(pulse_starts, pulse_ends):
    ax.hlines(y=0.404, xmin=start, xmax=end, colors='maroon', linestyles='-', lw=5)

# Empty plots for legend 
ax.plot([], [], color='r', label='Reliable')  
ax.plot([], [], color='darkorange', label='Unreliable ')

# Labels and Formatting
ax.set_xlabel('t [s]')
ax.set_ylabel('Bouton Sensitivity')
ax.set_ylim(0.4)  # Set y-axis limits
ax.legend(loc='best')  # Automatically position legend

fig.subplots_adjust(
    top=0.94,
    bottom=0.11,
    left=0.105,
    right=0.965,
    hspace=0.2,
    wspace=0.2
    )

plt.show()
#%% DAN array over time

# Create a figure and an axes object
fig, ax = plt.subplots()

# Set the window title (if supported by your backend)
fig.canvas.manager.set_window_title("Dan Activity")

# Plot DAN activity
# (Note: corrected the misspelling of "label")
ax.plot(tarr, danarr, label='DAN Activity', lw=2)

# Set axis labels
ax.set_xlabel("t [s]")
ax.set_ylabel("DAN Activity Level [a.u]")

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


fig.subplots_adjust(
    top=0.94,
    bottom=0.11,
    left=0.105,
    right=0.965,
    hspace=0.2,
    wspace=0.2
    )

# Display the plot
plt.show()
#%% DAN-Activity Bouton senitivity toghter

# Generate pulse start and end times
pulse_starts = [danon + i * (pulsdur + pulsint) for i in range(npules)]
pulse_ends = [start + pulsdur for start in pulse_starts]

#fig size
width_cm = 35.166
height_cm = 16.396
width_inches = width_cm / 2.54
height_inches = height_cm / 2.54

# Create a figure with 2 subplots, sharing the x-axis
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(width_inches, height_inches))



# Name the figure 
fig.canvas.manager.set_window_title("S1 DAN Activity and Bouton Scale")

# --- Subplot 1: DAN Activity ---
axs[0].plot(tarr, danarr, label='DAN', lw=2.5)


# Plot stimulus indicator (gray horizontal line from ston to stoff)
axs[0].hlines(y=0.002, xmin=ston, xmax=stoff, colors='gray', linestyles='-', lw=6.9)

# Plot DAN pulse indicators (maroon horizontal lines)
for start, end in zip(pulse_starts, pulse_ends):
    axs[0].hlines(y=0.002, xmin=start, xmax=end, colors='maroon', linestyles='-', lw=6.9)

# Axis labels and formatting
axs[0].set_ylabel("Activity [a.u]")
axs[0].set_ylim([-0.002, 0.26])
axs[0].legend(loc='upper left', frameon=False)

# --- Subplot 2: Bouton Scale Reduction ---
axs[1].plot(tarr, butscale[rs, :].T, color='r', alpha=0.4, lw=2)  
axs[1].plot(tarr, butscale[us, :].T, color='darkorange', alpha=0.4, lw=2)

# Empty plots for legend
axs[1].plot([], color='r', label='Reliable', lw=2.5)
axs[1].plot([], color='darkorange', label='Unreliable', lw=2.5)

# Plot stimulus and Dopamine indicators
axs[1].hlines(y=0.411, xmin=ston, xmax=stoff, colors='gray', linestyles='-', lw=6.9)
for start, end in zip(pulse_starts, pulse_ends):
    axs[1].hlines(y=0.411, xmin=start, xmax=end, colors='maroon', linestyles='-', lw=6.9)

# Labels and Formatting
axs[1].set_ylabel("b [a.u]")
axs[1].set_xlabel("t [s]")
axs[1].set_ylim([0.4, 1.0])  # Adjust if needed
axs[1].legend(loc='lower left', frameon=False)

# Adjust layout with manually extracted parameters
fig.subplots_adjust(
    top=0.985,
    bottom=0.115,
    left=0.08,
    right=0.99,
    hspace=0.2,
    wspace=0.2
)

#plt.savefig('/home/baxter/IBM/göttingen_poster/figurs/DAN_Activity_and_Bouton_Scale.svg.svg')
#%% Save inpscale array and responder indices 
butscale_controle = butscale[:, 0]
butscale_condtioned = butscale[:, -1]

# Define the absolute path to the save directory
save_path = '/home/baxter/IBM/code_repo/IBM_functions_package'

# Define a dictionary of arrays to save
data_to_save = {
    'butscale_controle': butscale_controle,
    'butscale_condtioned': butscale_condtioned,
    'inpscale': inpscale,
    'rs': rs,
    'us': us,
    'dan_0': dan_0,
    'r': r,
    'responders': responders
}

# Save all arrays into one .npz file
np.savez(os.path.join(save_path, 'dan_conditioning_results.npz'), **data_to_save)