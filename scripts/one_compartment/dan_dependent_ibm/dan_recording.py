#%% Base
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import io
import importlib.resources as pkg_resources  # Importing package resource handler
from IBM_functions_package import activity_model_functions as mdl
from IBM_functions_package import plotting_related_functions
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm

 
# Load the .npz parameter file from the package folder into memory
with pkg_resources.files("IBM_functions_package").joinpath("model_fit_normfac_3.885.npz").open("rb") as f:
    fitparams = np.load(io.BytesIO(f.read()))  # Read entire file into memory

# Load the .npz file "dan_conditioning_results.npz" from the package folder
with pkg_resources.files("IBM_functions_package").joinpath("dan_conditioning_results.npz").open("rb") as f:
    dan_conditioning_results = np.load(io.BytesIO(f.read()))    


# Global parameters  
nKCs       = int(fitparams['restparams'][0])      # Number of Kenyon Cells [Now 700; 700 before fitting]  
dt         = 0.01                                 # Time step for simulation (seconds) [Now 0.01; 0.001 before fitting]  
rng        = np.random.default_rng(420)           # Random number generator with a fixed seed [Now 420; 666 before fitting]  
bline      = fitparams['fittedpars_KD'][4]        # Baseline calcium concentration [Now 0.0; 0.02 before fitting]

# Input scale  
prr        = fitparams['restparams'][1]           # Rate of reliable responders [Now 0.05; 0.05 before fitting]  
pur        = fitparams['restparams'][2]           # Rate of unreliable responders [Now 0.15; 0.15 before fitting]  

# Null parameters  
tauinp     = fitparams['fittedpars_KD'][1]        # Input time constant (seconds) [Now 0.149; 0.2 before fitting]  
tauKCdec   = fitparams['fittedpars_KD'][0]        # Calcium decay time constant (seconds) [Now 0.44; 1.5 before fitting]  
n_str      = fitparams['fittedpars_KD'][5]        # Standard deviation of OU-process [Now 0.020; 0.01 before fitting]  

# Adaptation 
tauadapt   = fitparams['fittedpars_KD'][2]        # Time constant for adaptation (seconds) [Now 2.68; 2.0 before fitting]  
adaptscale = fitparams['fittedpars_KD'][3]        # Strength of adaptation [Now 0.58; 1.0 before fitting]  

# Inhibition parameters  
tauinh     = fitparams['fittedpars_WT'][0]        # Time constant for lateral inhibition (seconds) [Now 1.004; 1.5 before fitting]  
inhfactor  = fitparams['fittedpars_WT'][1]        # Inhibition scale factor [Now 15.53; 10 for sparse input, 1 for uniform input before fitting]  
infp       = fitparams['fittedpars_WT'][2]        # Midpoint of the sigmoidal function [Now 1.28; 0.5 before fitting]  
slf        = fitparams['fittedpars_WT'][3]        # Slope factor of the sigmoidal function [Now 0.16; 0.03 before fitting]

# Load the result arrays into variables with the same names as in the condtioning script
butscale_controle    = dan_conditioning_results["butscale_controle"]
butscale_conditioned = dan_conditioning_results["butscale_condtioned"]
inpscale             = dan_conditioning_results["inpscale"]
rs                   = dan_conditioning_results["rs"]
us                   = dan_conditioning_results["us"]
dan_0                = dan_conditioning_results["dan_0"]
r                    = dan_conditioning_results["r"]
responders           = dan_conditioning_results["responders"]


# Stimulus Generation
ston     = 5           # Stimulus onset time
stoff    = 10          # Stimulus offset time
tend     = 15          # End time of the simulation
tarr, starr = mdl.generate_step_stimulus(tend, ston, stoff, dt, 1 - bline)

# Generate Inhibition Matrix and Scale
inhscale = np.ones([nKCs, nKCs])
inhscale[np.diag_indices(nKCs)] = 0
inhscale /= np.sum(inhscale, axis=1)[:, None]
inhscale *= inhfactor

# Generate noise
n_str = n_str * np.sqrt(2 / tauKCdec * dt)
noise = rng.standard_normal([len(tarr) - 1, nKCs])

# Simulation preallocations pre condtioning array
CaKCcal         = np.zeros([len(tarr), nKCs])  # Calcium calixe
CaKCcal[0]      = bline                         # Initial calcium calix
CaKClobe_pre    = np.zeros([len(tarr), nKCs])  # Calcium compartment
CaKClobe_pre[0] = bline                        # Initial calcium compartment
adapt           = np.zeros([len(tarr), nKCs])  # Adaptation array
inh_pre         = np.zeros([len(tarr), nKCs])  # Lateral inhibition array

# Simulation preallocations post condtioning array
CaKClobe_post = CaKClobe_pre.copy()  # Untrained compartment 
inh_post      = inh_pre.copy()       # Untrained lateral inhibition state


# Simulate using Euler's method
for i in range(len(tarr) - 1):
    # Null
    CaKCcal[i + 1] = CaKCcal[i] + ((butscale_controle * inpscale * starr[i]) / tauinp - (CaKCcal[i] - bline) / tauKCdec) * dt
    CaKClobe_pre[i + 1] = CaKClobe_pre[i] + ((butscale_controle * inpscale * starr[i]) / tauinp - (CaKClobe_pre[i] - bline) / tauKCdec) * dt
    CaKClobe_post[i + 1] = CaKClobe_post[i] + ((butscale_conditioned * inpscale * starr[i]) / tauinp - (CaKClobe_post[i] - bline) / tauKCdec) * dt
    
    # ADD NOISE
    CaKCcal[i + 1] += noise[i] * n_str
    CaKClobe_pre[i + 1] += noise[i] * n_str
    CaKClobe_post[i + 1] += noise[i] * n_str

    # Adaptation
    adapt[i + 1] = mdl.adaptation_dynamics(adapt[i], CaKCcal[i], tauadapt, adaptscale, dt)
    CaKCcal[i + 1] -= (adapt[i] * CaKCcal[i]) * (1 / tauKCdec) * dt
    CaKClobe_pre[i + 1] -=  (adapt[i] * CaKClobe_pre[i]) * (1 / tauKCdec) * dt
    CaKClobe_post[i + 1] -= (adapt[i] * CaKClobe_post[i]) * (1 / tauKCdec) * dt
    
    # Inhibition 
    inh_pre[i + 1] = mdl.inhibition_dynamics(inh_pre[i], CaKClobe_pre[i], tauinh, butscale_controle * inhscale, dt) * mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_pre[i], infp, slf)
    inh_post[i + 1] = mdl.inhibition_dynamics(inh_post[i], CaKClobe_post[i], tauinh, butscale_conditioned * inhscale, dt) * mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_post[i], infp, slf)
    
    CaKClobe_pre[i + 1] -= (inh_pre[i]) * (1 / tauKCdec) * dt
    CaKClobe_post[i + 1] -= (inh_post[i]) * (1 / tauKCdec) * dt

    # Make sure everything is nonzero
    CaKCcal[i + 1] = np.clip(CaKCcal[i + 1], 0, None)
    CaKClobe_pre[i + 1] = np.clip(CaKClobe_pre[i + 1], 0, None)
    CaKClobe_post[i + 1] = np.clip(CaKClobe_post[i + 1], 0, None)
#%% Pre/Post Conditioning Activity Lineplot my style

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

# Name the figure 
fig.canvas.manager.set_window_title("KC Activity Pre vs Post Line")

# Plot 1: Pre-conditioning
axs[0].plot(tarr, CaKClobe_pre[:, rs], color='#2ca02c', alpha=0.2)  # Green for reliable
axs[0].plot(tarr, CaKClobe_pre[:, us], color='#9467bd', alpha=0.2)  # Purple for unreliable
axs[0].plot(tarr, np.mean(CaKClobe_pre[:, responders], axis=1), color='#A3144B', lw=4)  # Mean activity
axs[0].hlines(y=-0.01, xmin=ston, xmax=stoff, colors='k', linestyles='-', lw=5)  # Black stimulus line
axs[0].set_title('Pre')

# Plot 2: Post-conditioning
axs[1].plot(tarr, CaKClobe_post[:, rs], color='#2ca02c', alpha=0.2)  # Green for reliable
axs[1].plot(tarr, CaKClobe_post[:, us], color='#9467bd', alpha=0.2)  # Purple for unreliable
axs[1].plot(tarr, np.mean(CaKClobe_post[:, responders], axis=1), color='#A3144B', lw=4)  # Mean activity
axs[1].set_title('Post')

# Add legend to second subplot
axs[1].plot([], [], color='#2ca02c', lw=2.5, label='Reliable')  # Empty plot for legend 
axs[1].plot([], [], color='#9467bd', lw=2.5, label='Unreliable ')  # Empty plot for legend
# axs[1].plot([], [], color='#A3144B', lw=2.5, label='Mean Activity')
# axs[1].hlines(y=-0.01, xmin=ston, xmax=stoff, colors='k', linestyles='-', lw=5, label='Stimulus')  # Black stimulus line
axs[1].legend()  # Adjusted legend position

# Super labels
fig.supxlabel('Time (s)')  # Super x-axis label
fig.supylabel('Ca Activity (a.u)')  # Super y-axis label

plt.show()
#%% Pre/Post Conditioning Activity Lineplot ibo style

# Create figure and subplots with shared x/y axes
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

# Name the figure
fig.canvas.manager.set_window_title("KC Activity Pre vs. Post-Conditioning")

# Set y-axis limits for both plots
axs[0].set_ylim([-0.008, 2.2])

### --- Pre-conditioning Plot ---
# Compute mean and standard deviation for Reliable responders (Pre)
mean_rs_pre = np.mean(CaKClobe_pre[:, rs], axis=1)
std_rs_pre = np.std(CaKClobe_pre[:, rs], axis=1)

# Plot mean and standard deviation shaded area for Reliable responders (Pre)
axs[0].plot(tarr, mean_rs_pre, 'r', label='Reliable', lw=2)  # Mean reliable
axs[0].fill_between(tarr, 
                    np.clip(mean_rs_pre - std_rs_pre, 0, None),  # Prevent negative values
                    mean_rs_pre + std_rs_pre, 
                    color='r', alpha=0.25)  # ±1 standard deviation shading reliable

# Compute mean and standard deviation for Unreliable responders (Pre)
mean_us_pre = np.mean(CaKClobe_pre[:, us], axis=1)
std_us_pre = np.std(CaKClobe_pre[:, us], axis=1)

# Plot mean and standard deviation shaded area for Unreliable responders (Pre)
axs[0].plot(tarr, mean_us_pre, color='darkorange', label='Unreliable', lw=2)  # Mean unreliable
axs[0].fill_between(tarr, 
                    np.clip(mean_us_pre - std_us_pre, 0, None),  # Prevent negative values
                    mean_us_pre + std_us_pre, 
                    color='darkorange', alpha=0.25)  # ±1 standard deviation shading unreliable

# Labels, stimulus indicator, subplot title
axs[0].set_ylabel(r'Activity [a.u.]')
axs[0].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * 0.0035, '-', lw=5, color='gray')  # Stimulus indicator
axs[0].set_title('Pre')

### --- Post-conditioning Plot ---
# Compute mean and standard deviation for Reliable responders (Post)
mean_rs_post = np.mean(CaKClobe_post[:, rs], axis=1)
std_rs_post = np.std(CaKClobe_post[:, rs], axis=1)

# Plot mean and standard deviation shaded area for Reliable responders (Post)
axs[1].plot(tarr, mean_rs_post, 'r', label='Reliable', lw=2)  # Mean reliable
axs[1].fill_between(tarr, 
                    np.clip(mean_rs_post - std_rs_post, 0, None),  # Prevent negative values
                    mean_rs_post + std_rs_post, 
                    color='r', alpha=0.25)  # ±1 standard deviation shading reliable

# Compute mean and standard deviation for Unreliable responders (Post)
mean_us_post = np.mean(CaKClobe_post[:, us], axis=1)
std_us_post = np.std(CaKClobe_post[:, us], axis=1)

# Plot mean and standard deviation shaded area for Unreliable responders (Post)
axs[1].plot(tarr, mean_us_post, color='darkorange', label='Unreliable', lw=2)  # Mean unreliable
axs[1].fill_between(tarr, 
                    np.clip(mean_us_post - std_us_post, 0, None),  # Prevent negative values
                    mean_us_post + std_us_post, 
                    color='darkorange', alpha=0.25)  # ±1 standard deviation shading unreliable

# Stimulus indicator, subplot title, and legend
axs[1].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * 0.0035, '-', lw=5, color='gray')  # Stimulus indicator
axs[1].set_title('Post')
axs[1].legend()  # Adjusted legend position

# Add centered x-label
fig.text(0.531, 0.02, 't [s]', ha='center')


fig.subplots_adjust(
    top=0.935,
    bottom=0.1,
    left=0.11,
    right=0.965,
    hspace=0.2,
    wspace=0.2
    )


# Maximize figure window
manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.show()
#%% Pre/Post Conditioning Activity Colormatirx

# Sort responders by their input scale
sorted_indices = np.argsort(inpscale[responders])[::-1]  # Sort in descending order
responders_sorted = responders[sorted_indices]  # Reorder responders

# Find the index where input scale is closest to 0.5
threshold_index = np.argmax(inpscale[responders_sorted] < 0.5)  # First KC below 0.5

# **Convert threshold index to the correct y-position in imshow**
threshold_y_position = len(responders_sorted) - threshold_index  # Flip because of invert_yaxis()

# **Adjust time range for new x-axis labels**
t_start = ston - 1  # One second before stimulus onset
t_end = stoff + 2  # Two seconds after stimulus offset

# **Include -1s in the tick labels**
tick_positions = np.arange(t_start, t_end + 1, 1)  # Start from t_start (-1s)
tick_labels = [f"{t-ston:.0f}" for t in tick_positions]  # Adjust so ston = 0, keeping -1s

# Apply the time mask to the activity matrices
time_mask = (tarr >= t_start) & (tarr <= t_end)
CaKClobe_pre_sorted = CaKClobe_pre[time_mask][:, responders_sorted]
CaKClobe_post_sorted = CaKClobe_post[time_mask][:, responders_sorted]

# Define color scale limits across both plots
vmin = min(CaKClobe_pre_sorted.min(), CaKClobe_post_sorted.min())
vmax = max(CaKClobe_pre_sorted.max(), CaKClobe_post_sorted.max())

width_cm = 39.5
height_cm = 19.5
width_inches = width_cm / 2.54
height_inches = height_cm / 2.54

# Create figure and subplots, increase right margin to fit color bar
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(width_inches, height_inches))
fig.canvas.manager.set_window_title("KC Activity Pre vs. Post Conditioning Color Matrix")

# **Reduce space between subplots**
plt.subplots_adjust(wspace=0.09, bottom=0.15)  # Adjust spacing, leave space for x-label

# Pre-conditioning subplot - Sorted by Input Scale
img1 = axs[0].imshow(CaKClobe_pre_sorted.T, aspect='auto', cmap='viridis', interpolation='none', 
                      vmin=vmin, vmax=vmax, extent=[t_start, t_end, 0, len(responders_sorted)])

# **Fix horizontal line at correct input scale**
axs[0].hlines(y=threshold_y_position, xmin=t_start, xmax=t_end, color='white', linestyles='dashed', lw=2)

# **Add red vertical lines at stimulus onset and offset**
axs[0].axvline(x=ston, color='red', linestyle='-', linewidth=2)   # Stimulus ON
axs[0].axvline(x=stoff, color='red', linestyle='-', linewidth=2)  # Stimulus OFF

# Pre-conditioning subplot formatting
axs[0].set_title('Pre', fontsize = 32)

# **Ensure -1s is included in the x-ticks**
axs[0].set_xticks(tick_positions)
axs[0].set_xticklabels(tick_labels)  # Shift labels so stimulus onset is at 0, with -1s included

# Remove y-axis ticks and labels for both subplots
axs[0].set_yticks([])
axs[1].set_yticks([])

axs[0].invert_yaxis()  # Ensure correct order (largest input scale at bottom)

# Post-conditioning subplot - Sorted by Input Scale
img2 = axs[1].imshow(CaKClobe_post_sorted.T, aspect='auto', cmap='viridis', interpolation='none', 
                      vmin=vmin, vmax=vmax, extent=[t_start, t_end, 0, len(responders_sorted)])

# **Fix horizontal line at correct input scale**
axs[1].hlines(y=threshold_y_position, xmin=t_start, xmax=t_end, color='white', linestyles='dashed', lw=2)

# **Add red vertical lines at stimulus onset and offset**
axs[1].axvline(x=ston, color='red', linestyle='-', linewidth=2)   # Stimulus ON
axs[1].axvline(x=stoff, color='red', linestyle='-', linewidth=2)  # Stimulus OFF

# Post-conditioning subplot formatting
axs[1].set_title('Post', fontsize = 32)

# **Ensure -1s is included in the x-ticks**
axs[1].set_xticks(tick_positions)
axs[1].set_xticklabels(tick_labels)  # Ensure both subplots use the same x-labels

axs[1].invert_yaxis()

# **Create color bar to the right of the second subplot**
cbar_ax = fig.add_axes([0.918, 0.115, 0.03*0.5, 0.837])  # [left, bottom, width, height]
cbar = fig.colorbar(img1, cax=cbar_ax, orientation='vertical')
cbar.ax.tick_params()  # Adjust font size
cbar.set_label('Activity [a.u.]')

# **Add a single x-axis label for both subplots**
fig.text(0.476, 0.024, 't [s]', ha='center')
fig.text(0.0223, 0.23, 'Reliable', ha='center', va='center', rotation=90, fontsize = 32)
fig.text(0.0223, 0.63,  'Unreliable', ha='center', va='center', rotation=90, fontsize = 32)

# Adjust layout with manually extracted parameters
fig.subplots_adjust(
    top=0.955,
    bottom=0.12,
    left=0.04,
    right=0.91,
    hspace=0.075,
    wspace=0.08
    )

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

#plt.savefig('/home/baxter/IBM/göttingen_poster/figurs/KC Activity Pre vs. Post Conditioning Color Matrix.svg')
#%% Conditioning-Induced Activity Difference Plot ibo style

# --- Compute the differences (post-conditioning minus pre-conditioning) ---
diff_rs = CaKClobe_post[:, rs] - CaKClobe_pre[:, rs]
diff_us = CaKClobe_post[:, us] - CaKClobe_pre[:, us]

# Compute the mean and standard deviation across the responder cells at each time point
mean_diff_rs = np.mean(diff_rs, axis=1)
std_diff_rs = np.std(diff_rs, axis=1)

mean_diff_us = np.mean(diff_us, axis=1)
std_diff_us = np.std(diff_us, axis=1)

# --- Plotting ---
# Create figure and axis
fig, ax = plt.subplots()
fig.canvas.manager.set_window_title("Pre vs. Post-Conditioning Activity Difference")

# Reposition the x-axis (bottom spine) to y=0 (data coordinate)
ax.spines['bottom'].set_position(('data', -1.54))
ax.spines['left'].set_position(('data', 0))  # Move the y-axis to the 0s on the x-axis

# Define the time range for plotting (4 to 12 seconds)
time_mask = (tarr >= 4) & (tarr <= 12)

# Filter the data for the selected time range
tarr_filtered = tarr[time_mask]
mean_diff_rs_filtered = mean_diff_rs[time_mask]
std_diff_rs_filtered = std_diff_rs[time_mask]
mean_diff_us_filtered = mean_diff_us[time_mask]
std_diff_us_filtered = std_diff_us[time_mask]

# Adjusted time array so 4s is labeled as 0s
tarr_adjusted = tarr_filtered - 4

# Remove any data below 0s
tarr_adjusted = tarr_adjusted[tarr_adjusted >= 0]
mean_diff_rs_filtered = mean_diff_rs_filtered[tarr_adjusted >= 0]
std_diff_rs_filtered = std_diff_rs_filtered[tarr_adjusted >= 0]
mean_diff_us_filtered = mean_diff_us_filtered[tarr_adjusted >= 0]
std_diff_us_filtered = std_diff_us_filtered[tarr_adjusted >= 0]

# Plot the shaded standard deviation regions first (background)
ax.fill_between(
    tarr_adjusted,
    mean_diff_rs_filtered - std_diff_rs_filtered,
    mean_diff_rs_filtered + std_diff_rs_filtered,
    color='r',
    alpha=0.25,
    zorder=1
)

ax.fill_between(
    tarr_adjusted,
    mean_diff_us_filtered - std_diff_us_filtered,
    mean_diff_us_filtered + std_diff_us_filtered,
    color='darkorange',
    alpha=0.25,
    zorder=1
)

# Stimulus indicator (above shades)
ston_clipped = max(ston, 4)       # Ensure the lower bound is at least 4 s
stoff_clipped = min(stoff, 12)    # Ensure the upper bound is no more than 12 s
if ston_clipped < stoff_clipped:
    ax.hlines(
        y=-0.0008,
        xmin=ston_clipped - 4,  # Adjust stimulus onset time
        xmax=stoff_clipped - 4,  # Adjust stimulus offset time
        colors='gray',
        linestyles='-',
        lw=7,
        zorder=2
    )

# Dashed horizontal line at y=0 (above stimulus indicator)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, zorder=3)

# Plot the mean difference for Reliable responders (above everything else)
ax.plot(tarr_adjusted, mean_diff_rs_filtered, color='r', label='Reliable', lw=2.3, zorder=4)

# Plot the mean difference for Unreliable responders (above everything else)
ax.plot(tarr_adjusted, mean_diff_us_filtered, color='darkorange', alpha=0.5, label='Unreliable', lw=2.3, zorder=4)

# Add labels and legend
ax.set_xlabel('t [s]')
ax.set_ylabel('Activity Difference (a.u)')
ax.legend()

# Set the x-ticks so that every second is labeled and limit x-axis to start at 0
ax.set_xticks(np.arange(0, 9, 1))  # 0 to 8 seconds (corresponding to 4 to 12 original time)
ax.set_xticklabels([str(i) for i in range(0, 9)])
ax.set_xlim(left=0)  # Cut off the part of the x-axis below 0

# Set the lower y-limit (keeping the upper limit automatic)
ax.set_ylim(bottom=-1.53)

fig.subplots_adjust(
    top=0.96,
    bottom=0.125,
    left=0.125,
    right=0.965,
    hspace=0.2,
    wspace=0.2
    )


# Show plot
plt.show()
#%% Condtioning induced activity differnce heatmap

# Compute the differences (post-conditioning minus pre-conditioning)
diff = CaKClobe_post[:, responders] - CaKClobe_pre[:, responders]

# Ensure `responders` is properly indexed using `np.array(responders)`
sorted_indices = np.argsort(inpscale[responders])[::-1]  # Sort from highest to lowest

# Reorder the diff array based on the sorted indices
diff_sorted = diff[:, sorted_indices]

# Reorder the responders accordingly
responders_sorted = responders[sorted_indices]

# Set time range from -1s to 7s
t_start = ston - 1  # -1s
t_end = ston + 7    # 7s

# Find indices for the time range -1s to 7s
t_start_idx = np.argmin(np.abs(tarr - t_start))
t_end_idx = np.argmin(np.abs(tarr - t_end))

# Trim data to the desired range
diff_trimmed = diff_sorted[t_start_idx:t_end_idx, :]
tarr_trimmed = tarr[t_start_idx:t_end_idx]

# Shift time axis so that -1s corresponds to t_start
tarr_shifted = tarr_trimmed - ston  # Ensure that ston is 0s

# Normalize so that 0 corresponds to gray
norm = TwoSlopeNorm(vmin=np.min(diff_trimmed), vcenter=0, vmax=np.max(diff_trimmed))

width_cm = 24
height_cm = 19
width_inches = width_cm / 2.54
height_inches = height_cm / 2.54


# Create figure and single subplot
fig, ax = plt.subplots(figsize=(width_inches, height_inches))

# Name the figure
fig.canvas.manager.set_window_title("KC Activity Change (Post - Pre) Color Matrix (Sorted)")

# Plot only the selected time range with normalization
img = ax.imshow(diff_trimmed.T, aspect='auto', cmap='bwr', interpolation='none', norm=norm)

# Fix the x-axis tick issue (from -1s to 7s)
tick_positions = np.linspace(0, len(tarr_trimmed) - 1, num=9, dtype=int)
tick_labels = np.arange(-1, 8, 1)

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

# Remove y-ticks and labels
ax.set_yticks([])
ax.set_yticklabels([])

# Invert y-axis
ax.invert_yaxis()

# **Create color bar to the right of the second subplot**
cbar_ax = fig.add_axes([0.89, 0.107, 0.03*0.4, 0.87])  # [left, bottom, width, height]
cbar = fig.colorbar(img, cax=cbar_ax, orientation='vertical')
cbar.ax.tick_params()  # Adjust font size
cbar.set_label(r'$\Delta$ Activity [a.u.]')

# Fix separator line placement
sep_idx = len(responders_sorted)
if 0 < sep_idx < diff_trimmed.shape[1]:
    ax.axhline(y=sep_idx - 0.5, color='black', linestyle='--', linewidth=1.8)

# Add a dashed line corresponding to inpscale = 0.5
inpscale_sorted = inpscale[responders_sorted]
half_scale_idx = np.argmin(np.abs(inpscale_sorted - 0.5))
ax.axhline(y=half_scale_idx - 0.5, color='black', linestyle='--', linewidth=2)

# Add vertical red lines at stimulus onset and offset
ston_idx = np.argmin(np.abs(tarr_trimmed - ston)) if t_start <= ston <= t_end else None
stoff_idx = np.argmin(np.abs(tarr_trimmed - stoff)) if t_start <= stoff <= t_end else None

if ston_idx is not None:
    ax.axvline(x=ston_idx, color='red', linestyle='-', linewidth=2)
if stoff_idx is not None:
    ax.axvline(x=stoff_idx, color='red', linestyle='-', linewidth=2)

# **Add a single x-axis label for both subplots**
fig.text(0.453, 0.02, 't [s]', ha='center')
fig.text(0.0223, 0.22, 'Reliable', ha='center', va='center', rotation=90)
fig.text(0.0223, 0.63,  'Unreliable', ha='center', va='center', rotation=90)

fig.subplots_adjust(
    top=0.98,
    bottom=0.105,
    left=0.045,
    right=0.88,
    hspace=0.2,
    wspace=0.2
)

#plt.savefig('/home/baxter/IBM/göttingen_poster/figurs/KC Activity Change (Post - Pre) Color Matrix (Sorted).svg')
