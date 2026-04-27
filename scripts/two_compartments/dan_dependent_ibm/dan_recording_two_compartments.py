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
    
# Define save path for plots
save_path = "/home/baxter/IBM/presentations/figures/two_compartments"
os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
#%% Parameters

# Global parameters  
nKCs       = int(fitparams['restparams'][0]) # Number of Kenyon Cells  
dt         = 0.01                            # Time step for simulation (seconds) [0.01 => coarse Plotting, 0.001 thorough simulation]  
rng        = np.random.default_rng(420)      # Random number generator with a fixed seed [Now 420; 666 before fitting]  
bline      = fitparams['fittedpars_KD'][4]   # Baseline calcium concentration [Now 0.0; 0.02 before fitting]

# Input scale  
prr        = fitparams['restparams'][1]      # Rate of reliable responders to test odor  
pur        = fitparams['restparams'][2]      # Rate of unreliable responders to test odor   

# Null parameters  
tauinp     = fitparams['fittedpars_KD'][1]   # Input time constant (seconds) [Now 0.149; 0.2 before fitting]  
tauKCdec   = fitparams['fittedpars_KD'][0]   # Calcium decay time constant (seconds) [Now 0.44; 1.5 before fitting]  
n_str      = fitparams['fittedpars_KD'][5]   # Standard deviation of OU-process [Now 0.020; 0.01 before fitting]  

# Adaptation 
tauadapt   = fitparams['fittedpars_KD'][2]   # Time constant for adaptation (seconds) [Now 2.68; 2.0 before fitting]  
adaptscale = fitparams['fittedpars_KD'][3]   # Strength of adaptation [Now 0.58; 1.0 before fitting]  

# Inhibition parameters  
tauinh     = fitparams['fittedpars_WT'][0]   # Time constant for lateral inhibition (seconds) [Now 1.004; 1.5 before fitting]  
inhfactor  = fitparams['fittedpars_WT'][1]   # Inhibition scale factor [Now 15.53; 10 for sparse input, 1 for uniform input before fitting]  
infp       = fitparams['fittedpars_WT'][2]   # Midpoint of the sigmoidal function [Now 1.28; 0.5 before fitting]  
slf        = fitparams['fittedpars_WT'][3]   # Slope factor of the sigmoidal function [Now 0.16; 0.03 before fitting]

# Load the result arrays into variables with the same names as in the condtioning script
butscale_app  = dan_conditioning_results["butscale_appetetive"]
butscale_ave  = dan_conditioning_results["butscale_aversive"]
inpscale      = dan_conditioning_results["inpscale"]
rs            = dan_conditioning_results["rs"]
us            = dan_conditioning_results["us"]
dan_0         = dan_conditioning_results["dan_0"]
r             = dan_conditioning_results["r"]
responders    = dan_conditioning_results["responders"]
#%% Preallocations

# Generate tarr, starr (ODOR RECORING SETTING)
ston     = 5           # Stimulus onset time
stoff    = 10          # Stimulus offset time
tend     = 15          # End time of the simulation
tarr, starr = mdl.generate_step_stimulus(tend, ston, stoff, dt, 1 - bline) # Simulate a 5s long stimulus

# Responder rates and inputscale are inported form condtioning reslut arrarys 

# Generate Inhibition Matrix and Scale
inhscale = np.ones([nKCs, nKCs])              # Lateral inhibition connectivity matrix [inhibited KC X inhibiting KC]
inhscale[np.diag_indices(nKCs)] = 0           # Remove self-inhibition
inhscale /= np.sum(inhscale, axis=1)[:, None] # Normalize so that each KC receives the same amount of inhibition from all other KCs. 
inhscale *= inhfactor                         # Control the degree of inhibition 

# Generate noise
n_str = n_str * np.sqrt(2 / tauKCdec * dt)
noise = rng.standard_normal([len(tarr) - 1, nKCs])

# Generate Simulation arrays
CaKCcal         = np.zeros([len(tarr), nKCs]) # Calcium calixe
CaKClobe_ave    = np.zeros([len(tarr), nKCs]) # Aversive MB lobe compartment
CaKClobe_app    = np.zeros([len(tarr), nKCs]) # Appetetive MB lobe compartment
CaKCcal[0]      = bline                       # Initial calcium calix
CaKClobe_ave[0] = bline                       # Aversive MB lobe compartment initial calcium concentration
CaKClobe_app[0] = bline                       # Appetetive MB lobe compartment initial calcium concentration
adapt           = np.zeros([len(tarr), nKCs]) # Adaptation array
inh_ave = np.zeros([len(tarr), nKCs])         # Lateral inhibition array aversive comartment
inh_app = np.zeros([len(tarr), nKCs])         # Lateral inhibition array appetetive comartment
#%% Simulations

# Simulate calcium arrays useing Euler's method
for i in range(len(tarr) - 1):
    # Calyx activity
    CaKCcal[i + 1] = CaKCcal[i] + ((inpscale * starr[i]) / tauinp - (CaKCcal[i] - bline) / tauKCdec) * dt
    CaKClobe_app[i + 1] = CaKClobe_app[i] + ((butscale_app * inpscale * starr[i]) / tauinp - (CaKClobe_app[i] - bline) / tauKCdec) * dt
    CaKClobe_ave[i + 1] = CaKClobe_ave[i] + ((butscale_ave * inpscale * starr[i]) / tauinp - (CaKClobe_ave[i] - bline) / tauKCdec) * dt
    
    # Add noise
    CaKCcal[i + 1] += noise[i] * n_str
    CaKClobe_app[i + 1] += noise[i] * n_str
    CaKClobe_ave[i + 1] += noise[i] * n_str

    # Adaptation
    adapt[i + 1] = mdl.adaptation_dynamics(adapt[i], CaKCcal[i], tauadapt, adaptscale, dt)
    CaKCcal[i + 1] -= (adapt[i] * CaKCcal[i]) * (1 / tauKCdec) * dt
    CaKClobe_app[i + 1] -=  (adapt[i] * CaKClobe_app[i]) * (1 / tauKCdec) * dt
    CaKClobe_ave[i + 1] -=  (adapt[i] * CaKClobe_ave[i]) * (1 / tauKCdec) * dt
    
    # Inhibition 
    inh_app[i + 1] = mdl.inhibition_dynamics(inh_app[i], CaKClobe_app[i], tauinh, butscale_app * inhscale, dt) * mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_app[i], infp, slf)
    inh_ave[i + 1] = mdl.inhibition_dynamics(inh_ave[i], CaKClobe_ave[i], tauinh, butscale_ave * inhscale, dt) * mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_ave[i], infp, slf)
    CaKClobe_app[i + 1] -= (inh_app[i]) * (1 / tauKCdec) * dt
    CaKClobe_ave[i + 1] -= (inh_ave[i]) * (1 / tauKCdec) * dt

    # Ensure non-zero activity values
    CaKCcal[i + 1] = np.clip(CaKCcal[i + 1], 0, None)
    CaKClobe_ave[i + 1] = np.clip(CaKClobe_ave[i + 1], 0, None)
    CaKClobe_app[i + 1] = np.clip(CaKClobe_app[i + 1], 0, None)
#%% Aversive / Appetitive Post-Conditioning Activity Line Plot

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

# Name the figure
fig.canvas.manager.set_window_title('Aversive / Appetitive Post-Conditioning Activity Line Plot')

# Plot 2: Aversive Compartment
axs[0].plot(tarr, CaKClobe_ave[:, rs], color='r', alpha=0.2)  # aversive compartment reliable red
axs[0].plot(tarr, CaKClobe_ave[:, us], color='darkorange', alpha=0.2)  # aversive compartment unreliable orange
axs[0].plot(tarr, np.mean(CaKClobe_ave[:, responders], axis=1), color='black', lw=4)  # Mean activity
axs[0].hlines(y=-0.01, xmin=ston, xmax=stoff, colors='gray', linestyles='-', lw=4)  # Black stimulus line
axs[0].set_title('Aversive Compartment')

# Plot 1: Appetitive Compartment
axs[1].plot(tarr, CaKClobe_app[:, rs], color='r', alpha=0.2)  # appetetive compartment reliable red
axs[1].plot(tarr, CaKClobe_app[:, us], color='darkorange', alpha=0.2)  # aversive compartment unreliable orange
axs[1].plot(tarr, np.mean(CaKClobe_app[:, responders], axis=1), color='black', lw=4)  # Mean activity
axs[1].hlines(y=-0.01, xmin=ston, xmax=stoff, colors='k', linestyles='-', lw=5)  # Black stimulus line
axs[1].set_title('Appetitive Compartment')

# Add legend to aversive subplot
axs[1].plot([], [], color='r', lw=2.5, label='Reliable')  # Empty plot for legend 
axs[1].plot([], [], color='darkorange', lw=2.5, label='Unreliable') # Empty plot for legend
axs[1].plot([], [], color='black', lw=4, label='Mean Activity')
axs[1].hlines(y=-0.01, xmin=ston, xmax=stoff, colors='gray', linestyles='-', lw=4)  # Black stimulus line
axs[1].legend(loc='upper right', frameon= False, bbox_to_anchor=(1.1, 0.95))  # Adjusted legend position

# Super labels
fig.text(0.528, 0.02, 't [s]', ha='center')
fig.supylabel('Ca Activity (a.u.)')  # Super y-axis label

fig.subplots_adjust(
    top=0.94,
    bottom=0.095,
    left=0.085,
    right=0.955,
    hspace=0.2,
    wspace=0.2
    )

plt.show()
#%% Aversive / Appetetive Post Conditioning Activity Lineplot ibo style

# Create figure and subplots with shared x/y axes
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

# Name the figure
fig.canvas.manager.set_window_title("Aversive / Appetetive Post Conditioning Activity Lineplot")

# Set y-axis limits for both plots
axs[0].set_ylim([-0.008, 2.2])

#--- Aversive Compartment post conditioning Plot ---
# Compute mean and standard deviation for Reliable responders (Post)
mean_rs_ave = np.mean(CaKClobe_ave[:, rs], axis=1)
std_rs_ave = np.std(CaKClobe_ave[:, rs], axis=1)

# Plot mean and standard deviation shaded area for Reliable responders Aversive Comaprtment
axs[0].plot(tarr, mean_rs_ave, 'r', label='Reliable', lw=2)  # Mean reliable
axs[0].fill_between(tarr, np.clip(mean_rs_ave - std_rs_ave, 0, None),  # Prevent negative values
                    mean_rs_ave + std_rs_ave, color='r', alpha=0.25)   # ±1 standard deviation shading reliable

# Compute mean and standard deviation for Unreliable responders Aversive Comaprtment
mean_us_ave = np.mean(CaKClobe_ave[:, us], axis=1)
std_us_ave  = np.std(CaKClobe_ave[:, us], axis=1)

# Plot mean and standard deviation shaded area for Unreliable responders Aversive Comaprtment
axs[0].plot(tarr, mean_us_ave, color='darkorange', label='Unreliable', lw=2)    # Mean unreliable
axs[0].fill_between(tarr, np.clip(mean_us_ave - std_us_ave, 0, None),           # Prevent negative values
                    mean_us_ave + std_us_ave, color='darkorange', alpha=0.25) # ±1 standard deviation shading unreliable

# Stimulus indicator, subplot title, and legend
axs[0].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * 0.0035, '-', lw=5, color='gray')  # Stimulus indicator
axs[0].set_title('Aversive Compartment')
axs[0].set_ylabel(r'Activity [a.u.]')


### --- Appetetive Compartment Post conditioning Plot ---
# Compute mean and standard deviation for reliable responders appetetive compartment
mean_rs_app = np.mean(CaKClobe_app[:, rs], axis=1)
std_rs_app = np.std(CaKClobe_app[:, rs], axis=1)

# Plot mean and standard deviation shaded area for Reliable responders appetetive compartment
axs[1].plot(tarr, mean_rs_app, 'r', label='Reliable', lw=2)  # Mean reliable
axs[1].fill_between(tarr, np.clip(mean_rs_app - std_rs_app, 0, None), # Prevent negative values
                    mean_rs_app + std_rs_app, color='r', alpha=0.25)  # ±1 standard deviation shading reliable

# Compute mean and standard deviation for Unreliable responders appetetive compartment
mean_us_app = np.mean(CaKClobe_app[:, us], axis=1)
std_us_app = np.std(CaKClobe_app[:, us], axis=1)

# Plot mean and standard deviation shaded area for Unreliable responders appetetive compartment
axs[1].plot(tarr, mean_us_app, color='darkorange', label='Unreliable', lw=2)  # Mean unreliable
axs[1].fill_between(tarr, np.clip(mean_us_app - std_us_app, 0, None),         # Prevent negative values
                    mean_us_app + std_us_app, color='darkorange', alpha=0.25) # ±1 standard deviation shading unreliable

# Labels, stimulus indicator, subplot title
axs[1].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * 0.0035, '-', lw=5, color='gray')  # Stimulus indicator
axs[1].set_title('Appetetive Compartment')
axs[1].legend()


# Add centered x-label
fig.text(0.531, 0.02, 't [s]', ha='center')

fig.subplots_adjust(
    top=0.94,
    bottom=0.085,
    left=0.075,
    right=0.985,
    hspace=0.185,
    wspace=0.09
    )

# Maximize figure window
manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.show()
#%% Aversive / Appetetive Post Conditioning Activity Colormatirx

# Sort responders by their input scale
sorted_indices = np.argsort(inpscale[responders])[::-1]  # Sort in descending order
responders_sorted = responders[sorted_indices]           # Reorder responders

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
CaKClobe_app_sorted = CaKClobe_app[time_mask][:, responders_sorted]
CaKClobe_ave_sorted = CaKClobe_ave[time_mask][:, responders_sorted]

# Define color scale limits across both plots
vmin = min(CaKClobe_app_sorted.min(), CaKClobe_ave_sorted.min())
vmax = max(CaKClobe_app_sorted.max(), CaKClobe_ave_sorted.max())

# Create figure and subplots, increase right margin to fit color bar
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
fig.canvas.manager.set_window_title("Aversive vs. Appetetive post conditioning color matrix")

# **Reduce space between subplots**
plt.subplots_adjust(wspace=0.09, bottom=0.15)  # Adjust spacing, leave space for x-label

# Post-conditioning subplot - Sorted by Input Scale
img1 = axs[0].imshow(CaKClobe_ave_sorted.T, aspect='auto', cmap='viridis', interpolation='none', 
                      vmin=vmin, vmax=vmax, extent=[t_start, t_end, 0, len(responders_sorted)])

# **Fix horizontal line at correct input scale**
axs[0].hlines(y=threshold_y_position, xmin=t_start, xmax=t_end, color='white', linestyles='dashed', lw=2)

# **Add red vertical lines at stimulus onset and offset**
axs[0].axvline(x=ston, color='red', linestyle='-', linewidth=2)   # Stimulus ON
axs[0].axvline(x=stoff, color='red', linestyle='-', linewidth=2)  # Stimulus OFF

# Post-conditioning subplot formatting
axs[0].set_title('Aversive Comaprtment', fontsize = 32)

# **Ensure -1s is included in the x-ticks**
axs[0].set_xticks(tick_positions)
axs[0].set_xticklabels(tick_labels)  # Ensure both subplots use the same x-labels

axs[0].invert_yaxis()

# Pre-conditioning subplot - Sorted by Input Scale
img2 = axs[1].imshow(CaKClobe_app_sorted.T, aspect='auto', cmap='viridis', interpolation='none', 
                      vmin=vmin, vmax=vmax, extent=[t_start, t_end, 0, len(responders_sorted)])

# **Fix horizontal line at correct input scale**
axs[1].hlines(y=threshold_y_position, xmin=t_start, xmax=t_end, color='white', linestyles='dashed', lw=2)

# **Add red vertical lines at stimulus onset and offset**
axs[1].axvline(x=ston, color='red', linestyle='-', linewidth=2)   # Stimulus ON
axs[1].axvline(x=stoff, color='red', linestyle='-', linewidth=2)  # Stimulus OFF

# Pre-conditioning subplot formatting
axs[1].set_title('Appeteive Compartment', fontsize = 32)

# **Ensure -1s is included in the x-ticks**
axs[1].set_xticks(tick_positions)
axs[1].set_xticklabels(tick_labels)  # Shift labels so stimulus onset is at 0, with -1s included

# Remove y-axis ticks and labels for both subplots
axs[1].set_yticks([])
axs[0].set_yticks([])

axs[1].invert_yaxis()  # Ensure correct order (largest input scale at bottom)

# **Create color bar to the right of the second subplot**
cbar_ax = fig.add_axes([0.918, 0.105, 0.03*0.5, 0.85])  # [left, bottom, width, height]
cbar = fig.colorbar(img1, cax=cbar_ax, orientation='vertical')
cbar.ax.tick_params(labelsize=20)  # Adjust font size
cbar.set_label('Activity [a.u.]', fontsize = 30)

# **Add a single x-axis label for both subplots**
fig.text(0.472, 0.035, 't [s]', ha='center', fontsize = 30)
fig.text(0.0223, 0.23, 'Reliable', ha='center', va='center', rotation=90, fontsize = 32)
fig.text(0.0223, 0.63,  'Unreliable', ha='center', va='center', rotation=90, fontsize = 32)

# Adjust layout with manually extracted parameters
fig.subplots_adjust(
    top=0.955,
    bottom=0.105,
    left=0.04,
    right=0.905,
    hspace=0.075,
    wspace=0.06
    )

plt.show()
#%% Conditioning-Induced Activity Difference Aversive vs Appetetive Plot Ibo style

# --- Compute the differences (post-conditioning minus pre-conditioning) ---
diff_rs = CaKClobe_ave[:, rs] - CaKClobe_app[:, rs]
diff_us = CaKClobe_ave[:, us] - CaKClobe_app[:, us]

# Compute the mean and standard deviation across the responder cells at each time point
mean_diff_rs = np.mean(diff_rs, axis=1)
std_diff_rs = np.std(diff_rs, axis=1)

mean_diff_us = np.mean(diff_us, axis=1)
std_diff_us = np.std(diff_us, axis=1)

# --- Plotting ---
# Create figure and axis
fig, ax = plt.subplots()
fig.canvas.manager.set_window_title("Aversive vs. Appetetive Conditioning Induced Activity Difference")

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
    zorder=1)

ax.fill_between(
    tarr_adjusted,
    mean_diff_us_filtered - std_diff_us_filtered,
    mean_diff_us_filtered + std_diff_us_filtered,
    color='darkorange',
    alpha=0.25,
    zorder=1)

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
        zorder=2)

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

ax.fig.subplots_adjust(
    top=0.985,
    bottom=0.095,
    left=0.095,
    right=0.99,
    hspace=0.2,
    wspace=0.2)

# Show plot
plt.show()
#%% Aversive vs Appetetive Condtioning induced activity differnce heatmap


# Compute the differences (Aversvie - Appetetive Comaprtment Activity)
diff = CaKClobe_ave[:, responders] - CaKClobe_app[:, responders]

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

# Create figure and single subplot
fig, ax = plt.subplots()

# Name the figure
fig.canvas.manager.set_window_title("Conditioning induced activity differnce Aversive - Appetetive Color Matrix")

# Plot only the selected time range with normalization
img = ax.imshow(diff_trimmed.T, aspect='auto', cmap='bwr', interpolation='none', norm=norm)

# Fix the x-axis tick issue (from -1s to 7s)
tick_positions = np.linspace(0, len(tarr_trimmed) - 1, num=9, dtype=int)
tick_labels = np.arange(-1, 8, 1)

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize = 20)

# Remove y-ticks and labels
ax.set_yticks([])
ax.set_yticklabels([])

# Invert y-axis
ax.invert_yaxis()

# **Create color bar to the right of the second subplot**
cbar_ax = fig.add_axes([0.896, 0.105, 0.03*0.4, 0.87])  # [left, bottom, width, height]
cbar = fig.colorbar(img, cax=cbar_ax, orientation='vertical')
cbar.ax.tick_params(labelsize=20)  # Adjust font size
cbar.set_label(r'$\Delta$ Activity [a.u.]', fontsize=28)

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
fig.text(0.4675, 0.035, 't [s]', ha='center', fontsize=32)
fig.text(0.0223, 0.23, 'Reliable', ha='center', va='center', rotation=90, fontsize = 32)
fig.text(0.0223, 0.63,  'Unreliable', ha='center', va='center', rotation=90, fontsize = 32)

fig.subplots_adjust(
    top=0.98,
    bottom=0.105,
    left=0.045,
    right=0.89,
    hspace=0.2,
    wspace=0.2)

plt.show()