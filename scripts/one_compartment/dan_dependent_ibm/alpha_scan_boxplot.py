#%% Base

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import sys
import os
import io
import importlib.resources as pkg_resources  # Importing package resource handler
from IBM_functions_package import activity_model_functions as mdl
from IBM_functions_package import plotting_related_functions
import matplotlib.patches as mpatches

# Load the .npz parameter file from the package folder into memory
with pkg_resources.files("IBM_functions_package").joinpath("model_fit_normfac_3.885.npz").open("rb") as f:
    fitparams = np.load(io.BytesIO(f.read()))  # Read entire file into memory

# Global parameters  
nKCs       = int(fitparams['restparams'][0])  # Number of Kenyon Cells  
dt         = 0.01                             # Time step for simulation (seconds)  
rng        = np.random.default_rng(420)       # Random number generator with a fixed seed  
bline      = fitparams['fittedpars_KD'][4]      # Baseline calcium concentration

# Input scale  
prr        = fitparams['restparams'][1]         # Rate of reliable responders  
pur        = fitparams['restparams'][2]         # Rate of unreliable responders  

# Null parameters  
tauinp     = fitparams['fittedpars_KD'][1]        # Input time constant (seconds)  
tauKCdec   = fitparams['fittedpars_KD'][0]        # Calcium decay time constant (seconds)  
n_str      = fitparams['fittedpars_KD'][5]        # Standard deviation of OU-process  

# Adaptation parameters  
tauadapt   = fitparams['fittedpars_KD'][2]        # Time constant for adaptation (seconds)  
adaptscale = fitparams['fittedpars_KD'][3]        # Strength of adaptation  

# Inhibition parameters  
tauinh     = fitparams['fittedpars_WT'][0]        # Time constant for lateral inhibition (seconds)  
inhfactor  = fitparams['fittedpars_WT'][1]        # Inhibition scale factor  
infp       = fitparams['fittedpars_WT'][2]        # Midpoint of the sigmoidal function  
slf        = fitparams['fittedpars_WT'][3]        # Slope factor of the sigmoidal function

# DAN pulse timing and settings
npules     = 12               # Number of DAN pulses
pulsdur    = 1.25             # Pulse duration in seconds (on time)
pulsint    = 3.75             # Inter-pulse interval in seconds (off time)
danon      = 15               # DAN onset time at 15 seconds
danoff     = 75               # DAN offset time at 75 seconds
dan_0      = 0.2              # Initial pulse magnitude 
r          = 0.8              # Reduction factor 
alpha      = 0.5              # Learning Rate (default)

# Generate odor / input scale (Condtiong + Recording)
nrs  = rng.poisson(prr * nKCs)                        # Number of reliable responders
nurs = rng.poisson(pur * nKCs)                        # Number of unreliable responders
inpscale = np.zeros(nKCs)                             # Sensitivity of KCs to input stimulus
rs = rng.choice(np.arange(nKCs), nrs, replace=False)  # Indices of reliable responders
remaining_indices = list(set(np.arange(nKCs)) - set(rs))
us = rng.choice(np.array(remaining_indices), nurs, replace=False)  # Indices of unreliable responders
inpscale[rs] = rng.uniform(0.5, 1.0, nrs)           # Reliable responders
inpscale[us] = rng.uniform(0, 0.5, nurs)            # Unreliable responders
responders = np.concatenate([rs, us])               # List of responding KCs 
indx_active_kc = np.where(inpscale > 0)[0]          # Active cell indices

# Generate Inhibition Matrix and Scale
inhscale = np.ones([nKCs, nKCs])
inhscale[np.diag_indices(nKCs)] = 0
inhscale /= np.sum(inhscale, axis=1)[:, None]
inhscale *= inhfactor


# Define the desired alpha values
alpha_values = np.linspace(0.25, 1, num=4)  # Adjust endpoints and number as needed

print("Scanned Alpha values:")
for index, a_val in enumerate(alpha_values):
    print(f"Value {index}           Alpha = {a_val:.5f}")

# Generate condtioning time and stimulus array 
ston_con = 10   # odor onset in seconds
stdur_con = 60  # odor duration in seconds
stoff_con = 70  # odor offset (seconds)
tend_con = 80   # recording offset (seconds)
tarr_con, starr_con = mdl.generate_step_stimulus(tend_con, ston_con, stoff_con, dt, 1 - bline)

# Generate condtioning noise
n_str_con = n_str * np.sqrt(2 / tauKCdec * dt)
noise_con = rng.standard_normal([len(tarr_con) - 1, nKCs])

# Generate DAN-Array
__, danarr = mdl.generate_step_stimulus(tend_con, danon, danoff, dt)  # Array with 1s between danon and danoff
dan_start_idx = int(danon / dt)     # e.g., 15 s -> index 1500
dan_end_idx = int(danoff / dt)        # e.g., 75 s -> index 7500
pulse_duration_in_steps = int(pulsdur / dt) + 1             # "On" period steps
interval_duration_in_steps = int((pulsdur + pulsint) / dt)    # Total steps per pulse cycle

# Apply pulse pattern (1.25 s on, 3.75 s off) with decreasing magnitude to the DAN array
for pulse_idx in range(npules):
    current_danmag = dan_0 * (r ** pulse_idx)
    pulse_start_idx = dan_start_idx + pulse_idx * interval_duration_in_steps
    pulse_end_idx = pulse_start_idx + pulse_duration_in_steps
    danarr[pulse_start_idx:pulse_end_idx] = current_danmag  
    danarr[pulse_end_idx:pulse_start_idx + interval_duration_in_steps] = 0  
danarr[dan_end_idx:] = 0  # Ensure DAN is zero after offset
#%% Define Simulation Function
def condtioning_simulation(danarr, alpha_val, butscale_conditioned, tarr_con, starr_con, noise_con):
    """
    Run the condtioning simulation on a DAN array and a learning rate alpha_val.
    Returns simulation arrays for CaKCcal_con, CaKClobe_con, CaKClobe_con_ut , the updated butscale_conditioned, and the DAN array.
    """
    # Preallocate arrays for dopamine affected compartment
    CaKCcal_con  = np.zeros([len(tarr_con), nKCs])
    CaKClobe_con = np.zeros([len(tarr_con), nKCs])
    CaKCcal_con[0] = bline
    CaKClobe_con[0] = bline
    adapt_con = np.zeros([len(tarr_con), nKCs])
    inh_con   = np.zeros([len(tarr_con), nKCs])
    
    # Preallocate arrays for dopamine unaffected compartment
    CaKClobe_ut = CaKClobe_con.copy()  # Untrained compartment 
    inh_con_ut = inh_con.copy()       # Untrained lateral inhibition state
    
    
    for i in range(len(tarr_con) - 1):
        # Update calcium dynamics (null dynamics) for both compartments.
        CaKCcal_con[i + 1] = CaKCcal_con[i] + ((inpscale * starr_con[i]) / tauinp - (CaKCcal_con[i] - bline) / tauKCdec) * dt
        CaKClobe_con[i + 1] = CaKClobe_con[i] + ((inpscale * butscale_conditioned[:, i] * starr_con[i]) / tauinp - (CaKClobe_con[i] - bline) / tauKCdec) * dt
        CaKClobe_ut[i + 1] = CaKClobe_ut[i] + ((inpscale  * starr_con[i]) / tauinp - (CaKClobe_ut[i] - bline) / tauKCdec) * dt
        
        # Add noise
        CaKCcal_con[i + 1] += noise_con[i] * n_str_con
        CaKClobe_con[i + 1] += noise_con[i] * n_str_con
        CaKClobe_ut[i + 1] += noise_con[i] * n_str_con
        
        # Adaptation dynamics
        adapt_con[i + 1] = mdl.adaptation_dynamics(adapt_con[i], CaKCcal_con[i], tauadapt, adaptscale, dt)
        CaKCcal_con[i + 1] -= (adapt_con[i] * CaKCcal_con[i]) * (1 / tauKCdec) * dt
        CaKClobe_con[i + 1] -= (adapt_con[i] * CaKClobe_con[i]) * (1 / tauKCdec) * dt
        CaKClobe_ut[i + 1] -= (adapt_con[i] * CaKClobe_ut[i]) * (1 / tauKCdec) * dt
        
        # Inhibition dynamics (only applied to the conditioned compartment)
        inh_con[i + 1] = mdl.inhibition_dynamics(inh_con[i], CaKClobe_con[i], tauinh, inhscale, dt) * \
                     mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_con[i], infp, slf)
        inh_con_ut[i + 1] =mdl.inhibition_dynamics(inh_con_ut[i], CaKClobe_ut[i], tauinh, inhscale, dt) * \
                     mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_ut[i], infp, slf)
        
        CaKClobe_con[i + 1] -= inh_con[i] * (1 / tauKCdec) * dt
        CaKClobe_ut[i + 1]  -= inh_con_ut[i] * (1 / tauKCdec) * dt

        # Ensure non-negative calcium levels
        CaKCcal_con[i + 1] = np.clip(CaKCcal_con[i + 1], 0, None)
        CaKClobe_con[i + 1] = np.clip(CaKClobe_con[i + 1], 0, None)
        CaKClobe_ut[i + 1] = np.clip(CaKClobe_con[i + 1], 0, None)
        
        # Update bouton-scale in the conditioned compartment using DAN activity.
        butscale_conditioned[indx_active_kc, i + 1] = butscale_conditioned[indx_active_kc, i] - \
            (CaKClobe_con[i, indx_active_kc] * danarr[i] * alpha_val) * dt

    return CaKCcal_con, CaKClobe_con, CaKClobe_ut, butscale_conditioned

def simulation_recording(butscale_pre, butscale_post, tarr_rec, starr_rec, noise_rec):
    """
    Run the recording simulation.

    This function simulates calcium dynamics during a recording session.
    It uses two bouton-scale arrays:
      - butscale_pre: for the pre-conditioning (control) compartment.
      - butscale_conditioned: for the post-conditioning (conditioned) compartment.
    
    The simulation computes three calcium arrays:
      - CaKCcal: Global (null) dynamics.
      - CaKClobe_pre: Dynamics for the control compartment.
      - CaKClobe_post: Dynamics for the conditioned compartment.
    
    Returns:
      CaKCcal_rec, CaKClobe_pre, CaKClobe_post
    """
    # Preallocate calcium dynamics arrays.
    CaKCcal_rec = np.zeros([len(tarr_rec), nKCs])
    CaKClobe_pre = np.zeros([len(tarr_rec), nKCs])
    CaKClobe_post = np.zeros([len(tarr_rec), nKCs])
    
    # Set the initial calcium concentration.
    CaKCcal_rec[0] = bline
    CaKClobe_pre[0] = bline
    CaKClobe_post[0] = bline
    
    # Preallocate arrays for adaptation and inhibition.
    adapt_rec = np.zeros([len(tarr_rec), nKCs])
    inh_pre = np.zeros([len(tarr_rec), nKCs])
    inh_post = np.zeros([len(tarr_rec), nKCs])
    
    # Simulation loop.
    for i in range(len(tarr_rec) - 1):
        # --- Null dynamics update ---
        CaKCcal_rec[i + 1] = CaKCcal_rec[i] + (((butscale_pre * inpscale * starr_rec[i]) / tauinp) -
                                       ((CaKCcal_rec[i] - bline) / tauKCdec)) * dt
        CaKClobe_pre[i + 1] = CaKClobe_pre[i] + (((butscale_pre * inpscale * starr_rec[i]) / tauinp) -
                                                 ((CaKClobe_pre[i] - bline) / tauKCdec)) * dt
        CaKClobe_post[i + 1] = CaKClobe_post[i] + (((butscale_post * inpscale * starr_rec[i]) / tauinp) -
                                                   ((CaKClobe_post[i] - bline) / tauKCdec)) * dt
        
        # --- Add noise ---
        CaKCcal_rec[i + 1] += noise_rec[i] * n_str_rec
        CaKClobe_pre[i + 1] += noise_rec[i] * n_str_rec
        CaKClobe_post[i + 1] += noise_rec[i] * n_str_rec

        # --- Adaptation dynamics ---
        adapt_rec[i + 1] = mdl.adaptation_dynamics(adapt_rec[i], CaKCcal_rec[i], tauadapt, adaptscale, dt)
        CaKCcal_rec[i + 1] -= (adapt_rec[i] * CaKCcal_rec[i]) * (1 / tauKCdec) * dt
        CaKClobe_pre[i + 1] -= (adapt_rec[i] * CaKClobe_pre[i]) * (1 / tauKCdec) * dt
        CaKClobe_post[i + 1] -= (adapt_rec[i] * CaKClobe_post[i]) * (1 / tauKCdec) * dt

        # --- Inhibition dynamics ---
        inh_pre[i + 1] = mdl.inhibition_dynamics(inh_pre[i], CaKClobe_pre[i],
                                                 tauinh, (butscale_pre * inhscale), dt) * \
                         mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_pre[i], infp, slf)
        inh_post[i + 1] = mdl.inhibition_dynamics(inh_post[i], CaKClobe_post[i],
                                                  tauinh, (butscale_post * inhscale), dt) * \
                          mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe_post[i], infp, slf)
        
        CaKClobe_pre[i + 1] -= (inh_pre[i]) * (1 / tauKCdec) * dt
        CaKClobe_post[i + 1] -= (inh_post[i]) * (1 / tauKCdec) * dt

        # --- Ensure non-negative calcium levels ---
        CaKCcal_rec[i + 1] = np.clip(CaKCcal_rec[i + 1], 0, None)
        CaKClobe_pre[i + 1] = np.clip(CaKClobe_pre[i + 1], 0, None)
        CaKClobe_post[i + 1] = np.clip(CaKClobe_post[i + 1], 0, None)
    
    return CaKCcal_rec, CaKClobe_pre, CaKClobe_post
#%% Simulate Condtioning by Looping Over Alpha Values

# Empty dictionary to hold simulation results
condtioning_results = {}

for index, alpha_val in enumerate(alpha_values):
    print(f"Running condtioning simulation for alpha value [{index}] = {alpha_val}")
    
    # For each simulation run, create a fresh bouton-scale array to be alterd on the conditioned compartment by DAN activity.
    butscale_conditioned = np.ones([nKCs, len(tarr_con)])
    
    # Run the simulation using a fresh copy of the DAN array.
    CaKCcal_con, CaKClobe_con, CaKClobe_ut, butscale_conditioned = \
        condtioning_simulation(danarr, alpha_val, butscale_conditioned, tarr_con, starr_con, noise_con)
    
    # # Store gernated arrays in recording results dictionary.
    condtioning_results[alpha_val] = {
        #'CaKCcal': CaKCcal,
        #'CaKClobe': CaKClobe,
        #'CaKClobe_ut': CaKClobe_ut,
        'butscale_conditioned': butscale_conditioned,
        #'danarr': danarr_out
    }
#%% Recoring Simulation Preallocations

# Stimulus Generation for recording simulation:
ston_rec     = 5           # Stimulus onset time
stoff_rec    = 10          # Stimulus offset time
tend_rec     = 15          # End time of the simulation
tarr_rec, starr_rec = mdl.generate_step_stimulus(tend_rec, ston_rec, stoff_rec, dt, 1 - bline)

# Recompute noise as needed for recording simulation
n_str_rec = n_str * np.sqrt(2 / tauKCdec * dt)
noise_rec = rng.standard_normal([len(tarr_rec) - 1, nKCs])


# Empty dictionary to hold simulation results
recording_results = {}

# Create a pre-bouton scale as a 1D vector
butscale_pre = np.ones(nKCs)

for index, alpha_val in enumerate(alpha_values):
    print(f"Running recording simulation for alpha value [{index}] = {alpha_val}")
    
    # Create the post bouton-scale from the conditioning results dictionary
    butscale_post = condtioning_results[alpha_val]['butscale_conditioned'][:, -1]
    
    # Run the recording simulation
    CaKCcal_rec, CaKClobe_pre, CaKClobe_post = simulation_recording(butscale_pre, butscale_post, tarr_rec, starr_rec, noise_rec)
    
    # Store generated arrays in the recording results dictionary.
    recording_results[alpha_val] = {
        'CaKCcal_rec': CaKCcal_rec,
        'CaKClobe_pre': CaKClobe_pre,
        'CaKClobe_post': CaKClobe_post,
    }
#%% Pre/Post Conditioning Activity Boxplot at Odor Offset


# Define colors for Pre and Post
colors = ['#2E8B57', '#90EE90']  # Matte Green (Pre) & Lighter Green (Post)

# Define spacing for boxplot pairs
gap_between_pairs = 1.0  # Space between each alpha value's Pre/Post boxplots
box_offset = 0.13  # Offset between Pre and Post within a pair

# Ensure alpha values are in ascending order
alpha_values_sorted = sorted(alpha_values)

# Define x positions for each alpha value
num_alphas = len(alpha_values_sorted)
positions = np.arange(num_alphas) * gap_between_pairs  # Base positions

# Create a figure
fig, ax = plt.subplots()
fig.canvas.manager.set_window_title("KC Activity Boxplot at Odor Offset for All Alpha Values")

# Loop over all alpha values and create boxplots
for i, alpha_val in enumerate(alpha_values_sorted):
    # Extract data for current alpha value
    CaKClobe_pre = recording_results[alpha_val]['CaKClobe_pre']
    CaKClobe_post = recording_results[alpha_val]['CaKClobe_post']

    pre = CaKClobe_pre[tarr_rec == stoff_rec, responders]  # Only rs + us at offset
    post = CaKClobe_post[tarr_rec == stoff_rec, responders]  # Only rs + us at offset

    # Statistical testing
    tst = st.ttest_rel(pre, post)  # Paired (dependent) two-sample t-test
    p_value = tst.pvalue

    # Sanity check print
    print(f'Alpha Value: {alpha_val:<6.2f}  |  p-value: {p_value:.3e}')

    # Compute boxplot positions for current alpha (Pre and Post side by side)
    pre_pos = positions[i] - box_offset
    post_pos = positions[i] + box_offset

    # Create boxplots with outliers hidden
    bp = ax.boxplot(
        [pre, post],
        positions=[pre_pos, post_pos],
        patch_artist=True,
        widths=0.2,
        whis=1.5,
        showfliers=False,  # Hide outliers
        zorder=3,  # Boxplot on top of the background
        showmeans=True,  # Let means show
        meanprops={'marker': 'D', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 8}  # Controls mean markers
    )

    # Apply new green colors to the IQR (Interquartile Range) regions
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)  # Fill the IQR with color

    # Customize median lines
    for median in bp['medians']:
        median.set(color='black', linewidth=2.5)

    # Customize whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.2)

    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.5)

    # --- Add color-coded connecting lines ABOVE the boxplots ---
    for kc_idx, responder_idx in enumerate(responders):
        # Determine color based on index membership
        if responder_idx in rs:
            line_color = 'r'  # Reliable responders (Red)
        elif responder_idx in us:
            line_color = 'darkorange'  # Unreliable responders (Dark orange)
        else:
            line_color = 'dimgray'  # Default (if any index is uncategorized)

        ax.plot(
            [pre_pos, post_pos],  # X positions
            [pre[kc_idx], post[kc_idx]], 
            '-o', lw=1, ms=3, color=line_color, alpha=0.7, zorder=4  # Adjusted transparency for visibility
        )

    # --- Add significance stars based on p-value ---
    max_y = max(max(pre), max(post))  # Find max Y value of boxplots
    y_offset = 0.04 * max_y  # Offset above the boxplot

    # Determine the number of stars based on p-value
    if p_value < 0.001:
        star_label = '***'  # Highly significant
    elif p_value < 0.01:
        star_label = '**'   # Moderately significant
    elif p_value < 0.05:
        star_label = '*'    # Weakly significant
    else:
        star_label = ''  # No significance

    if star_label:
        # Draw a horizontal line above the boxplots
        ax.plot([pre_pos, post_pos], [max_y + y_offset] * 2, lw=2, color='black')

        # Annotate with stars
        ax.text((pre_pos + post_pos) / 2, max_y + 0.8 * y_offset, star_label, 
                ha='center', va='bottom', fontsize=20, fontweight='bold')

# Set x-axis labels and ticks
ax.set_xticks(positions)
ax.set_xticklabels([f"{a:.2f}" for a in alpha_values_sorted])  # Only 2 decimal places, no rotation

# Set y-axis labels
ax.set_ylabel('Activity [a.u.]')
ax.set_xlabel('Modulation Rate')  # Render as LaTeX

# Define legend patches for Pre/Post
legend_patches = [
    mpatches.Patch(color=colors[0], label="Pre"),
    mpatches.Patch(color=colors[1], label="Post"),
    plt.Line2D([], [], color='red', label='Reliable', lw=2),
    plt.Line2D([], [], color='darkorange', label='Unreliable', lw=2)
]

# Single legend combining all labels
ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0.835, 0.95))

# Reduce white space in y-axis
ax.set_ylim(0)

fig.subplots_adjust(
    top=0.92,
    bottom=0.115,
    left=0.1,
    right=0.895,
    hspace=0.2,
    wspace=0.2
    )

# Display plot
plt.show()
#%% Singel Apha box plot 

# Define colors for Pre and Post
colors = ['#2E8B57', '#90EE90']  # Matte Green (Pre) & Lighter Green (Post)

# Alpha value to plot
alpha_val = 0.5

# Define boxplot positions
pre_pos = 0.22 - 0.13 / 2
post_pos = 0.22 + 0.13 / 2

# Create a figure

width_cm = 10.769
height_cm = 18.480
width_inches = width_cm / 2.54
height_inches = height_cm / 2.54


# Create figure and single subplot
fig, ax = plt.subplots(figsize=(width_inches, height_inches))
fig.canvas.manager.set_window_title("KC Activity Boxplot at Odor Offset for Alpha = 0.5")

# Extract data for alpha = 0.5
CaKClobe_pre = recording_results[alpha_val]['CaKClobe_pre']
CaKClobe_post = recording_results[alpha_val]['CaKClobe_post']

pre = CaKClobe_pre[tarr_rec == stoff_rec, responders]  # Only rs + us at offset
post = CaKClobe_post[tarr_rec == stoff_rec, responders]  # Only rs + us at offset

# Perform statistical testing
tst = st.ttest_rel(pre, post)
p_value = tst.pvalue

# Create boxplot
bp = ax.boxplot(
    [pre, post],
    positions=[pre_pos, post_pos],
    patch_artist=True,
    widths=0.1,
    whis=1.5,
    showfliers=False,  # Hide outliers
    zorder=3,  # Boxplot on top of the background
    showmeans=True,  # Let means show
    meanprops={'marker': 'D', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 8}  # Controls mean markers
)

# Apply new green colors to the IQR (Interquartile Range) regions
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)  # Fill the IQR with color

# Customize median lines
for median in bp['medians']:
    median.set(color='black', linewidth=0)

# Customize whiskers and caps
for whisker in bp['whiskers']:
    whisker.set(color='black', linewidth=1.2)

for cap in bp['caps']:
    cap.set(color='black', linewidth=1.5)

# --- Add color-coded connecting lines ABOVE the boxplots ---
for kc_idx, responder_idx in enumerate(responders):
    # Determine color based on index membership
    if responder_idx in rs:
        line_color = 'r'  # Reliable responders (Red)
    elif responder_idx in us:
        line_color = 'darkorange'  # Unreliable responders (Dark orange)
    else:
        line_color = 'dimgray'  # Default (if any index is uncategorized)

    ax.plot(
        [pre_pos, post_pos],  # X positions
        [pre[kc_idx], post[kc_idx]], 
        '-o', lw=1.8, ms=3, color=line_color, alpha=0.6, zorder=4  # Adjusted transparency for visibility
    )

# --- Add significance stars based on p-value ---
max_y = max(max(pre), max(post))  # Find max Y value of boxplots
y_offset = 0.04 * max_y  # Offset above the boxplot

if p_value < 0.001:
    star_label = '***'  # Highly significant
elif p_value < 0.01:
    star_label = '**'   # Moderately significant
elif p_value < 0.05:
    star_label = '*'    # Weakly significant
else:
    star_label = ''  # No significance

if star_label:
    # Draw a horizontal line above the boxplots
    ax.plot([pre_pos, post_pos], [max_y + y_offset] * 2, lw=2, color='black')

    # Annotate with stars
    ax.text((pre_pos + post_pos) / 2, max_y + 0.8 * y_offset, star_label, 
            ha='center', va='bottom', fontsize=20, fontweight='bold')

# Legend
ax.plot([], [], color='red', label='Reliable', lw=2.5)
ax.plot([], [], color='darkorange', label='Unreliable', lw=2.5)
ax.legend(bbox_to_anchor=(0.375, 0.8), loc='center left', frameon=False, fontsize=30)

# Set x-axis labels and ticks correctly
ax.set_xticks([pre_pos, post_pos])
ax.set_xticklabels(["Pre", "Post"])

# Set y-axis label
ax.set_ylabel('Activity [a.u.]')

# Reduce white space in y-axis and fix x-axis limits
ax.set_ylim(0)
ax.set_xlim(pre_pos - 0.1, post_pos + 0.1)  # Ensuring boxplots are within visible range

# Adjust figure layout
fig.subplots_adjust(
    top=0.943,
    bottom=0.094,
    left=0.242,
    right=0.734,
    hspace=0.2,
    wspace=0.2
)

#plt.savefig('/home/baxter/IBM/göttingen_poster/figurs/KC Activity Boxplot at Odor Offset for Alpha = 0.5.svg')