#%% Base

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

from IBM_functions_package import singel_compartment
#%% Comaprtment Object

# Create reate model object
activity_model = singel_compartment.Compartment()

#Time and Stimulus array 
tarr  = activity_model.tarr
starr = activity_model.starr

#Random KCs for individual plotting
selected_KCs = activity_model.rng.choice(activity_model.nKCs, 100, replace=False)

#Active KCs
inpscale = activity_model.inpscale
responders = np.where(inpscale > 0)[0]

#%% Null Simulation

# Null model iterative variables
tauKCdec_values = [0.5, 1.0, 1.5]   # Calcium decay time constants (s)         default = 1.5
tauinp_values = [0.2, 0.4, 0.6]     # Input drive time constants (s)           default = 0.2
n_str_values = [0.01, 0.03, 0.07]   # Standard deviations of the OU process    default = 0.01  
bline_values = [0.02, 0.04, 0.08]   # Baseline calcium concentration           default = 0.02

# Initialize lists to store results
tauKCdec_results = []
tauinp_results = []
n_str_results = []
bline_results = []

# Iterate over tauKCdec values and store results
for tauKCdec in tauKCdec_values:
    activity_model.tauKCdec = tauKCdec
    null = activity_model.simulate_null()
    tauKCdec_results.append(null)
activity_model.tauKCdec = 1.5

# Iterate over tauinp values and store results
for tauinp in tauinp_values:
    activity_model.tauinp = tauinp
    null = activity_model.simulate_null()
    tauinp_results.append(null)
activity_model.tauinp = 0.2

# Iterate over n_str values and store results
for n_str in n_str_values:
    activity_model.n_str = n_str
    null = activity_model.simulate_null()
    n_str_results.append(null)
activity_model.n_str = 0.01

# Iterate over bline values and store results
for bline in bline_values:
    activity_model.bline = bline
    null = activity_model.simulate_null()
    bline_results.append(null)
activity_model.bline = 0.02

#%% Null Mean Bouton Response Plots

# Tau Calcium Decay
plt.figure('Calcium Decay')
for i, tauKCdec_result in enumerate(tauKCdec_results):
    plt.plot(tarr, np.mean(tauKCdec_result[:, responders], axis=1), label=rf'$\tau_{{Ca^{{2+}}}}$ = {tauKCdec_values[i]}s')
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
plt.title(r'Calcium Decay Constant Parameter Scan')
plt.xlabel('Time (s)')
plt.ylabel('Ca Activity (a.u)')
plt.legend()
plt.show()

# Tau Input Drive
plt.figure('Stimulus Drive')
for i, tauinp_result in enumerate(tauinp_results):
    plt.plot(tarr, np.mean(tauinp_result[:, responders], axis=1), label = rf'$\tau_{{s}}$ = {tauinp_values[i]}s')
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
plt.title('Stimulus Drive Constant Parameter Scan')
plt.xlabel('Time (s)')
plt.ylabel('Ca Activity (a.u)')
plt.legend()
plt.show()

# Standart Deviation OU Process
fig, axs = plt.subplots(1, len(n_str_values), figsize=(20, 5), sharey=True)

for i, n_str_result in enumerate(n_str_results):
    ax = axs[i]
    ax.plot(tarr, n_str_result[:, responders], alpha=0.4)
    ax.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
    ax.set_title(rf'$\sigma$ = {n_str_values[i]}')
fig.suptitle('SD of OU-Process Parameter Scan')
fig.supxlabel('Time (s)')
fig.supylabel('Ca Activity (a.u)')
plt.tight_layout()
plt.show()

# Baseline Calcium
plt.figure('Baseline Calcium')
for i, bline_result in enumerate(bline_results):
    plt.plot(tarr, np.mean(bline_result[:, responders], axis=1), label = rf'${{Ca}}^{{2+}}_{0}$ = {bline_values[i]}')
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
plt.title('Baseline Calcium Parameter Scan')
plt.xlabel('Time (s)')
plt.ylabel('Ca Activity (a.u)')
plt.legend()
plt.show()
#%%  Null Mean Bouton Response Subplot

fig = plt.figure(figsize=(10, 8))

# Define the outer grid with 2 rows and 2 columns
outer_grid = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.4)

# Baseline Calcium Concentration
ax1 = plt.Subplot(fig, outer_grid[0, 0])
fig.add_subplot(ax1)
for i, bline_result in enumerate(bline_results):
    ax1.plot(tarr, np.mean(bline_result[:,responders], axis=1), label = rf'${{Ca}}^{{2+}}_{0}$ = {bline_values[i]}')
ax1.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
ax1.set_title('Baseline Calcium', fontsize=18)
ax1.legend(fontsize=16, loc='upper right')

# Tau Input Drive
ax2 = plt.Subplot(fig, outer_grid[0, 1])
fig.add_subplot(ax2)
for i, tauinp_result in enumerate(tauinp_results):
    ax2.plot(tarr, np.mean(tauinp_result[:, responders], axis=1), label = rf'$\tau_{{s}}$ = {tauinp_values[i]}s')
ax2.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
ax2.set_title('Stimulus Drive Constant', fontsize=18)
ax2.legend(fontsize=15.5, loc='upper right')

# Tau Calcium Decay
ax3 = plt.Subplot(fig, outer_grid[1, 0])
fig.add_subplot(ax3)
for i, tauKCdec_result in enumerate(tauKCdec_results):
    ax3.plot(tarr, np.mean(tauKCdec_result[:, responders], axis=1), label=rf'$\tau_{{Ca^{{2+}}}}$ = {tauKCdec_values[i]}s')
ax3.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
ax3.set_title('Calcium Decay Constant', fontsize=18)
ax3.legend(fontsize=16, loc='upper right')

# Standard Deviation OU-Process
inner_grid = gridspec.GridSpecFromSubplotSpec(1, len(n_str_values), subplot_spec=outer_grid[1, 1], wspace=0.3, hspace=0.3)

for i, n_str_result in enumerate(n_str_results):
    ax = plt.Subplot(fig, inner_grid[0, i])
    fig.add_subplot(ax)
    ax.plot(tarr, n_str_result[:, responders], alpha=0.4)
    ax.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
    ax.text(1.2, 0.83, rf'$\sigma$ = {n_str_values[i]}', fontsize=16, ha='right', va='top', transform=ax.transAxes)
    if i == 0:
        ax.set_ylabel('Ca Activity (a.u)', fontsize=18)
        ax.set_xlabel('Time (s)', fontsize=18)
    if i > 0:
       ax.set_yticklabels([])

# Add a title specifically for the sub-subplots
fig.text(0.75, 0.46 ,'SD of OU-Process', ha='center', fontsize=15)

# Super labels
fig.suptitle('Null Parameter Scan', fontsize=24)
fig.supxlabel('Time (s)', fontsize=23)
fig.supylabel('Ca Activity (a.u)', fontsize=23)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

#%% Adaptation Only Simulation

# Adapation model iterative variables
tauadapt_values = [0.5, 2, 4]          # Adaptation time constant in s  default = 2
adaptscale_values = [0.5, 1, 1.5]      # Adaptation strength            default = 1

# Initialize lists to store results
tauadapt_results = []
adaptscale_results = []

# Iterate over tauadapt values and store results
for tauadapt in tauadapt_values:
    activity_model.tauadapt = tauadapt
    adaptation = activity_model.simulate_adaptation()
    tauadapt_results.append(adaptation)
activity_model.tauadapt = 2

# Iterate over adaptscale values and store results
for adaptscale in adaptscale_values:
    activity_model.adaptscale = adaptscale
    adaptation = activity_model.simulate_adaptation()
    adaptscale_results.append(adaptation)
activity_model.adaptscale = 1
#%% Adaptation Only Mean Bouton Response Plots

# Tau Adaptation
plt.figure('Tau Adapation')
for i, tauadapt_result in enumerate(tauadapt_results):
    plt.plot(tarr, np.mean(tauadapt_result[:,responders], axis=1), label = rf'$\tau_{{A}}$ = {tauadapt_values[i]}s')
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
plt.title('Adaptation Constant Parameter Scan')
plt.xlabel('Time (s)')
plt.ylabel('Ca Activity (a.u)')
plt.legend()
plt.show()

# Adapation Scale
plt.figure('Adapation Scale')
for i, adaptscale_result in enumerate(adaptscale_results):
    plt.plot(tarr, np.mean(adaptscale_result[:, responders], axis=1), label=rf'$q_{{A}}$ = {adaptscale_values[i]}')
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
plt.title('Adaptation Scale Parameter Scan')
plt.xlabel('Time (s)')
plt.ylabel('Ca Activity (a.u)')
plt.legend()
plt.show()

#%% Adaptation Only Mean Bouton Response Subplot

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Tau Adaptation
for i, tauadapt_result in enumerate(tauadapt_results):
    axs[0].plot(tarr, np.mean(tauadapt_result[:,responders], axis=1), label = rf'$\tau_{{A}}$ = {tauadapt_values[i]}s')
axs[0].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
axs[0].set_title('Adaptation Constant')
axs[0].legend(loc='upper right', fontsize=18)

# Adaptation Scale
for i, adaptscale_result in enumerate(adaptscale_results):
    axs[1].plot(tarr, np.mean(adaptscale_result[:,responders], axis=1), label=rf'$q_{{A}}$ = {adaptscale_values[i]}')
axs[1].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
axs[1].set_title('Adapation Scale')
axs[1].legend(loc='upper right', fontsize=18)

# Super title for the entire figure
fig.suptitle('Adaptation Only Parameter Scan')
fig.supxlabel('Time (s)')
fig.supylabel('Ca Activity (a.u)')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

#%% Inhibition Only Simulation

# Inhebion only model iterative variables
tauinh_only_values = [1, 1.5, 2]            # Lateral inhibition time constant in s             default = 1.5       
inhscale_only_factors = [1, 1.3, 1.5]       # Ramp up the degree of inhibition showcase effect       default = 1

# Initialize lists to store results
tauinh_only_results = []
inhscale_only_results = []

# Iterate over tauinh values and store results
for tauinh in tauinh_only_values:
    activity_model.tauinh = tauinh
    inhebion_only = activity_model.simulate_inhebition_only()
    tauinh_only_results.append(inhebion_only)
activity_model.tauinh = 1.5

# Iterate over inhscale factors and store results 
for inhscale_only_factor in inhscale_only_factors:
    activity_model.inhscale = activity_model.inhscale * inhscale_only_factor
    inhebion_only = activity_model.simulate_inhebition_only()
    inhscale_only_results.append(inhebion_only)
activity_model.inhcale_factor = 1
#%% Inhebition Only Mean Bouton Response Plots

# Tau Inhebiotion
plt.figure('Inhibition Constant')
for i, tauinh_only_result in enumerate(tauinh_only_results):
    plt.plot(tarr, np.mean(tauinh_only_result[:,responders], axis=1), label=rf'$\tau_{{I}}$ = {tauinh_only_values[i]}s')
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
plt.title('Inhibition Constant Parameter Scan')
plt.xlabel('Time (s)')
plt.ylabel('Ca Activity (a.u)')
plt.legend()
plt.show()

# Inhscale
plt.figure('Inhibition Scale')
for i, inhscale_only_result in enumerate(inhscale_only_results):
    plt.plot(tarr, np.mean(inhscale_only_result[:,responders], axis=1), label=rf'$q_{{I}}$ = {inhscale_only_factors[i]}')
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
plt.title('Inhibition Scale Parameter Scan')
plt.xlabel('Time (s)')
plt.ylabel('Ca Activity (a.u)')
plt.legend()
plt.show()

#%% Inhebition Only Mean Bouton Response Subplot

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Tau Inhibition
for i, tauinh_only_result in enumerate(tauinh_only_results):
    axs[0].plot(tarr, np.mean(tauinh_only_result[:,responders], axis=1), label=rf'$\tau_{{I}}$ = {tauinh_only_values[i]}s')
axs[0].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
axs[0].set_title('Inhibition Constant')
axs[0].legend(loc='upper right', fontsize=18)

# Inhibition Scale
for i, inhscale_only_result in enumerate(inhscale_only_results):
    axs[1].plot(tarr, np.mean(inhscale_only_result[:,responders], axis=1), label=rf'$Q_{{I}}$ = {inhscale_only_factors[i]}')
axs[1].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
axs[1].set_title('Inhibition Scale')
axs[1].legend(loc='upper right', fontsize=18)

# Super title for the entire figure
fig.suptitle('Inhibition Only Parameter Scan')
fig.supxlabel('Time (s)')
fig.supylabel('Ca Activity (a.u)')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

#%% Inhibition & Adaptation Simulation

# Inhebion iterative variables
tauinh_values = [1, 1.5, 2]            # Lateral inhibition time constant in s                  default = 1.5       
inhscale_factors = [1, 1.3, 1.5]       # Ramp up the degree of inhibition showcase effect       default = 1

# Adapation iterative variables
tauadapt_values = [0.5, 2, 4]          # Adaptation time constant in s                           default = 2
adaptscale_values = [0.5, 1, 1.5]      # Adaptation strength                                     default = 1


# Initialize lists to store results
tauinh_results = []
inhscale_results = []
tauadapt_results = []
adaptscale_results = []

# Iterate over tauinh values and store results
for tauinh in tauinh_values:
    activity_model.tauinh = tauinh
    inhibition_adaptation = activity_model.simulate_inhibition_adaptation()
    tauinh_results.append(inhibition_adaptation)
activity_model.tauinh = 1.5

# Iterate over inhscale factors and store results    
for inhscale_factor in inhscale_factors:
    activity_model.inhscale = activity_model.inhscale * inhscale_factor
    inhibition_adaptation = activity_model.simulate_inhibition_adaptation()
    inhscale_results.append(inhibition_adaptation)
activity_model.inhcale_factor = 1

# Iterate over tauadapt values and store results
for tauadapt in tauadapt_values:
    activity_model.tauadapt = tauadapt
    adaptation = activity_model.simulate_inhibition_adaptation()
    tauadapt_results.append(adaptation)
activity_model.tauadapt = 2

# Iterate over adaptscale values and store results
for adaptscale in adaptscale_values:
    activity_model.adaptscale = adaptscale
    adaptation = activity_model.simulate_inhibition_adaptation()
    adaptscale_results.append(adaptation)
activity_model.adaptscale = 1
#%% Inhibition & Adaptation Mean Bouton Response Plots

# Tau Inhebiotion
plt.figure('Inhibition Constant')
for i, tauinh_result in enumerate(tauinh_results):
    plt.plot(tarr, np.mean(tauinh_result[:,responders], axis=1), label=rf'$\tau_{{I}}$ = {tauinh_values[i]}s')
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
plt.title('Inhibition Constant Parameter Scan')
plt.xlabel('Time (s)')
plt.ylabel('Ca Activity (a.u)')
plt.legend()
plt.show()

# Inhibition Scale
plt.figure('Inhibition Scale')
for i, inhscale_result in enumerate(inhscale_results):
    plt.plot(tarr, np.mean(inhscale_result[:,responders], axis=1), label=rf'$q_{{I}}$ = {inhscale_factors[i]}')
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
plt.title('Inhibition Scale Parameter Scan')
plt.xlabel('Time (s)')
plt.ylabel('Ca Activity (a.u)')
plt.legend()
plt.show()

# Tau Adaptation
plt.figure('Tau Adapation')
for i, tauadapt_result in enumerate(tauadapt_results):
    plt.plot(tarr, np.mean(tauadapt_result[:,responders], axis=1), label = rf'$\tau_{{A}}$ = {tauadapt_values[i]}s')
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
plt.title('Adaptation Constant Parameter Scan')
plt.xlabel('Time (s)')
plt.ylabel('Ca Activity (a.u)')
plt.legend()
plt.show()

# Adapation Scale
plt.figure('Adapation Scale')
for i, adaptscale_result in enumerate(adaptscale_results):
    plt.plot(tarr, np.mean(adaptscale_result[:, responders], axis=1), label=rf'$q_{{A}}$ = {adaptscale_values[i]}')
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
plt.title('Adaptation Scale Parameter Scan')
plt.xlabel('Time (s)')
plt.ylabel('Ca Activity (a.u)')
plt.legend()
plt.show()
#%% Inhibition & Adaptation Mean Bouton Response Subplots

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))  # 2 rows, 2 columns

# Tau Inhibition
for i, tauinh_result in enumerate(tauinh_results):
    axs[0, 0].plot(tarr, np.mean(tauinh_result[:, responders], axis=1), label=rf'$\tau_{{I}}$ = {tauinh_values[i]}s')
axs[0, 0].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
axs[0, 0].set_title('Inhibition Constant')
axs[0, 0].legend(loc='upper right', fontsize=18)

# Inhibition Scale
for i, inhscale_result in enumerate(inhscale_results):
    axs[0, 1].plot(tarr, np.mean(inhscale_result[:, responders], axis=1), label=rf'$q_{{I}}$ = {inhscale_factors[i]}')
axs[0, 1].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
axs[0, 1].set_title('Inhibition Scale')
axs[0, 1].legend(loc='upper right', fontsize=18)

# Tau Adaptation
for i, tauadapt_result in enumerate(tauadapt_results):
    axs[1, 0].plot(tarr, np.mean(tauadapt_result[:, responders], axis=1), label=rf'$\tau_{{A}}$ = {tauadapt_values[i]}s')
axs[1, 0].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
axs[1, 0].set_title('Adaptation Constant')
axs[1, 0].legend(loc='upper right', fontsize=18)

# Adapation Scale
for i, adaptscale_result in enumerate(adaptscale_results):
    axs[1, 1].plot(tarr, np.mean(adaptscale_result[:, responders], axis=1), label=rf'$q_{{A}}$ = {adaptscale_values[i]}')
axs[1, 1].plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
axs[1, 1].set_title('Adaptation Scale')
axs[1, 1].legend(loc='upper right', fontsize=18)

# Super title for the entire figure
fig.suptitle('Inhibition & Adaptation Parameter Scan')
fig.supxlabel('Time (s)')
fig.supylabel('Ca Activity (a.u)')

# Adjust layout and display the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent overlap with the supertitle
plt.show()
#%% Modulated Inhebition Simulation

# Modulated Inhibition iterative variables
infp_values = [0.3, 0.5, 0.7]     # midpoint of the sigmoidal function      # default = 0.5
slf_values = [0.01, 0.03, 0.05]   # slope factor of the sigmoidal function, # Default = 0.03
                                  # this factor is inverse to the slope at inflection point

# Initialize lists to store results
infp_results = []
slf_results = []

# Iterate over infp values and store results 
for infp_value in infp_values:
    activity_model.infp = infp_value
    full_model = activity_model.simulate_full_model()
    infp_results.append(full_model)
activity_model.infp = 0.5

for slf in slf_values:
    activity_model.slf = slf
    full_model = activity_model.simulate_full_model()
    slf_results.append(full_model)
activity_model.slf = 0.03

#Full Model 
modulated_inhebion_model = activity_model.simulate_full_model()
#%% Modulated Inhibition Individual Bouton Response Plots

# Slope Factor
fig, axs = plt.subplots(1, len(slf_values), figsize=(20, 5), sharey=True)
for i, slf_result in enumerate(slf_results):
    ax = axs[i]
    ax.plot(tarr, slf_result[:, responders], alpha=0.5)
    ax.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
    ax.set_title(rf'$k$ = {slf_values[i]}')
fig.suptitle('Slope Factor Modulation Function')
fig.supxlabel('Time (s)')
fig.supylabel('Ca Activity (a.u)')
plt.tight_layout()

# Midpiont
fig, axs = plt.subplots(1, len(infp_values), figsize=(20, 5), sharey=True)
for i, infp_result in enumerate(infp_results):
    ax = axs[i]
    ax.plot(tarr, infp_result[:, responders], alpha=0.5)
    ax.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
    ax.set_title(rf'$\mu$ = {infp_values[i]}')
fig.suptitle('Midpoint Modulation Function')
fig.supxlabel('Time (s)')
fig.supylabel('Ca Activity (a.u)')
plt.tight_layout()

#Full Model 
plt.figure('Modulated Inhibition')
plt.plot(tarr, modulated_inhebion_model[:, responders], alpha=0.5)
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus')
plt.title('Modulated Inhibition Default')
plt.xlabel('Time (s)')
plt.ylabel('Ca Activity (a.u)')
#%% Full Model Subplot

fig = plt.figure(figsize=(14, 14))  # Increased height to accommodate the third row

# Define the outer grid with 3 rows and 2 columns (added 3rd row)
outer_grid = gridspec.GridSpec(3, 2, wspace=0.4, hspace=0.4)

# Slope Factor subplots (first row)
inner_grid2 = gridspec.GridSpecFromSubplotSpec(1, len(slf_values), subplot_spec=outer_grid[0, :], wspace=0.3, hspace=0.3)
for i, slf_result in enumerate(slf_results):
    ax = plt.Subplot(fig, inner_grid2[0, i])
    fig.add_subplot(ax)
    ax.plot(tarr, slf_result[:, responders], alpha=0.5)
    ax.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
    ax.set_title(rf'$k$ = {slf_values[i]}', fontsize=14, pad=10)  # Title for slope factor subplots
    if i > 0:
        ax.set_yticklabels([])

# Midpoint subplots (second row)
inner_grid1 = gridspec.GridSpecFromSubplotSpec(1, len(infp_values), subplot_spec=outer_grid[1, :], wspace=0.3, hspace=0.3)
for i, infp_result in enumerate(infp_results):
    ax = plt.Subplot(fig, inner_grid1[0, i])
    fig.add_subplot(ax)
    ax.plot(tarr, infp_result[:, responders], alpha=0.5)
    ax.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
    ax.set_title(rf'$\mu$ = {infp_values[i]}', fontsize=14, pad=10)  # Title for midpoint subplots
    if i > 0:
        ax.set_yticklabels([])
    ax.set_xticklabels([])

# New row for the "Modulated Inhibition Model with Default Parameters" (third row)
# Span across both columns in the third row
ax_default = plt.Subplot(fig, outer_grid[2, :])
fig.add_subplot(ax_default)
ax_default.plot(tarr, modulated_inhebion_model[:, responders], alpha=0.5)
ax_default.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus')
ax_default.set_title('Modulated Inhibition Model with Default Parameters', fontsize=16)
ax_default.set_xlabel('Time (s)', fontsize=14)
ax_default.set_ylabel('Ca Activity (a.u)', fontsize=14)

# Add centered and slightly above row titles
fig.text(0.5, 0.92, 'Slope Factor', ha='center', fontsize=15)  # Title above the first row
fig.text(0.5, 0.64, 'Midpoint', ha='center', fontsize=15)  # Title above the second row

# Super labels
fig.suptitle('Modulated Inhibition Parameter Scan', fontsize=20)
fig.supxlabel('Time (s)', fontsize=18)
fig.supylabel('Ca Activity (a.u)', fontsize=18)

# Adjust layout and display the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


"""
fig = plt.figure(figsize=(14, 10))

# Define the outer grid with 2 rows and 2 columns
outer_grid = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.4)

# Slope Factor subplots (now first)
inner_grid2 = gridspec.GridSpecFromSubplotSpec(1, len(slf_values), subplot_spec=outer_grid[0, :], wspace=0.3, hspace=0.3)
for i, slf_result in enumerate(slf_results):
    ax = plt.Subplot(fig, inner_grid2[0, i])
    fig.add_subplot(ax)
    ax.plot(tarr, slf_result[:, responders], alpha=0.5)
    ax.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
    ax.set_title(rf'$k$ = {slf_values[i]}', fontsize=14, pad=10)  # Title for slope factor subplots
    if i > 0:
        ax.set_yticklabels([])

# Midpoint subplots (now second)
inner_grid1 = gridspec.GridSpecFromSubplotSpec(1, len(infp_values), subplot_spec=outer_grid[1, :], wspace=0.3, hspace=0.3)
for i, infp_result in enumerate(infp_results):
    ax = plt.Subplot(fig, inner_grid1[0, i])
    fig.add_subplot(ax)
    ax.plot(tarr, infp_result[:, responders], alpha=0.5)
    ax.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)
    ax.set_title(rf'$\mu$ = {infp_values[i]}', fontsize=14, pad=10)  # Title for midpoint subplots
    if i > 0:
        ax.set_yticklabels([])
    ax.set_xticklabels([])

# Add centered and slightly above row titles
fig.text(0.5, 0.92, 'Slope Factor Modulation Function', ha='center', fontsize=16)  # Title above the first row
fig.text(0.5, 0.47, 'Midpoint Modulation Function', ha='center', fontsize=16)  # Title above the second row

# Super labels
fig.suptitle('Modulated Inhibition Parameter Scan', fontsize=20)
fig.supxlabel('Time (s)', fontsize=18)
fig.supylabel('Ca Activity (a.u)', fontsize=18)

# Adjust layout and display the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.figure('Modulated Inhibition')
plt.plot(tarr, modulated_inhebion_model[:, responders], alpha=0.5)
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus')
plt.title('Modulated Inhibition Default')
plt.xlabel('Time (s)')
plt.ylabel('Ca Activity (a.u)')
"""