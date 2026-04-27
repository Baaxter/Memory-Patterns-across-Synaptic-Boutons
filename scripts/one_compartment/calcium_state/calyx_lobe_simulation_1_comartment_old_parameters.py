#%% Base

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

from IBM_functions_package import singel_compartment
#%% Compartment Object

# Create reate model object
activity_model = singel_compartment.Compartment()

#Time and Stimulus array 
tarr  = activity_model.tarr
starr = activity_model.starr

#Active KCs
inpscale = activity_model.inpscale
responders = np.where(inpscale > 0)[0]
rs = activity_model.rs
us = activity_model.us

# Calcium  arrays
null_model = activity_model.simulate_null()
adaptation_model = activity_model.simulate_adaptation()
inhibition_model = activity_model.simulate_inhibition_adaptation()
full_model = activity_model.simulate_full_model()
#%% Mean Subplot

# Create a figure with a 16:9 aspect ratio for Beamer slides
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(3, 2)

# Define colors using the default matplotlib color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Null Model Plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(tarr, np.mean(null_model[:, responders], axis=1), color=colors[0], lw=2)
ax1.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=3)  # Stimulus marker
ax1.set_title('Null', fontsize=22)

# Adaptation Only Model Plot (shares y-axis with ax1)
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax2.plot(tarr, np.mean(adaptation_model[:, responders], axis=1), color=colors[1], lw=2)
ax2.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=3)  # Stimulus marker
ax2.set_title('Adaptation', fontsize=22)

# Inhibition Only Model Plot (shares y-axis with ax1)
ax3 = fig.add_subplot(gs[1, 0], sharey=ax1)
ax3.plot(tarr, np.mean(inhibition_model[:, responders], axis=1), color=colors[2], lw=2)
ax3.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=3)  # Stimulus marker
ax3.set_title('Inhibition', fontsize=22)

# Full Model Plot (shares y-axis with ax1)
ax4 = fig.add_subplot(gs[1, 1], sharey=ax1)
ax4.plot(tarr, np.mean(full_model[:, responders], axis=1), color=colors[3], lw=2)
ax4.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=3)  # Stimulus marker
ax4.set_title('Modulated Inhibition', fontsize=22)

# Combined Plot (does not share y-axis since it spans both columns)
ax5 = fig.add_subplot(gs[2, :])
ax5.plot(tarr, np.mean(null_model[:, responders], axis=1), label='Null', color=colors[0], lw=2)
ax5.plot(tarr, np.mean(adaptation_model[:, responders], axis=1), label='Adaptation', color=colors[1], lw=2)
ax5.plot(tarr, np.mean(inhibition_model[:, responders], axis=1), label='Inhibition', color=colors[2], lw=2)
ax5.plot(tarr, np.mean(full_model[:, responders], axis=1), label='Modulated Inhibition', color=colors[3], lw=2)
ax5.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=3)  # Stimulus marker
ax5.set_title('All', fontsize=22)
ax5.legend(fontsize=16, loc='upper center', frameon=True, bbox_to_anchor=(0.8, 1.25), edgecolor='black')

# Add global axis labels and title
fig.supxlabel('Time (s)', fontsize=24)
fig.supylabel('Ca Activity (a.u.)', fontsize=24)
# fig.suptitle('Mean Bouton Response Dynamics', fontsize=24)  # Optional: Uncomment for a global title

# Optimize layout and adjust margins
plt.tight_layout()
fig.subplots_adjust(top=0.9)  # Adjust top margin for titles
plt.savefig(os.path.expanduser('~/mean_response_dynamics.png'), dpi=300, bbox_inches='tight')  # Save figure
plt.show()
#%% Individual Subplot 

# Create a figure with a 16:9 aspect ratio for Beamer slides
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 2)  # Set up a 2x2 grid layout

# Null Model Plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(tarr, null_model[:, responders], alpha=0.5)  # Plot individual responder traces
ax1.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k', lw=5)  # Stimulus marker
ax1.set_title('Null', fontsize=24)

# Adaptation Only Model Plot
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(tarr, adaptation_model[:, responders], alpha=0.5)  # Plot individual responder traces
ax2.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)  # Stimulus marker
ax2.set_title('Adaptation', fontsize=24)

# Inhibition Only Model Plot
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(tarr, inhibition_model[:, responders], alpha=0.5)  # Plot individual responder traces
ax3.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)  # Stimulus marker
ax3.set_title('Inhibition', fontsize=24)

# Full Model (Modulated Inhibition) Plot
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(tarr, full_model[:, responders], alpha=0.5)  # Plot individual responder traces
ax4.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Simulus')  # Stimulus marker
ax4.set_title('Modulated Inhibition', fontsize=24)
ax4.legend(fontsize=14, loc='upper center', frameon=True, bbox_to_anchor=(0.8, 0.9), edgecolor='black')

# Add global axis labels
fig.supxlabel('Time (s)', fontsize=26)
fig.supylabel('Ca Activity (a.u.)', fontsize=26)

# Optimize layout and adjust margins
plt.tight_layout()
fig.subplots_adjust(top=0.9)  # Ensure sufficient space at the top
plt.savefig(os.path.expanduser('~/individual_response_dynamics.png'), dpi=300, bbox_inches='tight')  # Save figure
plt.show()
#%% Mean + Individual Subplot

# Create a figure with a 16:9 aspect ratio for Beamer slides
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)  # Set up a 2x2 grid layout with spacing

# Set transparency for individual responder lines
al = 0.2

# Null Model Plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(tarr, null_model[:, responders], alpha=al)  # Plot individual responder traces
ax1.plot(tarr, np.mean(null_model[:, responders], axis=1), color='#D81B60', lw=3.5)  # Plot mean activity
ax1.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)  # Stimulus marker
ax1.set_title('Null', fontsize=24)

# Adaptation Only Model Plot (shares y-axis with ax1)
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax2.plot(tarr, adaptation_model[:, responders], alpha=al)  # Plot individual responder traces
ax2.plot(tarr, np.mean(adaptation_model[:, responders], axis=1), color='#D81B60', lw=3.5)  # Plot mean activity
ax2.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)  # Stimulus marker
ax2.set_title('Adaptation', fontsize=24)

# Inhibition Only Model Plot (shares y-axis with ax1)
ax3 = fig.add_subplot(gs[1, 0], sharey=ax1)
ax3.plot(tarr, inhibition_model[:, responders], alpha=al)  # Plot individual responder traces
ax3.plot(tarr, np.mean(inhibition_model[:, responders], axis=1), color='#D81B60', lw=3.5)  # Plot mean activity
ax3.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)  # Stimulus marker
ax3.set_title('Inhibition', fontsize=24)

# Modulated Inhibition Model Plot (shares y-axis with ax1)
ax4 = fig.add_subplot(gs[1, 1], sharey=ax1)
ax4.plot(tarr, full_model[:, responders], alpha=al)  # Plot individual responder traces
ax4.plot(tarr, np.mean(full_model[:, responders], axis=1), color='#D81B60', lw=3.5, label='Mean Activity')  # Plot mean activity
ax4.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus')  # Stimulus marker
ax4.set_title('Modulated Inhibition', fontsize=24)
ax4.legend(fontsize=16, loc='upper center', frameon=True, bbox_to_anchor=(0.8, 0.9), edgecolor='black')

# Add global axis labels
fig.supxlabel('Time (s)', fontsize=26)
fig.supylabel('Ca Activity (a.u.)', fontsize=26)

# Optimize layout and adjust margins
plt.tight_layout()
fig.subplots_adjust(top=0.9)  # Ensure consistent top margin
plt.savefig(os.path.expanduser('~/mean_and_individual_response_dynamics.png'), dpi=300, bbox_inches='tight')  # Save figure
plt.show()
#%% Mean + Individual Subplot with Comparisent 

# Create a figure with a 16:9 aspect ratio for Beamer slides
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(3, 2)  # Set up a 3x2 grid layout

# Set transparency for individual responder lines
al = 0.2

# Null Model Plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(tarr, null_model[:, responders], alpha=al)  # Plot individual responder traces
ax1.plot(tarr, np.mean(null_model[:, responders], axis=1), color='#D81B60', lw=3.5)  # Plot mean activity
ax1.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)  # Stimulus marker
ax1.set_title('Null', fontsize=24)

# Adaptation Only Model Plot (shares y-axis with ax1)
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax2.plot(tarr, adaptation_model[:, responders], alpha=al)  # Plot individual responder traces
ax2.plot(tarr, np.mean(adaptation_model[:, responders], axis=1), color='#D81B60', lw=3.5)  # Plot mean activity
ax2.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)  # Stimulus marker
ax2.set_title('Adaptation', fontsize=24)

# Inhibition Only Model Plot (shares y-axis with ax1)
ax3 = fig.add_subplot(gs[1, 0], sharey=ax1)
ax3.plot(tarr, inhibition_model[:, responders], alpha=al)  # Plot individual responder traces
ax3.plot(tarr, np.mean(inhibition_model[:, responders], axis=1), color='#D81B60', lw=3.5)  # Plot mean activity
ax3.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)  # Stimulus marker
ax3.set_title('Inhibition', fontsize=24)

# Modulated Inhibition Model Plot (shares y-axis with ax1)
ax4 = fig.add_subplot(gs[1, 1], sharey=ax1)
ax4.plot(tarr, full_model[:, responders], alpha=al)  # Plot individual responder traces
ax4.plot(tarr, np.mean(full_model[:, responders], axis=1), color='#D81B60', lw=3.5, label='Mean Activity')  # Plot mean activity
ax4.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus')  # Stimulus marker
ax4.set_title('Modulated Inhibition', fontsize=24)
ax4.legend(fontsize=16, loc='upper center', frameon=True, bbox_to_anchor=(0.8, 0.9), edgecolor='black')

# Combined Mean Activity Plot (doesn't need shared y-axis since it spans the bottom row)
ax5 = fig.add_subplot(gs[2, :])
ax5.plot(tarr, np.mean(null_model[:, responders], axis=1), label='Null', lw=2.5)  # Null model mean activity
ax5.plot(tarr, np.mean(adaptation_model[:, responders], axis=1), label='Adaptation', lw=2.5)  # Adaptation mean activity
ax5.plot(tarr, np.mean(inhibition_model[:, responders], axis=1), label='Inhibition', lw=2.5)  # Inhibition mean activity
ax5.plot(tarr, np.mean(full_model[:, responders], axis=1), label='Modulated Inhibition', lw=2.5)  # Full model mean activity
ax5.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)  # Stimulus marker
ax5.set_title('Mean Activity Comparison', fontsize=24)
ax5.legend(fontsize=14, frameon=True, loc='upper right', edgecolor='black')  # Add legend

# Add global axis labels
fig.supxlabel('Time (s)', fontsize=26)
fig.supylabel('Ca Activity (a.u.)', fontsize=26)

# Optimize layout and adjust margins
plt.tight_layout()
fig.subplots_adjust(top=0.9)  # Ensure consistent top margin
plt.savefig(os.path.expanduser('~/mean_and_individual_response_dynamics_comparison.png'), dpi=300, bbox_inches='tight')  # Save figure
plt.show()

#%% Full Model Individual Responses with Mean (Standalone)

# Create a figure with a 16:9 aspect ratio
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(1, 1, 1)  # Single subplot

# Plot individual responses and mean activity
ax.plot(tarr, full_model[:, responders], alpha=0.3)  # Individual responder traces
ax.plot(tarr, np.mean(full_model[:, responders], axis=1), color='#D81B60', lw=3.5, label='Mean Activity')  # Mean activity
ax.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus')  # Stimulus marker

# Add labels, title, and legend
ax.set_xlabel('Time (s)', fontsize=28)
ax.set_ylabel('Ca Activity (a.u.)', fontsize=28)
ax.legend(fontsize=24, frameon=True, loc='upper right', edgecolor='black')  # Add legend with appropriate font size and location

# Optimize layout and save the figure
plt.tight_layout()
plt.savefig(os.path.expanduser('~/full_model_individual_and_mean_responses.png'), dpi=300, bbox_inches='tight')  # Save the figure
plt.show()
#%% Reliabile Unreliabile Subplot  

# Create a figure with a 16:9 aspect ratio for Beamer slides
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 2)  # Set up a 2x2 grid layout

# Null Model Plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(tarr, null_model[:, rs], color='#2ca02c', alpha=0.3)  # Reliable responders
ax1.plot(tarr, null_model[:, us], color='#9467bd', alpha=0.3)  # Unreliable responders
ax1.plot(tarr, np.mean(null_model[:, responders], axis=1), color='#D81B60', lw=3.5)  # Mean activity
ax1.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)  # Stimulus marker
ax1.set_title('Null', fontsize=24)

# Adaptation Model Plot (shares y-axis with ax1)
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax2.plot(tarr, adaptation_model[:, rs], color='#2ca02c', alpha=0.3)  # Reliable responders
ax2.plot(tarr, adaptation_model[:, us], color='#9467bd', alpha=0.3)  # Unreliable responders
ax2.plot(tarr, np.mean(adaptation_model[:, responders], axis=1), color='#D81B60', lw=3.5)  # Mean activity
ax2.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)  # Stimulus marker
ax2.set_title('Adaptation', fontsize=24)

# Inhibition Model Plot (shares y-axis with ax1)
ax3 = fig.add_subplot(gs[1, 0], sharey=ax1)
ax3.plot(tarr, inhibition_model[:, rs], color='#2ca02c', alpha=0.3)  # Reliable responders
ax3.plot(tarr, inhibition_model[:, us], color='#9467bd', alpha=0.3)  # Unreliable responders
ax3.plot(tarr, np.mean(inhibition_model[:, responders], axis=1), color='#D81B60', lw=3.5)  # Mean activity
ax3.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5)  # Stimulus marker
ax3.set_title('Inhibition', fontsize=24)

# Full Model Plot (shares y-axis with ax1)
ax4 = fig.add_subplot(gs[1, 1], sharey=ax1)
ax4.plot(tarr, full_model[:, rs], color='#2ca02c', alpha=0.3)  # Reliable responders
ax4.plot(tarr, full_model[:, us], color='#9467bd', alpha=0.3)  # Unreliable responders
ax4.plot(tarr, np.mean(full_model[:, responders], axis=1), color='#D81B60', lw=3.5)  # Mean activity
# Add dummy lines for legend
ax4.plot([], [], color='#2ca02c', lw=2.5, label='Reliable Responders')
ax4.plot([], [], color='#9467bd', lw=2.5, label='Unreliable Responders')
ax4.plot([], [], color='#D81B60', lw=2.5, label='Mean Activity')
ax4.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus')  # Stimulus marker
ax4.legend(fontsize=14, loc='upper right', frameon=True, bbox_to_anchor=(1.2, 0.95), edgecolor='black')  # Adjusted legend position
ax4.set_title('Modulated Inhibition', fontsize=20)

# Add global axis labels
fig.supxlabel('Time (s)', fontsize=26)
fig.supylabel('Ca Activity (a.u.)', fontsize=26)

# Optimize layout and save the figure
plt.tight_layout()
fig.subplots_adjust(top=0.9)  # Ensure consistent top margin
plt.savefig(os.path.expanduser('~/individual_response_dynamics_reliable_unreliable.png'), dpi=300, bbox_inches='tight')  # Save figure
plt.show()
#%% Full Model Showcase Reliabil and Unreliabile

# Create a figure with a 16:9 aspect ratio for Beamer slides
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 2, wspace=0.4)  # 1x2 grid layout with spacing between subplots

# Subplot 1: Individual Responses with Mean
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(tarr, full_model[:, responders], alpha=0.3)  # Individual responder traces
ax1.plot(tarr, np.mean(full_model[:, responders], axis=1), color='#D81B60', lw=3.5)  # Bold mean activity line

# Subplot 2: Reliable and Unreliable Responses with Mean (shares y-axis with ax1)
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax2.plot(tarr, full_model[:, rs], color='#2ca02c', alpha=0.3)  # Reliable responders
ax2.plot(tarr, full_model[:, us], color='#9467bd', alpha=0.3)  # Unreliable responders
ax2.plot(tarr, np.mean(full_model[:, responders], axis=1), color='#D81B60', lw=3.5)  # Bold mean activity line
# Add dummy lines for legend
ax2.plot([], [], color='#2ca02c', lw=2.5, label='Reliable Responders')
ax2.plot([], [], color='#9467bd', lw=2.5, label='Unreliable Responders')
ax2.plot([], [], color='#D81B60', lw=2.5, label='Mean Activity')
ax2.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus')  # Stimulus marker
ax2.legend(fontsize=16, frameon=True, loc='upper right', bbox_to_anchor=(1.25, 0.95), edgecolor='black')  # Adjusted legend position

# Add global axis labels for the figure
fig.supxlabel('Time (s)', fontsize=26)
fig.supylabel('Ca Activity (a.u.)', fontsize=26)

# Optimize layout for clean presentation
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust top margin for global labels

# Save and show the figure
plt.savefig(os.path.expanduser('~/full_model_individual_and_rs_us_showcase.png'), dpi=300, bbox_inches='tight')  # Save as high-resolution image
plt.show()
#%% Mean Plots 

# Null Model
plt.figure('Null Model', figsize=(16, 9))
plt.plot(tarr, np.mean(null_model[:, responders], axis=1), label='Mean Activity', lw=2.5)
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus')
plt.xlabel('Time (s)', fontsize=22)
plt.ylabel('Ca Activity (a.u.)', fontsize=26)
plt.title('Null Model', fontsize=26)
plt.legend(fontsize=22, frameon=True)
plt.tight_layout()

# Adaptation Model
plt.figure('Adaptation Model', figsize=(16, 9))
plt.plot(tarr, np.mean(adaptation_model[:, responders], axis=1), label='Mean Activity', lw=2.5)
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus')
plt.xlabel('Time (s)', fontsize=22)
plt.ylabel('Ca Activity (a.u.)', fontsize=26)
plt.title('Adaptation Model', fontsize=26)
plt.legend(fontsize=22, frameon=True)
plt.tight_layout()

# Inhibition Model
plt.figure('Inhibition Model', figsize=(16, 9))
plt.plot(tarr, np.mean(inhibition_model[:, responders], axis=1), label='Mean Activity', lw=2.5)
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus')
plt.xlabel('Time (s)', fontsize=22)
plt.ylabel('Ca Activity (a.u.)', fontsize=26)
plt.title('Inhibition Model', fontsize=26)
plt.legend(fontsize=22, frameon=True)
plt.tight_layout()

# Modulated Inhibition Model
plt.figure('Modulated Inhibition Model', figsize=(16, 9))
plt.plot(tarr, np.mean(full_model[:, responders], axis=1), label='Mean Activity', lw=2.5)
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus')
plt.xlabel('Time (s)', fontsize=22)
plt.ylabel('Ca Activity (a.u.)', fontsize=26)
plt.title('Modulated Inhibition Model', fontsize=26)
plt.legend(fontsize=22, frameon=True)
plt.tight_layout()

# All Together
plt.figure('All Together', figsize=(16, 9))
plt.plot(tarr, np.mean(null_model[:, responders], axis=1), label='Null', lw=2.5)
plt.plot(tarr, np.mean(adaptation_model[:, responders], axis=1), label='Adaptation', lw=2.5)
plt.plot(tarr, np.mean(inhibition_model[:, responders], axis=1), label='Inhibition', lw=2.5)
plt.plot(tarr, np.mean(full_model[:, responders], axis=1), label='Modulated Inhibition', lw=2.5)
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus')
plt.xlabel('Time (s)', fontsize=22)
plt.ylabel('Ca Activity (a.u.)', fontsize=26)
#plt.title('Model Variations', fontsize=26)
plt.legend(fontsize=22, frameon=True)
plt.tight_layout()
#plt.savefig(os.path.expanduser('~/mean_activity_compaisent.png'), dpi=300, bbox_inches='tight')  # Save as high-resolution image

plt.show()
#%% Individual Plots

# Null Model
plt.figure('Null Model', figsize=(16, 9))
plt.plot(tarr, null_model[:, responders], alpha=0.5)  # Individual responders with transparency
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus', alpha=0.5)
plt.xlabel('Time (s)', fontsize=26)
plt.ylabel('Ca Activity (a.u.)', fontsize=26)
plt.title('Null Model', fontsize=26)
plt.tight_layout()

# Adaptation Model
plt.figure('Adaptation Model', figsize=(16, 9))
plt.plot(tarr, adaptation_model[:, responders], alpha=0.5)  # Individual responders with transparency
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus', alpha=0.5)
plt.xlabel('Time (s)', fontsize=26)
plt.ylabel('Ca Activity (a.u.)', fontsize=26)
plt.title('Adaptation Model', fontsize=26)
plt.tight_layout()

# Inhibition Model
plt.figure('Inhibition & Adaptation Model', figsize=(16, 9))
plt.plot(tarr, inhibition_model[:, responders], alpha=0.5)  # Individual responders with transparency
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus', alpha=0.5)
plt.xlabel('Time (s)', fontsize=26)
plt.ylabel('Ca Activity (a.u.)', fontsize=26)
plt.title('Inhibition & Adaptation Model', fontsize=26)
plt.tight_layout()

# Full Model
plt.figure('Modulated Inhibition Model', figsize=(16, 9))
plt.plot(tarr, full_model[:, responders], alpha=0.5)  # Individual responders with transparency
plt.plot(tarr[starr > 0], np.ones(len(tarr[starr > 0])) * -0.01, 'k-', lw=5, label='Stimulus', alpha=0.5)
plt.xlabel('Time (s)', fontsize=26)
plt.ylabel('Ca Activity (a.u.)', fontsize=26)
plt.title('Modulated Inhibition Model', fontsize=26)
plt.tight_layout()

plt.show()