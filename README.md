# Simulating Calcium Dynamics and Independent Bouton Modulation in the Mushroom Body γ-Lobe

This repository contains code for simulating calcium activity in individual γ-lobe boutons using both single-compartment and multi-compartment model variants. The calcium-activity model is fitted to experimental data from Manoim et al. (2022) (https://doi.org/10.1016/j.cub.2022.09.007
) and incorporates inhibitory lateral KC–KC interactions (Moshe Panas, internal communications).

The implementation is based on the original KC_KC_interactions_project repository by Ibrahim Tunc and has been significantly extended for research and experimentation purposes.

In addition, the repository explores mechanisms underlying independent bouton modulation as described in Bilz et al. (2020) (https://doi.org/10.1016/j.neuron.2020.03.010
) by simulating calcium dynamics in individual γ-lobe boutons during aversive olfactory conditioning.

These simulations can be directly compared to calcium trace recordings reported in Bilz et al. (2020), as the corresponding experimental dataset is included in this repository, along with code for analysis and visualization of the recordings.



---

# Project Layout

```text
📦 <root>/
├── 📦 IBM_functions_package/                 # Core package for IBM & dataset processing
│   ├── 📦 xlsxDataSource/                    # Experimental dataset (Bilz et al., 2020)
│   ├── 📝 activity_model_functions.py        # Calcium dynamics simulation functions
│   ├── 📝 plotting_related_functions.py      # Helper functions for plotting results
│   ├── 📦 model_fit_normfac_3.885.npz        # Fitted parameters for model simulations
│   ├── 📦 MBON_sigmoid_tuning_noisy.npz      # Fitted parameters for model simulations
│   └── 📦 dan_conditioning_results.npz       # Arrays from aversive conditioning (overwritten each run)
│
├── 📦 scripts/                               # Scripts for simulations and data analysis
│   ├── 📦 bilz_data_visualization/           # Load, analyze, and visualize experimental data
│   │   └── 📦 metadata_fitting_and_questions/ # Dataset descriptions and notes
│   │
│   ├── 📦 one_compartment/                   # Single-compartment γ-lobe simulations
│   │   ├── 📦 calcium_state/                 # Calyx vs. lobe comparisons; parameter scans
│   │   └── 📦 dan_dependent_ibm/             # IBM-related scripts
│   │       ├── 📝 dan_conditioning.py        # IBM during aversive olfactory conditioning
│   │       ├── 📝 dan_recording.py           # IBM effects on simulated odor recordings
│   │       └── 📝 alpha_scan_boxplot.py      # Parameter scan of IBM strength (α)
│   │
│   ├── 📦 two_compartments/                  # Two-compartment (appetitive & aversive) models
│   │   ├── 📦 calcium_state/                 # Calcium activity across calyx & two lobes
│   │   └── 📦 dan_dependent_ibm/             # IBM scripts for the two-compartment model
│   │       ├── 📝 conditioning & recording scripts   # Dopamine-pulse learning rule
│   │       └── 📦 prediction_error_functionality/    # Prediction-error learning scripts
│   │           ├── 📝 error_learning_no_ibm.py       # Prediction-error learning without IBM
│   │           ├── 📝 error_learning_ibm.py          # Prediction-error learning with IBM
│   │           └── 📝 pulse_learning_ibm.py          # Control script
│
├── 📝 MANIFEST.in                           # Includes dataset files during packaging
└── 📝 setup.py                              # Package installation and dataset initialization
