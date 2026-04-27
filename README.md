# Simulating Calcium Dynamics and Independent Bouton Modulation in the Mushroom Body γ-Lobe

Computational model of calcium activity in individual γ-lobe boutons in Drosophila melanogaster, implemented using single- and multi-compartment frameworks.

The model is fitted to experimental data from Manoim et al. (2022) (https://doi.org/10.1016/j.cub.2022.09.007
) and incorporates inhibitory KC–KC interactions. It builds upon the KC_KC_interactions project by Ibrahim Tunc and has been significantly extended during my Bachelor’s thesis and subsequent research assistant position in the Nawrot Lab.

The repository also investigates independent bouton modulation as described in Bilz et al. (2020) (https://doi.org/10.1016/j.neuron.2020.03.010
), enabling direct comparison between simulated calcium dynamics and in vivo calcium imaging data during olfactory conditioning.

---

## 🔬 Key Features
- Single- and multi-compartment bouton activity models  
- KC–KC inhibitory interaction dynamics  
- Data-driven parameter fitting  
- Comparison with experimental calcium traces  

## 🧾 Conference Poster on this project

<p align="center">
  <img src="poster/IBM_goettingen_poster.svg" width="750"/>
</p>

## 📊 Data
Includes experimental datasets and analysis tools for calcium imaging data from Bilz et al. (2020).

## ⚙️ Tech
Python · NumPy · SciPy · Matplotlib

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
