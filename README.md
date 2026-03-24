# Whole-Brain Hopf Model Reconstruction

The goal of this project is to reproduce and extend the whole-brain model presented in *Functional connectotomy of a whole-brain model* by rebuilding the system step by step, beginning with a single Hopf oscillator, then a coupled two-node system, and finally the full structurally connected brain network. The repository is intended to provide a clear implementation of the model, simulation pipeline, and fitting procedure described in the paper, while also serving as a base for further extensions and experiments.

## Aim

The initial focus of the repository is faithful reconstruction of the baseline model and fitting pipeline described in the paper. Once that baseline is validated, the codebase will be used to explore extensions such as alternative connectivity inputs, parameter sweeps, and modifications to the node or coupling dynamics.


## Structure

* `src/`

  * `hopf_model.py` – Defines the coupled Hopf dynamics, integrates the network ODEs, and helpers to run simulations and sample initial conditions.
  * `signal_processing.py` – Turns simulated (or empirical) time series into band-limited signals, instantaneous phases, and pairwise phase-lag–based connectivity measures used in later analyses.
  * `utils.py` – plotting and notebook helpers shared across experiments.

* `data/`

  * Structural connectivity matrices and processed empirical inputs
* `notebooks/`
  
  * `01_single_node.ipynb` - Single node model
  * `02_two_node.ipynb` - Two node model

* `figures/`

  * Figures generated from notebooks

## Paper reference

This repository is based on:

> Alexandersen CG, Douw L, Zimmermann MLM, Bick C, Goriely A. *Functional connectotomy of a whole-brain model reveals tumor-induced alterations to neuronal dynamics in glioma patients.* Netw Neurosci. 2025 Mar 20;9(1):280-302.

## Requirements

Install the Python dependencies needed to run the simulations and regenerate results:

```bash
pip install -r requirements.txt
```

