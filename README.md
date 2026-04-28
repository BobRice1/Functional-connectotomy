# Whole-Brain Hopf Model Reconstruction

The goal of this project is to reproduce and extend the whole-brain model
presented in *Functional connectotomy of a whole-brain model*. The code rebuilds
the system step by step, beginning with a single Hopf oscillator, then a coupled
two-node system, and finally the full structurally connected brain network.

The repository is intended to provide a clear implementation of the model,
simulation pipeline, and fitting procedure described in the paper, while also
serving as a base for extensions such as virtual lesioning and structural
connectivity perturbation experiments.

## Structure

* `src/`
  * `hopf_model.py` - coupled Hopf dynamics, integration, and initial-condition
    helpers.
  * `signal_processing.py` - bandpass filtering, instantaneous phase, PLI, and
    thresholding helpers.
  * `network_analysis.py` - shared network analysis utilities: SC
    normalization, topology generation, PLI fitting, perturbation helpers,
    AAL78 labels, hubness, ranking, and plotting helpers.
  * `project_paths.py` - project-root, data, figure, and paper path helpers.
  * `results_io.py` - schema validation for cached `.npz` result files.
  * `utils.py` - small plotting helpers for early notebooks.

* `data/`
  * Empirical inputs: `structural_connectivity_matrix.csv`,
    `control_PLI_raw.csv`, `glioma_PLI_raw.csv`,
    `control_PLI_top3percent.csv`, `glioma_PLI_top3percent.csv`, and
    `exp_frequencies.csv`.
  * Cached intermediate outputs: `vla_results.npz` from notebook `05`,
    `sc_perturbation_results.npz` from notebook `06`, and
    `part_c_summary.npz` / `part_c_master_table.csv` from notebook `07`.

* `notebooks/`
  * `01_single_node.ipynb` - single-node Hopf dynamics, bifurcation, amplitude,
    and signal-pipeline checks.
  * `02_two_node.ipynb` - two coupled Hopf nodes: coupling/excitability sweeps,
    PLI behavior, and Arnold tongue.
  * `03_network_stability.ipynb` - synthetic network sanity checks and
    reproduction of the paper's network-scale analyses.
  * `04_validation_and_fitting.ipynb` - validation of simulation and parameter
    fitting against synthetic/empirical PLI targets.
  * `05_virtual_lesioning.ipynb` - Part A: virtual lesioning atlas; per-region
    delta-K rankings without modifying the structural matrix.
  * `06_sc_perturbation.ipynb` - Part B: structural edge-scaling lesions and
    dose-response of fitted `K*`.
  * `07_extension_synthesis.ipynb` - Part C: synthesis of Parts A and B,
    including ranking, hubness, and tumour-overlap metrics.

* `figures/`
  * Generated figures under `figures/hopf model/`, grouped by experiment.

* `scripts/`
  * Stand-alone utilities used outside the notebook flow.

* `Paper/`
  * LaTeX write-up of the project (`paper.tex`) and generated paper outputs.

## Reproducible Workflow

Install dependencies:

```bash
pip install -r requirements.txt
```

The extension notebooks consume each other's cached outputs:

1. Run `notebooks/05_virtual_lesioning.ipynb` to create
   `data/vla_results.npz`.
2. Run `notebooks/06_sc_perturbation.ipynb` to create
   `data/sc_perturbation_results.npz`.
3. Run `notebooks/07_extension_synthesis.ipynb` to create
   `data/part_c_summary.npz` and `data/part_c_master_table.csv`.

Notebook `06` is computationally expensive. Set `HOPF_N_JOBS` to control
parallelism, for example:

```bash
$env:HOPF_N_JOBS = "8"   # PowerShell
```

Existing cached `.npz` files are intentionally kept in `data/` so downstream
notebooks can be inspected without rerunning the long sweeps.

## Fast Validation

The lightweight checks avoid expensive scientific reruns:

```bash
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest discover -s tests
python -m py_compile src\hopf_model.py src\signal_processing.py src\network_analysis.py src\project_paths.py src\results_io.py scripts\patch_nb06_robustness.py
```

The tests cover deterministic initial conditions, Hopf simulation shapes, PLI
matrix behavior, network helper consistency, perturbation helpers, and cache
schema validation.

## Paper Reference

This repository is based on:

> Alexandersen CG, Douw L, Zimmermann MLM, Bick C, Goriely A. *Functional
> connectotomy of a whole-brain model reveals tumor-induced alterations to
> neuronal dynamics in glioma patients.* Network Neuroscience. 2025
> Mar 20;9(1):280-302.
