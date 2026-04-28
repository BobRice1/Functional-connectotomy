"""Validation helpers for cached analysis outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


EXPECTED_NPZ_KEYS = {
    "vla_results": {
        "dK_ctrl", "dK_glio", "dK_diff",
        "mean_ctrl", "mean_glio", "mean_diff",
        "Kfull_ctrl", "Kfull_glio",
        "VLA_K", "VLA_LAM",
    },
    "sc_perturbation_results": {
        "regions", "region_kind", "damage_levels", "damage_mild",
        "Kstar_ctrl", "Kstar_glio",
        "mean_Kc", "mean_Kg", "std_Kc", "std_Kg",
        "slopes_ctrl_mild", "slopes_ctrl_full",
        "slopes_glio_mild", "slopes_glio_full",
        "N_TOP", "N_NULL", "SC_N_IC",
        "VLA_K", "VLA_LAM",
        "mean_ctrl_A", "mean_glio_A",
    },
    "part_c_summary": {
        "eigcent", "degree", "strength",
        "rho_eig_ctrl", "p_eig_ctrl",
        "rho_eig_glio", "p_eig_glio",
        "rho_partA_partB_mild", "p_partA_partB_mild",
        "rho_partA_partB_full", "p_partA_partB_full",
        "tau_partA_hub", "p_tau_partA_hub",
        "tau_partA_partB", "p_tau_partA_partB",
        "tau_hub_partB", "p_tau_hub_partB",
        "paper_tumour_set", "overlap_Ns",
        "overlap_partA_ctrl", "overlap_partA_glio", "overlap_partA_diff",
        "overlap_eigcent", "overlap_partB_mild",
        "p_partA_ctrl", "p_partA_glio", "p_partA_diff",
        "p_eigcent", "p_partB_mild",
    },
}


def cache_name_from_path(path):
    """Return the expected-key registry name for a cache path."""
    return Path(path).stem


def expected_keys(cache_name):
    """Return required keys for a known cache name."""
    try:
        return EXPECTED_NPZ_KEYS[cache_name]
    except KeyError as exc:
        known = ", ".join(sorted(EXPECTED_NPZ_KEYS))
        raise KeyError(f"Unknown cache {cache_name!r}; known caches: {known}") from exc


def validate_npz_keys(path, cache_name=None, required_keys=None, allow_extra=True):
    """Validate an .npz cache's key set and return its sorted keys."""
    path = Path(path)
    if cache_name is None:
        cache_name = cache_name_from_path(path)
    required = set(expected_keys(cache_name) if required_keys is None else required_keys)

    with np.load(path, allow_pickle=True) as cache:
        keys = set(cache.files)

    missing = sorted(required - keys)
    extra = sorted(keys - required)
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")
    if extra and not allow_extra:
        raise KeyError(f"{path} has unexpected keys: {extra}")
    return sorted(keys)


def load_validated_npz(path, cache_name=None, allow_extra=True):
    """Validate and open an .npz cache."""
    validate_npz_keys(path, cache_name=cache_name, allow_extra=allow_extra)
    return np.load(path, allow_pickle=True)
