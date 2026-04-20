"""Shared helpers for 78x78 whole-brain Hopf analyses.

These utilities are used across the network-scale notebooks (the main
reproduction of Alexandersen et al. 2025 and the virtual-lesioning
extension). Extracted here to keep notebooks lean and consistent.
"""

from __future__ import annotations

import numpy as np

from hopf_model import random_initial_conditions, simulate_hopf
from signal_processing import compute_phase, compute_pli


# ---------------------------------------------------------------------------
# Matrix / correlation helpers
# ---------------------------------------------------------------------------

def upper_triangle_values(matrix):
    """Return the strict upper-triangle entries of a square matrix as a 1-D array."""
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def matrix_correlation(a, b):
    """Pearson correlation between the upper triangles of two square matrices."""
    x = upper_triangle_values(np.asarray(a, dtype=float))
    y = upper_triangle_values(np.asarray(b, dtype=float))
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    if np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def masked_correlation(sim_pli, emp_pli, mask_indices):
    """Pearson correlation of two PLI matrices after removing rows/columns in mask_indices."""
    keep = np.ones(sim_pli.shape[0], dtype=bool)
    keep[list(mask_indices)] = False
    sub_sim = sim_pli[np.ix_(keep, keep)]
    sub_emp = emp_pli[np.ix_(keep, keep)]
    return matrix_correlation(sub_sim, sub_emp)


def normalize_weights(W, method="spectral"):
    """Scale a connectivity matrix to unit spectral radius (default) or unit max/row-sum."""
    W = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(W, 0.0)
    if method == "max":
        scale = np.max(np.abs(W))
    elif method == "row":
        scale = np.max(np.sum(np.abs(W), axis=1))
    elif method == "spectral":
        scale = np.max(np.abs(np.linalg.eigvalsh(W)))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    if not np.isfinite(scale) or scale == 0:
        return W
    return W / scale


def load_cortical_frequencies(path, n_cortical=78):
    """Subject-mean peak frequency for each of the first n_cortical AAL regions."""
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    per_subject = raw[:, 1:n_cortical + 1]
    return np.nanmean(per_subject, axis=0)


# ---------------------------------------------------------------------------
# Whole-brain simulation wrapper
# ---------------------------------------------------------------------------

def simulate_pli_matrix(
    W, K, lam, freq_hz, C=20.0, seed=None, t_total=8.0, t_discard=1.5, fs=300
):
    """Run one Hopf simulation and return its PLI matrix."""
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    omega = 2 * np.pi * np.asarray(freq_hz, dtype=float)
    z0 = random_initial_conditions(n, rng=seed)
    _, x, _ = simulate_hopf(
        n, W, K, lam, C, omega, z0,
        t_total=t_total, t_discard=t_discard, fs=fs,
    )
    phases = compute_phase(x, fs=fs)
    return compute_pli(phases)


# ---------------------------------------------------------------------------
# AAL78 region metadata and labelled-matrix plotting
# ---------------------------------------------------------------------------

AAL78_LABELS = [
    "Precentral L", "Precentral R",
    "Frontal Sup L", "Frontal Sup R",
    "Frontal Sup Orb L", "Frontal Sup Orb R",
    "Frontal Mid L", "Frontal Mid R",
    "Frontal Mid Orb L", "Frontal Mid Orb R",
    "Frontal Inf Oper L", "Frontal Inf Oper R",
    "Frontal Inf Tri L", "Frontal Inf Tri R",
    "Frontal Inf Orb L", "Frontal Inf Orb R",
    "Rolandic Oper L", "Rolandic Oper R",
    "SMA L", "SMA R",
    "Olfactory L", "Olfactory R",
    "Frontal Sup Med L", "Frontal Sup Med R",
    "Frontal Med Orb L", "Frontal Med Orb R",
    "Rectus L", "Rectus R",
    "Insula L", "Insula R",
    "Cingulate Ant L", "Cingulate Ant R",
    "Cingulate Mid L", "Cingulate Mid R",
    "Cingulate Post L", "Cingulate Post R",
    "Hippocampus L", "Hippocampus R",
    "ParaHippocampal L", "ParaHippocampal R",
    "Amygdala L", "Amygdala R",
    "Calcarine L", "Calcarine R",
    "Cuneus L", "Cuneus R",
    "Lingual L", "Lingual R",
    "Occipital Sup L", "Occipital Sup R",
    "Occipital Mid L", "Occipital Mid R",
    "Occipital Inf L", "Occipital Inf R",
    "Fusiform L", "Fusiform R",
    "Postcentral L", "Postcentral R",
    "Parietal Sup L", "Parietal Sup R",
    "Parietal Inf L", "Parietal Inf R",
    "SupraMarginal L", "SupraMarginal R",
    "Angular L", "Angular R",
    "Precuneus L", "Precuneus R",
    "Paracentral Lob L", "Paracentral Lob R",
    "Caudate L", "Caudate R",
    "Putamen L", "Putamen R",
    "Pallidum L", "Pallidum R",
    "Thalamus L", "Thalamus R",
]

NETWORK_NAMES = [
    "Somatomotor", "Somatomotor",
    "Frontoparietal", "Frontoparietal",
    "Limbic", "Limbic",
    "Ventral Attn", "Ventral Attn",
    "Limbic", "Limbic",
    "Ventral Attn", "Ventral Attn",
    "Frontoparietal", "Frontoparietal",
    "Limbic", "Limbic",
    "Somatomotor", "Somatomotor",
    "Somatomotor", "Somatomotor",
    "Limbic", "Limbic",
    "Default Mode", "Default Mode",
    "Limbic", "Limbic",
    "Limbic", "Limbic",
    "Ventral Attn", "Ventral Attn",
    "Default Mode", "Default Mode",
    "Default Mode", "Default Mode",
    "Default Mode", "Default Mode",
    "Limbic", "Limbic",
    "Limbic", "Limbic",
    "Limbic", "Limbic",
    "Visual", "Visual",
    "Visual", "Visual",
    "Visual", "Visual",
    "Visual", "Visual",
    "Visual", "Visual",
    "Visual", "Visual",
    "Visual", "Visual",
    "Somatomotor", "Somatomotor",
    "Dorsal Attn", "Dorsal Attn",
    "Dorsal Attn", "Dorsal Attn",
    "Ventral Attn", "Ventral Attn",
    "Default Mode", "Default Mode",
    "Default Mode", "Default Mode",
    "Somatomotor", "Somatomotor",
    "Subcortical", "Subcortical",
    "Subcortical", "Subcortical",
    "Subcortical", "Subcortical",
    "Subcortical", "Subcortical",
]

NETWORK_COLORS = {
    "Visual":         "#9B59B6",
    "Somatomotor":    "#3498DB",
    "Dorsal Attn":    "#2ECC71",
    "Ventral Attn":   "#E74C3C",
    "Limbic":         "#F39C12",
    "Frontoparietal": "#1ABC9C",
    "Default Mode":   "#E91E63",
    "Subcortical":    "#7F8C8D",
}


def plot_labelled_matrix(ax, mat, title, cmap="plasma", vmin=None, vmax=None,
                         labels=None, networks=None, net_colors=None,
                         label_fontsize=3.2, cbar=True, fig=None):
    """Plot a 78x78 matrix with AAL78 region labels coloured by functional network."""
    if labels is None:
        labels = AAL78_LABELS
    if networks is None:
        networks = NETWORK_NAMES
    if net_colors is None:
        net_colors = NETWORK_COLORS

    n = mat.shape[0]
    if vmax is None:
        vmax = mat.max() if mat.max() > 0 else 1
    if vmin is None:
        vmin = 0

    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=9, pad=4)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels[:n], fontsize=label_fontsize, rotation=90)
    ax.set_yticklabels(labels[:n], fontsize=label_fontsize)

    for idx in range(n):
        col = net_colors.get(networks[idx], "#333333")
        ax.get_xticklabels()[idx].set_color(col)
        ax.get_yticklabels()[idx].set_color(col)

    ax.tick_params(length=0)

    if cbar and fig is not None:
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

    return im
