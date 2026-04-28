"""Shared helpers for 78x78 whole-brain Hopf analyses.

These utilities are used across the network-scale notebooks (the main
reproduction of Alexandersen et al. 2025 and the virtual-lesioning
extension). Extracted here to keep notebooks lean and consistent.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import hypergeom, rankdata, wilcoxon

try:
    from .hopf_model import random_initial_conditions, simulate_hopf
    from .signal_processing import compute_phase, compute_pli
except ImportError:  # notebooks add src/ directly to sys.path
    from hopf_model import random_initial_conditions, simulate_hopf
    from signal_processing import compute_phase, compute_pli


# ---------------------------------------------------------------------------
# Matrix / correlation helpers
# ---------------------------------------------------------------------------

def upper_triangle_values(matrix):
    """Return the strict upper-triangle entries of a square matrix as a 1-D array."""
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def nonzero_upper_triangle_values(matrix):
    """Return non-zero strict upper-triangle entries of a square matrix."""
    values = upper_triangle_values(np.asarray(matrix, dtype=float))
    return values[np.abs(values) > 0]


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


def pairwise_consistency(matrices):
    """Mean pairwise upper-triangle correlation over a list of matrices."""
    scores = []
    for a, b in combinations(matrices, 2):
        score = matrix_correlation(a, b)
        if np.isfinite(score):
            scores.append(score)
    if not scores:
        return np.nan
    return float(np.mean(scores))


def edge_density(W):
    """Fraction of non-zero strict upper-triangle entries."""
    upper = upper_triangle_values(np.asarray(W, dtype=float))
    return float(np.mean(np.abs(upper) > 0))


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


def structural_hubness(W, threshold=1e-12, as_tuple=False):
    """Return weighted degree, strength, and eigenvector centrality per region."""
    A = np.abs(np.asarray(W, dtype=float)).copy()
    np.fill_diagonal(A, 0.0)
    A = (A + A.T) / 2.0

    degree = (A > threshold).sum(axis=1)
    strength = A.sum(axis=1)

    eigvals, eigvecs = np.linalg.eigh(A)
    dominant = eigvecs[:, np.argmax(np.abs(eigvals))]
    if np.sum(dominant) < 0:
        dominant = -dominant
    eigcent = np.abs(dominant)

    if as_tuple:
        return degree, strength, eigcent
    return {"degree": degree, "strength": strength, "eigcent": eigcent}


def load_cortical_frequencies(path, n_cortical=78):
    """Subject-mean peak frequency for each of the first n_cortical AAL regions."""
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    per_subject = raw[:, 1:n_cortical + 1]
    return np.nanmean(per_subject, axis=0)


def load_empirical_frequency_bank(path):
    """Return all finite empirical frequency values after the first CSV column."""
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    if raw.ndim == 1:
        values = raw[1:]
    else:
        values = raw[:, 1:]
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


# ---------------------------------------------------------------------------
# Synthetic and empirical-scale topology helpers
# ---------------------------------------------------------------------------

def ring_network(n, weight=1.0):
    """Undirected nearest-neighbour ring network."""
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        W[i, (i - 1) % n] = weight
        W[i, (i + 1) % n] = weight
    return W


def erdos_renyi_network(n, p=0.2, rng=None, weighted=True):
    """Undirected Erdos-Renyi graph with optional random positive weights."""
    rng = np.random.default_rng(rng)
    mask = np.triu(rng.uniform(size=(n, n)) < p, k=1)
    if weighted:
        weights = rng.uniform(0.2, 1.0, size=(n, n))
        W = np.where(mask, weights, 0.0)
    else:
        W = mask.astype(float)
    return W + W.T


def modular_network(n, modules=3, p_in=0.7, p_out=0.08, rng=None):
    """Undirected block-modular weighted graph."""
    rng = np.random.default_rng(rng)
    groups = np.array_split(np.arange(n), modules)
    W = np.zeros((n, n), dtype=float)
    for group in groups:
        for i in group:
            for j in group:
                if i < j and rng.uniform() < p_in:
                    W[i, j] = rng.uniform(0.4, 1.0)
    for group_i, group_j in combinations(groups, 2):
        for i in group_i:
            for j in group_j:
                if rng.uniform() < p_out:
                    W[i, j] = rng.uniform(0.1, 0.5)
    return W + W.T


def sample_weighted_graph_from_mask(mask, weight_bank, rng=None):
    """Sample weights into the upper-triangle positions marked by mask."""
    rng = np.random.default_rng(rng)
    W = np.zeros(mask.shape, dtype=float)
    edge_count = int(np.count_nonzero(mask))
    if edge_count > 0:
        sampled = rng.choice(weight_bank, size=edge_count, replace=True)
        W[mask] = sampled
    return W + W.T


def density_matched_random_network(reference_W, rng=None):
    """Random graph with density and weight distribution matched to reference_W."""
    rng = np.random.default_rng(rng)
    n = reference_W.shape[0]
    density = edge_density(reference_W)
    weight_bank = nonzero_upper_triangle_values(reference_W)
    mask = np.triu(rng.uniform(size=(n, n)) < density, k=1)
    return normalize_weights(sample_weighted_graph_from_mask(mask, weight_bank, rng=rng))


def density_matched_modular_network(reference_W, modules=4, rng=None):
    """Modular graph with approximate density and weights matched to reference_W."""
    rng = np.random.default_rng(rng)
    n = reference_W.shape[0]
    density = edge_density(reference_W)
    weight_bank = nonzero_upper_triangle_values(reference_W)
    groups = np.array_split(np.arange(n), modules)
    mask = np.zeros((n, n), dtype=bool)

    p_in = min(0.75, max(0.18, density * 2.8))
    p_out = min(0.18, max(0.01, density * 0.35))

    for group in groups:
        for i in group:
            for j in group:
                if i < j and rng.uniform() < p_in:
                    mask[i, j] = True
    for group_i, group_j in combinations(groups, 2):
        for i in group_i:
            for j in group_j:
                if rng.uniform() < p_out:
                    mask[i, j] = True

    if not np.any(mask):
        return density_matched_random_network(reference_W, rng=rng)
    return normalize_weights(sample_weighted_graph_from_mask(mask, weight_bank, rng=rng))


def make_topology(name, n, rng=None):
    """Construct a normalized synthetic topology by name."""
    rng = np.random.default_rng(rng)
    if name == "ring":
        W = ring_network(n)
    elif name == "random":
        p = min(0.35, max(0.08, 3.0 / max(n - 1, 1)))
        W = erdos_renyi_network(n, p=p, rng=rng)
    elif name == "modular":
        W = modular_network(n, modules=min(4, max(2, n // 10 or 2)), rng=rng)
    else:
        raise ValueError(f"Unknown topology: {name}")

    if not np.any(W):
        W = ring_network(n)
    return normalize_weights(W)


def make_empirical_scale_topology(name, reference_W, rng=None):
    """Construct a normalized topology with the same size as reference_W."""
    reference_W = np.asarray(reference_W, dtype=float)
    n = reference_W.shape[0]
    if name == "ring":
        return normalize_weights(ring_network(n))
    if name == "random":
        return density_matched_random_network(reference_W, rng=rng)
    if name == "modular":
        return density_matched_modular_network(reference_W, rng=rng)
    if name == "empirical":
        return normalize_weights(reference_W)
    raise ValueError(f"Unknown empirical-scale topology: {name}")


def shuffle_sc(W, rng=None):
    """Randomly reassign upper-triangle edge weights, preserving distribution."""
    rng = np.random.default_rng(rng)
    W = np.asarray(W, dtype=float)
    upper_idx = np.triu_indices_from(W, k=1)
    upper = W[upper_idx].copy()
    rng.shuffle(upper)
    out = np.zeros_like(W)
    out[upper_idx] = upper
    return out + out.T


def sample_frequency_vector(n, rng=None, mode="alpha_jitter", empirical_bank=None):
    """Sample alpha-band frequency vectors for synthetic network checks."""
    rng = np.random.default_rng(rng)
    if mode == "alpha_jitter":
        hz = rng.normal(loc=10.0, scale=0.8, size=n)
    elif mode == "empirical_bank":
        if empirical_bank is None or len(empirical_bank) == 0:
            raise ValueError("empirical_bank mode requires a non-empty empirical bank")
        hz = rng.choice(empirical_bank, size=n, replace=True)
    elif mode == "smooth_gradient":
        base = np.linspace(8.5, 10.5, n)
        hz = base + rng.normal(scale=0.15, size=n)
    else:
        raise ValueError(f"Unknown frequency mode: {mode}")
    return np.clip(hz, 8.0, 12.0)


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


def _run_one_repeat(W, K, lam, seed, freq_hz, t_total, t_discard, fs):
    """Single picklable simulation worker returning a PLI matrix."""
    return simulate_pli_matrix(
        W, K=K, lam=lam, freq_hz=freq_hz, seed=seed,
        t_total=t_total, t_discard=t_discard, fs=fs,
    )


def repeated_simulation_summary(
    W,
    K,
    lam,
    repeats=3,
    freq_mode="alpha_jitter",
    empirical_bank=None,
    fixed_freq=None,
    base_seed=0,
    t_total=8.0,
    t_discard=1.5,
    fs=300,
    n_jobs=1,
):
    """Run repeated simulations and summarize mean PLI and consistency."""
    seeds = [base_seed + rep for rep in range(repeats)]
    if fixed_freq is not None:
        freqs = [fixed_freq] * repeats
    else:
        freqs = [
            sample_frequency_vector(
                W.shape[0], rng=seed, mode=freq_mode, empirical_bank=empirical_bank
            )
            for seed in seeds
        ]

    plis = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_repeat)(W, K, lam, seed, freq, t_total, t_discard, fs)
        for seed, freq in zip(seeds, freqs)
    )
    return {
        "mean_pli": np.mean(plis, axis=0),
        "consistency": pairwise_consistency(plis),
        "mean_strength": float(np.mean([upper_triangle_values(p).mean() for p in plis])),
        "n_repeats": repeats,
    }


def _eval_one_gridpoint(
    i,
    j,
    W,
    K,
    lam,
    repeats,
    target_fc,
    fixed_freq,
    freq_mode,
    empirical_bank,
    t_total,
    t_discard,
    fs,
):
    """Run all repeats for one (K, lambda) grid point."""
    plis = []
    scores = []
    for rep in range(repeats):
        seed = 1000 + 100 * i + 10 * j + rep
        if fixed_freq is not None:
            freq_hz = fixed_freq
        else:
            freq_hz = sample_frequency_vector(
                W.shape[0], rng=seed, mode=freq_mode, empirical_bank=empirical_bank
            )
        pli = simulate_pli_matrix(
            W, K=K, lam=lam, freq_hz=freq_hz, seed=seed,
            t_total=t_total, t_discard=t_discard, fs=fs,
        )
        plis.append(pli)
        scores.append(matrix_correlation(pli, target_fc))
    return i, j, np.nanmean(scores), pairwise_consistency(plis)


def evaluate_empirical_fit(
    W,
    target_fc,
    K_values,
    lam_values,
    repeats=3,
    empirical_bank=None,
    fixed_freq=None,
    freq_mode="empirical_bank",
    t_total=8.0,
    t_discard=1.5,
    fs=300,
    n_jobs=1,
):
    """Grid-search Hopf parameters against a target FC/PLI matrix."""
    fit = np.full((len(lam_values), len(K_values)), np.nan)
    stability = np.full_like(fit, np.nan)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_one_gridpoint)(
            i, j, W, K, lam, repeats, target_fc, fixed_freq,
            freq_mode, empirical_bank, t_total, t_discard, fs,
        )
        for i, lam in enumerate(lam_values)
        for j, K in enumerate(K_values)
    )
    for i, j, score, stab in results:
        fit[i, j] = score
        stability[i, j] = stab
    return fit, stability


# ---------------------------------------------------------------------------
# Virtual lesioning / perturbation helpers
# ---------------------------------------------------------------------------

def fit_virtual_lesion_one_ic(ic, lam, K_vals, W, freq, emp_ctrl, emp_glio, sim_kw):
    """Sweep K for one IC and return full and per-region-masked K* values."""
    seed = 90_000 + ic
    n_K = len(K_vals)
    n = W.shape[0]

    corr_full_ctrl = np.empty(n_K)
    corr_full_glio = np.empty(n_K)
    corr_mask_ctrl = np.empty((n_K, n))
    corr_mask_glio = np.empty((n_K, n))

    for k, K in enumerate(K_vals):
        pli = simulate_pli_matrix(W, K=K, lam=lam, freq_hz=freq, seed=seed, **sim_kw)
        corr_full_ctrl[k] = matrix_correlation(pli, emp_ctrl)
        corr_full_glio[k] = matrix_correlation(pli, emp_glio)
        for region in range(n):
            corr_mask_ctrl[k, region] = masked_correlation(pli, emp_ctrl, [region])
            corr_mask_glio[k, region] = masked_correlation(pli, emp_glio, [region])

    return (
        ic,
        K_vals[np.nanargmax(corr_full_ctrl)],
        K_vals[np.nanargmax(corr_full_glio)],
        K_vals[np.nanargmax(corr_mask_ctrl, axis=0)],
        K_vals[np.nanargmax(corr_mask_glio, axis=0)],
    )


def wilcoxon_pvalue_safe(vec, min_nonzero=10):
    """Wilcoxon p-value that returns NaN when the vector is too sparse/degenerate."""
    vec = np.asarray(vec, dtype=float)
    nonzero = vec[vec != 0]
    if len(nonzero) < min_nonzero:
        return np.nan
    try:
        _, p_value = wilcoxon(nonzero)
        return p_value
    except ValueError:
        return np.nan


def summarise_delta_k(delta_k, min_nonzero=10):
    """Summarize per-region delta-K samples with mean, CI, and Wilcoxon p-values."""
    delta_k = np.asarray(delta_k, dtype=float)
    return {
        "mean": delta_k.mean(axis=0),
        "ci_lo": np.percentile(delta_k, 2.5, axis=0),
        "ci_hi": np.percentile(delta_k, 97.5, axis=0),
        "pvals": np.array([
            wilcoxon_pvalue_safe(delta_k[:, region], min_nonzero=min_nonzero)
            for region in range(delta_k.shape[1])
        ]),
    }


def top_k_drivers(summary, k=10):
    """Most-negative mean delta-K entries, matching the Part A driver convention."""
    return np.argsort(summary["mean"])[:k]


def damage_region(W, region, damage_fraction):
    """Scale all edges touching region by (1 - damage_fraction)."""
    factor = 1.0 - float(damage_fraction)
    W_damaged = np.asarray(W, dtype=float).copy()
    W_damaged[region, :] *= factor
    W_damaged[:, region] *= factor
    return W_damaged


def fit_damaged_region_kstars(
    region, damage, ic, W_base, K_vals, lam, freq, emp_ctrl, emp_glio, sim_kw
):
    """Return K* fits for one structurally damaged region/IC condition."""
    seed = 100_000 + region * 1000 + int(round(damage * 100)) * 10 + ic
    W = damage_region(W_base, region, damage)

    corr_ctrl = np.empty(len(K_vals))
    corr_glio = np.empty(len(K_vals))
    for k, K in enumerate(K_vals):
        pli = simulate_pli_matrix(W, K=K, lam=lam, freq_hz=freq, seed=seed, **sim_kw)
        corr_ctrl[k] = matrix_correlation(pli, emp_ctrl)
        corr_glio[k] = matrix_correlation(pli, emp_glio)

    Kstar_ctrl = K_vals[np.nanargmax(corr_ctrl)]
    Kstar_glio = K_vals[np.nanargmax(corr_glio)]
    return region, float(damage), ic, float(Kstar_ctrl), float(Kstar_glio), corr_ctrl, corr_glio


def compute_damage_slopes(mean_Kc, mean_Kg, damage_levels, mild_levels=None):
    """Compute full-range and mild-range linear K* damage slopes."""
    damage_levels = np.asarray(damage_levels, dtype=float)
    mean_Kc = np.asarray(mean_Kc, dtype=float)
    mean_Kg = np.asarray(mean_Kg, dtype=float)
    if mild_levels is None:
        mild_levels = damage_levels[:3]
        mild_slice = slice(0, 3)
    else:
        mild_levels = np.asarray(mild_levels, dtype=float)
        mild_slice = [int(np.where(damage_levels == level)[0][0]) for level in mild_levels]

    return {
        "slopes_ctrl_full": np.array([
            np.polyfit(damage_levels, mean_Kc[i], 1)[0]
            for i in range(mean_Kc.shape[0])
        ]),
        "slopes_glio_full": np.array([
            np.polyfit(damage_levels, mean_Kg[i], 1)[0]
            for i in range(mean_Kg.shape[0])
        ]),
        "slopes_ctrl_mild": np.array([
            np.polyfit(mild_levels, mean_Kc[i, mild_slice], 1)[0]
            for i in range(mean_Kc.shape[0])
        ]),
        "slopes_glio_mild": np.array([
            np.polyfit(mild_levels, mean_Kg[i, mild_slice], 1)[0]
            for i in range(mean_Kg.shape[0])
        ]),
    }


def driver_strength(delta_k):
    """Driver strength convention used in Part C: larger means stronger driver."""
    return -np.asarray(delta_k, dtype=float)


def rank_high_first(values):
    """Return ranks where the largest value is rank 1."""
    return rankdata(-np.asarray(values), method="average")


def topn_indices(values, n, valid_mask=None):
    """Indices of the n largest finite values, optionally masked."""
    values = np.asarray(values, dtype=float).copy()
    if valid_mask is not None:
        values = np.where(valid_mask, values, -np.inf)
    values = np.where(np.isfinite(values), values, -np.inf)
    return np.argsort(-values)[:n]


def hypergeom_pvalue(hits, top_n, set_size, total):
    """One-sided P(X >= hits) for top_n draws from total items."""
    return float(hypergeom.sf(hits - 1, total, set_size, top_n))


def overlap_test(top_indices, target_indices, total=78):
    """Return overlap count and hypergeometric p-value."""
    hits = len(set(top_indices) & set(target_indices))
    return hits, hypergeom_pvalue(hits, len(top_indices), len(target_indices), total)


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
