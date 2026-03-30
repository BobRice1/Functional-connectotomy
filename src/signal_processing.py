import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert


def bandpass_filter(signal, fs, low=8.0, high=12.0, order=4):
    """Zero-phase Butterworth bandpass along the last axis."""
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=-1)


def compute_phase(signal, fs, low=8.0, high=12.0, order=4):
    """Bandpass then Hilbert transform to get instantaneous phase."""
    filtered = bandpass_filter(signal, fs, low, high, order)
    return np.angle(hilbert(filtered, axis=-1))


def compute_pli(phases, _max_elements=50_000_000):
    """Phase-lag index matrix: PLI_ij = |<sign(sin(theta_i - theta_j))>_t|.

    Uses a full broadcast when the intermediate array fits in memory,
    otherwise falls back to a row-wise loop (O(N*T) instead of O(N^2*T)).
    """
    N, T = phases.shape
    if N * N * T <= _max_elements:
        diff = phases[:, np.newaxis, :] - phases[np.newaxis, :, :]
        pli = np.abs(np.mean(np.sign(np.sin(diff)), axis=-1))
    else:
        pli = np.empty((N, N), dtype=float)
        for i in range(N):
            diff = phases[i] - phases          # (N, T)
            pli[i] = np.abs(np.mean(np.sign(np.sin(diff)), axis=-1))
    np.fill_diagonal(pli, 0.0)
    return pli


def threshold_pli(pli, percentile=97):
    """Zero out PLI values below the given percentile of the upper triangle."""
    upper = pli[np.triu_indices_from(pli, k=1)]
    out = pli.copy()
    out[out < np.percentile(upper, percentile)] = 0.0
    return out
