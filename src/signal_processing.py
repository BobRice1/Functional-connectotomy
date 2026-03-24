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


def compute_pli(phases):
    """Phase-lag index matrix: PLI_ij = |<sign(sin(theta_i - theta_j))>_t|."""
    N = phases.shape[0]
    pli = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            diff = phases[i] - phases[j]
            pli[i, j] = pli[j, i] = np.abs(np.mean(np.sign(np.sin(diff))))
    return pli


def threshold_pli(pli, percentile=97):
    """Zero out PLI values below the given percentile of the upper triangle."""
    upper = pli[np.triu_indices_from(pli, k=1)]
    out = pli.copy()
    out[out < np.percentile(upper, percentile)] = 0.0
    return out
