import numpy as np
import matplotlib.pyplot as plt


def plot_time_series(t, signals, labels=None, title=None, ylabel="x(t)"):
    """Plot one or more time series on a shared axis."""
    fig, ax = plt.subplots(figsize=(10, 3))
    signals = np.atleast_2d(signals)
    labels = labels or [None] * len(signals)
    for sig, lab in zip(signals, labels):
        ax.plot(t, sig, label=lab, lw=0.7)
    ax.set(xlabel="Time (s)", ylabel=ylabel, title=title)
    if any(l is not None for l in labels):
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_phase_portrait(x, y, title=None):
    """Plot a 2-D phase portrait (x vs y)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x, y, lw=0.5)
    ax.set(xlabel="x", ylabel="y", title=title, aspect="equal")
    fig.tight_layout()
    return fig, ax


def plot_power_spectrum(signal, fs, title=None, freq_range=(0, 50)):
    """Plot the one-sided power spectrum."""
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(signal)) ** 2 / len(signal)
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freqs[mask], spectrum[mask], lw=0.8)
    ax.set(xlabel="Frequency (Hz)", ylabel="Power", title=title)
    fig.tight_layout()
    return fig, ax
