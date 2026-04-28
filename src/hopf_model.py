import numpy as np
from scipy.integrate import solve_ivp


def hopf_rhs(_t, state, _N, W, K, lam, C, omega):
    """Coupled Hopf oscillator ODE. State layout: [x1, y1, ..., xN, yN]."""
    x = state[0::2]
    y = state[1::2]
    r2 = x**2 + y**2

    dx = x * (lam - r2) - omega * y + K * np.tanh(C * W @ x)
    dy = y * (lam - r2) + omega * x

    out = np.empty_like(state)
    out[0::2] = dx
    out[1::2] = dy
    return out


def simulate_hopf(
    N,
    W,
    K,
    lam,
    C,
    omega,
    z0=None,
    t_total=14.5,
    t_discard=1.0,
    fs=1250,
    rtol=None,
    atol=None,
    method=None,
):
    """Integrate the Hopf model, discard transient, return (t, x, y)."""
    if z0 is None:
        z0 = random_initial_conditions(N, rng=42)

    z0 = np.asarray(z0)
    if np.iscomplexobj(z0):
        if z0.ndim == 0:
            z0 = np.full(N, z0, dtype=complex)
        state0 = np.empty(2 * N)
        state0[0::2] = z0.real
        state0[1::2] = z0.imag
    else:
        z0 = np.asarray(z0, dtype=float)
        if z0.ndim == 0:
            state0 = np.zeros(2 * N, dtype=float)
            state0[0::2] = float(z0)
        elif z0.shape == (N,):
            state0 = np.zeros(2 * N, dtype=float)
            state0[0::2] = z0
        else:
            state0 = z0

    if state0.shape != (2 * N,):
        raise ValueError(f"Initial state must contain {N} complex or {2 * N} real values")

    t_eval = np.arange(t_discard, t_total, 1.0 / fs)

    solve_kwargs = {
        "t_eval": t_eval,
        "args": (N, W, K, lam, C, omega),
    }
    if rtol is not None:
        solve_kwargs["rtol"] = rtol
    if atol is not None:
        solve_kwargs["atol"] = atol
    if method is not None:
        solve_kwargs["method"] = method

    sol = solve_ivp(
        hopf_rhs, (0.0, t_total), state0,
        **solve_kwargs,
    )
    if sol.status != 0:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol.t - t_discard, sol.y[0::2], sol.y[1::2]


def random_initial_conditions(N, rng=None):
    """Sample N complex values uniformly in the unit disk."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    r = np.sqrt(rng.uniform(0, 1, N))
    theta = rng.uniform(0, 2 * np.pi, N)
    return r * np.exp(1j * theta)


def gaussian_initial_conditions(N, rng=None, scale=0.1):
    """Sample N complex Gaussian initial conditions with a configurable scale."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    return scale * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
