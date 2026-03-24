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


def simulate_hopf(N, W, K, lam, C, omega, z0,
                  t_total=14.5, t_discard=1.0, fs=1250):
    """Integrate the Hopf model, discard transient, return (t, x, y)."""
    if np.iscomplexobj(z0):
        state0 = np.empty(2 * N)
        state0[0::2] = z0.real
        state0[1::2] = z0.imag
    else:
        state0 = np.asarray(z0, dtype=float)

    t_eval = np.arange(t_discard, t_total, 1.0 / fs)

    sol = solve_ivp(
        hopf_rhs, (0.0, t_total), state0,
        t_eval=t_eval, args=(N, W, K, lam, C, omega),
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
