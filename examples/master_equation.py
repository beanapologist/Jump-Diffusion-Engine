"""
master_equation.py — Pauli master equation via markov_generator

Demonstrates that the generator matrix L built by JumpDiffusionEngine.markov_generator
is the exact matrix encoding of the Pauli (Kolmogorov forward) master equation:

    dp_n/dt = Σ_{m≠n} [ W_{n←m} p_m  −  W_{m←n} p_n ]
                          ^^^^^^^^^^^^^    ^^^^^^^^^^^^^
                              in              out

In matrix form this is simply:

    dp/dt = L · p

where L[n, m] = W_{n←m}  (off-diagonal, ≥ 0)  and
      L[n, n] = −Σ_{m≠n} W_{m←n} = −(rate_up[n] + rate_down[n])

The diagonal is set by the *safer column-generator approach*:

    L[n, n] = −(rate_up[n] + rate_down[n])

so each column of L sums to exactly zero, which is the matrix-level statement
of probability conservation: d/dt Σ_n p_n = 0.

This script:
  1. Builds L for the default engine (constant Λ, nonlinear sink f).
  2. Starts from a narrow Gaussian initial distribution p(0).
  3. Integrates dp/dt = L·p forward in time using scipy.integrate.solve_ivp.
  4. Plots the evolving distribution and verifies that probability is conserved
     at every output time.

Run:
    python3 master_equation.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from jump_diffusion_engine import JumpDiffusionEngine


# ---------------------------------------------------------------------------
# 1. Set up the engine
# ---------------------------------------------------------------------------
LAMBDA_VAL = 0.5        # constant forcing Λ
SIGMA      = 0.4        # diffusion strength
K          = 0.8        # linear restoring rate
G          = 0.5        # nonlinear gain
K_SAT      = 2.0        # saturation scale

eng = JumpDiffusionEngine(
    lambda_func=lambda t: LAMBDA_VAL,
    sigma=SIGMA,
    jump_rate=0.0,       # pure diffusion + drift for clarity
    dt=0.01,
    seed=42,
    k=K, g=G, K=K_SAT,
)

# ---------------------------------------------------------------------------
# 2. Build the Markov generator matrix
# ---------------------------------------------------------------------------
X_RANGE = (-8.0, 12.0)
N       = 300            # grid nodes  (matrix is N × N)

out   = eng.markov_generator(lambda_val=LAMBDA_VAL, x_range=X_RANGE, n_points=N)
L     = out['L']         # (N, N) generator — columns sum to zero
x     = out['x']         # grid positions
dx    = out['dx']

# Verify the column-sum property (should be ~machine epsilon)
max_col_err = np.abs(L.sum(axis=0)).max()
print(f"Max |column sum| of L:  {max_col_err:.2e}  (should be < 1e-12)")

# ---------------------------------------------------------------------------
# 3. Initial distribution — narrow Gaussian centred at x = 2.0
# ---------------------------------------------------------------------------
x0_dist = 2.0            # initial mean
sig0    = 0.4            # initial width

p0 = np.exp(-0.5 * ((x - x0_dist) / sig0) ** 2)
_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
p0 /= _trapz(p0, x)   # normalise to a proper density

# ---------------------------------------------------------------------------
# 4. Integrate dp/dt = L · p forward in time
# ---------------------------------------------------------------------------
T_MAX    = 4.0
T_REPORT = [0.0, 0.5, 1.0, 2.0, 4.0]   # snapshot times

def rhs(t, p):
    """dp/dt = L p"""
    return L @ p

sol = solve_ivp(
    rhs,
    t_span=(0.0, T_MAX),
    y0=p0,
    method='RK45',
    t_eval=T_REPORT,
    rtol=1e-6,
    atol=1e-9,
)

assert sol.success, f"ODE solver failed: {sol.message}"

# ---------------------------------------------------------------------------
# 5. Verify probability conservation at every snapshot
# ---------------------------------------------------------------------------
print("\nProbability conservation check:")
print(f"{'Time':>8s}   {'∫p dx':>12s}   {'max p':>10s}")
for i, t in enumerate(sol.t):
    p_t = sol.y[:, i]
    norm = _trapz(p_t, x)
    print(f"{t:8.2f}   {norm:12.6f}   {p_t.max():10.6f}")

# ---------------------------------------------------------------------------
# 6. Plot the evolving distribution
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: probability distributions at each snapshot
ax = axes[0]
colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(sol.t)))
for i, t in enumerate(sol.t):
    ax.plot(x, sol.y[:, i], color=colours[i], label=f"t = {t:.1f}")

# Mark the stable fixed point
fps = eng.find_fixed_points(LAMBDA_VAL)
stable = [fp for fp in fps if fp['stable']]
if stable:
    xs = stable[0]['x_star']
    ax.axvline(xs, color='crimson', linestyle='--', linewidth=1.2,
               label=f"Δ* = {xs:.2f}")

ax.set_xlabel("State Δ")
ax.set_ylabel("Probability density p(Δ, t)")
ax.set_title("Master equation: dp/dt = L · p")
ax.legend(fontsize=8)
ax.set_xlim(X_RANGE)
ax.set_ylim(bottom=0)

# Right: probability norm over time (conservation check)
norms = [_trapz(sol.y[:, i], x) for i in range(len(sol.t))]
ax2 = axes[1]
ax2.plot(sol.t, norms, 'o-', color='steelblue')
ax2.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
ax2.set_xlabel("Time t")
ax2.set_ylabel("∫ p(Δ, t) dΔ")
ax2.set_title("Probability conservation (should stay = 1)")
ax2.set_ylim(0.98, 1.02)

plt.tight_layout()
fig.savefig("/tmp/master_equation.png", dpi=120)
print("\nFigure saved to /tmp/master_equation.png")
