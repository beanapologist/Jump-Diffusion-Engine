"""
fock_ladder_generator.py — Column generator for a Fock-space ladder

Demonstrates five properties of the Markov generator matrix L for a
truncated quantum harmonic oscillator / photon-number ladder:

  dp/dt = L p,   with the column-generator convention  1^T L = 0

The single-step transitions are standard bosonic ladder rates:
  W_{n-1 <- n} = gd * n          (emission   / stimulated decay)
  W_{n+1 <- n} = gu * (n+1)      (absorption / stimulated creation)

Optional pair processes add dn=±2 jumps:
  W_{n-2 <- n} = pair_dn * n*(n-1)       (two-photon loss)
  W_{n+2 <- n} = pair_up * (n+1)*(n+2)   (two-photon gain)

Demonstrations
--------------
1. COLUMN GENERATOR, dn=±1 only
   - Columns sum to zero (1^T L = 0)
   - Matrix is tridiagonal (bandwidth ≤ 1)
   - Stationary distribution is geometric: p_n ∝ r^n, r = gu/gd

2. BOUNDARY TERMS — truncation matters
   - Lower wall n=0: emission rate gd*0 = 0  → natural reflecting wall
   - Upper wall n=N-1: absorption would leave space, so it is dropped → reflecting
   - n_bar converges to r/(1-r) as N → ∞; finite-N bias is truncation, not physics

3. PAIR PROCESSES: dn=±2 widens the kernel support
   - Bandwidth becomes 2; 1^T L = 0 still holds

4. PARITY: pure dn=±2 splits Fock space into even/odd sectors
   - Off-diagonal even↔odd block is exactly zero: parity is conserved
   - Squeezing / two-photon processes cannot change photon-number parity

5. DETAILED BALANCE with both dn=±1 and dn=±2 present
   - Net current on each bond J_{a→b} = L[b,a]*p[a] − L[a,b]*p[b]
   - Non-zero currents signal irreversibility when competing processes coexist

Run:
    python3 fock_ladder_generator.py
"""

import numpy as np


def L_gen(N, gd, gu, pair_up=0.0, pair_dn=0.0):
    """
    Build the Markov generator matrix for a truncated Fock-space ladder.

    Convention: L[n, m] = W_{n←m}  (off-diagonal, ≥ 0),
                L[m, m] = −Σ_{n≠m} W_{n←m}  (diagonal, ≤ 0).

    This is the *column-generator* form: dp/dt = L p, 1^T L = 0.

    Parameters
    ----------
    N        : int   — number of Fock states  {0, 1, …, N-1}
    gd       : float — single-photon loss rate (emission)
    gu       : float — single-photon gain rate (absorption)
    pair_up  : float — two-photon gain rate (dn = +2)
    pair_dn  : float — two-photon loss rate (dn = −2)

    Returns
    -------
    L : (N, N) ndarray
    """
    L = np.zeros((N, N))

    def add(dest, src, rate):
        """Add a transition src→dest with given rate (ignored if out of bounds)."""
        if 0 <= dest < N and rate > 0:
            L[dest, src] += rate
            L[src, src]  -= rate

    for n in range(N):
        add(n - 1, n, gd * n)                  # emission:    dn = −1
        add(n + 1, n, gu * (n + 1))             # absorption:  dn = +1
        add(n - 2, n, pair_dn * n * (n - 1))   # pair loss:   dn = −2
        add(n + 2, n, pair_up * (n + 1) * (n + 2))  # pair gain: dn = +2

    return L


def stationary(L):
    """Return the normalised stationary distribution (eigenvector for eigenvalue 0)."""
    w, v = np.linalg.eig(L)
    i = np.argmin(np.abs(w))
    p = np.real(v[:, i])
    p /= p.sum()
    return p


def net_current(p, L, a, b):
    """Net probability current on bond a↔b: J = L[b,a]*p[a] − L[a,b]*p[b]."""
    return L[b, a] * p[a] - L[a, b] * p[b]


# ---------------------------------------------------------------------------
# 1. COLUMN GENERATOR, dn=±1 only
# ---------------------------------------------------------------------------
N = 8
gd, gu = 1.0, 0.3

L = L_gen(N, gd, gu)
print("1. COLUMN GENERATOR, dn=+-1 only")
print("   1^T L = ", np.round(L.sum(0), 12))

idx = np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
bw_ok = np.all(idx[np.abs(L) > 1e-12] <= 1)
print(f"   bandwidth: nonzero only on |n-m|<=1 -> {bw_ok}")

p = stationary(L)
r = gu / gd
th = r ** np.arange(N)
th /= th.sum()
print(f"   stationary matches geometric r^n: {np.allclose(p, th)}")

# ---------------------------------------------------------------------------
# 2. BOUNDARY TERMS — truncation matters
# ---------------------------------------------------------------------------
print("\n2. BOUNDARY TERMS — the truncation matters")
print("   at n=0: emission rate gd*0 = 0 (no leak below vacuum)  -> reflecting  OK")
print(f"   at n=N-1: absorption gu*N = {gu*N:.2f} would leave the space; we DROP it.")
print("   dropping = reflecting wall. This is why n_bar was slightly below r/(1-r):")

nbar = np.sum(np.arange(N) * p)
print(f"   n_bar(N={N}) = {nbar:.6f}   vs   r/(1-r) = {r/(1-r):.6f}")

for NN in (8, 16, 32, 64):
    LL = L_gen(NN, gd, gu)
    pp = stationary(LL)
    print(f"     N={NN:>3}: n_bar={np.sum(np.arange(NN)*pp):.9f}")

print(f"   -> converges to {r/(1-r):.9f} as the wall recedes. Truncation, not physics.")

# ---------------------------------------------------------------------------
# 3. PAIR PROCESSES: dn=±2 widens the support of W
# ---------------------------------------------------------------------------
print("\n3. PAIR PROCESSES: dn=+-2 widens the support of W")
L2 = L_gen(N, gd, gu, pair_up=0.02, pair_dn=0.05)
bw = np.max(idx[np.abs(L2) > 1e-12])
print(f"   bandwidth now = {bw} (was 1). Kernel support = {{+-1, +-2}}")
print("   1^T L = ", np.round(L2.sum(0), 12))

# ---------------------------------------------------------------------------
# 4. PARITY: pure dn=±2 splits Fock space into even/odd sectors
# ---------------------------------------------------------------------------
print("\n4. PARITY: pure dn=+-2 splits Fock space into even/odd sectors")
Lp = L_gen(N, 0, 0, pair_up=0.02, pair_dn=0.05)
ev = np.arange(0, N, 2)
od = np.arange(1, N, 2)
cross = np.abs(Lp[np.ix_(od, ev)]).max()
print(f"   coupling even->odd block: {cross:.3e}")
print("   -> exactly zero: parity is CONSERVED. Two disconnected ladders.")
print("      Squeezing/two-photon processes cannot change photon-number parity.")

# ---------------------------------------------------------------------------
# 5. DETAILED BALANCE with dn=±1 and dn=±2 present
# ---------------------------------------------------------------------------
print("\n5. DETAILED BALANCE with dn=+-2 present")
p2 = stationary(L2)
print("   net current on each bond (0 = reversible):")
for n in range(4):
    j1 = net_current(p2, L2, n, n + 1)
    j2 = net_current(p2, L2, n, n + 2)
    print(f"     {n}->{n+1}: {j1:+.3e}    {n}->{n+2}: {j2:+.3e}")
