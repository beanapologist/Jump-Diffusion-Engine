"""
closure.py — Phase/log closure as a memory-integrity primitive.
CPU (NumPy) and GPU (CuPy/PyTorch) ready.

The notebook's identity:  A = B*C  =>  log A - log B - log C = 0
                                   =>  arg A - arg B - arg C = 0 (mod 2*pi)
                                   =>  A/(B*C) = 1  (closure to the origin)

Instead of guarding stored values with thresholds (gapmem), guard the
RELATIONS between them. If the pipeline guarantees A = B@C, then the
closure residual is an exact invariant: zero means intact, nonzero means
silent corruption — no ground truth, no tuned barrier, and common gain
errors cancel (g*A vs g*B leaves the closure of ratios untouched).

This is used in production today under other names:
  * Freivalds' check — verify A@B = C in O(n^2) instead of O(n^3)
  * ABFT checksums   — checksum rows/cols ride through the matmul
  * VLBI closure phase — antenna phase errors cancel in triangle sums
Silent data corruption in fleet GPUs/CPUs is a real, documented failure
mode; closure checks catch it at a few percent overhead.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Freivalds closure check: does A @ B == C ?  Cost O(k n^2), not O(n^3)
# ---------------------------------------------------------------------------
def freivalds(A, B, C, k=3, rtol=1e-6, xp=np, rng=None):
    """Probabilistic verification of the closure A = B @ C.

    Projects the residual onto k random +-1 vectors:
        r = A @ v - B @ (C @ v)
    If C is correct, r = 0 exactly. If corrupted, P(miss) <= 2^-k.
    Returns (ok, max_residual).
    """
    rng = rng or np.random.default_rng()
    n = B.shape[1]
    scale = xp.abs(C).max() + 1e-300
    worst = 0.0
    for _ in range(k):
        v = rng.choice(np.asarray([-1.0, 1.0]), size=n)
        v = xp.asarray(v)
        r = A @ v - B @ (C @ v)
        worst = max(worst, float(xp.abs(r).max()) / float(scale))
        if worst > rtol:
            return False, worst
    return True, worst


# ---------------------------------------------------------------------------
# 2. Log/phase closure residuals for stored triples (A, B, C) with A = B*C
# ---------------------------------------------------------------------------
def log_closure(A, B, C, xp=np):
    """log|A| - log|B| - log|C|  (elementwise, 0 iff magnitudes close)."""
    return xp.log(xp.abs(A)) - xp.log(xp.abs(B)) - xp.log(xp.abs(C))


def phase_closure(A, B, C, xp=np):
    """arg A - arg B - arg C, wrapped to (-pi, pi].

    The wrap is the winding-aware step the notebook flags: tan(phi) = -1
    alone cannot distinguish 3*pi/4 from -pi/4; only the two-argument
    angle (and the mod 2*pi) closes the triangle correctly.
    """
    res = xp.angle(A) - xp.angle(B) - xp.angle(C)
    return (res + np.pi) % (2 * np.pi) - np.pi


def closure_ok(A, B, C, tol=1e-8, xp=np):
    """Full closure: A/(B*C) = 1, i.e. both residuals at the origin."""
    lr = log_closure(A, B, C, xp=xp)
    pr = phase_closure(A, B, C, xp=xp)
    return bool(xp.abs(lr).max() < tol and xp.abs(pr).max() < tol), \
        float(xp.abs(lr).max()), float(xp.abs(pr).max())


# ---------------------------------------------------------------------------
# 3. Closure-guarded memory: store relations, not just values
# ---------------------------------------------------------------------------
class ClosureMemory:
    """Holds matrices with a declared closure A = B @ C.

    write(): stores B, C and the computed product A plus checksum vectors.
    read():  O(n^2) Freivalds verification before the value is trusted;
             on failure, recompute A from B, C (self-healing) and report.
    """

    def __init__(self, k=3, xp=np, rng=None):
        self.k = k
        self.xp = xp
        self.rng = rng or np.random.default_rng()
        self._store = {}

    def write(self, key, B, C):
        A = B @ C
        self._store[key] = (A, B, C)
        return A

    def read(self, key):
        A, B, C = self._store[key]
        ok, res = freivalds(A, B, C, k=self.k, xp=self.xp, rng=self.rng)
        if ok:
            return A, {"verified": True, "residual": res, "healed": False}
        A_fixed = B @ C                       # closure broken: recompute
        self._store[key] = (A_fixed, B, C)
        return A_fixed, {"verified": False, "residual": res, "healed": True}

    # deliberate corruption hook for testing
    def _flip_bit(self, key, i, j, bit=40):
        A, B, C = self._store[key]
        A = A.copy()
        u = A[i, j].view() if hasattr(A[i, j], 'view') else A[i, j]
        raw = np.float64(A[i, j]).view(np.uint64) ^ np.uint64(1 << bit)
        A[i, j] = raw.view(np.float64)
        self._store[key] = (A, B, C)


# ---------------------------------------------------------------------------
# 4. The semantic triangle:  A = median, B = source, C = sink
#    Stationary balance of  d(Delta) = [B - C*Delta] dt + sigma dW :
#        median(Delta) = source / sink   =>   A * C = B
#    Closure residual:  log A + log C - log B = 0
#    Median (not mean): robust to the jump term J dN.
#    Residual drift is an early warning of the fold (sink -> 0 is 1/0).
# ---------------------------------------------------------------------------
class BalanceMonitor:
    """Streams Delta samples; checks the median-source-sink closure."""

    def __init__(self, source, sink, xp=np):
        self.B = float(source)   # epsilon, the drive
        self.C = float(sink)     # k, the dissipation
        self.xp = xp
        self.residuals = []

    def closure_residual(self, block):
        """log(median) + log(sink) - log(source); 0 iff balance holds."""
        xp = self.xp
        A = float(xp.median(xp.asarray(block)))
        if A <= 0 or self.B <= 0 or self.C <= 0:
            return float('inf')  # sign break: the triangle has torn
        r = np.log(A) + np.log(self.C) - np.log(self.B)
        self.residuals.append(r)
        return r

    def fold_warning(self, window=5, tol=0.1):
        """Rising |residual| = the sink is failing = approaching 1/0."""
        if len(self.residuals) < window:
            return False
        recent = np.abs(np.asarray(self.residuals[-window:]))
        return bool(np.all(np.diff(recent) > 0) or recent[-1] > tol)
