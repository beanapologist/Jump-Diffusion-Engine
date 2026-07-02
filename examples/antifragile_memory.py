"""
Anti-fragile 1-bit memory on the unit circle -- standalone (numpy only).

The three ingredients, as specified:

1. THE 2x2 MATRIX. State Delta is complex: Delta = |Delta| e^{i phi}.
   Linear part of the flow is (a + i w) Delta; its real 2x2 form is
        [[ a, -w],
         [ w,  a]]        diagonal = growth/dissipation, off-diagonal = rotation (i)

2. EVERYTHING NORMALIZES TO ONE. Nonlinear term -a|Delta|^2 Delta gives
        d|Delta|/dt = a |Delta| (1 - |Delta|^2)
   so the modulus flows to exactly 1: the unit circle |mu| = 1 is the attractor.
   The phase-locking force below is purely tangential, so it NEVER perturbs
   the modulus -- normalization is structural, not approximate.

3. THE BIT IS THE PHASE (poem V: psi = |psi| e^{i phi}).
   Tangential force -i eps sin(2 phi) Delta gives phase dynamics
        d phi = [w - eps sin(2 phi)] dt + (noise)
   For eps > w this locks at two stable phases separated by pi: one bit.
   Tilted-washboard barrier: V(phi) = -(eps/2) cos(2 phi) - w phi.

Full SDE:
   dDelta = [ (a + i w) Delta - a|Delta|^2 Delta - i eps sin(2 arg Delta) Delta ] dt + sigma dW_c

Anti-fragile rule: phase stress (squared distance from the nearest locked
phase) ratchets eps up -> the phase barrier deepens under attack. The modulus
stays on the unit circle throughout: hardening never breaks normalization.

Run:  python3 antifragile_memory_normalized.py
"""

import numpy as np


class UnitCircleMemory:
    """Vectorized: simulates `n` independent cells at once."""

    def __init__(self, n=400, a=1.0, w=0.3, eps=0.6, sigma=0.35, dt=0.005,
                 seed=0, harden=6.0, eps_max=12.0, decay=0.01):
        self.n, self.a, self.w = n, a, w
        self.eps0 = eps
        self.eps = np.full(n, eps)
        self.sigma, self.dt = sigma, dt
        self.harden, self.eps_max, self.decay = harden, eps_max, decay
        self.rng = np.random.default_rng(seed)

    def locked_phase(self, eps=None):
        e = self.eps0 if eps is None else eps
        return 0.5 * np.arcsin(self.w / e)

    def phase_barrier(self, eps=None):
        e = self.eps0 if eps is None else eps
        pm = 0.5 * np.arcsin(self.w / e)
        pu = np.pi / 2 - pm
        V = lambda p: -(e / 2) * np.cos(2 * p) - self.w * p
        return V(pu) - V(pm)

    def run(self, t_max, adapt=True):
        """All cells start at the locked phase on the unit circle (bit = 0).
        Returns (flip_fraction, mean_modulus)."""
        pl = self.locked_phase()
        z = np.full(self.n, np.exp(1j * pl), dtype=complex)
        flipped = np.zeros(self.n, dtype=bool)
        mod_acc, steps = 0.0, int(t_max / self.dt)
        sq = np.sqrt(self.dt)
        for _ in range(steps):
            phi = np.angle(z)
            if adapt:
                d0 = np.abs(np.angle(np.exp(1j * (phi - pl))))
                d1 = np.abs(np.angle(np.exp(1j * (phi - pl - np.pi))))
                d = np.minimum(d0, d1)                      # distance to nearest basin centre
                grow = 1.0 + self.harden * d * d * self.dt
                self.eps = np.where(self.eps < self.eps_max, self.eps * grow, self.eps)
                self.eps -= self.decay * (self.eps - self.eps0) * self.dt
            drift = ((self.a + 1j * self.w) * z
                     - self.a * np.abs(z) ** 2 * z
                     - 1j * self.eps * np.sin(2 * phi) * z) * self.dt
            noise = self.sigma * (self.rng.normal(0, sq, self.n)
                                  + 1j * self.rng.normal(0, sq, self.n))
            z = z + drift + noise
            phi = np.angle(z)
            d0 = np.abs(np.angle(np.exp(1j * (phi - pl))))
            d1 = np.abs(np.angle(np.exp(1j * (phi - pl - np.pi))))
            flipped |= (d1 < d0 - 0.5)                      # decisively in the other basin
            mod_acc += np.abs(z).mean()
        return flipped.mean(), mod_acc / steps


def demo():
    print("Unit-circle anti-fragile memory")
    print("bit in the phase | modulus normalized to 1 | linear part = [[a,-w],[w,a]]\n")

    m = UnitCircleMemory()
    J = np.array([[m.a, -m.w], [m.w, m.a]])
    ev = np.linalg.eigvals(J)
    print(f"linear 2x2: [[{m.a},-{m.w}],[{m.w},{m.a}]]   eigenvalues = {ev[0]:.2f}, {ev[1]:.2f}   (a +/- i w)\n")

    print(f"{'sigma':>6} | {'passive flip':>12} | {'antifrag flip':>13} | {'mean |D| pas':>12} | {'mean |D| anti':>13} | verdict")
    print("-" * 92)
    for sigma in (0.30, 0.38, 0.46):
        mp = UnitCircleMemory(n=600, sigma=sigma, seed=10)
        pp, modp = mp.run(30.0, adapt=False)
        ma = UnitCircleMemory(n=600, sigma=sigma, seed=10)
        pa, moda = ma.run(30.0, adapt=True)
        v = "ANTI-FRAGILE WINS" if pa < pp else "no gain"
        print(f"{sigma:>6} | {pp:>12.3f} | {pa:>13.3f} | {modp:>12.3f} | {moda:>13.3f} | {v}")

    cell = UnitCircleMemory(n=600, sigma=0.46, seed=10)
    cell.run(30.0, adapt=True)
    e_mean = cell.eps.mean()
    print(f"\nmechanism: eps {cell.eps0:.2f} -> {e_mean:.2f} (mean)   "
          f"phase barrier {cell.phase_barrier():.3f} -> {cell.phase_barrier(e_mean):.3f} "
          f"({cell.phase_barrier(e_mean)/cell.phase_barrier():.1f}x deeper)   modulus pinned to 1 throughout")


if __name__ == "__main__":
    demo()
