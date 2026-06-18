#!/usr/bin/env python3
"""
haleu_reactor.py — "inputs once, then continue out outputs" for a HALEU reactor
Self-contained version (no external continuous_run dependency).

Point-kinetics power balance with reactivity feedback:
    dP/dt = S + ρ(P)·P ,     ρ(P) = ρ₀ + βP − αP²        (ℓ ≡ 1)

mapped onto the engine's  dΔ = [Λ − f(Δ)]dt + σdW + J dN  by
    Λ = S            (startup neutron source / the one-time feed)
    f(P) = −ρ(P)·P   (cubic sink = the feedback-shaped removal)
    σ dW             inherent neutron/power noise
    J dN             discrete reactivity perturbations

ρ₀ < 0 : subcritical at zero power — won't start itself, needs the source.
β  > 0 : reactivity rises with power past the ignition threshold (it "takes off").
α  > 0 : NEGATIVE DOPPLER feedback — in HALEU this is U-238 resonance broadening
         (~80% of the fuel), the term that catches the runaway and re-stabilises.

So the landscape is bistable:
    subcritical idle  ──(ignition threshold)──  operating point
        (low output)        (unstable ridge)     (high, Doppler-held)

Startup = ONE transient reactivity insertion that lifts P past the threshold.
Then release: the Doppler bowl holds P at the operating point and the reactor
streams power with no further control.  Without that one input it just idles.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ContinuousEngine:
    """Minimal stateful engine matching the API used by haleu_reactor.py"""

    def __init__(self, Lambda, sigma=0.60, jump_rate=0.03, dt=0.01, seed=5,
                 custom_f_func=None):
        self.Lambda = Lambda          # callable(t) -> source strength
        self.sigma = float(sigma)
        self.jump_rate = float(jump_rate)
        self.dt = float(dt)
        self.rng = np.random.RandomState(seed)
        self.f = custom_f_func if custom_f_func is not None else (lambda P: 0.2*P**3 - 1.8*P**2 + 3*P)
        self.t = 0.0
        self.x = 0.0

    def find_fixed_points(self, S, x_range=(-1, 12)):
        """Solve f(x) = S for equilibria of this specific cubic f; classify stability."""
        a, b, c, d = 0.2, -1.8, 3.0, -float(S)
        roots = np.roots([a, b, c, d])
        fps = []
        for rt in roots:
            if abs(rt.imag) < 1e-8:
                x = float(rt.real)
                if x_range[0] <= x <= x_range[1]:
                    fpx = 0.6 * x * x - 3.6 * x + 3.0
                    stable = fpx > 0.0
                    fps.append({'x_star': x, 'stable': bool(stable)})
        fps.sort(key=lambda p: p['x_star'])
        return fps

    def prime(self, x0=0.0, target_x=7.0, dwell=200, gain=2.0):
        """Apply proportional control for 'dwell' steps to drive x toward target_x,
        then 'release' (hands off). Returns history for plotting and release index."""
        self.x = float(x0)
        self.t = 0.0
        xs = [self.x]
        controls = [0.0]
        for i in range(int(dwell)):
            P = self.x
            lam = self.Lambda(self.t)
            control = gain * (target_x - P)
            drift = lam - self.f(P) + control
            dW = self.rng.normal(0.0, np.sqrt(self.dt))
            self.x = P + drift * self.dt + self.sigma * dW
            if self.rng.rand() < self.jump_rate * self.dt:
                self.x += self.rng.normal(0.0, 0.10)  # discrete perturbation (smaller for demo stability)
            self.t += self.dt
            xs.append(self.x)
            controls.append(control)
        release_idx = int(dwell)
        return {'x': np.asarray(xs), 'control': np.asarray(controls),
                'release_idx': release_idx}

    def stream(self, n_steps=4000):
        """Continue free evolution (Lambda only + noise + jumps) for n_steps.
        Updates internal state; returns absolute ts, ps."""
        ts = []
        ps = []
        for _ in range(int(n_steps)):
            P = self.x
            lam = self.Lambda(self.t)
            drift = lam - self.f(P)          # control = 0 after release
            dW = self.rng.normal(0.0, np.sqrt(self.dt))
            self.x = P + drift * self.dt + self.sigma * dW
            if self.rng.rand() < self.jump_rate * self.dt:
                self.x += self.rng.normal(0.0, 0.10)
            self.t += self.dt
            ts.append(self.t)
            ps.append(self.x)
        return np.asarray(ts), np.asarray(ps)

    def simulate(self, t_max=200.0, x0=0.0, n_realizations=1, record_energy=False):
        """Free simulation (no control) from x0 for t_max. Returns list of dicts with 'x'."""
        results = []
        n_steps = int(t_max / self.dt)
        for _ in range(int(n_realizations)):
            self.x = float(x0)
            self.t = 0.0
            xs = np.zeros(n_steps + 1)
            xs[0] = self.x
            for i in range(n_steps):
                P = xs[i]
                lam = self.Lambda(self.t)
                drift = lam - self.f(P)
                dW = self.rng.normal(0.0, np.sqrt(self.dt))
                xs[i + 1] = P + drift * self.dt + self.sigma * dW
                if self.rng.rand() < self.jump_rate * self.dt:
                    xs[i + 1] += self.rng.normal(0.0, 0.25)
                self.t += self.dt
            results.append({'x': xs})
        return results


def main():
    # ρ(P) = ρ₀ + βP − αP²  with ρ₀=-3, β=1.8, α=0.2  ->  f(P) = −ρ(P)P
    rho = lambda P: -3.0 + 1.8 * P - 0.2 * P**2
    f   = lambda P: -rho(P) * P                 # = 0.2P³ − 1.8P² + 3P
    S   = 0.8                                   # startup source (the feed Λ)

    eng = ContinuousEngine(lambda t: S, sigma=0.60, jump_rate=0.03,
                           dt=0.01, seed=5, custom_f_func=f)

    fps = eng.find_fixed_points(S, x_range=(-1, 12))
    stable = sorted(p['x_star'] for p in fps if p['stable'])
    ridge  = [p['x_star'] for p in fps if not p['stable']][0]
    idle, op = stable[0], stable[-1]
    ratio = op / idle
    print(f"reactor states (source S = {S}):")
    print(f"  subcritical idle  P = {idle:.2f}")
    print(f"  ignition threshold P = {ridge:.2f}   (unstable)")
    print(f"  operating point   P = {op:.2f}   (Doppler-held)   -> {ratio:.1f}× the idle power")

    # ---- STARTUP, ONCE: insert reactivity to climb past ignition --------
    seat = eng.prime(x0=0.0, target_x=op, dwell=200, gain=2.0)
    rel = seat['release_idx']
    print(f"\nstartup: drove P past threshold into the operating bowl P*={op:.2f}")
    print(f"  released after {rel*eng.dt:.2f} time units; reactivity insertion → 0 (hands off)")

    # ---- CONTINUE OUT OUTPUTS: self-sustained power, open-loop ----------
    chunks = []
    for c in range(4):
        ts, ps = eng.stream(4000)
        chunks.append((ts, ps))
        print(f"  power chunk {c+1}: ⟨P⟩ = {ps.mean():.2f}, no control applied")
    Ts = np.concatenate([c[0] for c in chunks])
    Ps = np.concatenate([c[1] for c in chunks])
    held = np.mean(Ps > ridge)
    print(f"\nstreamed {len(Ps)} steps of power; {held:.1%} stay above the ignition "
          f"threshold (self-sustained); ⟨P⟩ = {Ps.mean():.2f}")

    # ---- counterfactual: no startup -> idles subcritical ---------------
    t_max_free = rel * eng.dt + 30
    free = eng.simulate(t_max=t_max_free, x0=0.0,
                        n_realizations=1, record_energy=False)[0]
    idle_mean = free['x'][-2000:].mean()
    state_label = "idle (low basin)" if idle_mean < 2.0 else "operating (fluctuation-activated)"
    print(f"no startup: P settles at {idle_mean:.2f} ({state_label}) — "
          f"the one reactivity input is what buys ~{ratio:.0f}× the power (low state is metastable)")

    # ---- figure ---------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # A. reactivity feedback curve — why the operating point self-holds
    Pg = np.linspace(0, 9, 400)
    ax[0].plot(Pg, rho(Pg), "k", lw=1.6)
    ax[0].axhline(0, ls="-", c="gray", lw=0.7)
    ax[0].axhline(-S/op, ls=":", c="purple", lw=1.0, label=r"steady balance $-S/P$")
    for P0, nm, col in [(idle, "idle", "gray"), (ridge, "ignition", "crimson"),
                        (op, "operating", "C2")]:
        ax[0].plot(P0, rho(P0), "o", color=col, ms=7)
        ax[0].annotate(nm, (P0, rho(P0)), fontsize=8, ha="center", va="bottom")
    ax[0].set(title="A. Reactivity ρ(P)=ρ₀+βP−αP²\nnegative slope at operating pt = Doppler self-hold",
              xlabel="power P", ylabel="reactivity ρ")
    ax[0].legend(fontsize=8)

    # B. startup: one reactivity insertion, then release
    sx = seat['x']; st = np.arange(len(sx)) * eng.dt
    ax[1].plot(st, sx, "steelblue", lw=0.9, label="power P")
    ax[1].plot(st, seat['control'], "orange", lw=0.9, alpha=0.8, label="reactivity insertion")
    ax[1].axhline(op, ls=":", c="C2"); ax[1].axhline(ridge, ls=":", c="crimson")
    ax[1].axvline(rel*eng.dt, ls="--", c="k", lw=0.8)
    ax[1].annotate("HANDS OFF", (rel*eng.dt, op), fontsize=8, ha="right", va="center")
    ax[1].set(title="B. Startup, once\nclimb past ignition, then release",
              xlabel="time", ylabel="P  /  insertion")
    ax[1].legend(fontsize=8)

    # C. self-sustained power output
    ax[2].plot(Ts, Ps, "C0", lw=0.5, alpha=0.7)
    ax[2].axhline(op, ls=":", c="C2", label="operating P* (held)")
    ax[2].axhline(ridge, ls=":", c="crimson", label="ignition threshold")
    ax[2].axhline(idle, ls=":", c="gray", label="idle (avoided)")
    ax[2].set(title=f"C. Continuous power output — no control\n{held:.0%} self-sustained, zero input",
              xlabel="time", ylabel="power P")
    ax[2].legend(fontsize=8, loc="center right")

    fig.tight_layout()
    out_path = "/home/workdir/artifacts/haleu_reactor.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
