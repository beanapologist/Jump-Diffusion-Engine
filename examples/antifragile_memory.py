import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Dict
import warnings

class JumpDiffusionEngineAntiDeg:
    """
    Bistable memory cell with anti‑degradation (antifragile) dynamics.

    SDE:
        dΔ = [Λ(t) − f(Δ)] dt + σ dW + J dN
    with f(Δ) = k Δ³ − g Δ   (odd cubic, double‑well).

    Anti‑degradation rules (when use_adaptive=True):
        dg/dt = -β (g - g0) + α (Δ - Δ*)²
        dk/dt =  γ (k_max - k) (Δ - Δ*)²
    where Δ* is the stable fixed point nearest to the current position.
    Stress accumulator: dS = (Δ - Δ*)² dt.
    """

    def __init__(self,
                 lambda_func: Callable[[float], float],
                 sigma: float = 0.5,
                 jump_rate: float = 0.05,
                 jump_size_dist: Optional[Callable] = None,
                 dt: float = 0.01,
                 seed: int = 42,
                 # Bistable parameters
                 k: float = 0.8,
                 g: float = 0.5,
                 # Anti‑degradation parameters
                 g0: Optional[float] = None,
                 k_max: float = 5.0,
                 stress_alpha: float = 0.1,
                 stress_beta: float = 0.05,
                 harden_rate: float = 0.02,
                 ):
        self.lambda_func = lambda_func
        self.sigma = sigma
        self.jump_rate = jump_rate
        self.jump_size_dist = jump_size_dist
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # Cubic sink parameters
        self.k = k
        self.g = g
        self.g0 = g0 if g0 is not None else g   # relaxation target
        self.k_max = k_max

        # Anti‑degradation rates
        self.stress_alpha = stress_alpha
        self.stress_beta = stress_beta
        self.harden_rate = harden_rate

        # Internal state for stress accumulation
        self.S = 0.0

        self.f_func = self._cubic_sink

    def _cubic_sink(self, x: float) -> float:
        return self.k * x**3 - self.g * x

    # ----------------------------------------------------------------------
    # Analytic cubic solver with numerical guard
    # ----------------------------------------------------------------------
    @staticmethod
    def _cubic_roots(k: float, g: float, lam: float) -> List[Dict]:
        """
        Solve k x^3 - g x - lam = 0 analytically.
        Returns list of {'x_star', 'stable', 'f_prime'}.
        """
        if abs(k) < 1e-12:
            if abs(g) > 1e-12:
                x = -lam / g
                f_prime = -g
                return [{'x_star': x, 'stable': f_prime > 0, 'f_prime': f_prime}]
            return []

        p = -g / k
        q = -lam / k
        disc = (q/2)**2 + (p/3)**3

        roots = []
        if disc > 1e-12:
            # One real root
            sqrt_disc = np.sqrt(disc)
            u = (-q/2 + sqrt_disc) ** (1/3)
            v = (-q/2 - sqrt_disc) ** (1/3)
            x_real = u + v
            f_prime = 3 * k * x_real**2 - g
            roots.append({'x_star': x_real, 'stable': f_prime > 0, 'f_prime': f_prime})
        elif abs(disc) <= 1e-12:
            # Multiple roots
            u = (-q/2) ** (1/3) if q <= 0 else -((-q/2) ** (1/3))
            x1 = 2*u
            x2 = -u
            for x in set([x1, x2]):
                f_prime = 3 * k * x**2 - g
                roots.append({'x_star': x, 'stable': f_prime > 0, 'f_prime': f_prime})
        else:
            # Three distinct real roots: trigonometric with guard
            r = 2 * np.sqrt(-p/3)
            # Guard against floating-point overflow/domain errors
            arg = (3*q) / (2*p) * np.sqrt(-3/p)
            arg = np.clip(arg, -1.0, 1.0)
            theta = np.arccos(arg) / 3
            for i in range(3):
                x = r * np.cos(theta - 2*np.pi*i/3)
                f_prime = 3 * k * x**2 - g
                roots.append({'x_star': x, 'stable': f_prime > 0, 'f_prime': f_prime})

        # Deduplicate
        roots.sort(key=lambda d: d['x_star'])
        unique = []
        for r in roots:
            if not any(np.isclose(r['x_star'], u['x_star'], atol=1e-8) for u in unique):
                unique.append(r)
        return unique

    def find_fixed_points(self, lambda_val: float,
                         x_range: Tuple[float, float] = (-20, 20),
                         n_points: int = 2000) -> List[Dict]:
        """Fast analytic solver – x_range and n_points kept for API compatibility."""
        return self._cubic_roots(self.k, self.g, lambda_val)

    # ----------------------------------------------------------------------
    # Potential and basin depth
    # ----------------------------------------------------------------------
    def potential(self, x_grid: np.ndarray, lambda_val: float) -> np.ndarray:
        return (self.k / 4.0) * x_grid**4 - (self.g / 2.0) * x_grid**2 - lambda_val * x_grid

    def basin_depth(self, lambda_val: float, x_range: Tuple[float, float] = (-15, 15), n_points: int = 5000) -> List[Dict]:
        fps = self.find_fixed_points(lambda_val)
        minima = [fp for fp in fps if fp['stable']]
        maxima = [fp for fp in fps if not fp['stable']]

        basins = []
        for min_fp in minima:
            xm = min_fp['x_star']
            left_bar = max([m['x_star'] for m in maxima if m['x_star'] < xm], default=None)
            right_bar = min([m['x_star'] for m in maxima if m['x_star'] > xm], default=None)

            Vm = self.potential(np.array([xm]), lambda_val)[0]
            depths = []
            if left_bar is not None:
                Vl = self.potential(np.array([left_bar]), lambda_val)[0]
                depths.append(Vl - Vm)
            if right_bar is not None:
                Vr = self.potential(np.array([right_bar]), lambda_val)[0]
                depths.append(Vr - Vm)

            if depths:
                basins.append({
                    "x_star": float(xm),
                    "depth": float(min(depths)),
                    "lambda_val": lambda_val
                })
        return basins

    # ----------------------------------------------------------------------
    # Main simulation (bug‑fixed, fast, with optional g/k recording)
    # ----------------------------------------------------------------------
    def simulate(self, t_max: float, x0: float = 0.0,
                 n_realizations: int = 1,
                 record_energy: bool = True,
                 lambda_val_for_energy: Optional[float] = None,
                 use_adaptive: bool = False,
                 record_stress: bool = False,
                 record_gk: bool = False) -> List[Dict]:
        dt = self.dt
        n_steps = int(t_max / dt) + 1
        results = []

        k_initial = self.k
        g_initial = self.g

        for r in range(n_realizations):
            t = np.linspace(0, t_max, n_steps)
            x = np.zeros(n_steps)
            x[0] = x0
            energy = np.zeros(n_steps) if record_energy else None
            stress_hist = np.zeros(n_steps) if record_stress else None
            g_hist = np.zeros(n_steps) if record_gk else None
            k_hist = np.zeros(n_steps) if record_gk else None

            self.S = 0.0
            self.g = g_initial
            self.k = k_initial

            for i in range(1, n_steps):
                t_prev = t[i-1]
                x_prev = x[i-1]
                lam_now = self.lambda_func(t_prev)

                if use_adaptive:
                    fps = self._cubic_roots(self.k, self.g, lam_now)
                    stable = [fp for fp in fps if fp['stable']]
                    if stable:
                        x_star = min(stable, key=lambda fp: abs(fp['x_star'] - x_prev))['x_star']
                    else:
                        x_star = 0.0

                    stress = (x_prev - x_star)**2
                    self.S += stress * dt

                    dg = (-self.stress_beta * (self.g - self.g0) + self.stress_alpha * stress) * dt
                    dk = self.harden_rate * (self.k_max - self.k) * stress * dt
                    self.g += dg
                    self.k += dk
                    self.g = max(self.g, 1e-6)
                    self.k = max(self.k, 0.01)

                    if record_stress:
                        stress_hist[i] = self.S
                    if record_gk:
                        g_hist[i] = self.g
                        k_hist[i] = self.k

                drift = (lam_now - self._cubic_sink(x_prev)) * dt
                dW = self.rng.normal(0, np.sqrt(dt))
                diffusion = self.sigma * dW
                jumps = 0.0
                if self.jump_rate > 0:
                    n_jumps = self.rng.poisson(self.jump_rate * dt)
                    for _ in range(n_jumps):
                        jumps += self.rng.normal(0, 1.0) if self.jump_size_dist is None else self.jump_size_dist()

                x[i] = x_prev + drift + diffusion + jumps

                if record_energy:
                    if lambda_val_for_energy is not None:
                        energy[i] = self.potential(np.array([x[i]]), lambda_val_for_energy)[0]
                    else:
                        energy[i] = self.potential(np.array([x[i]]), lam_now)[0]

            res = {'t': t, 'x': x}
            if record_energy and energy is not None:
                res['energy'] = energy
            if record_stress and stress_hist is not None:
                res['stress'] = stress_hist
                res['final_S'] = self.S
            if record_gk and g_hist is not None:
                res['g_hist'] = g_hist
                res['k_hist'] = k_hist
                res['final_g'] = self.g
                res['final_k'] = self.k

            results.append(res)

            # Restore parameters for next realisation
            self.g = g_initial
            self.k = k_initial
            self.S = 0.0

        return results

    # ----------------------------------------------------------------------
    # New method: visualise evolving potential from recorded g/k histories
    # ----------------------------------------------------------------------
    def plot_evolving_potential(self, result: Dict, 
                                snapshots: Optional[List[float]] = None,
                                x_range: Tuple[float, float] = (-4, 4),
                                n_pts: int = 500,
                                figsize: Tuple[int, int] = (10, 6)):
        """
        Plot the quartic potential V(x; g(t), k(t)) at selected time snapshots.
        result must contain 't', 'g_hist', 'k_hist' (from simulate with record_gk=True).
        If snapshots is None, uses np.linspace(0, len(t)-1, 5) int indices.
        """
        if 'g_hist' not in result or 'k_hist' not in result:
            raise ValueError("result must contain 'g_hist' and 'k_hist' (set record_gk=True in simulate)")

        t = result['t']
        g_hist = result['g_hist']
        k_hist = result['k_hist']
        x = result['x']  # optional, for overlay

        if snapshots is None:
            idxs = np.linspace(0, len(t)-1, 5, dtype=int)
        else:
            idxs = [int(np.argmin(np.abs(t - tau))) for tau in snapshots]

        x_grid = np.linspace(x_range[0], x_range[1], n_pts)
        fig, ax = plt.subplots(figsize=figsize)

        for idx in idxs:
            g_now = g_hist[idx]
            k_now = k_hist[idx]
            lam_now = self.lambda_func(t[idx])
            # Potential uses current k,g and lambda (which is encoded in the drift)
            # For visualisation we plot V(x) = (k/4)x^4 - (g/2)x^2 - lambda*x
            # But lambda is time‑varying; we take the value at that time.
            V = (k_now / 4.0) * x_grid**4 - (g_now / 2.0) * x_grid**2 - lam_now * x_grid
            # Shift so minimum is at 0 for clarity
            V -= np.min(V)
            ax.plot(x_grid, V, label=f"t={t[idx]:.1f}, g={g_now:.2f}, k={k_now:.2f}")

        # Optionally overlay the trajectory if x is in result
        if 'x' in result:
            ax2 = ax.twinx()
            ax2.plot(t, result['x'], color='gray', alpha=0.3, label='trajectory')
            ax2.set_ylabel('Δ(t)', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')

        ax.set_xlabel('Δ')
        ax.set_ylabel('V(Δ) (shifted)')
        ax.set_title('Evolving double‑well potential under anti‑degradation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------------
    # Other methods (unchanged)
    # ----------------------------------------------------------------------
    def identify_boundary(self, lambda_val: Optional[float] = None,
                          x_range: Tuple[float, float] = (-20, 20),
                          n_points: int = 4000,
                          target_x: Optional[float] = None) -> Dict:
        lam = lambda_val if lambda_val is not None else self.lambda_func(0)
        fps = self.find_fixed_points(lam)
        if not fps:
            return {'x_star': None, 'left_wall': None, 'right_wall': None,
                    'half_width': None, 'sigma_envelope': None, 'all_fixed': []}

        stable = [fp for fp in fps if fp['stable']]
        unstable = [fp for fp in fps if not fp['stable']]

        if target_x is not None and stable:
            x_star = min(stable, key=lambda fp: abs(fp['x_star'] - target_x))
        elif stable:
            x_star = max(stable, key=lambda fp: fp['f_prime'])
        else:
            x_star = fps[0]
        xs = x_star['x_star']
        fprime = x_star['f_prime']

        below = [fp['x_star'] for fp in unstable if fp['x_star'] < xs]
        above = [fp['x_star'] for fp in unstable if fp['x_star'] > xs]
        left_wall = max(below) if below else None
        right_wall = min(above) if above else None

        gaps = []
        if left_wall is not None:
            gaps.append(xs - left_wall)
        if right_wall is not None:
            gaps.append(right_wall - xs)
        half_width = float(min(gaps)) if gaps else None

        env = float(np.sqrt(self.sigma**2 / (2 * fprime))) if fprime > 0 else None

        return {
            'x_star': float(xs),
            'left_wall': float(left_wall) if left_wall is not None else None,
            'right_wall': float(right_wall) if right_wall is not None else None,
            'half_width': half_width,
            'sigma_envelope': env,
            'all_fixed': fps,
        }

    def seat_and_release(self, t_max: float, x0: float = 0.0,
                         lambda_val: Optional[float] = None,
                         target_x: Optional[float] = None,
                         settle_tol: Optional[float] = None,
                         dwell: int = 200,
                         gain: float = 2.0) -> Dict:
        b = self.identify_boundary(lambda_val, target_x=target_x)
        xs = b['x_star']
        if xs is None:
            raise ValueError("No stable bowl found for these inputs; nothing to seat into.")

        lam = lambda_val if lambda_val is not None else self.lambda_func(0)
        if settle_tol is None:
            settle_tol = (b['sigma_envelope'] or 0.5) * 1.5

        dt = self.dt
        n_steps = int(t_max / dt) + 1
        x = np.zeros(n_steps)
        x[0] = x0
        control = np.zeros(n_steps)
        released = False
        release_idx = None
        inside_run = 0

        for i in range(1, n_steps):
            xp = x[i-1]
            u = 0.0
            if not released:
                if abs(xp - xs) <= settle_tol:
                    inside_run += 1
                    if inside_run >= dwell:
                        released = True
                        release_idx = i
                else:
                    inside_run = 0
                    if abs(xp - xs) > settle_tol:
                        u = -gain * (xp - xs)
            drift = (self.lambda_func((i-1)*dt) - self._cubic_sink(xp) + u) * dt
            dW = self.rng.normal(0, np.sqrt(dt))
            xnew = xp + drift + self.sigma * dW
            if self.jump_rate > 0:
                n_j = self.rng.poisson(self.jump_rate * dt)
                for _ in range(n_j):
                    xnew += self.rng.normal(0, 1.0) if self.jump_size_dist is None else self.jump_size_dist()
            x[i] = xnew
            control[i] = u

        post = x[release_idx:] if release_idx is not None else np.array([])
        contained = bool(np.all(np.abs(post - xs) <= (b['half_width'] or np.inf))) if post.size else None

        return {
            't': np.linspace(0, t_max, n_steps),
            'x': x,
            'control': control,
            'boundary': b,
            'x_star': xs,
            'settle_tol': float(settle_tol),
            'release_idx': release_idx,
            'released': released,
            'control_after_release': float(np.sum(np.abs(control[release_idx:]))) if release_idx else None,
            'contained_after_release': contained,
        }

    def stationary_density(self, lambda_val: float, x_range: Tuple[float, float] = (-10, 10), n_points: int = 2000):
        x = np.linspace(x_range[0], x_range[1], n_points)
        V = self.potential(x, lambda_val)
        V_shifted = V - np.min(V)
        p = np.exp(-2 * V_shifted / (self.sigma ** 2 + 1e-12))
        norm = np.trapezoid(p, x) if hasattr(np, 'trapezoid') else np.trapz(p, x)
        p /= (norm + 1e-12)
        return x, p

    def escape_probability(self, threshold: float, t_max: float, x0: Optional[float] = None,
                           x_star: Optional[float] = None, n_trials: int = 500) -> float:
        if x_star is None:
            fps = self.find_fixed_points(self.lambda_func(0))
            stable = [fp for fp in fps if fp['stable']]
            x_star = stable[0]['x_star'] if stable else 0.0
        if x0 is None:
            x0 = x_star
        escapes = 0
        for _ in range(n_trials):
            res = self.simulate(t_max=t_max, x0=x0, n_realizations=1, record_energy=False, use_adaptive=False)[0]
            if np.any(np.abs(res["x"] - x_star) > threshold):
                escapes += 1
        return escapes / n_trials

    def parameter_sweep(self, param_name: str, param_values: np.ndarray,
                        lambda_val: Optional[float] = None,
                        metrics: Optional[List[str]] = None,
                        x_range: Tuple[float, float] = (-15, 15),
                        n_points: int = 2000, plot: bool = True) -> Dict:
        if metrics is None:
            metrics = ['fixed_points', 'basin_depth', 'escape_prob', 'stationary_mass']

        results = []
        original_k, original_g = self.k, self.g

        for val in param_values:
            if param_name == 'k':
                self.k = val
            elif param_name == 'g':
                self.g = val
            else:
                setattr(self, param_name, val)

            sweep_lambda = lambda_val if lambda_val is not None else self.lambda_func(0)
            sweep_res = {'param_value': val, 'param_name': param_name}

            try:
                fps = self.find_fixed_points(sweep_lambda)
                stable_fps = [fp for fp in fps if fp['stable']]
                sweep_res['num_stable'] = len(stable_fps)
                sweep_res['positivity_ok'] = len(stable_fps) >= 2

                if 'basin_depth' in metrics:
                    basins = self.basin_depth(sweep_lambda)
                    sweep_res['avg_basin_depth'] = np.mean([b['depth'] for b in basins]) if basins else 0.0

                if 'escape_prob' in metrics:
                    x_star_esc = stable_fps[0]['x_star'] if stable_fps else None
                    esc = self.escape_probability(threshold=5.0, t_max=100.0,
                                                  x_star=x_star_esc, n_trials=100)
                    sweep_res['escape_prob'] = esc

                if 'stationary_mass' in metrics:
                    x_sd, p = self.stationary_density(sweep_lambda)
                    if stable_fps:
                        x0c = stable_fps[0]['x_star']
                        bowl = np.abs(x_sd - x0c) <= 5.0
                        mass = (np.trapezoid(p[bowl], x_sd[bowl]) if hasattr(np, 'trapezoid')
                                else np.trapz(p[bowl], x_sd[bowl]))
                    else:
                        mass = (np.trapezoid(p, x_sd) if hasattr(np, 'trapezoid')
                                else np.trapz(p, x_sd))
                    sweep_res['stationary_mass_in_bowl'] = float(mass)
            except Exception as e:
                sweep_res['error'] = str(e)

            results.append(sweep_res)

            self.k, self.g = original_k, original_g

        if plot:
            self._plot_sweep(results, param_name, metrics)

        return {'sweep': results, 'param_name': param_name}

    def _plot_sweep(self, results: List[Dict], param_name: str, metrics: List[str]):
        plt.figure(figsize=(12, 8))
        vals = [r['param_value'] for r in results]
        if any(m in metrics for m in ['basin_depth', 'avg_basin_depth']):
            depths = [r.get('avg_basin_depth', 0) for r in results]
            plt.subplot(211)
            plt.plot(vals, depths, 'g-o')
            plt.xlabel(param_name)
            plt.ylabel('Avg Basin Depth')
            plt.grid(True)
        if 'escape_prob' in metrics:
            probs = [r.get('escape_prob', 0) for r in results]
            plt.subplot(212)
            plt.plot(vals, probs, 'r-o')
            plt.xlabel(param_name)
            plt.ylabel('Escape Prob')
            plt.grid(True)
        plt.suptitle(f'Sweep on {param_name}')
        plt.tight_layout()
        plt.show()

    def plot_trajectories(self, results: List[Dict], title: str = "Jump-Diffusion Trajectories", show_energy: bool = False):
        has_energy = show_energy and any('energy' in r for r in results)
        n_panels = 2 if has_energy else 1
        fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), squeeze=False)
        ax_traj = axes[0, 0]

        for i, r in enumerate(results):
            label = f"Run {i + 1}" if len(results) > 1 else None
            ax_traj.plot(r['t'], r['x'], lw=0.8, alpha=0.7, label=label)

        ax_traj.set_xlabel("Time")
        ax_traj.set_ylabel("Δ")
        ax_traj.set_title(title)
        ax_traj.grid(True, alpha=0.3)
        if len(results) > 1:
            ax_traj.legend(fontsize=8)

        if has_energy:
            ax_en = axes[1, 0]
            for i, r in enumerate(results):
                if 'energy' in r:
                    label = f"Run {i + 1}" if len(results) > 1 else None
                    ax_en.plot(r['t'], r['energy'], lw=0.8, alpha=0.7, label=label)
            ax_en.set_xlabel("Time")
            ax_en.set_ylabel("Energy V(Δ)")
            ax_en.set_title("Potential Energy")
            ax_en.grid(True, alpha=0.3)
            if len(results) > 1:
                ax_en.legend(fontsize=8)

        plt.tight_layout()
        plt.show()
        return fig


# ======================================================================
# Demo and cubic solver test
# ======================================================================
if __name__ == "__main__":
    # ---------- quick test of the analytic solver ----------
    def test_cubic_solver():
        engine = JumpDiffusionEngineAntiDeg(lambda_func=lambda t: 0.0, k=1.0, g=1.0)
        # At lambda=0, roots should be x=0 (unstable) and x=±1 (stable)
        roots = engine._cubic_roots(1.0, 1.0, 0.0)
        expected = sorted([-1.0, 0.0, 1.0])
        got = sorted([r['x_star'] for r in roots])
        assert np.allclose(got, expected, atol=1e-8), f"Expected {expected}, got {got}"
        # Stability: x=0 unstable, ±1 stable
        for r in roots:
            if abs(r['x_star']) < 0.1:
                assert not r['stable'], "x=0 should be unstable"
            else:
                assert r['stable'], "±1 should be stable"
        print("Cubic solver test passed.")

    test_cubic_solver()

    # ---------- main demo ----------
    def lambda_func(t):
        return 0.5 + 0.3 * np.sin(2 * np.pi * 0.1 * t)

    engine = JumpDiffusionEngineAntiDeg(
        lambda_func,
        k=0.8, g=0.5,
        g0=0.5, k_max=3.0,
        stress_alpha=0.2, stress_beta=0.05, harden_rate=0.02,
        sigma=0.4, dt=0.01
    )

    # Simulate with adaptation, record g and k histories
    results = engine.simulate(
        t_max=50.0, x0=1.0,
        n_realizations=1,
        use_adaptive=True,
        record_stress=True,
        record_gk=True,
        record_energy=False
    )
    res = results[0]
    print(f"Final stress S = {res['final_S']:.2f}")
    print(f"Final g = {res['final_g']:.2f}, final k = {res['final_k']:.2f}")

    # Plot trajectory, stress, and parameters
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax1.plot(res['t'], res['x'])
    ax1.set_ylabel('Δ')
    ax1.set_title('Trajectory with anti‑degradation')
    ax1.grid(True)

    ax2.plot(res['t'], res['stress'])
    ax2.set_ylabel('Cumulative stress S')
    ax2.set_title('Stress accumulation')
    ax2.grid(True)

    ax3.plot(res['t'], res['g_hist'], label='g(t)')
    ax3.plot(res['t'], res['k_hist'], label='k(t)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Parameters')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    # ---------- evolving potential visualisation ----------
    engine.plot_evolving_potential(res, snapshots=[0, 10, 25, 40, 50])