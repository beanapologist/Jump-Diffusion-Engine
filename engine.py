import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Dict

class JumpDiffusionEngine:
    """
    Universal Jump-Diffusion Simulator & Stochastic Stability Framework.

    Implements the SDE:
        dΔ = [Λ(t) − f(Δ)] dt + σ dW + J dN

    Core sink: f(Δ) = k Δ + g Δ² / (K² + Δ²) — fully self-contained via attributes.
    Supports restless/adaptive k, dynamic sweeps, and coherence checks.
    """

    def __init__(self,
                 lambda_func: Callable[[float], float],
                 sigma: float = 0.5,
                 jump_rate: float = 0.05,
                 jump_size_dist: Optional[Callable] = None,
                 dt: float = 0.01,
                 seed: int = 42,
                 k: float = 0.8,
                 g: float = 0.5,
                 K: float = 2.0,
                 custom_f_func: Optional[Callable[[float], float]] = None):
        self.lambda_func = lambda_func
        self.sigma = sigma
        self.jump_rate = jump_rate
        self.jump_size_dist = jump_size_dist
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # Core parameters for the nonlinear sink
        self.k = k
        self.g = g
        self.K = K

        # Use custom f_func if provided, otherwise default to self-contained nonlinear sink
        if custom_f_func is not None:
            self.f_func = custom_f_func
            self._uses_custom_f = True
        else:
            self.f_func = self._default_sink  # now internal and uses self attrs
            self._uses_custom_f = False

    def _default_sink(self, d: float) -> float:
        """Default nonlinear sink using self attributes (refactored)."""
        linear = self.k * d
        nonlinear = self.g * (d**2) / (self.K**2 + d**2) if abs(d) > 1e-12 or self.K > 0 else 0.0
        return linear + nonlinear

    def adaptive_k(self, recent_x: np.ndarray,
                   target_reversion: Optional[float] = None,
                   alpha: float = 0.3) -> float:
        """
        Adaptive/restless k: variance-matching + EMA smoothing.
        Stronger effective reversion when system is far out (ties to z-score fade).

        NOTE: if a custom f_func was supplied, it may not read self.k, in which
        case updating self.k here has no effect on the dynamics — a silent no-op.
        We warn once so the caller isn't misled into thinking adaptation is active.
        """
        if getattr(self, '_uses_custom_f', False) and not getattr(self, '_warned_adaptive_custom', False):
            import warnings
            warnings.warn(
                "adaptive_k is updating self.k, but a custom f_func is in use and "
                "may ignore self.k — adaptation may have no effect. Pass a custom_f_func "
                "that reads self.k, or disable use_adaptive_k.",
                RuntimeWarning, stacklevel=2)
            self._warned_adaptive_custom = True

        if len(recent_x) < 10:
            return self.k

        var = np.var(recent_x)
        k_new = self.sigma**2 / (2 * (var + 1e-8))

        if target_reversion is not None:
            k_new = 0.5 * k_new + 0.5 * target_reversion

        k_new = np.clip(k_new, 0.05, 5.0)
        self.k = (1 - alpha) * self.k + alpha * k_new
        return self.k

    def simulate(self, t_max: float, x0: float = 0.0,
                 n_realizations: int = 1,
                 record_energy: bool = True,
                 lambda_val_for_energy: Optional[float] = None,
                 use_adaptive_k: bool = False,
                 adaptive_window: int = 50) -> List[Dict]:
        """Monte Carlo with optional on-the-fly adaptive k."""
        dt = self.dt
        n_steps = int(t_max / dt) + 1
        results = []

        for r in range(n_realizations):
            t = np.linspace(0, t_max, n_steps)
            x = np.zeros(n_steps)
            x[0] = x0
            energy = np.zeros(n_steps) if record_energy else None

            for i in range(1, n_steps):
                t_prev = t[i-1]
                x_prev = x[i-1]

                # Adaptive k modulation (updates self.k, which _default_sink reads)
                if use_adaptive_k and i > adaptive_window:
                    recent = x[max(0, i-adaptive_window):i]
                    self.adaptive_k(recent)

                # Drift using current f_func (now always respects self.k/g/K)
                drift = (self.lambda_func(t_prev) - self.f_func(x_prev)) * dt

                # Diffusion + Jumps (unchanged)
                dW = self.rng.normal(0, np.sqrt(dt))
                diffusion = self.sigma * dW
                jumps = 0.0
                if self.jump_rate > 0:
                    n_jumps = self.rng.poisson(self.jump_rate * dt)
                    for _ in range(n_jumps):
                        if self.jump_size_dist is None:
                            jumps += self.rng.normal(0, 1.0)
                        else:
                            jumps += self.jump_size_dist()

                x[i] = x_prev + drift + diffusion + jumps

                if record_energy:
                    if lambda_val_for_energy is not None:
                        energy[i] = self._potential_at(x[i], lambda_val_for_energy)
                    else:
                        energy[i] = 0.5 * x[i]**2

            res = {'t': t, 'x': x}
            if record_energy and energy is not None:
                res['energy'] = energy
            results.append(res)

        return results

    def _potential_at(self, x: float, lambda_val: float) -> float:
        """
        Potential at a single point: V(x) = ∫_0^x (f(Δ) − Λ) dΔ.

        Matches the convention of potential() (which integrates from the left
        edge of its grid); here we anchor V(0)=0. Uses a modest trapezoid
        quadrature from 0 to x so energy recording in simulate() is meaningful
        instead of stuck at zero. Sign of x is handled (integral reverses).
        """
        if x == 0.0:
            return 0.0
        n = max(2, int(abs(x) / max(self.dt, 1e-3)) + 1)
        n = min(n, 2000)  # cap cost per step
        grid = np.linspace(0.0, x, n)
        integrand = np.array([self.f_func(xi) for xi in grid]) - lambda_val
        return float(np.trapezoid(integrand, grid) if hasattr(np, 'trapezoid')
                     else np.trapz(integrand, grid))

    # === Analysis methods (find_fixed_points, potential, basin_depth, etc.) remain unchanged ===
    # They already call self.f_func(x) which now reliably uses self.k/g/K

    def find_fixed_points(self, lambda_val: float,
                         x_range: Tuple[float, float] = (-20, 20),
                         n_points: int = 2000) -> List[Dict]:
        """Find stable fixed points (positivity/coherence proxy)."""
        def eq(x):
            return self.f_func(x) - lambda_val

        fixed_points = []
        x_grid = np.linspace(x_range[0], x_range[1], n_points)
        y = np.array([self.f_func(xi) for xi in x_grid])  # vector-safe

        sign_changes = np.where(np.diff(np.sign(y - lambda_val)))[0]

        for idx in sign_changes:
            try:
                sol = root_scalar(eq, bracket=[x_grid[idx], x_grid[idx+1]], method='brentq')
                if sol.converged:
                    x_star = sol.root
                    h = 1e-5
                    f_prime = (self.f_func(x_star + h) - self.f_func(x_star - h)) / (2 * h)
                    stable = f_prime > 0
                    fixed_points.append({
                        'x_star': float(x_star),
                        'stable': stable,
                        'f_prime': float(f_prime),
                        'lambda_val': lambda_val
                    })
            except:
                pass

        # Deduplicate
        unique = []
        for fp in fixed_points:
            if not any(np.isclose(fp['x_star'], u['x_star'], atol=1e-4) for u in unique):
                unique.append(fp)
        return unique

    def find_folds(self, param_name: str, param_values: np.ndarray,
                   lambda_val: Optional[float] = None,
                   x_range: Tuple[float, float] = (-20, 20),
                   n_points: int = 4000) -> Dict:
        """
        Domain-agnostic bifurcation locator.

        Scans `param_name` over `param_values` and reports where the number of
        STABLE fixed points of f(Δ)=Λ changes — i.e. the folds/cusps where bowls
        are born or destroyed. Makes NO assumption about which parameter or what
        values matter: the caller supplies the axis, the engine reads the
        topology of its own potential. Restores the swept attribute afterward.

        Returns:
            {
              'param_name', 'param_values',
              'stable_counts': list[int],            # # stable fp at each value
              'folds': list of {'low','high','n_before','n_after'},
              'bistable_windows': list of (lo, hi)   # contiguous spans with >1 bowl
            }
        """
        lam_val = lambda_val if lambda_val is not None else self.lambda_func(0)
        original = getattr(self, param_name, None)

        counts = []
        for v in param_values:
            setattr(self, param_name, v)
            fps = self.find_fixed_points(lam_val, x_range, n_points)
            counts.append(sum(1 for fp in fps if fp.get('stable', False)))

        if original is not None:
            setattr(self, param_name, original)

        counts = np.asarray(counts)
        param_values = np.asarray(param_values)

        edges = np.where(np.diff(counts) != 0)[0]
        folds = [{
            'low': float(param_values[i]),
            'high': float(param_values[i + 1]),
            'n_before': int(counts[i]),
            'n_after': int(counts[i + 1]),
        } for i in edges]

        # Contiguous spans where more than one stable bowl coexists.
        windows = []
        in_win = False
        start = None
        for i, c in enumerate(counts):
            if c > 1 and not in_win:
                in_win, start = True, param_values[i]
            elif c <= 1 and in_win:
                in_win = False
                windows.append((float(start), float(param_values[i - 1])))
        if in_win:
            windows.append((float(start), float(param_values[-1])))

        return {
            'param_name': param_name,
            'param_values': param_values.tolist(),
            'stable_counts': counts.tolist(),
            'folds': folds,
            'bistable_windows': windows,
        }

    def potential(self, x_grid: np.ndarray, lambda_val: float) -> np.ndarray:
        """V(Δ) = ∫ (f - Λ) dΔ"""
        y = np.array([self.f_func(xi) for xi in x_grid]) - lambda_val
        dx = np.diff(x_grid)
        V = np.zeros_like(x_grid)
        V[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
        return V

    def basin_depth(self, lambda_val: float, x_range: Tuple[float, float] = (-15, 15), n_points: int = 5000) -> List[Dict]:
        """Basin resilience."""
        x = np.linspace(x_range[0], x_range[1], n_points)
        V = self.potential(x, lambda_val)
        # ... (rest of basin_depth implementation unchanged - minima/maxima logic)
        minima = []
        maxima = []
        for i in range(1, len(x)-1):
            if V[i] < V[i-1] and V[i] < V[i+1]:
                minima.append(i)
            if V[i] > V[i-1] and V[i] > V[i+1]:
                maxima.append(i)

        basins = []
        for m_idx in minima:
            m = x[m_idx]
            left_barrier = right_barrier = None
            for b_idx in maxima:
                if b_idx < m_idx:
                    left_barrier = b_idx
                if b_idx > m_idx:
                    right_barrier = b_idx
                    break
            depths = []
            if left_barrier is not None:
                depths.append(V[left_barrier] - V[m_idx])
            if right_barrier is not None:
                depths.append(V[right_barrier] - V[m_idx])
            if depths:
                basins.append({
                    "x_star": float(m),
                    "depth": float(min(depths)),
                    "lambda_val": lambda_val
                })
        return basins

    def escape_probability(self, threshold: float, t_max: float, x0: Optional[float] = None,
                           x_star: Optional[float] = None, n_trials: int = 500) -> float:
        if x_star is None:
            fps = self.find_fixed_points(self.lambda_func(0))
            x_star = fps[0]['x_star'] if fps else 0.0
        # PATCH 1: Start trials AT the equilibrium unless x0 explicitly given,
        # so escape is measured relative to the bowl we're testing.
        if x0 is None:
            x0 = x_star
        escapes = 0
        for _ in range(n_trials):
            res = self.simulate(t_max=t_max, x0=x0, n_realizations=1, record_energy=False)[0]
            if np.any(np.abs(res["x"] - x_star) > threshold):
                escapes += 1
        return escapes / n_trials

    def identify_boundary(self, lambda_val: Optional[float] = None,
                          x_range: Tuple[float, float] = (-20, 20),
                          n_points: int = 4000,
                          target_x: Optional[float] = None) -> Dict:
        """
        STEP 1 — find the boundary of the live region from f alone.

        The bowl is the stable fixed point Δ*; its walls are the adjacent
        unstable fixed points (ridges) on either side. Between those ridges is
        the basin Δ may roam without escaping. No assumed values: read from the
        potential of whatever f is loaded.

        Returns:
            {
              'x_star'        : chosen stable equilibrium (the seat),
              'left_wall'     : nearest unstable fp below x_star (or None = open),
              'right_wall'    : nearest unstable fp above x_star (or None = open),
              'half_width'    : distance to the nearer wall (the bound radius),
              'sigma_envelope': sqrt(sigma^2 / (2 f'(x_star))) — natural OU spread,
              'all_fixed'     : every fixed point found
            }
        """
        lam = lambda_val if lambda_val is not None else self.lambda_func(0)
        fps = self.find_fixed_points(lam, x_range, n_points)
        if not fps:
            return {'x_star': None, 'left_wall': None, 'right_wall': None,
                    'half_width': None, 'sigma_envelope': None, 'all_fixed': []}

        stable = [fp for fp in fps if fp.get('stable', False)]
        unstable = [fp for fp in fps if not fp.get('stable', False)]

        # choose the seat: nearest stable fp to target_x, else the deepest-held one
        if target_x is not None and stable:
            x_star = min(stable, key=lambda fp: abs(fp['x_star'] - target_x))
        elif stable:
            # prefer the stable fp with the steepest restoring slope (tightest bowl)
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
        """
        STEP 2 — drive Δ into the stable bowl, then RELEASE.

        Identifies the boundary, applies a transient corrective push only while
        Δ is outside the settle band around Δ*. Once Δ has dwelled inside that
        band for `dwell` consecutive steps, control switches OFF for good — the
        bowl holds Δ unaided thereafter. The whole point: once seated with exact
        inputs, you never touch it again. Returns the trace plus the release
        index so you can verify control did nothing after release.

        settle_tol defaults to the natural OU envelope (the breathing room the
        cloud is *supposed* to have); the band is not a clamp, it's the basin.
        """
        b = self.identify_boundary(lambda_val, target_x=target_x)
        xs = b['x_star']
        if xs is None:
            raise ValueError("No stable bowl found for these inputs; nothing to seat into.")

        lam = lambda_val if lambda_val is not None else self.lambda_func(0)
        if settle_tol is None:
            settle_tol = (b['sigma_envelope'] or 0.5) * 1.5  # ~1.5σ band = the live cloud

        dt = self.dt
        n_steps = int(t_max / dt) + 1
        x = np.zeros(n_steps)
        x[0] = x0
        control = np.zeros(n_steps)     # corrective drift actually applied
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
                    # transient push toward the seat, only while outside the band
                    if abs(xp - xs) > settle_tol:
                        u = -gain * (xp - xs)
            # natural dynamics: drift from f + the (possibly zero) control + noise
            drift = (self.lambda_func((i-1)*dt) - self.f_func(xp) + u) * dt
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
        # Shift V by its minimum before exponentiation to avoid numerical overflow.
        # The constant shift cancels exactly in the normalisation.
        V_shifted = V - np.min(V)
        p = np.exp(-2 * V_shifted / (self.sigma ** 2 + 1e-12))
        norm = np.trapezoid(p, x) if hasattr(np, 'trapezoid') else np.trapz(p, x)
        p /= (norm + 1e-12)
        return x, p

    def parameter_sweep(self, param_name: str, param_values: np.ndarray,
                        lambda_val: Optional[float] = None,
                        metrics: Optional[List[str]] = None,
                        x_range: Tuple[float, float] = (-15, 15),
                        n_points: int = 2000, plot: bool = True) -> Dict:
        """Dynamic sweep — now cleaner since f_func always reads self attrs."""
        if metrics is None:
            metrics = ['fixed_points', 'basin_depth', 'escape_prob', 'stationary_mass']

        results = []
        original_k, original_g, original_K = self.k, self.g, self.K

        for val in param_values:
            if param_name == 'k':
                self.k = val
            elif param_name == 'g':
                self.g = val
            elif param_name == 'K':
                self.K = val
            elif hasattr(self, param_name):
                setattr(self, param_name, val)

            sweep_lambda = lambda_val if lambda_val is not None else self.lambda_func(0)
            sweep_res = {'param_value': val, 'param_name': param_name}

            try:
                fps = self.find_fixed_points(sweep_lambda, x_range, n_points)
                stable_fps = [fp for fp in fps if fp.get('stable', False)]
                sweep_res['num_stable'] = len(stable_fps)
                sweep_res['positivity_ok'] = len(stable_fps) > 0

                if 'basin_depth' in metrics:
                    basins = self.basin_depth(sweep_lambda, x_range)
                    sweep_res['avg_basin_depth'] = np.mean([b['depth'] for b in basins]) if basins else 0.0

                if 'escape_prob' in metrics:
                    # PATCH 1 (coupled): derive x_star from the SAME lambda this
                    # iteration uses, instead of letting escape_probability re-derive
                    # one from lambda_func(0) at a mismatched reference point.
                    x_star_esc = stable_fps[0]['x_star'] if stable_fps else None
                    esc = self.escape_probability(threshold=5.0, t_max=100.0,
                                                  x_star=x_star_esc, n_trials=100)
                    sweep_res['escape_prob'] = esc

                if 'stationary_mass' in metrics:
                    # PATCH 2: p is a normalized density; mass = ∫ p dx, not Σ p.
                    # Σ p scales with n_points and is meaningless. Restrict to the
                    # bowl (±threshold around the stable x*) for a real probability.
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

            # Restore originals
            self.k, self.g, self.K = original_k, original_g, original_K

        if plot:
            self._plot_sweep(results, param_name, metrics)

        return {'sweep': results, 'param_name': param_name}

    def _plot_sweep(self, results: List[Dict], param_name: str, metrics: List[str]):
        # (unchanged plotting logic)
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
        """Plot simulated trajectories from simulate().

        Parameters
        ----------
        results : list of dicts returned by simulate()
        title   : figure title
        show_energy : if True and energy was recorded, add a second panel
        """
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

# Quick test / usage
if __name__ == "__main__":
    def lambda_func(t):
        return 0.5 + 0.3 * np.sin(2 * np.pi * 0.1 * t)

    engine = JumpDiffusionEngine(lambda_func, k=0.8, g=0.5, K=2.0)  # uses default nonlinear sink

    # Adaptive + sweep ready
    sweep = engine.parameter_sweep('k', np.linspace(0.1, 2.0, 15), plot=True)
    print("Sweep complete. Positivity OK regions:", [r for r in sweep['sweep'] if r.get('positivity_ok')])
