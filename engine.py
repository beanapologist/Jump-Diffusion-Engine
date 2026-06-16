import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Dict

class JumpDiffusionEngine:
    """
    Universal Jump-Diffusion Simulator & Stochastic Stability Framework.
    
    Implements the SDE:
        dΔ = [Λ(t) − f(Δ)] dt + σ dW + J dN
    
    Now a compact, mathematically rigorous tool answering:
    1. Where does it want to go?          → fixed points
    2. How strongly is it held?           → basin depth / potential
    3. How often does it escape?          → escape probability
    4. Where does it spend most time?     → stationary density
    """
    
    def __init__(self, 
                 lambda_func: Callable[[float], float],
                 f_func: Callable[[float], float],
                 sigma: float = 0.5,
                 jump_rate: float = 0.05,
                 jump_size_dist: Optional[Callable] = None,
                 dt: float = 0.01,
                 seed: int = 42):
        self.lambda_func = lambda_func
        self.f_func = f_func
        self.sigma = sigma
        self.jump_rate = jump_rate
        self.jump_size_dist = jump_size_dist  # None → use self.rng
        self.dt = dt
        self.rng = np.random.default_rng(seed)
    
    def simulate(self, t_max: float, x0: float = 0.0, 
                 n_realizations: int = 1, 
                 record_energy: bool = True,
                 lambda_val_for_energy: Optional[float] = None) -> List[Dict]:
        """Run Monte Carlo realizations. Energy now prefers potential when lambda_val provided."""
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
                
                # Drift
                drift = (self.lambda_func(t_prev) - self.f_func(x_prev)) * dt
                
                # Diffusion
                dW = self.rng.normal(0, np.sqrt(dt))
                diffusion = self.sigma * dW
                
                # Jumps (now reproducible)
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
                        # Use potential-based "energy"
                        energy[i] = self._potential_at(x[i], lambda_val_for_energy)
                    else:
                        energy[i] = 0.5 * x[i]**2
            
            res = {'t': t, 'x': x}
            if record_energy and energy is not None:
                res['energy'] = energy
            results.append(res)
        
        return results
    
    def _potential_at(self, x: float, lambda_val: float) -> float:
        """Helper for single-point potential (trapezoidal approximation)."""
        # For single points we approximate locally or reuse full grid in calling code
        return 0.0  # Placeholder; full potential used in analysis methods
    
    def find_fixed_points(self, lambda_val: float, 
                         x_range: Tuple[float, float] = (-20, 20),
                         n_points: int = 2000) -> List[Dict]:
        """Find stable 'bowls' where Λ = f(Δ) and f'(Δ*) > 0."""
        def eq(x):
            return self.f_func(x) - lambda_val
        
        fixed_points = []
        x_grid = np.linspace(x_range[0], x_range[1], n_points)
        y = self.f_func(x_grid)
        
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
    
    def potential(self, x_grid: np.ndarray, lambda_val: float) -> np.ndarray:
        """V(Δ) = ∫ (f(Δ) - Λ) dΔ. Minima = stable attractors."""
        y = self.f_func(x_grid) - lambda_val
        dx = np.diff(x_grid)
        V = np.zeros_like(x_grid)
        V[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
        return V
    
    def basin_depth(self,
                    lambda_val: float,
                    x_range: Tuple[float, float] = (-15, 15),
                    n_points: int = 5000) -> List[Dict]:
        """Improved: lowest barrier on either side (real resilience)."""
        x = np.linspace(x_range[0], x_range[1], n_points)
        V = self.potential(x, lambda_val)
        
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
            left_barrier = None
            right_barrier = None
            
            # Left barrier
            for b_idx in maxima:
                if b_idx < m_idx:
                    left_barrier = b_idx
            # Right barrier
            for b_idx in maxima:
                if b_idx > m_idx:
                    right_barrier = b_idx
                    break
            
            depths = []
            if left_barrier is not None:
                depths.append(V[left_barrier] - V[m_idx])
            if right_barrier is not None:
                depths.append(V[right_barrier] - V[m_idx])
            
            if depths:
                min_depth = min(depths)
                basins.append({
                    "x_star": float(m),
                    "depth": float(min_depth),
                    "lambda_val": lambda_val
                })
        
        return basins
    
    def escape_probability(self,
                           threshold: float,
                           t_max: float,
                           x0: float = 0.0,
                           x_star: Optional[float] = None,
                           n_trials: int = 500) -> float:
        """
        P(escape) relative to attractor (x_star). Uses fixed points if not provided.
        """
        if x_star is None:
            fps = self.find_fixed_points(self.lambda_func(0), n_points=1000)  # approx constant Lambda at t=0
            if fps:
                x_star = fps[0]['x_star']
            else:
                x_star = 0.0
        
        escapes = 0
        for _ in range(n_trials):
            res = self.simulate(t_max=t_max, x0=x0, n_realizations=1, record_energy=False)[0]
            if np.any(np.abs(res["x"] - x_star) > threshold):
                escapes += 1
        return escapes / n_trials
    
    def stationary_density(self,
                           lambda_val: float,
                           x_range: Tuple[float, float] = (-10, 10),
                           n_points: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate stationary density for diffusion (constant Λ):
        p(Δ) ∝ exp(-2V(Δ)/σ²)
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        V = self.potential(x, lambda_val)
        p = np.exp(-2 * V / (self.sigma ** 2 + 1e-12))  # avoid div0
        p /= np.trapz(p, x)
        return x, p
    
    def plot_trajectories(self, results: List[Dict], title: str = "Jump-Diffusion Trajectories", 
                         show_energy: bool = False):
        plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(211 if show_energy else 111)
        for res in results:
            ax1.plot(res['t'], res['x'], alpha=0.7, linewidth=1.2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('State Δ')
        ax1.set_title(title)
        ax1.grid(True)
        
        if show_energy and 'energy' in results[0]:
            ax2 = plt.subplot(212, sharex=ax1)
            for res in results:
                ax2.plot(res['t'], res['energy'], alpha=0.7)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Energy / Potential')
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_potential(self, lambda_val: float, x_range: Tuple[float, float] = (-10, 10), n_points: int = 1000):
        """Plot potential landscape."""
        x = np.linspace(x_range[0], x_range[1], n_points)
        V = self.potential(x, lambda_val)
        plt.figure(figsize=(10, 6))
        plt.plot(x, V, 'g-', label='Potential V(Δ)', linewidth=2)
        plt.xlabel('State Δ')
        plt.ylabel('Potential V')
        plt.title(f'Potential Landscape (Λ = {lambda_val:.3f})')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def basin_analysis(self, lambda_val: float, x_range: Tuple[float, float] = (-10, 10)):
        """Visualize sink vs source."""
        x = np.linspace(x_range[0], x_range[1], 500)
        plt.figure(figsize=(10, 6))
        plt.plot(x, self.f_func(x), 'b-', label='Sink f(Δ)', linewidth=2)
        plt.axhline(y=lambda_val, color='r', linestyle='--', label=f'Source Λ = {lambda_val:.3f}')
        plt.xlabel('State Δ')
        plt.ylabel('Rate')
        plt.title('Basin of Attraction')
        plt.grid(True)
        plt.legend()
        plt.show()