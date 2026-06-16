import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Dict

class JumpDiffusionEngine:
    """
    Universal Jump-Diffusion Simulator for Dissipative Systems.
    
    Implements the SDE:
        dΔ = [Λ(t) − f(Δ)] dt + σ dW + J dN
    
    Plug-and-play across domains.
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
        self.jump_size_dist = jump_size_dist or (lambda: np.random.normal(0, 1.0))
        self.dt = dt
        self.rng = np.random.default_rng(seed)
    
    def simulate(self, t_max: float, x0: float = 0.0, 
                 n_realizations: int = 1, 
                 record_energy: bool = True) -> List[Dict]:
        """Run Monte Carlo realizations."""
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
                
                # Jumps
                jumps = 0.0
                if self.jump_rate > 0:
                    n_jumps = self.rng.poisson(self.jump_rate * dt)
                    for _ in range(n_jumps):
                        jumps += self.jump_size_dist()
                
                x[i] = x_prev + drift + diffusion + jumps
                
                if record_energy:
                    energy[i] = 0.5 * x[i]**2
            
            res = {'t': t, 'x': x}
            if record_energy:
                res['energy'] = energy
            results.append(res)
        
        return results
    
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
    
    def plot_trajectories(self, results: List[Dict], title: str = "Jump-Diffusion Trajectories", 
                         show_energy: bool = False):
        """Plot trajectories (and energy)."""
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
            ax2.set_ylabel('Energy')
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def basin_analysis(self, lambda_val: float, x_range: Tuple[float, float] = (-10, 10)):
        """Visualize the restoring bowl."""
        x = np.linspace(x_range[0], x_range[1], 500)
        plt.figure(figsize=(10, 6))
        plt.plot(x, self.f_func(x), 'b-', label='Sink f(Δ)', linewidth=2)
        plt.axhline(y=lambda_val, color='r', linestyle='--', label=f'Source Λ = {lambda_val:.3f}')
        plt.xlabel('State Δ')
        plt.ylabel('Rate')
        plt.title('Basin of Attraction (the Bowl)')
        plt.grid(True)
        plt.legend()
        plt.show()