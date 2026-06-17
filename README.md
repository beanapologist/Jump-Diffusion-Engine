# Jump-Diffusion-Engine – Stochastic Stability Analysis
[![CI](https://github.com/beanapologist/Jump-Diffusion-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/beanapologist/Jump-Diffusion-Engine/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub release](https://img.shields.io/github/v/release/beanapologist/Jump-Diffusion-Engine)](https://github.com/beanapologist/Jump-Diffusion-Engine/releases)

**Simulate, analyse, and steer noisy nonlinear systems.**

A simulation framework for systems with a Source (Λ(t)), a Medium (Δ), and a nonlinear Sink (f(Δ))
subject to continuous diffusion and discrete jump noise.

## Core Equation

`dΔ = [Λ(t) − f(Δ)] dt + σ dW + J dN`

- `Λ(t) = ε₀ + A·sin(ωt)` — the source (constant or time-varying)
- `f(Δ) = kΔ + gΔ²/(K²+Δ²)` — the nonlinear sink (linear + saturating)
- `Δ* : Λ = f(Δ), f′(Δ*) > 0` — a stable equilibrium (basin centre)

Use `jump_diffusion_engine.py` to **analyse** stochastic systems and **steer** trajectories toward stable basins:

| # | Action | Method | What it does |
|:-|:---|:---|:---|
| 1 | **Seat** into a stable basin | `seat_and_release()` | Applies a transient corrective push, then releases control. Basin strength determines how well it holds. |
| 2 | **Adapt** to volatility | `adaptive_k()` | Updates the reversion coefficient based on recent variance. |
| 3 | **Map** equilibria | `find_fixed_points()` | Finds stable (`f′>0`) and unstable fixed points for a given Λ. |
| 4 | **Measure** basin depth | `basin_depth()` | Quantifies the potential-energy barrier around each basin. |
| 5 | **Estimate** escape risk | `escape_probability()` | Empirical escape rate via Monte Carlo trials. |
| 6 | **Visualise** the stationary distribution | `stationary_density()` | Computes the Boltzmann-weighted PDF p(Δ) ∝ e^{−2V/σ²}. |

## Installation

> **Note:** PyPI publishing is not yet configured. Install directly from a GitHub release or from a local clone.

**Install from a GitHub release (end users)**

After the `v0.1.0` tag is published as a GitHub Release, install with:

```bash
pip install "jump-diffusion-engine @ https://github.com/beanapologist/Jump-Diffusion-Engine/archive/refs/tags/v0.1.0.tar.gz"
```

Replace `v0.1.0` with the tag you want. All releases are listed on the
[Releases page](https://github.com/beanapologist/Jump-Diffusion-Engine/releases).

**Editable install from source (contributors)**

```bash
git clone https://github.com/beanapologist/Jump-Diffusion-Engine.git
cd Jump-Diffusion-Engine
pip install -e .
```

**Manual dependency install**

If you prefer not to install the package, install dependencies directly:

```bash
pip install numpy scipy matplotlib
```

Then add the repository root to your Python path and import:

```python
from jump_diffusion_engine import JumpDiffusionEngine
```

## Quick Start

```python
import numpy as np
from jump_diffusion_engine import JumpDiffusionEngine

# Define a source: constant with a small oscillation
def lambda_func(t):
    return 0.5 + 0.1 * np.sin(2 * np.pi * 0.1 * t)

eng = JumpDiffusionEngine(lambda_func, sigma=0.3, jump_rate=0.05, seed=42)

# 1. Find stable equilibria
fps = eng.find_fixed_points(lambda_val=0.5)
print("Fixed points:", fps)

# 2. Simulate a few trajectories
results = eng.simulate(t_max=20.0, x0=0.0, n_realizations=3)
eng.plot_trajectories(results)

# 3. Steer into a stable basin, then release control
result = eng.seat_and_release(t_max=15.0, x0=3.0, lambda_val=0.5)
print(f"Released at step {result['release_idx']}, seated at Δ* ≈ {result['x_star']:.3f}")

# 4. Estimate escape probability from the basin
p_escape = eng.escape_probability(threshold=2.0, t_max=10.0, n_trials=200)
print(f"Empirical escape probability: {p_escape:.3f}")

# 5. Stationary density
x, p = eng.stationary_density(lambda_val=0.5)
```

## Use Cases

Use this on any system that needs to maintain a steady state while being bombarded by both constant noise and sudden, unpredictable shocks.

---

### 1. Renewable Energy Grid Management

Power grids are dissipative systems because electricity is used up as soon as it is made.

- **The Source (`Λ`):** The fluctuating energy from wind turbines or solar panels.
- **The Sink (`f`):** The battery storage and consumer demand that drains the energy.
- **The Jumps (`J`):** Sudden lightning strikes or a power plant going offline.
- **Use Case:** Using your code to ensure the grid frequency stays in the bowl (for example, `60 Hz`) and does not crash during a sudden surge.

---

### 2. Biological Homeostasis (Synthetic Biology)

In genetic engineering, scientists create circuits inside cells to produce insulin or other chemicals.

- **The Source (`Λ`):** The nutrients fed to the cell.
- **The Sink (`f`):** The metabolic rate at which the cell uses those nutrients.
- **The Jumps (`J`):** Sudden changes in temperature or pH levels.
- **Use Case:** Programming a cell to keep its internal chemical levels stable even if the environment becomes noisy or shocked.

---

### 3. Automated Financial Trading (Hedge Funds)

Markets are the definition of jump-diffusion.

- **The Source (`Λ`):** The underlying growth or drift of an asset.
- **The Sink (`f`):** The mean-reverting force (investors selling when the price is too high).
- **The Jumps (`J`):** A flash crash or a sudden news event.
- **Use Case:** A trading bot that identifies the bowl (the fair value) and does not panic-sell during a Poisson shock, knowing the sink force will pull the price back.

---

### 4. Spacecraft Orientation & Satellite Control

Satellites must point their antennas precisely at Earth while floating in noisy space.

- **The Source (`Λ`):** The thrusters or reaction wheels.
- **The Sink (`f`):** The friction of the gyroscopes or the magnetic pull of Earth.
- **The Jumps (`J`):** Tiny meteoroid impacts or solar flares.
- **Use Case:** An autopilot system that keeps the satellite centered in its orientation bowl regardless of constant vibrations or sudden bumps.

---

### 5. Supply Chain & Inventory Control

A warehouse needs to keep enough stock in the bowl without overflowing or running out.

- **The Source (`Λ`):** The incoming shipments of goods.
- **The Sink (`f`):** The customer orders draining the inventory.
- **The Jumps (`J`):** A sudden viral trend causing a massive spike in orders.
- **Use Case:** An AI manager that uses your restoring-force logic to automatically adjust order rates so the warehouse does not stay empty after a shock.

---

### 6. Climate Modeling (Resilience Thresholds)

Ecologists use these models to see whether a forest or ocean can survive climate change.

- **The Source (`Λ`):** Rainfall and sunlight.
- **The Sink (`f`):** Evaporation and consumption by animals.
- **The Jumps (`J`):** Forest fires or extreme heatwaves.
- **Use Case:** Determining the tipping point—how big a jump (`J`) it takes to push the system out of its bowl so that it can never recover.

---

### 7. AI Training & Safe Learning

When training a robot to walk, you do not want it to explore so far that it breaks itself.

- **The Source (`Λ`):** The robot's drive to move forward.
- **The Sink (`f`):** A penalty in the code that gets stronger as the robot gets closer to falling.
- **The Jumps (`J`):** A person pushing the robot or an uneven floor.
- **Use Case:** Ensuring the robot's brain always stays within a safe operational basin.

## Citation

If you use this software in your research, please cite it as below.
A Zenodo DOI will be added after this release is archived on Zenodo.

```bibtex
@software{SarahMarin_JumpDiffusionEngine_2026,
  author       = {Sarah Marin},
  title        = {JumpDiffusionEngine: A Universal Framework for Multistable Stochastic Control},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/beanapologist/Jump-Diffusion-Engine},
  note         = {Version 0.1.0 – initial release; Zenodo archiving pending}
}
```

## License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for details.
