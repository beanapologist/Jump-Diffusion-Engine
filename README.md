# Jump-Diffusion-Engine – Perfectly Balanced Stochastic Control
**One set of inputs. Continuous, self-regulating outputs.**

A universal stability port for any dynamic system with a Source (Λ(t)), a Medium (Δ), and a Sink (f(Δ)).

## Core Equation

`dΔ = [Λ(t) − f(Δ)] dt + σ dW + J dN`

- `Λ(t) = ε₀ + A·sin(ωt)` — the source, steady and singing
- `f(Δ) = kΔ + gΔ²/(K²+Δ²)` — the sink, linear then saturating
- `Δ* : Λ = f(Δ), f′(Δ*) > 0` — the bowl, the stable held place

Use `engine.py` to **force** any stochastic system into its stable basin, then **verify** its resilience:

| # | Action | Method | What it does |
|:-|:---|:---|:---|
| 1 | **FORCE** it into the bowl | `seat_and_release()` | Applies a transient push, then **releases control to zero**. The basin holds it forever. |
| 2 | **BREATHE** with the noise | `adaptive_k()` | Dynamically stiffens when calm, relaxes when volatile. Prevents numerical blow-up while maximizing rejection. |
| 3 | **MAP** where the bowls are | `find_fixed_points()` | Finds all stable (`f′>0`) and unstable equilibria before you force it. |
| 4 | **MEASURE** how deep the bowl is | `basin_depth()` | Quantifies the energy barrier—how hard you'd have to push to knock it out. |
| 5 | **PREDICT** escape risk | `escape_probability()` | Empirical escape rate over Monte Carlo trials. Should be near-zero after seating. |
| 6 | **VISUALIZE** the long-term cloud | `stationary_density()` | The normalized PDF \( p(\Delta) \propto e^{-2V/\sigma^2} \)—where it lives once forced. |

## Installation

This project requires Python 3 and the libraries used by `engine.py`:

- `numpy`
- `scipy`
- `matplotlib`

Install them with pip:

```bash
pip install numpy scipy matplotlib
```

Then import the engine in your Python code:

```python
from engine import JumpDiffusionEngine
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

## License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for details.
