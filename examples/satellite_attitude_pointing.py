"""
Satellite attitude-pointing demo on JumpDiffusionEngine.

Δ        = pointing error (deg) off boresight (target Δ*=0, centered on Earth)
Λ(t)     = thruster / reaction-wheel torque command  (the Source)
f(Δ)     = restoring sink: gyroscopic/structural damping + a soft magnetic-torque
           term that grows with |Δ| but saturates -- must be ODD in Δ since the
           satellite has no preferred drift direction. f(Δ) = k*Δ + g*Δ**3/(K**2+Δ**2)
           (cubic-over-quadratic instead of the engine's default square-over-square,
           specifically to restore odd symmetry: f(-Δ) = -f(Δ)).
J dN     = meteoroid impacts / solar-flare torque kicks (Poisson jumps)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jump_diffusion_engine import JumpDiffusionEngine

k_gyro = 1.2     # gyroscopic/structural damping rate
g_mag  = 0.6     # magnetic-torque restoring strength
K_sat  = 3.0     # saturation scale (deg)

def f_attitude(d):
    return k_gyro * d + g_mag * d**3 / (K_sat**2 + d**2)

# sanity check: odd symmetry
test = np.linspace(-5, 5, 11)
print("symmetry check  f(Δ) vs -f(-Δ):")
for x in test:
    print(f"  Δ={x:6.2f}  f(Δ)={f_attitude(x):8.4f}   -f(-Δ)={-f_attitude(-x):8.4f}")

# thruster command: small constant bias + slow sinusoidal disturbance torque
Lambda = lambda t: 0.05 * np.sin(2*np.pi*0.02*t)

# meteoroid/solar-flare jumps: rare, small-to-moderate kicks
jump_dist = lambda: np.random.normal(0, 0.3)

eng = JumpDiffusionEngine(Lambda, sigma=0.15, jump_rate=0.01, dt=0.01, seed=11,
                          jump_size_dist=jump_dist, custom_f_func=f_attitude)

b = eng.identify_boundary(target_x=0.0)
print("\nboundary / bowl around target attitude:")
for kk, vv in b.items():
    if kk != 'all_fixed':
        print(f"  {kk}: {vv}")

res = eng.seat_and_release(t_max=200.0, x0=4.0, target_x=0.0, dwell=300, gain=3.0)
print(f"\nseat_and_release: released={res['released']}  release_idx={res['release_idx']}  "
      f"x_star={res['x_star']:.3f}  settle_tol={res['settle_tol']:.3f}")
print(f"control activity AFTER release (should be ~0): {res['control_after_release']}")
hw = b['half_width']
hw_str = f"{hw:.3f}" if hw is not None else "None (single global bowl, no opposing wall in range)"
print(f"contained after release within half_width={hw_str}: {res['contained_after_release']}")

esc = eng.escape_probability(threshold=3.0, t_max=300.0, x_star=0.0, n_trials=1000)
print(f"\nescape probability (|Δ|>3deg within 300 time units, from rest at boresight): {esc:.3f}")

# plot
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
t = res['t']; x = res['x']; ctrl = res['control']
ax[0].plot(t, x, lw=0.8, color='steelblue', label='pointing error Δ (deg)')
ax[0].plot(t, ctrl, lw=0.8, color='orange', alpha=0.7, label='thruster correction')
ax[0].axhline(0, ls=':', c='green', label='boresight target')
if res['release_idx']:
    ax[0].axvline(t[res['release_idx']], ls='--', c='k', lw=0.8)
    ax[0].annotate('AUTOPILOT OFF', (t[res['release_idx']], 3.5), fontsize=8, ha='left')
ax[0].set(title="Satellite attitude: seat then release", xlabel="time", ylabel="deg / torque")
ax[0].legend(fontsize=8)

dg = np.linspace(-6, 6, 300)
ax[1].plot(dg, [f_attitude(xx) for xx in dg], 'k', lw=1.5)
ax[1].axhline(0, c='gray', lw=0.6)
ax[1].axvline(0, c='gray', lw=0.6)
ax[1].set(title="Restoring sink f(Δ) — odd-symmetric", xlabel="pointing error Δ (deg)", ylabel="f(Δ)")

fig.tight_layout()
